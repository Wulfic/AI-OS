"""Brain registry core - main BrainRegistry dataclass with brain creation and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import time

from aios.core.brains.protocol import Brain
from aios.core.brains.numpy_brain import NumpyMLPBrain
from aios.core.brains.actv1_brain import ACTv1Brain
from aios.core.brains import registry_storage
from aios.core.brains import registry_stats
from aios.core.brains import registry_management

try:
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover - fallback for bootstrap
    system_paths = None


def _legacy_repo_brains_root() -> str:
    return str((Path(__file__).resolve().parents[3] / "artifacts" / "brains").resolve())


@dataclass
class BrainRegistry:
    """Keeps track of sub-brains and supports dynamic creation under a global storage budget."""

    brains: Dict[str, Brain] = field(default_factory=dict)
    total_storage_limit_mb: Optional[float] = None
    storage_limit_mb_by_modality: Dict[str, float] = field(default_factory=dict)
    store_dir: Optional[str] = field(default_factory=lambda: str(system_paths.get_brains_root()) if system_paths else _legacy_repo_brains_root())
    # usage stats
    usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pinned: set[str] = field(default_factory=set)
    masters: set[str] = field(default_factory=set)
    parent: Dict[str, Optional[str]] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store_dir:
            self.store_dir = self._normalize_store_dir(self.store_dir)

    def __setattr__(self, key, value):
        if key == "store_dir" and value is not None:
            value = self._normalize_store_dir(value)
        super().__setattr__(key, value)

    @staticmethod
    def _normalize_store_dir(value: str | None) -> str:
        if not value:
            if system_paths is not None:
                return str(system_paths.get_brains_root())
            return _legacy_repo_brains_root()
        path = Path(value)
        if path.is_absolute():
            return str(path)
        if system_paths is not None:
            try:
                return str(system_paths.resolve_artifact_path(path))
            except Exception:
                pass
        return os.path.abspath(path)

    def _used_bytes(self) -> int:
        return sum(max(0, int(b.size_bytes())) for b in self.brains.values())

    def _within_budget(self, add_bytes: int) -> bool:
        if self.total_storage_limit_mb is None or self.total_storage_limit_mb <= 0:
            return True
        limit = int(self.total_storage_limit_mb * 1024 * 1024)
        return (self._used_bytes() + max(0, int(add_bytes))) <= limit

    def _within_modality_budget(self, modalities: List[str], add_bytes: int) -> bool:
        if not self.storage_limit_mb_by_modality:
            return True
        add_b = max(0, int(add_bytes))
        # Use first modality as the key for caps (simple policy)
        key = str(modalities[0]).strip().lower() if modalities else None
        if not key or key not in self.storage_limit_mb_by_modality:
            return True
        cap_mb = float(self.storage_limit_mb_by_modality.get(key, 0) or 0)
        if cap_mb <= 0:
            return True
        cap_bytes = int(cap_mb * 1024 * 1024)
        # sum bytes of brains matching this modality
        used = 0
        for n, b in self.brains.items():
            mods = (self.usage.get(n, {}).get("modalities") or [])
            if mods and str(mods[0]).strip().lower() == key:
                used += max(0, int(b.size_bytes()))
        return (used + add_b) <= cap_bytes

    def get(self, name: str) -> Optional[Brain]:
        b = self.brains.get(name)
        if b is not None:
            return b
        return registry_storage.try_load_offloaded(self, name)

    def list(self) -> List[str]:
        """List all brains, including loaded brains and offloaded ACTV1 brain bundles."""
        names = set(self.brains.keys())
        # Also include pinned and master brains even if not loaded
        names.update(self.pinned)
        names.update(self.masters)
        # Scan for ACTV1 brain bundles on disk
        try:
            if self.store_dir:
                actv1_base = os.path.join(self.store_dir, "actv1")
                if os.path.isdir(actv1_base):
                    for entry in os.listdir(actv1_base):
                        entry_path = os.path.join(actv1_base, entry)
                        if os.path.isdir(entry_path):
                            # Check if it looks like a brain bundle (has brain.json or actv1_student.safetensors)
                            has_brain = os.path.exists(os.path.join(entry_path, "brain.json"))
                            has_model = os.path.exists(os.path.join(entry_path, "actv1_student.safetensors"))
                            if has_brain or has_model:
                                names.add(entry)
        except Exception:
            pass
        return sorted(list(names))

    def record_use(self, name: str, modalities: Optional[List[str]] = None) -> None:
        now = time.time()
        meta = self.usage.get(name) or {}
        meta.setdefault("created_at", now)
        meta["last_used"] = now
        meta["hits"] = int(meta.get("hits", 0)) + 1
        if modalities is not None:
            meta.setdefault("modalities", list(modalities))
        self.usage[name] = meta

    def record_training_steps(self, name: str, steps: int) -> None:
        """Record training steps completed for a brain model.
        
        Args:
            name: Brain name
            steps: Number of training steps completed in this session
        """
        meta = self.usage.get(name) or {}
        current_steps = int(meta.get("training_steps", 0))
        meta["training_steps"] = current_steps + int(steps)
        meta["last_trained"] = time.time()
        self.usage[name] = meta

    def stats(self) -> Dict[str, Any]:
        """Compute comprehensive stats for all brains in registry."""
        return registry_stats.compute_registry_stats(self)

    def create_numpy_mlp(self, name: str, modalities: List[str], cfg_overrides: Optional[Dict[str, Any]] = None) -> Optional[Brain]:
        from aios.core.train import TrainConfig

        cfg = TrainConfig()
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        brain = NumpyMLPBrain(name=name, modalities=list(modalities), cfg=cfg)
        if not self._within_budget(brain.size_bytes()):
            return None
        if not self._within_modality_budget(modalities, brain.size_bytes()):
            return None
        self.brains[name] = brain
        self.record_use(name, modalities)
        return brain

    def create_actv1(
        self,
        name: str,
        modalities: List[str],
        checkpoint_path: str,
        brain_config_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        inference_device: Optional[str] = None,
    ) -> Optional[Brain]:
        """Create an ACTv1 brain from a trained checkpoint.
        
        Args:
            name: Brain name
            modalities: List of modalities this brain handles
            checkpoint_path: Path to actv1_student.safetensors
            brain_config_path: Optional path to brain.json (auto-detected if None)
            max_seq_len: Optional max sequence length override
            inference_device: Optional specific device for inference
            
        Returns:
            ACTv1Brain instance or None if budget exceeded
        """
        brain = ACTv1Brain(
            name=name,
            modalities=list(modalities),
            checkpoint_path=checkpoint_path,
            brain_config_path=brain_config_path,
            max_seq_len=max_seq_len,
            inference_device=inference_device,
        )
        # Respect storage budgets
        if not self._within_budget(brain.size_bytes()):
            return None
        if not self._within_modality_budget(modalities, brain.size_bytes()):
            return None
        self.brains[name] = brain
        self.record_use(name, modalities)
        # Establish parent/child relations if a master exists for this modality
        try:
            # Choose a canonical parent: the first master for the same modality prefix
            parent_name = next((m for m in sorted(self.masters) if (self.usage.get(m, {}).get("modalities") or []) == list(modalities)), None)
            self.parent[name] = parent_name
            if parent_name:
                self.children.setdefault(parent_name, [])
                if name not in self.children[parent_name]:
                    self.children[parent_name].append(name)
        except Exception:
            pass
        return brain

    # --- Offload/Restore support ---
    def offload(self, name: str) -> bool:
        """Offload brain to disk to free memory."""
        return registry_storage.offload_brain(self, name)

    # --- Pinning helpers (persisted in store_dir if set) ---
    def load_pinned(self) -> bool:
        """Load pinned brains list from disk."""
        return registry_management.load_pinned_brains(self)

    def load_masters(self) -> bool:
        """Load master brains list from disk."""
        return registry_management.load_master_brains(self)

    def save_pinned(self) -> bool:
        """Save pinned brains list to disk."""
        return registry_management.save_pinned_brains(self)

    def save_masters(self) -> bool:
        """Save master brains list to disk."""
        return registry_management.save_master_brains(self)

    def pin(self, name: str) -> None:
        self.pinned.add(name)
        self.save_pinned()

    def unpin(self, name: str) -> None:
        # Do not allow unpinning a master brain
        if name in self.masters:
            return
        if name in self.pinned:
            self.pinned.remove(name)
            self.save_pinned()

    def mark_master(self, name: str) -> None:
        self.masters.add(name)
        # Masters are always pinned
        self.pinned.add(name)
        self.save_masters()
        self.save_pinned()

    def unmark_master(self, name: str) -> None:
        """Remove master flag (does not unpin)."""
        if name in self.masters:
            self.masters.remove(name)
            self.save_masters()

    def set_parent(self, child: str, parent: Optional[str]) -> None:
        """Set or clear the parent of a brain; updates children map accordingly."""
        registry_management.set_brain_parent(self, child, parent)

    def clear_parent(self, child: str) -> None:
        self.set_parent(child, None)

    def prune(self, target_mb: Optional[float] = None, offload: bool = False) -> List[str]:
        """Evict brains (LRU) until within target_mb or global limit. Returns evicted names.

        If offload=True and store_dir is configured, checkpoints brains before eviction."""
        evicted: List[str] = []
        def over_limit() -> bool:
            if target_mb is not None and target_mb > 0:
                return self._used_bytes() > int(target_mb * 1024 * 1024)
            return not self._within_budget(0)
        if not over_limit():
            return evicted
        # Sort by last_used ascending (oldest first)
        order = sorted(self.brains.keys(), key=lambda n: float(self.usage.get(n, {}).get("last_used", 0)))
        for n in order:
            if not over_limit():
                break
            # Skip pinned brains
            if n in self.pinned:
                continue
            try:
                if offload and self.store_dir:
                    self.offload(n)
                else:
                    del self.brains[n]
                evicted.append(n)
            except Exception:
                pass
        return evicted

    # --- Management helpers: rename, master/parent relations ---
    def rename(self, old: str, new: str) -> bool:
        """Rename a brain in memory and on disk (offloaded files) when present."""
        return registry_management.rename_brain(self, old, new)
