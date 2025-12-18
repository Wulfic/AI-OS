"""Router - Simple router that picks sub-brains by modality and task hash."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
from pathlib import Path
import hashlib
import json
import logging
import os

from aios.core.brains.registry_core import BrainRegistry

logger = logging.getLogger(__name__)


@dataclass
class Router:
    """Simple router that picks sub-brains by modality and task hash. Creates on-demand."""

    registry: BrainRegistry
    default_modalities: List[str] = field(default_factory=lambda: ["text"])
    brain_prefix: str = "brain"
    create_cfg: Dict[str, Any] = field(default_factory=dict)
    modality_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategy: str = "hash"  # 'hash' | 'round_robin'
    expert_registry_path: Optional[str] = None
    _rr_idx: Dict[str, int] = field(default_factory=dict)
    expert_registry: Optional[Any] = field(default=None, init=False, repr=False)
    _active_goal_ids: Set[str] = field(default_factory=set, init=False, repr=False)
    _loaded_expert_ids: Set[str] = field(default_factory=set, init=False, repr=False)
    inference_device_getter: Optional[Callable[[], Optional[str]]] = field(default=None, repr=False)

    def __post_init__(self):
        """Load expert registry if path provided."""
        if self.expert_registry_path:
            try:
                registry_path = Path(self.expert_registry_path)
                if registry_path.exists():
                    # Lazy import to avoid circular dependencies
                    from aios.core.hrm_models.expert_metadata import ExpertRegistry
                    self.expert_registry = ExpertRegistry.load(str(registry_path))
                else:
                    # Create empty registry
                    from aios.core.hrm_models.expert_metadata import ExpertRegistry
                    self.expert_registry = ExpertRegistry()
                    # Ensure directory exists
                    registry_path.parent.mkdir(parents=True, exist_ok=True)
                    self.expert_registry.save(str(registry_path))
            except Exception:
                # Log but don't crash - expert registry is optional
                from aios.core.hrm_models.expert_metadata import ExpertRegistry
                self.expert_registry = ExpertRegistry()

    def _brain_name_for(self, modalities: Iterable[str], payload: Any) -> str:
        # Stable but cheap name based on modalities and a hash of a payload sketch
        mods = ",".join(sorted({str(m).strip().lower() for m in modalities if str(m).strip()})) or ",".join(self.default_modalities)
        h = hashlib.sha1(str(type(payload)).encode("utf-8")).hexdigest()[:8]
        return f"{self.brain_prefix}-{mods}-{h}"

    def _select_existing_rr(self, modalities: List[str]) -> Optional[str]:
        mods_key = ",".join(sorted({str(m).strip().lower() for m in modalities if str(m).strip()})) or ",".join(self.default_modalities)
        names = [n for n in self.registry.list() if (self.registry.usage.get(n, {}).get("modalities") or []) == list(modalities)]
        if not names:
            return None
        i = int(self._rr_idx.get(mods_key, 0)) % len(names)
        self._rr_idx[mods_key] = i + 1
        return names[i]

    def handle(self, task: Dict[str, Any]) -> Dict[str, Any]:
        modalities = task.get("modalities") or self.default_modalities
        payload = task.get("payload")
        task_options = task.get("options") if isinstance(task.get("options"), dict) else {}
        strict_master = bool(task.get("strict_master")) or (task_options.get("allow_fallback") is False)
        
        # Check if there's a master brain for these modalities - use it directly if available.
        # Prioritise recently used masters (descending last_used) so the brain the user just loaded
        # handles requests ahead of legacy entries that might be placeholders.
        matching_masters: List[str] = []
        master_errors: List[str] = []
        if self.registry.masters:
            def _master_sort_key(name: str) -> tuple[float, str]:
                usage = self.registry.usage.get(name, {})
                last_used = float(usage.get("last_used", 0.0) or 0.0)
                # Negative for descending and name for deterministic ordering
                return (-last_used, name)

            for m in sorted(self.registry.masters, key=_master_sort_key):
                # Try to get or load the master brain to check its modalities
                test_brain = self.registry.get(m)
                if test_brain is None:
                    # Master not loaded and can't be loaded - skip it
                    continue

                # Get modalities from the loaded brain or usage record
                if hasattr(test_brain, "modalities"):
                    m_mods = getattr(test_brain, "modalities")
                else:
                    m_mods = self.registry.usage.get(m, {}).get("modalities", [])

                if m_mods == list(modalities):
                    matching_masters.append(m)

        # Try each matching master in priority order until one succeeds
        for master_name in matching_masters:
            load_error: Optional[str] = None
            brain = self.registry.get(master_name)
            if brain is None:
                # Try to load master from disk (ACTv1 bundle)
                try:
                    if self.registry.store_dir:
                        brain_dir = os.path.join(self.registry.store_dir, "actv1", master_name)
                        checkpoint_path = os.path.join(brain_dir, "actv1_student.safetensors")
                        brain_json_path = os.path.join(brain_dir, "brain.json")

                        if os.path.exists(checkpoint_path) and os.path.exists(brain_json_path):
                            # Read brain.json to get configuration
                            with open(brain_json_path, "r", encoding="utf-8") as f:
                                brain_data = json.load(f)

                            # Get max_seq_len from brain.json (this is the trained context length)
                            trained_max_seq_len = brain_data.get("max_seq_len", 2048)

                            # Create ACTv1 brain
                            brain = self.registry.create_actv1(
                                name=master_name,
                                modalities=list(modalities),
                                checkpoint_path=checkpoint_path,
                                brain_config_path=brain_json_path,
                                max_seq_len=trained_max_seq_len,
                                inference_device=self._resolve_inference_device(),
                            )
                except Exception as e:
                    logger.error(f"[Router] Failed to load master {master_name}: {e}")
                    brain = None
                    load_error = str(e)

            if brain is None:
                if load_error is None:
                    load_error = "required files not found"
                master_errors.append(f"{master_name}: {load_error}")
                continue

            try:
                res = brain.run(task)
                self.registry.record_use(master_name, list(modalities))
                return res
            except Exception as e:
                logger.error(
                    "[Router] Master brain '%s' failed during run: %s",
                    master_name,
                    e,
                )
                # DO NOT automatically unload master brains on error - they should stay loaded
                # unless explicitly unloaded by the user. Errors can be transient and should
                # not cause the brain to be removed from memory.
                master_errors.append(f"{master_name}: {e}")
                # Try next master (if any)
                continue

        if matching_masters:
            error_msg = "; ".join(master_errors) if master_errors else "Master brain selection failed"
            return {"ok": False, "error": error_msg}

        if strict_master:
            if self.registry.masters:
                return {
                    "ok": False,
                    "error": (
                        "No master brain available for requested modalities; fallback is disabled. "
                        "Load the desired brain and ensure its modalities match."
                    ),
                }
            return {
                "ok": False,
                "error": "No master brain configured. Load a brain before starting chat.",
            }
        
        # Fallback to original behavior: hash-based or round-robin brain selection
        if self.strategy == "round_robin":
            name = self._select_existing_rr(list(modalities)) or self._brain_name_for(modalities, payload)
        else:
            name = self._brain_name_for(modalities, payload)
        brain = self.registry.get(name)
        if brain is None:
            # Brain not found - for ACTv1 brains, they must be pre-trained
            # Dynamic brain creation is not supported for ACTv1
            return {
                "ok": False,
                "error": f"Brain '{name}' not found. ACTv1 brains must be trained first using HRM training.",
            }
        
        # Run the brain
        try:
            res = brain.run(task)
            # update usage on success
            self.registry.record_use(name, list(modalities))
            return res
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _resolve_inference_device(self) -> Optional[str]:
        """Return preferred inference device for brain loading."""

        if callable(self.inference_device_getter):
            try:
                device = self.inference_device_getter()
                if device:
                    return str(device)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Inference device getter failed: %s", exc)
        return None

    def update_active_goals(self, goal_ids: List[str]) -> Dict[str, Any]:
        """Update active goals and manage expert loading/unloading.
        
        Args:
            goal_ids: List of active goal IDs (as strings)
        
        Returns:
            Dict with:
                - newly_activated: List of goal IDs that became active
                - newly_deactivated: List of goal IDs that were deactivated
                - linked_experts: List of expert IDs linked to active goals
                - loaded_experts: List of currently loaded expert IDs
        """
        goal_ids_set = set(goal_ids)
        
        # Find changes
        newly_activated = goal_ids_set - self._active_goal_ids
        newly_deactivated = self._active_goal_ids - goal_ids_set
        
        # Update active goals
        self._active_goal_ids = goal_ids_set
        
        # Get experts linked to all active goals
        linked_experts = self.get_experts_for_goals(list(goal_ids_set))
        
        # Update loaded expert IDs (placeholder for Task 12 when experts are actually loaded)
        # For now, just track what SHOULD be loaded based on goals
        self._loaded_expert_ids = set(linked_experts)
        
        return {
            "newly_activated": list(newly_activated),
            "newly_deactivated": list(newly_deactivated),
            "linked_experts": linked_experts,
            "loaded_experts": list(self._loaded_expert_ids),
        }

    def get_loaded_experts(self) -> List[str]:
        """Get list of currently loaded expert IDs.
        
        Returns:
            List of expert IDs that are currently loaded in memory
        """
        return list(self._loaded_expert_ids)

    def get_experts_for_goals(self, goal_ids: List[str]) -> List[str]:
        """Get expert IDs linked to given goals.
        
        Args:
            goal_ids: List of goal IDs
        
        Returns:
            List of unique expert IDs linked to any of the given goals
        """
        if not self.expert_registry:
            return []
        
        expert_ids = set()
        for goal_id in goal_ids:
            # Get experts linked to this goal
            experts = self.expert_registry.get_experts_by_goal(goal_id)
            for expert in experts:
                expert_ids.add(expert.expert_id)
        
        return list(expert_ids)

    def get_active_goals(self) -> List[str]:
        """Get currently active goal IDs.
        
        Returns:
            List of active goal IDs
        """
        return list(self._active_goal_ids)
