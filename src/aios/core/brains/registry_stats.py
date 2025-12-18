"""Brain registry statistics generation - usage tracking and ACTV1 bundle discovery."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.registry_core import BrainRegistry


def discover_actv1_bundles(registry: "BrainRegistry") -> Dict[str, int]:
    """Discover ACTV1 brain bundles on disk and their sizes.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        Dict mapping brain name to size in bytes
    """
    out: Dict[str, int] = {}
    try:
        if not registry.store_dir:
            return out
        base = os.path.join(registry.store_dir, "actv1")
        if not os.path.isdir(base):
            return out
        for entry in sorted(os.listdir(base)):
            try:
                p = os.path.join(base, entry)
                if not os.path.isdir(p):
                    continue
                # Skip internal system directories (parallel training checkpoints, etc.)
                if entry in ('parallel_checkpoints', 'checkpoints', 'temp', 'tmp', '.git') or entry.startswith('_'):
                    continue
                # Prefer size of actv1_student.safetensors; fall back to total dir size
                pt = os.path.join(p, "actv1_student.safetensors")
                if os.path.exists(pt):
                    sz = int(os.path.getsize(pt))
                else:
                    # Fallback: sum all files but EXCLUDE parallel_checkpoints and other temp dirs
                    total = 0
                    for r, dirs, files in os.walk(p):
                        # Modify dirs in-place to skip temp/checkpoint directories
                        dirs[:] = [d for d in dirs if d not in ('parallel_checkpoints', 'checkpoints', 'temp', 'tmp', '.git', '__pycache__') and not d.startswith('_')]
                        for f in files:
                            try:
                                total += int(os.path.getsize(os.path.join(r, f)))
                            except Exception:
                                continue
                    sz = int(total)
                out[str(entry)] = max(0, int(sz))
            except Exception:
                continue
    except Exception:
        return out
    return out


def load_brain_metadata_from_disk(registry: "BrainRegistry", brain_name: str) -> Dict[str, Any]:
    """Load training metadata from brain.json on disk.
    
    Args:
        registry: BrainRegistry instance
        brain_name: Brain name
        
    Returns:
        Dict with training_steps, last_trained, dataset_stats, dataset_history
    """
    meta_from_disk = {}
    try:
        if registry.store_dir:
            brain_json = os.path.join(registry.store_dir, "actv1", brain_name, "brain.json")
            if os.path.exists(brain_json):
                with open(brain_json, "r", encoding="utf-8") as f:
                    disk_data = json.load(f) or {}
                    # Extract relevant training metadata
                    if "training_steps" in disk_data:
                        meta_from_disk["training_steps"] = int(disk_data.get("training_steps", 0))
                    if "last_trained" in disk_data:
                        meta_from_disk["last_trained"] = float(disk_data.get("last_trained", 0))
                    # Extract dataset tracking information
                    if "dataset_stats" in disk_data:
                        meta_from_disk["dataset_stats"] = disk_data.get("dataset_stats", {})
                    if "dataset_history" in disk_data:
                        # Only include recent history (last 20 sessions) to avoid bloat
                        history = disk_data.get("dataset_history", [])
                        meta_from_disk["dataset_history"] = history[-20:] if isinstance(history, list) else []
    except Exception:
        pass
    return meta_from_disk


def compute_registry_stats(registry: "BrainRegistry") -> Dict[str, Any]:
    """Compute comprehensive stats for all brains in registry.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        Dict with used_bytes and brains dict
    """
    from aios.core.brains.registry_storage import get_offloaded_size
    
    # Start with loaded brains
    entries: Dict[str, Any] = {}
    for n, b in registry.brains.items():
        try:
            sz = int(b.size_bytes())
        except Exception:
            sz = 0
        entries[n] = {
            **registry.usage.get(n, {}),
            "size_bytes": sz,
            "pinned": n in registry.pinned,
            "master": n in registry.masters,
            "parent": registry.parent.get(n),
            "children": sorted(registry.children.get(n, [])),
        }
    
    # Ensure pinned/masters also appear even if not loaded
    def _ensure(name: str) -> None:
        if name not in entries:
            peek = get_offloaded_size(registry, name)
            entries[name] = {
                **registry.usage.get(name, {}),
                "size_bytes": int(peek),
                "pinned": name in registry.pinned,
                "master": name in registry.masters,
                "parent": registry.parent.get(name),
                "children": sorted(registry.children.get(name, [])),
            }
    for n in sorted(registry.pinned | registry.masters):
        _ensure(n)

    # Also surface ACTV1 brain bundles found on disk under store_dir/actv1
    try:
        actv1_sizes = discover_actv1_bundles(registry)
        for n, sz in actv1_sizes.items():
            if n not in entries:
                # Load training_steps and dataset info from brain.json if available
                meta_from_disk = load_brain_metadata_from_disk(registry, n)
                entries[n] = {
                    **registry.usage.get(n, {}),
                    **meta_from_disk,  # Override with disk metadata if present
                    "size_bytes": int(sz),
                    "pinned": n in registry.pinned,
                    "master": n in registry.masters,
                    "parent": registry.parent.get(n),
                    "children": sorted(registry.children.get(n, [])),
                }
            else:
                # If present but size unknown, fill it in
                try:
                    cur = int(entries[n].get("size_bytes", 0) or 0)
                    if cur <= 0 and sz > 0:
                        entries[n]["size_bytes"] = int(sz)
                    # Also load training_steps and dataset info from disk if not already present
                    if "training_steps" not in entries[n] or entries[n].get("training_steps", 0) == 0:
                        meta_from_disk = load_brain_metadata_from_disk(registry, n)
                        entries[n].update(meta_from_disk)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Compute total used bytes from entries, falling back to live sum
    try:
        total_used = sum(int(v.get("size_bytes", 0) or 0) for v in entries.values())
    except Exception:
        total_used = sum(max(0, int(b.size_bytes())) for b in registry.brains.values())
    
    return {"used_bytes": int(total_used), "brains": entries}


def merge_orphaned_parallel_checkpoints(registry: "BrainRegistry", brain_name: str, remove_after_merge: bool = True) -> bool:
    """Merge orphaned parallel GPU checkpoints into a single actv1_student.safetensors file.
    
    This is useful when parallel training completed but didn't create the final merged checkpoint.
    The parallel checkpoints should be in the parallel_checkpoints subdirectory.
    
    Args:
        registry: BrainRegistry instance
        brain_name: Brain name to merge checkpoints for
        remove_after_merge: If True, remove parallel checkpoints after successful merge
        
    Returns:
        True if merge successful, False otherwise
    """
    try:
        if not registry.store_dir:
            return False
        
        brain_dir = os.path.join(registry.store_dir, "actv1", brain_name)
        if not os.path.isdir(brain_dir):
            return False
        
        # Check if actv1_student.safetensors already exists
        final_checkpoint = os.path.join(brain_dir, "actv1_student.safetensors")
        if os.path.exists(final_checkpoint):
            return False  # Already exists, no merge needed
        
        # Look for parallel checkpoints
        parallel_dir = os.path.join(brain_dir, "parallel_checkpoints")
        if not os.path.isdir(parallel_dir):
            return False
        
        # Find all GPU checkpoint files
        import glob
        checkpoint_files = sorted(glob.glob(os.path.join(parallel_dir, "gpu*_final.safetensors")))
        if not checkpoint_files:
            checkpoint_files = sorted(glob.glob(os.path.join(parallel_dir, "gpu*.safetensors")))
        
        if len(checkpoint_files) < 1:
            return False  # No checkpoints to merge
        
        # If only one checkpoint, just copy it
        if len(checkpoint_files) == 1:
            import shutil
            shutil.copy2(checkpoint_files[0], final_checkpoint)
            if remove_after_merge:
                try:
                    os.remove(checkpoint_files[0])
                except Exception:
                    pass
            return True
        
        # Merge multiple checkpoints by averaging weights
        try:
            from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
            import torch
            
            # Load all checkpoints
            state_dicts = []
            for cp in checkpoint_files:
                try:
                    sd = load_safetensors(cp, device='cpu')
                    state_dicts.append(sd)
                except Exception:
                    pass
            
            if not state_dicts:
                return False
            
            # Average weights across all checkpoints
            merged_state = {}
            for key in state_dicts[0].keys():
                tensors = [sd[key].float() for sd in state_dicts]
                stacked = torch.stack(tensors)
                merged_state[key] = stacked.mean(dim=0)
            
            # Save merged checkpoint
            save_safetensors(merged_state, final_checkpoint)
            
            # Cleanup parallel checkpoints if requested
            if remove_after_merge:
                for cp in checkpoint_files:
                    try:
                        os.remove(cp)
                    except Exception:
                        pass
            
            return True
        except Exception:
            return False
    except Exception:
        return False
