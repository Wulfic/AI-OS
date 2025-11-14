"""Brain registry storage operations - offload/restore and checkpoint management."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.registry_core import BrainRegistry
    from aios.core.brains.numpy_brain import NumpyMLPBrain

logger = logging.getLogger(__name__)


def get_store_paths(registry: "BrainRegistry", name: str) -> Tuple[str, str]:
    """Get checkpoint and metadata paths for a brain.
    
    Args:
        registry: BrainRegistry instance
        name: Brain name
        
    Returns:
        Tuple of (checkpoint_path, metadata_path)
        
    Raises:
        RuntimeError: If store_dir not configured
    """
    if not registry.store_dir:
        raise RuntimeError("store_dir not configured")
    d = os.path.abspath(registry.store_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.npz"), os.path.join(d, f"{name}.json")


def offload_brain(registry: "BrainRegistry", name: str) -> bool:
    """Offload a brain to disk to free memory.
    
    Args:
        registry: BrainRegistry instance
        name: Brain name to offload
        
    Returns:
        True if offloaded successfully, False otherwise
    """
    from aios.core.brains.numpy_brain import NumpyMLPBrain
    
    b = registry.brains.get(name)
    if b is None or not isinstance(b, NumpyMLPBrain):
        return False
    try:
        npz, meta_path = get_store_paths(registry, name)
        # Save checkpoint
        tr = b._trainer_ready()
        tr.save_checkpoint(npz, {"name": name})
        # Save meta
        meta = {
            "name": name,
            "modalities": list(b.modalities),
            "cfg": {k: getattr(b.cfg, k) for k in ("input_dim", "hidden", "output_dim", "dynamic_width", "width_storage_limit_mb") if hasattr(b.cfg, k)},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        # Remove from memory after successful save
        del registry.brains[name]
        return True
    except Exception:
        return False


def try_load_offloaded(registry: "BrainRegistry", name: str) -> Optional["NumpyMLPBrain"]:
    """Try to load an offloaded brain from disk.
    
    Args:
        registry: BrainRegistry instance
        name: Brain name to load
        
    Returns:
        Loaded brain or None if not found or failed to load
    """
    from aios.core.brains.numpy_brain import NumpyMLPBrain
    from aios.core.train import TrainConfig
    
    if not registry.store_dir:
        return None
    
    # First try to load ACTv1 brain from actv1/ subdirectory
    actv1_brain = try_load_actv1(registry, name)
    if actv1_brain is not None:
        return actv1_brain
    
    # Fall back to NumpyMLPBrain offloaded format
    try:
        npz, meta_path = get_store_paths(registry, name)
        if not (os.path.exists(npz) and os.path.exists(meta_path)):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        modalities = list(meta.get("modalities") or [])
        cfg_dict = dict(meta.get("cfg") or {})
        cfg = TrainConfig()
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        brain = NumpyMLPBrain(name=name, modalities=modalities or ["text"], cfg=cfg)
        tr = brain._trainer_ready()
        tr.load_checkpoint(npz)
        registry.brains[name] = brain
        registry.record_use(name, modalities)
        return brain
    except Exception:
        return None


def try_load_actv1(registry: "BrainRegistry", name: str) -> Optional[any]:
    """Try to load an ACTv1 brain from actv1/ subdirectory.
    
    Args:
        registry: BrainRegistry instance
        name: Brain name to load
        
    Returns:
        Loaded ACTv1Brain or None if not found or failed to load
    """
    from aios.core.brains.actv1_brain import ACTv1Brain
    
    if not registry.store_dir:
        return None
    
    try:
        # Look for ACTv1 brain bundle in actv1/<name>/
        brain_dir = os.path.join(registry.store_dir, "actv1", name)
        checkpoint_path = os.path.join(brain_dir, "actv1_student.safetensors")
        brain_config_path = os.path.join(brain_dir, "brain.json")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        if not os.path.exists(brain_config_path):
            return None
        
        # Load brain.json to get modalities
        with open(brain_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        modalities = config.get("modalities", ["text"])
        if isinstance(modalities, str):
            modalities = [modalities]
        
        # Create ACTv1 brain
        brain = ACTv1Brain(
            name=name,
            modalities=list(modalities),
            checkpoint_path=checkpoint_path,
            brain_config_path=brain_config_path,
        )
        
        # Add to registry
        registry.brains[name] = brain
        registry.record_use(name, modalities)
        
        return brain
        
    except Exception as e:
        # Silent fail - let caller handle missing brain
        logger.warning(f"[registry_storage] Failed to load ACTv1 brain {name}: {e}")
        return None


def get_offloaded_size(registry: "BrainRegistry", name: str) -> int:
    """Get size of offloaded checkpoint when not loaded.
    
    Args:
        registry: BrainRegistry instance
        name: Brain name
        
    Returns:
        Size in bytes, or 0 if not found
    """
    try:
        if not registry.store_dir:
            return 0
        import os as _os
        npz, _meta = get_store_paths(registry, name)
        return int(_os.path.getsize(npz)) if _os.path.exists(npz) else 0
    except Exception:
        return 0
