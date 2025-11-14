"""Brain registry management operations - pinning, masters, rename, parent/child relations."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.registry_core import BrainRegistry

logger = logging.getLogger(__name__)


def get_pins_path(registry: "BrainRegistry") -> Optional[str]:
    """Get path to pinned brains JSON file.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        Path to pinned.json or None if store_dir not configured
    """
    if not registry.store_dir:
        return None
    d = os.path.abspath(registry.store_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "pinned.json")


def get_masters_path(registry: "BrainRegistry") -> Optional[str]:
    """Get path to master brains JSON file.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        Path to masters.json or None if store_dir not configured
    """
    if not registry.store_dir:
        return None
    d = os.path.abspath(registry.store_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "masters.json")


def load_pinned_brains(registry: "BrainRegistry") -> bool:
    """Load pinned brains list from disk.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        True if loaded successfully, False otherwise
    """
    p = get_pins_path(registry)
    if not p:
        logger.debug("Cannot load pinned brains: no store directory configured")
        return False
    
    if not os.path.exists(p):
        logger.debug(f"Pinned brains registry not found at {p}, will create on first save")
        return False
    
    logger.info(f"Loading pinned brains from {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = set(str(n) for n in (data or []))
        registry.pinned = names
        logger.info(f"Loaded {len(names)} pinned brains")
        logger.debug(f"Pinned brains: {sorted(names)}")
        return True
    except Exception as e:
        logger.error(f"Failed to load pinned brains from {p}: {e}")
        return False


def load_master_brains(registry: "BrainRegistry") -> bool:
    """Load master brains list from disk.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        True if loaded successfully, False otherwise
    """
    p = get_masters_path(registry)
    if not p:
        logger.debug("Cannot load master brains: no store directory configured")
        return False
    
    if not os.path.exists(p):
        logger.debug(f"Master brains registry not found at {p}, will create on first save")
        return False
    
    logger.info(f"Loading master brains from {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = set(str(n) for n in (data or []))
        registry.masters = names
        # Ensure masters are pinned as well
        registry.pinned |= registry.masters
        logger.info(f"Loaded {len(names)} master brains")
        logger.debug(f"Master brains: {sorted(names)}")
        return True
    except Exception as e:
        logger.error(f"Failed to load master brains from {p}: {e}")
        return False


def save_pinned_brains(registry: "BrainRegistry") -> bool:
    """Save pinned brains list to disk.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        True if saved successfully, False otherwise
    """
    p = get_pins_path(registry)
    if not p:
        logger.warning("Cannot save pinned brains: no store directory configured")
        return False
    
    logger.info("Saving pinned brains registry")
    logger.debug(f"Saving {len(registry.pinned)} pinned brains to {p}")
    
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted(registry.pinned), f)
        
        # Log file size for debugging
        file_size = os.path.getsize(p)
        logger.info(f"Saved {len(registry.pinned)} pinned brains")
        logger.debug(f"File size: {file_size} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to save pinned brains to {p}: {e}")
        return False


def save_master_brains(registry: "BrainRegistry") -> bool:
    """Save master brains list to disk.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        True if saved successfully, False otherwise
    """
    p = get_masters_path(registry)
    if not p:
        logger.warning("Cannot save master brains: no store directory configured")
        return False
    
    logger.info("Saving master brains registry")
    logger.debug(f"Saving {len(registry.masters)} master brains to {p}")
    
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted(registry.masters), f)
        
        # Log file size for debugging
        file_size = os.path.getsize(p)
        logger.info(f"Saved {len(registry.masters)} master brains")
        logger.debug(f"File size: {file_size} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to save master brains to {p}: {e}")
        return False


def rename_brain(registry: "BrainRegistry", old: str, new: str) -> bool:
    """Rename a brain in memory and on disk (offloaded files) when present.
    
    Args:
        registry: BrainRegistry instance
        old: Current brain name
        new: New brain name
        
    Returns:
        True on success, False if new name already exists or old not found
    """
    from aios.core.brains.registry_storage import get_store_paths
    
    old = str(old)
    new = str(new)
    if old == new:
        return True
    if new in registry.brains:
        return False
    b = registry.brains.pop(old, None)
    if b is not None:
        registry.brains[new] = b
    # move usage/meta
    if old in registry.usage:
        registry.usage[new] = registry.usage.pop(old)
    # pin/master sets
    if old in registry.pinned:
        registry.pinned.remove(old)
        registry.pinned.add(new)
        save_pinned_brains(registry)
    if old in registry.masters:
        registry.masters.remove(old)
        registry.masters.add(new)
        save_master_brains(registry)
    # parent/children maps
    if old in registry.parent:
        registry.parent[new] = registry.parent.pop(old)
    for p, kids in list(registry.children.items()):
        if old in kids:
            kids = [new if k == old else k for k in kids]
            registry.children[p] = kids
    if old in registry.children:
        registry.children[new] = registry.children.pop(old)
    # rename offloaded files (best-effort)
    try:
        if registry.store_dir:
            from os import path as _p, rename as _mv
            onpz, ojson = get_store_paths(registry, old)
            nnpz, njson = get_store_paths(registry, new)
            # Only rename if source exists and dest doesn't
            if _p.exists(onpz) and not _p.exists(nnpz):
                _mv(onpz, nnpz)
            if _p.exists(ojson) and not _p.exists(njson):
                _mv(ojson, njson)
    except Exception:
        pass
    return True


def set_brain_parent(registry: "BrainRegistry", child: str, parent: Optional[str]) -> None:
    """Set or clear the parent of a brain; updates children map accordingly.
    
    Args:
        registry: BrainRegistry instance
        child: Child brain name
        parent: Parent brain name or None to clear
    """
    c = str(child)
    p = str(parent) if parent else None
    # Remove from existing parent's children list
    try:
        cur = registry.parent.get(c)
        if cur and cur in registry.children:
            registry.children[cur] = [k for k in registry.children[cur] if k != c]
    except Exception:
        pass
    # Set new parent
    registry.parent[c] = p
    if p:
        registry.children.setdefault(p, [])
        if c not in registry.children[p]:
            registry.children[p].append(c)
