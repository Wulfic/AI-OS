"""Brain registry management operations - pinning, masters, rename, parent/child relations."""

from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.registry_core import BrainRegistry


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
    if not p or not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = set(str(n) for n in (data or []))
        registry.pinned = names
        return True
    except Exception:
        return False


def load_master_brains(registry: "BrainRegistry") -> bool:
    """Load master brains list from disk.
    
    Args:
        registry: BrainRegistry instance
        
    Returns:
        True if loaded successfully, False otherwise
    """
    p = get_masters_path(registry)
    if not p or not os.path.exists(p):
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = set(str(n) for n in (data or []))
        registry.masters = names
        # Ensure masters are pinned as well
        registry.pinned |= registry.masters
        return True
    except Exception:
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
        return False
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted(registry.pinned), f)
        return True
    except Exception:
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
        return False
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted(registry.masters), f)
        return True
    except Exception:
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
