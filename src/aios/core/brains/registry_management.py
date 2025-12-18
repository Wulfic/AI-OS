"""Brain registry management operations - pinning, masters, rename, parent/child relations."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.brains.registry_core import BrainRegistry

logger = logging.getLogger(__name__)


def _safe_load_json_list(filepath: str) -> tuple[bool, List[Any]]:
    """Safely load a JSON file expected to contain a list.
    
    Handles empty files, corrupted content, and missing files gracefully.
    If the file is empty or corrupted, it will be repaired with an empty list.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Tuple of (success, data). Success is True if file was loaded or repaired,
        False only if the file couldn't be read or written.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Handle empty file
        if not content:
            logger.warning(f"File {filepath} is empty, initializing with empty list")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("[]")
                return True, []
            except Exception as write_err:
                logger.error(f"Failed to repair empty file {filepath}: {write_err}")
                return True, []  # Still return success with empty list
        
        # Try to parse JSON
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                logger.warning(f"File {filepath} contains non-list data, treating as empty list")
                return True, []
            return True, data
        except json.JSONDecodeError as e:
            logger.warning(f"File {filepath} contains invalid JSON ({e}), repairing with empty list")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("[]")
            except Exception as write_err:
                logger.error(f"Failed to repair corrupted file {filepath}: {write_err}")
            return True, []
            
    except FileNotFoundError:
        return False, []
    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        return False, []


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
    
    Handles empty or corrupted files gracefully by repairing them.
    
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
    success, data = _safe_load_json_list(p)
    if not success:
        logger.debug(f"Could not read pinned brains from {p}")
        return False
    
    names = set(str(n) for n in data)
    registry.pinned = names
    logger.info(f"Loaded {len(names)} pinned brains")
    logger.debug(f"Pinned brains: {sorted(names)}")
    return True


def load_master_brains(registry: "BrainRegistry") -> bool:
    """Load master brains list from disk.
    
    Handles empty or corrupted files gracefully by repairing them.
    
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
    success, data = _safe_load_json_list(p)
    if not success:
        logger.debug(f"Could not read master brains from {p}")
        return False
    
    names = set(str(n) for n in data)
    registry.masters = names
    # Ensure masters are pinned as well
    registry.pinned |= registry.masters
    logger.info(f"Loaded {len(names)} master brains")
    logger.debug(f"Master brains: {sorted(names)}")
    return True


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
