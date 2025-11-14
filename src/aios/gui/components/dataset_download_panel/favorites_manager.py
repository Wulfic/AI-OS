"""
Favorites Management for Dataset Downloads

Functions for managing user's favorite datasets.
Favorites are stored in ~/.config/aios/favorite_datasets.json
"""

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any, Set

logger = logging.getLogger(__name__)

_favorites_cache: List[Dict[str, Any]] | None = None
_favorites_mtime: float | None = None
_favorites_lock = Lock()


def _clone_favorites(favorites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a shallow copy of the favorites list."""
    return [dict(item) for item in favorites]


def _get_favorites_path() -> Path:
    """Get the path to the favorites config file."""
    config_dir = Path.home() / ".config" / "aios"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "favorite_datasets.json"


def _refresh_favorites_cache(favorites_path: Path) -> List[Dict[str, Any]]:
    """Reload favorites from disk and update the in-memory cache."""
    global _favorites_cache, _favorites_mtime

    if not favorites_path.exists():
        logger.debug(f"Favorites file not found: {favorites_path}, creating new")
        _favorites_cache = []
        _favorites_mtime = None
        return []

    try:
        logger.info(f"Loading dataset favorites from {favorites_path}")
        with open(favorites_path, 'r', encoding='utf-8') as f:
            favorites = json.load(f)
        logger.info(f"Loaded {len(favorites)} favorite dataset(s)")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(f"Failed to load favorites from {favorites_path}: {exc}")
        favorites = []

    _favorites_cache = favorites
    try:
        _favorites_mtime = favorites_path.stat().st_mtime
    except OSError:
        _favorites_mtime = None

    return favorites


def load_favorites() -> List[Dict[str, Any]]:
    """
    Load favorite datasets from config file.
    
    Returns:
        List of dataset info dictionaries
    """
    favorites_path = _get_favorites_path()

    with _favorites_lock:
        global _favorites_cache, _favorites_mtime

        try:
            current_mtime = favorites_path.stat().st_mtime
        except OSError:
            current_mtime = None

        cache_stale = (
            _favorites_cache is None
            or _favorites_mtime != current_mtime
        )

        if cache_stale:
            favorites = _refresh_favorites_cache(favorites_path)
        else:
            favorites = _favorites_cache or []
            logger.debug(
                "Returning %d cached favorite dataset(s)",
                len(favorites),
            )

        return _clone_favorites(favorites)


def save_favorites(favorites: List[Dict[str, Any]]) -> bool:
    """
    Save favorite datasets to config file.
    
    Args:
        favorites: List of dataset info dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    favorites_copy = _clone_favorites(favorites)

    try:
        favorites_path = _get_favorites_path()
        logger.info(f"Saving {len(favorites_copy)} favorite dataset(s)")
        logger.debug(f"Saving favorites to {favorites_path}")
        with _favorites_lock:
            with open(favorites_path, 'w', encoding='utf-8') as f:
                json.dump(favorites_copy, f, indent=2)

            # Update cache immediately so UI can reflect new state without reload
            global _favorites_cache, _favorites_mtime
            _favorites_cache = favorites_copy
            try:
                _favorites_mtime = favorites_path.stat().st_mtime
            except OSError:
                _favorites_mtime = None

        logger.info("Favorites saved successfully")
        return True
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"Failed to save favorites: {e}")
        return False


def add_favorite(dataset_info: Dict[str, Any]) -> bool:
    """
    Add a dataset to favorites.
    
    Args:
        dataset_info: Dataset information dictionary
        
    Returns:
        True if added, False if already favorited
    """
    favorites = load_favorites()
    
    # Check if already favorited
    dataset_id = dataset_info.get("id") or dataset_info.get("path")
    for fav in favorites:
        if fav.get("id") == dataset_id or fav.get("path") == dataset_id:
            logger.info(f"Dataset already in favorites: {dataset_id}")
            return False  # Already favorited
    
    # Add timestamp
    dataset_info["favorited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    favorites.append(dataset_info)
    
    logger.info(f"User action: Added dataset to favorites: {dataset_id}")
    return save_favorites(favorites)


def remove_favorite(dataset_id: str) -> bool:
    """
    Remove a dataset from favorites.
    
    Args:
        dataset_id: Dataset ID or path to remove
        
    Returns:
        True if removed, False if not found
    """
    favorites = load_favorites()
    original_len = len(favorites)
    
    # Filter out the dataset
    favorites = [f for f in favorites if f.get("id") != dataset_id and f.get("path") != dataset_id]
    
    if len(favorites) < original_len:
        logger.info(f"User action: Removed dataset from favorites: {dataset_id}")
        return save_favorites(favorites)
    logger.warning(f"Dataset not found in favorites: {dataset_id}")
    return False


def get_favorite_ids() -> Set[str]:
    """Return a set containing all favorited dataset identifiers."""
    favorites = load_favorites()
    favorite_ids: Set[str] = set()

    for fav in favorites:
        primary = fav.get("id")
        secondary = fav.get("path")

        if primary:
            favorite_ids.add(str(primary))
        if secondary:
            favorite_ids.add(str(secondary))

    return favorite_ids


def is_favorited(dataset_id: str) -> bool:
    """
    Check if a dataset is favorited.
    
    Args:
        dataset_id: Dataset ID or path to check
        
    Returns:
        True if favorited, False otherwise
    """
    if not dataset_id:
        return False

    favorites = get_favorite_ids()
    return str(dataset_id) in favorites
