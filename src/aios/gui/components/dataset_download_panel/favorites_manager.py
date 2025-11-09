"""
Favorites Management for Dataset Downloads

Functions for managing user's favorite datasets.
Favorites are stored in ~/.config/aios/favorite_datasets.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any


def _get_favorites_path() -> Path:
    """Get the path to the favorites config file."""
    config_dir = Path.home() / ".config" / "aios"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "favorite_datasets.json"


def load_favorites() -> List[Dict[str, Any]]:
    """
    Load favorite datasets from config file.
    
    Returns:
        List of dataset info dictionaries
    """
    favorites_path = _get_favorites_path()
    if not favorites_path.exists():
        return []
    
    try:
        with open(favorites_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_favorites(favorites: List[Dict[str, Any]]) -> bool:
    """
    Save favorite datasets to config file.
    
    Args:
        favorites: List of dataset info dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    try:
        favorites_path = _get_favorites_path()
        with open(favorites_path, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, indent=2)
        return True
    except Exception:
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
            return False  # Already favorited
    
    # Add timestamp
    dataset_info["favorited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    favorites.append(dataset_info)
    
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
        return save_favorites(favorites)
    return False


def is_favorited(dataset_id: str) -> bool:
    """
    Check if a dataset is favorited.
    
    Args:
        dataset_id: Dataset ID or path to check
        
    Returns:
        True if favorited, False otherwise
    """
    favorites = load_favorites()
    for fav in favorites:
        if fav.get("id") == dataset_id or fav.get("path") == dataset_id:
            return True
    return False
