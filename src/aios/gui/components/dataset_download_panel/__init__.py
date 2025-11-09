"""
Dataset Download Panel

A comprehensive panel for searching, downloading, and managing HuggingFace datasets.

This module provides:
- HuggingFace dataset search with filtering
- Favorites management
- Dataset download with streaming support
- Progress tracking and cancellation
- HuggingFace authentication

Example:
    from aios.gui.components.dataset_download_panel import DatasetDownloadPanel
    
    panel = DatasetDownloadPanel(parent, log_callback)
"""

# Import main class for backward compatibility
from .panel_main import DatasetDownloadPanel

# Import favorites management functions for backward compatibility
from .favorites_manager import (
    load_favorites,
    save_favorites,
    add_favorite,
    remove_favorite,
    is_favorited
)

# Re-export all public components
__all__ = [
    "DatasetDownloadPanel",
    "load_favorites",
    "save_favorites", 
    "add_favorite",
    "remove_favorite",
    "is_favorited"
]
