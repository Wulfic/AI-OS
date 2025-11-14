"""Cache management functionality for Settings panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from tkinter import messagebox

if TYPE_CHECKING:
    from .panel_main import SettingsPanel

logger = logging.getLogger(__name__)

# Import config persistence functions
from . import config_persistence


def load_cache_size(panel: "SettingsPanel") -> None:
    """Load cache size configuration from config file.
    
    Args:
        panel: The settings panel instance
    """
    try:
        settings = config_persistence.load_settings_from_config()
        max_size_mb = settings.get('cache_max_size_mb', 1000.0)
        panel.cache_size_var.set(str(int(max_size_mb)))
        logger.debug(f"Loaded cache size from config: {max_size_mb} MB")
    except Exception as e:
        logger.error(f"Error loading cache size: {e}")
        panel.cache_size_var.set("1000")  # Default


def save_cache_size(panel: "SettingsPanel") -> None:
    """Save cache size configuration to config file.
    
    Args:
        panel: The settings panel instance
    """
    try:
        # Validate input
        try:
            size_mb = float(panel.cache_size_var.get())
            if size_mb <= 0:
                raise ValueError("Size must be positive")
        except ValueError:
            logger.warning(f"Invalid cache size input: {panel.cache_size_var.get()}")
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid positive number for cache size."
            )
            return
        
        logger.info(f"User action: Setting cache size limit to {size_mb} MB")
        
        # Save using config_persistence
        success = config_persistence.save_settings_to_config({
            'cache_max_size_mb': size_mb
        })
        
        if not success:
            messagebox.showerror(
                "Error",
                "Failed to save cache size configuration"
            )
            return
        
        logger.info(f"Successfully saved cache size limit: {size_mb} MB")
        messagebox.showinfo(
            "Settings Saved",
            f"Cache size limit set to {size_mb} MB.\n\n"
            "Changes will take effect on next training run."
        )
        
        # Refresh stats to show updated limit
        refresh_cache_stats(panel)
        
    except Exception as e:
        logger.error(f"Failed to save cache size: {e}", exc_info=True)
        messagebox.showerror(
            "Error",
            f"Failed to save cache size:\n{str(e)}"
        )


def refresh_cache_stats(panel: "SettingsPanel") -> None:
    """Refresh and display cache statistics.
    
    Args:
        panel: The settings panel instance
    """
    try:
        from ....data.streaming_cache import get_cache
        
        # Reload cache size from config in case it changed
        load_cache_size(panel)
        
        cache = get_cache()
        stats = cache.get_cache_stats()
        
        # Format the stats display
        size_status = stats.get('size_limit_status', 'Unknown')
        blocks = stats.get('total_chunks', 0)
        datasets = stats.get('datasets_cached', 0)
        
        panel.cache_stats_label.config(
            text=f"{size_status} | {blocks} blocks | {datasets} datasets"
        )
    except Exception as e:
        panel.cache_stats_label.config(
            text=f"Error: {str(e)[:40]}"
        )


def clear_cache(panel: "SettingsPanel") -> None:
    """Clear all cached dataset blocks.
    
    Args:
        panel: The settings panel instance
    """
    try:
        from ....data.streaming_cache import get_cache
        
        logger.info("User action: Requesting to clear dataset cache")
        
        # Confirm with user
        result = messagebox.askyesno(
            "Clear Cache",
            "Are you sure you want to clear all cached dataset blocks?\n\n"
            "This will free up disk space but subsequent training runs\n"
            "will need to re-download data from HuggingFace.",
            icon='warning'
        )
        
        if not result:
            logger.info("User cancelled cache clear operation")
            return
        
        cache = get_cache()
        stats = cache.get_cache_stats()
        dataset_count = len(stats.get('chunks_per_dataset', {}))
        
        logger.info(f"Clearing cache for {dataset_count} dataset(s)")
        
        # Clear all datasets
        total_removed = 0
        for dataset in list(stats.get('chunks_per_dataset', {}).keys()):
            removed = cache.clear_dataset_cache(dataset)
            total_removed += removed
            logger.debug(f"Removed {removed} blocks from dataset '{dataset}'")
        
        logger.info(f"Successfully cleared {total_removed} cached blocks from {dataset_count} datasets")
        
        # Refresh stats
        refresh_cache_stats(panel)
        
        # Show success message
        messagebox.showinfo(
            "Cache Cleared",
            f"Successfully cleared {total_removed} cached block(s)."
        )
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}", exc_info=True)
        messagebox.showerror(
            "Error",
            f"Failed to clear cache:\n{str(e)}"
        )
