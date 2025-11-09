"""Cache management for dataset streaming."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from tkinter import messagebox

if TYPE_CHECKING:
    from .panel_main import SettingsPanel

logger = logging.getLogger(__name__)


def load_cache_size(panel: "SettingsPanel") -> None:
    """Load cache size configuration from config file.
    
    Args:
        panel: The settings panel instance
    """
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path.cwd() / "config" / "default.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                cache_config = config.get('streaming_cache', {})
                max_size_mb = cache_config.get('max_size_mb', 100)
                panel.cache_size_var.set(str(int(max_size_mb)))
    except Exception as e:
        logger.error(f"Error loading cache size: {e}")
        panel.cache_size_var.set("100")  # Default


def save_cache_size(panel: "SettingsPanel") -> None:
    """Save cache size configuration to config file.
    
    Args:
        panel: The settings panel instance
    """
    try:
        import yaml
        from pathlib import Path
        
        # Validate input
        try:
            size_mb = float(panel.cache_size_var.get())
            if size_mb <= 0:
                raise ValueError("Size must be positive")
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid positive number for cache size."
            )
            return
        
        config_path = Path.cwd() / "config" / "default.yaml"
        if not config_path.exists():
            messagebox.showerror(
                "Error",
                f"Config file not found: {config_path}"
            )
            return
        
        # Load existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Ensure streaming_cache section exists
        if 'streaming_cache' not in config:
            config['streaming_cache'] = {}
        
        # Update max_size_mb
        config['streaming_cache']['max_size_mb'] = float(size_mb)
        
        # Save config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        messagebox.showinfo(
            "Settings Saved",
            f"Cache size limit set to {size_mb} MB.\n\n"
            "Changes will take effect on next training run."
        )
        
        # Refresh stats to show updated limit
        refresh_cache_stats(panel)
        
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to save cache size:\n{str(e)}"
        )
        logger.error(f"Error saving cache size: {e}")


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
        
        # Confirm with user
        result = messagebox.askyesno(
            "Clear Cache",
            "Are you sure you want to clear all cached dataset blocks?\n\n"
            "This will free up disk space but subsequent training runs\n"
            "will need to re-download data from HuggingFace.",
            icon='warning'
        )
        
        if not result:
            return
        
        cache = get_cache()
        stats = cache.get_cache_stats()
        
        # Clear all datasets
        total_removed = 0
        for dataset in list(stats.get('chunks_per_dataset', {}).keys()):
            removed = cache.clear_dataset_cache(dataset)
            total_removed += removed
        
        # Refresh stats
        refresh_cache_stats(panel)
        
        # Show success message
        messagebox.showinfo(
            "Cache Cleared",
            f"Successfully cleared {total_removed} cached block(s)."
        )
        
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to clear cache:\n{str(e)}"
        )
        logger.error(f"Error clearing cache: {e}")
