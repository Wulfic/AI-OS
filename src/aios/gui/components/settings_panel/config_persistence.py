"""Configuration file persistence for Settings panel.

This module provides functions to load/save settings from/to config/default.yaml,
making the YAML config file the single source of truth for settings configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

# Import centralized config loader
try:
    from ...config_loader import get_config_path, load_config as load_full_config, save_config as save_full_config
except ImportError:
    # Fallback for older code structure
    def get_config_path() -> Path:
        """Get the path to the user's config file."""
        import os
        env_path = os.environ.get("AIOS_CONFIG")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p
        user_config = Path.home() / ".config" / "aios" / "config.yaml"
        if user_config.exists():
            return user_config
        # Walk up from this file to find config/default.yaml
        current = Path(__file__).resolve()
        for parent in current.parents:
            config_file = parent / "config" / "default.yaml"
            if config_file.exists():
                return config_file
        return Path.cwd() / "config" / "default.yaml"
    
    def load_full_config() -> dict:
        """Fallback config loader."""
        try:
            config_path = get_config_path()
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            pass
        return {}
    
    def save_full_config(config: dict) -> bool:
        """Fallback config saver."""
        try:
            config_path = get_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            return True
        except Exception:
            return False


def load_settings_from_config() -> dict[str, Any]:
    """Load settings from config file.
    
    Returns:
        Dict with settings values:
        - theme: str (Light Mode, Dark Mode, etc.)
        - log_level: str (Normal, Advanced, DEBUG)
        - cache_max_size_mb: float
    """
    try:
        config = load_full_config()
        
        if not config:
            logger.debug("No config loaded for settings")
            return {}
        
        settings = {}
        
        # UI theme (if we add it to config later)
        if 'ui' in config and isinstance(config['ui'], dict):
            theme = config['ui'].get('theme')
            if theme in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                settings['theme'] = theme
        
        # Logging level (if we add it to config later)
        if 'logging' in config and isinstance(config['logging'], dict):
            log_level = config['logging'].get('level')
            if log_level in ("Normal", "Advanced", "DEBUG"):
                settings['log_level'] = log_level
        
        # Cache settings from streaming_cache
        if 'streaming_cache' in config and isinstance(config['streaming_cache'], dict):
            cache = config['streaming_cache']
            max_size = cache.get('max_size_mb')
            if max_size is not None:
                try:
                    settings['cache_max_size_mb'] = float(max_size)
                except (ValueError, TypeError):
                    pass
        
        # Data section for additional cache settings
        if 'data' in config and isinstance(config['data'], dict):
            data = config['data']
            if 'streaming_cache' in data and isinstance(data['streaming_cache'], dict):
                cache = data['streaming_cache']
                max_size = cache.get('max_size_mb')
                if max_size is not None:
                    try:
                        settings['cache_max_size_mb'] = float(max_size)
                    except (ValueError, TypeError):
                        pass
        
        if settings:
            logger.info(f"Loaded settings from config: {settings}")
        else:
            logger.debug("No settings found in config")
        
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings from config: {e}", exc_info=True)
        return {}


def save_settings_to_config(settings_values: dict[str, Any]) -> bool:
    """Save settings to config file.
    
    Args:
        settings_values: Dict with settings from SettingsPanel.get_state()
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        logger.info("Saving settings to config")
        logger.debug(f"Settings values: {settings_values}")
        
        # Load existing config
        config = load_full_config()
        
        # Update UI section (if we want to persist theme)
        # For now, theme is saved only in gui_state.json, not config.yaml
        # Uncomment this if we want theme in config:
        # if 'ui' not in config:
        #     config['ui'] = {}
        # if 'theme' in settings_values:
        #     config['ui']['theme'] = settings_values['theme']
        
        # Update logging section (if we want to persist log level)
        # For now, log_level is saved only in gui_state.json, not config.yaml
        # Uncomment this if we want log_level in config:
        # if 'logging' not in config:
        #     config['logging'] = {}
        # if 'log_level' in settings_values:
        #     config['logging']['level'] = settings_values['log_level']
        
        # Update cache settings - these SHOULD go to config
        cache_size = settings_values.get('cache_max_size_mb')
        if cache_size is not None:
            # Update streaming_cache section (top-level)
            if 'streaming_cache' not in config:
                config['streaming_cache'] = {}
            config['streaming_cache']['max_size_mb'] = float(cache_size)
            
            # Also update data.streaming_cache if it exists
            if 'data' in config and isinstance(config['data'], dict):
                if 'streaming_cache' not in config['data']:
                    config['data']['streaming_cache'] = {}
                config['data']['streaming_cache']['max_size_mb'] = float(cache_size)
        
        # Save config back to file
        success = save_full_config(config)
        
        if success:
            logger.info("Successfully saved settings to config")
        else:
            logger.error("Failed to save settings to config")
        
        return success
    except Exception as e:
        logger.error(f"Failed to save settings to config: {e}", exc_info=True)
        return False


__all__ = [
    'load_settings_from_config',
    'save_settings_to_config',
]
