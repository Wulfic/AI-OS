"""Centralized config file loader for GUI components.

This module provides a single source of truth for loading configuration
files, ensuring consistency between GUI and CLI components.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover - fallback in early bootstrap
    system_paths = None


def get_config_path() -> Path:
    """Get the path to the user's config file, following same logic as CLI.
    
    Priority order:
    1. AIOS_CONFIG environment variable
    2. User config at ~/.config/aios/config.yaml  
    3. Repo default at config/default.yaml (from repo root)
    4. Repo default relative to this module
    
    Returns:
        Path to config file (may not exist)
    """
    # Check environment variable first (matches CLI behavior)
    env_path = os.environ.get("AIOS_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            logger.debug(f"Using config from AIOS_CONFIG: {p}")
            return p
        logger.warning(f"AIOS_CONFIG set to {env_path} but file doesn't exist")
    
    # Check user config directory (~/.config/aios/config.yaml)
    if system_paths is not None:
        try:
            user_config_dir = system_paths.get_user_config_dir()
        except Exception:
            logger.debug("Failed to resolve user config dir via helper", exc_info=True)
            user_config_dir = Path.home() / ".config" / "aios"
    else:
        user_config_dir = Path.home() / ".config" / "aios"

    user_config = user_config_dir / "config.yaml"
    if user_config.exists():
        logger.debug(f"Using user config: {user_config}")
        return user_config
    
    # Try to find repo root by looking for config/default.yaml
    # Start from current file location and walk up
    try:
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            config_candidate = parent / "config" / "default.yaml"
            if config_candidate.exists():
                logger.debug(f"Using repo default config: {config_candidate}")
                return config_candidate
    except Exception as e:
        logger.debug(f"Failed to find config relative to module: {e}")
    
    # Fall back to CWD only as last resort
    fallback = Path.cwd() / "config" / "default.yaml"
    if fallback.exists():
        logger.debug(f"Using CWD config: {fallback}")
        return fallback
    
    # If nothing found, return user config path (will be created on save)
    logger.warning(f"Config file not found, will use user config path on save")
    return Path.home() / ".config" / "aios" / "config.yaml"


def get_writable_config_path() -> Path:
    """Get a writable config path, falling back to user config if system path is not writable.
    
    This function checks if the normal config path is writable. If not (e.g., when installed
    system-wide to /opt or similar), it falls back to the user's config directory.
    
    Returns:
        Path to a writable config file location
    """
    # Check environment variable first
    env_path = os.environ.get("AIOS_CONFIG")
    if env_path:
        p = Path(env_path)
        # Check if file exists and is writable, or if parent is writable for new file
        if p.exists() and os.access(p, os.W_OK):
            return p
        if not p.exists() and p.parent.exists() and os.access(p.parent, os.W_OK):
            return p
    
    # Try to find repo root and check if writable
    try:
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            config_candidate = parent / "config" / "default.yaml"
            if config_candidate.exists():
                if os.access(config_candidate, os.W_OK):
                    logger.debug(f"Using writable repo config: {config_candidate}")
                    return config_candidate
                # File exists but not writable, continue to user config
                logger.debug(f"Repo config not writable: {config_candidate}")
                break
            # Check if parent directory would be writable for creating new file
            if config_candidate.parent.exists() and os.access(config_candidate.parent, os.W_OK):
                logger.debug(f"Using writable repo config dir: {config_candidate}")
                return config_candidate
    except Exception as e:
        logger.debug(f"Failed to check repo config path: {e}")
    
    # Try CWD config
    cwd_config = Path.cwd() / "config" / "default.yaml"
    if cwd_config.exists() and os.access(cwd_config, os.W_OK):
        return cwd_config
    if not cwd_config.exists() and cwd_config.parent.exists() and os.access(cwd_config.parent, os.W_OK):
        return cwd_config
    
    # Fall back to user config directory (always writable by user)
    if system_paths is not None:
        try:
            user_config_dir = system_paths.get_user_config_dir()
        except Exception:
            user_config_dir = Path.home() / ".config" / "aios"
    else:
        user_config_dir = Path.home() / ".config" / "aios"
    
    user_config = user_config_dir / "config.yaml"
    logger.debug(f"Using user config path for writing: {user_config}")
    return user_config


def load_config() -> dict[str, Any]:
    """Load configuration from the user's config file.
    
    Returns:
        Config dictionary, or empty dict if file doesn't exist or can't be loaded
    """
    try:
        import yaml
        
        config_path = get_config_path()
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}")
            return {}
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        if not isinstance(config, dict):
            logger.error(f"Config file is not a dict: {type(config)}")
            return {}
        
        logger.debug(f"Loaded config with {len(config)} top-level keys")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        return {}


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to the user's config file.
    
    Uses a writable config path, falling back to user config directory
    if the default config path is not writable (e.g., system-wide install).
    
    Args:
        config: Configuration dictionary to save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import yaml
        
        # Use writable path to avoid permission errors on system installs
        config_path = get_writable_config_path()
        logger.info(f"Saving configuration to {config_path}")
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info("Configuration saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}", exc_info=True)
        return False
