"""Configuration file persistence for Evaluation panel settings.

This module provides functions to load/save evaluation settings from/to config/default.yaml,
making the YAML config file the single source of truth for evaluation configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """Get the path to the user's config file, following same logic as CLI.
    
    Returns:
        Path to user config file (respects AIOS_CONFIG env var, falls back to repo default)
    """
    import os
    
    # Check environment variable first (matches CLI behavior)
    env_path = os.environ.get("AIOS_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    
    # Check user config directory (~/.config/aios/config.yaml)
    user_config = Path.home() / ".config" / "aios" / "config.yaml"
    if user_config.exists():
        return user_config
    
    # Fall back to repo default (current working directory)
    config_path = Path.cwd() / "config" / "default.yaml"
    if config_path.exists():
        return config_path
    
    # Try relative to this file (for installed packages)
    try:
        module_path = Path(__file__).resolve()
        repo_root = module_path.parents[5]  # Up from src/aios/gui/components/evaluation_panel
        config_path = repo_root / "config" / "default.yaml"
        if config_path.exists():
            return config_path
    except Exception:
        pass
    
    # Fall back to cwd path even if it doesn't exist
    return Path.cwd() / "config" / "default.yaml"


def get_writable_config_path() -> Path:
    """Get a writable config path, falling back to user config if system path is not writable.
    
    Returns:
        Path to a writable config file location
    """
    import os
    
    # Check environment variable first
    env_path = os.environ.get("AIOS_CONFIG")
    if env_path:
        p = Path(env_path)
        # Check if parent exists and is writable, or if file exists and is writable
        if p.exists() and os.access(p, os.W_OK):
            return p
        if not p.exists() and p.parent.exists() and os.access(p.parent, os.W_OK):
            return p
    
    # Try current working directory config
    cwd_config = Path.cwd() / "config" / "default.yaml"
    if cwd_config.exists() and os.access(cwd_config, os.W_OK):
        return cwd_config
    if not cwd_config.exists() and cwd_config.parent.exists() and os.access(cwd_config.parent, os.W_OK):
        return cwd_config
    
    # Try repo config relative to this file
    try:
        module_path = Path(__file__).resolve()
        repo_root = module_path.parents[5]
        repo_config = repo_root / "config" / "default.yaml"
        if repo_config.exists() and os.access(repo_config, os.W_OK):
            return repo_config
        if not repo_config.exists() and repo_config.parent.exists() and os.access(repo_config.parent, os.W_OK):
            return repo_config
    except Exception:
        pass
    
    # Fall back to user config directory (always writable by user)
    user_config = Path.home() / ".config" / "aios" / "config.yaml"
    return user_config


def load_evaluation_from_config() -> dict[str, Any]:
    """Load evaluation settings from config/default.yaml.
    
    Returns:
        Dict of evaluation settings, or empty dict if config doesn't exist or has no evaluation section
    """
    try:
        config_path = get_config_path()
        logger.info(f"Loading evaluation config from {config_path}")
        
        if not config_path.exists():
            logger.debug(f"Config file not found: {config_path}")
            return {}
        
        # Use safe_load with timeout protection (YAML parsing can be slow)
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                # Corrupted YAML, return empty
                logger.error(f"YAML parsing error in config file: {e}")
                return {}
        
        if not isinstance(config, dict):
            logger.warning(f"Config file is not a dict: {type(config)}")
            return {}
        
        evaluation = config.get('evaluation', {})
        if not isinstance(evaluation, dict):
            logger.warning("Evaluation section in config is not a dict")
            return {}
        
        logger.info("Successfully loaded evaluation config")
        logger.debug(f"Evaluation config: {evaluation}")
        return evaluation
    except Exception as e:
        # Any file I/O error, return empty config
        logger.error(f"Failed to load evaluation config: {e}")
        return {}


def save_evaluation_to_config(evaluation_values: dict[str, Any]) -> bool:
    """Save evaluation settings to config/default.yaml.
    
    Args:
        evaluation_values: Dict of evaluation settings from EvaluationPanel.get_state()
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Use writable config path to avoid permission errors on system installs
        config_path = get_writable_config_path()
        logger.info(f"Saving evaluation config to {config_path}")
        logger.debug(f"Evaluation values: {evaluation_values}")
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config (try to read from original location first for merging)
        config: dict[str, Any] = {}
        read_path = get_config_path()
        if read_path.exists():
            try:
                with open(read_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                pass
        # Also try writable path if different and exists
        if config_path != read_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    # Merge user config over base config
                    if isinstance(user_config, dict):
                        config.update(user_config)
            except Exception:
                pass
        
        if not isinstance(config, dict):
            config = {}
        
        # Update evaluation section
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        evaluation = config['evaluation']
        if not isinstance(evaluation, dict):
            evaluation = {}
            config['evaluation'] = evaluation
        
        # Update evaluation fields
        # Configuration settings (commonly changed)
        evaluation['batch_size'] = evaluation_values.get('batch_size', 'auto')
        evaluation['limit'] = evaluation_values.get('limit', '0')
        evaluation['num_fewshot'] = evaluation_values.get('num_fewshot', '5')
        evaluation['output_path'] = evaluation_values.get('output_path', 'artifacts/evaluation')
        
        # Advanced options (less commonly changed)
        evaluation['log_samples'] = evaluation_values.get('log_samples', False)
        evaluation['cache_requests'] = evaluation_values.get('cache_requests', True)
        evaluation['check_integrity'] = evaluation_values.get('check_integrity', False)
        
        # Optional: Save last used model/benchmarks for convenience (but don't load them automatically)
        # Users might want these to persist or might not - we'll save them separately
        if 'last_used' not in evaluation:
            evaluation['last_used'] = {}
        
        evaluation['last_used']['model_source'] = evaluation_values.get('model_source', 'huggingface')
        evaluation['last_used']['model_name'] = evaluation_values.get('model_name', '')
        evaluation['last_used']['selected_benchmarks'] = evaluation_values.get('selected_benchmarks', '')
        
        # Write config back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        logger.info("Successfully saved evaluation config")
        return True
    except Exception as e:
        logger.error(f"Failed to save evaluation config: {e}", exc_info=True)
        return False


def merge_config_with_defaults(config_values: dict[str, Any], default_values: dict[str, Any]) -> dict[str, Any]:
    """Merge config values with defaults, preferring config values when present.
    
    Args:
        config_values: Values loaded from config file
        default_values: Default values (e.g., from GUI state or hardcoded defaults)
    
    Returns:
        Merged dict with config values taking precedence
    """
    result = dict(default_values)
    
    # Override with config values for settings (not last_used)
    for key in ['batch_size', 'limit', 'num_fewshot', 'output_path', 
                'log_samples', 'cache_requests', 'check_integrity']:
        if key in config_values:
            result[key] = config_values[key]
    
    return result


__all__ = [
    'get_config_path',
    'load_evaluation_from_config',
    'save_evaluation_to_config',
    'merge_config_with_defaults',
]
