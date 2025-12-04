"""Configuration file persistence for Resources panel settings.

This module provides functions to load/save resources settings from/to config/default.yaml,
making the YAML config file the single source of truth for resource configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
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


def load_resources_from_config() -> dict[str, Any]:
    """Load resources settings from config/default.yaml.
    
    Returns:
        Dict of resources settings, or empty dict if config doesn't exist or has no resources section
    """
    try:
        config = load_full_config()
        
        if not config:
            logger.debug("No config loaded")
            return {}
        
        resources = config.get('resources', {})
        if not isinstance(resources, dict):
            logger.warning("Resources section in config is not a dict")
            return {}
        art_dir = resources.get('artifacts_dir')
        if art_dir is not None and not isinstance(art_dir, str):
            try:
                resources['artifacts_dir'] = str(art_dir)
            except Exception:
                resources.pop('artifacts_dir', None)
        
        logger.info("Successfully loaded resource config")
        logger.debug(f"Resource config: {resources}")
        return resources
    except Exception as e:
        logger.error(f"Failed to load resource config: {e}", exc_info=True)
        return {}


def save_resources_to_config(resources_values: dict[str, Any]) -> bool:
    """Save resources settings to config/default.yaml.
    
    Args:
        resources_values: Dict of resources settings from ResourcesPanel.get_values()
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        logger.info("Saving resource config")
        logger.debug(f"Resource values: {resources_values}")
        
        # Load existing config
        config = load_full_config()
        
        # Update resources section
        if 'resources' not in config:
            config['resources'] = {}
        
        # Merge new values into resources section
        resources = config['resources']
        if not isinstance(resources, dict):
            resources = {}
            config['resources'] = resources
        
        # Update all resource fields
        resources['cpu_threads'] = resources_values.get('cpu_threads', 0)
        resources['gpu_mem_pct'] = resources_values.get('gpu_mem_pct', 90)
        resources['cpu_util_pct'] = resources_values.get('cpu_util_pct', 0)
        resources['gpu_util_pct'] = resources_values.get('gpu_util_pct', 0)
        
        # Device selections
        resources['train_device'] = resources_values.get('train_device', 'auto')
        resources['train_cuda_selected'] = resources_values.get('train_cuda_selected', [])
        resources['train_cuda_mem_pct'] = resources_values.get('train_cuda_mem_pct', {})
        resources['train_cuda_util_pct'] = resources_values.get('train_cuda_util_pct', {})
        
        # Training mode (DDP vs Parallel)
        resources['training_mode'] = resources_values.get('training_mode', 'ddp')
        
        resources['run_device'] = resources_values.get('run_device', 'auto')
        resources['run_cuda_selected'] = resources_values.get('run_cuda_selected', [])
        resources['run_cuda_mem_pct'] = resources_values.get('run_cuda_mem_pct', {})
        resources['run_cuda_util_pct'] = resources_values.get('run_cuda_util_pct', {})
        
        # Storage caps
        resources['dataset_cap'] = resources_values.get('dataset_cap', '')

        # DeepSpeed ZeRO stage selection (shared with training panel)
        resources['zero_stage'] = resources_values.get('zero_stage', 'none')
        
        # System RAM limit
        system_mem_limit = resources_values.get('system_mem_limit_gb', None)
        if system_mem_limit and str(system_mem_limit).strip():
            try:
                resources['system_mem_limit_gb'] = float(system_mem_limit)
            except (ValueError, TypeError):
                resources['system_mem_limit_gb'] = None
        else:
            resources['system_mem_limit_gb'] = None

        # Artifacts directory override
        artifacts_dir = (resources_values.get('artifacts_dir') or '').strip()
        if artifacts_dir:
            resources['artifacts_dir'] = artifacts_dir
        else:
            resources.pop('artifacts_dir', None)
        
        # Save config back to file
        success = save_full_config(config)
        
        if success:
            logger.info("Successfully saved resource config")
            return True
        else:
            logger.error("Failed to save resource config")
            raise RuntimeError("Failed to write config file")
        
    except Exception as e:
        logger.error(f"Failed to save resource config: {e}", exc_info=True)
        raise e


def merge_config_with_defaults(config_values: dict[str, Any], default_values: dict[str, Any]) -> dict[str, Any]:
    """Merge config values with defaults, preferring config values when present.
    
    Args:
        config_values: Values loaded from config file
        default_values: Default values (e.g., from GUI state or hardcoded defaults)
    
    Returns:
        Merged dict with config values taking precedence
    """
    result = dict(default_values)
    
    # Override with config values
    for key, value in config_values.items():
        # Only override if config value is meaningful (not None, not empty string for most fields)
        if key in ['cpu_threads', 'gpu_mem_pct', 'cpu_util_pct', 'gpu_util_pct']:
            if isinstance(value, (int, float)) and value >= 0:
                result[key] = value
        elif key in ['train_device', 'run_device']:
            if value in ['auto', 'cpu', 'cuda']:
                result[key] = value
        elif key in ['train_cuda_selected', 'run_cuda_selected']:
            if isinstance(value, list):
                result[key] = value
        elif key in ['train_cuda_mem_pct', 'train_cuda_util_pct', 'run_cuda_mem_pct', 'run_cuda_util_pct']:
            if isinstance(value, dict):
                result[key] = value
        elif key in ['dataset_cap', 'model_cap', 'per_brain_cap', 'zero_stage']:
            # Allow empty strings for storage caps
            result[key] = value
        elif key == 'system_mem_limit_gb':
            result[key] = value
        else:
            result[key] = value
    
    return result


__all__ = [
    'get_config_path',
    'load_resources_from_config',
    'save_resources_to_config',
    'merge_config_with_defaults',
]
