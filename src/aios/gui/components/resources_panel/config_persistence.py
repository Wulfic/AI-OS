"""Configuration file persistence for Resources panel settings.

This module provides functions to load/save resources settings from/to config/default.yaml,
making the YAML config file the single source of truth for resource configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import yaml


def get_config_path() -> Path:
    """Get the path to the config/default.yaml file.
    
    Returns:
        Path to config/default.yaml (may not exist yet)
    """
    # Try current working directory first
    config_path = Path.cwd() / "config" / "default.yaml"
    if config_path.exists():
        return config_path
    
    # Try relative to this file (for installed packages)
    try:
        module_path = Path(__file__).resolve()
        repo_root = module_path.parents[5]  # Up from src/aios/gui/components/resources_panel
        config_path = repo_root / "config" / "default.yaml"
        if config_path.exists():
            return config_path
    except Exception:
        pass
    
    # Fall back to cwd path even if it doesn't exist
    return Path.cwd() / "config" / "default.yaml"


def load_resources_from_config() -> dict[str, Any]:
    """Load resources settings from config/default.yaml.
    
    Returns:
        Dict of resources settings, or empty dict if config doesn't exist or has no resources section
    """
    try:
        config_path = get_config_path()
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        if not isinstance(config, dict):
            return {}
        
        resources = config.get('resources', {})
        if not isinstance(resources, dict):
            return {}
        
        return resources
    except Exception:
        return {}


def save_resources_to_config(resources_values: dict[str, Any]) -> bool:
    """Save resources settings to config/default.yaml.
    
    Args:
        resources_values: Dict of resources settings from ResourcesPanel.get_values()
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        config_path = get_config_path()
        
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        config: dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        
        if not isinstance(config, dict):
            config = {}
        
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
        
        # System RAM limit
        system_mem_limit = resources_values.get('system_mem_limit_gb', None)
        if system_mem_limit and str(system_mem_limit).strip():
            try:
                resources['system_mem_limit_gb'] = float(system_mem_limit)
            except (ValueError, TypeError):
                resources['system_mem_limit_gb'] = None
        else:
            resources['system_mem_limit_gb'] = None
        
        # Write config back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return True
    except Exception as e:
        print(f"[Resources] Failed to save config: {e}")
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
        elif key in ['dataset_cap', 'model_cap', 'per_brain_cap']:
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
