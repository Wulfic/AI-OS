"""Configuration file persistence for Evaluation panel settings.

This module provides functions to load/save evaluation settings from/to config/default.yaml,
making the YAML config file the single source of truth for evaluation configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
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
        repo_root = module_path.parents[5]  # Up from src/aios/gui/components/evaluation_panel
        config_path = repo_root / "config" / "default.yaml"
        if config_path.exists():
            return config_path
    except Exception:
        pass
    
    # Fall back to cwd path even if it doesn't exist
    return Path.cwd() / "config" / "default.yaml"


def load_evaluation_from_config() -> dict[str, Any]:
    """Load evaluation settings from config/default.yaml.
    
    Returns:
        Dict of evaluation settings, or empty dict if config doesn't exist or has no evaluation section
    """
    try:
        config_path = get_config_path()
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        if not isinstance(config, dict):
            return {}
        
        evaluation = config.get('evaluation', {})
        if not isinstance(evaluation, dict):
            return {}
        
        return evaluation
    except Exception:
        return {}


def save_evaluation_to_config(evaluation_values: dict[str, Any]) -> bool:
    """Save evaluation settings to config/default.yaml.
    
    Args:
        evaluation_values: Dict of evaluation settings from EvaluationPanel.get_state()
    
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
        
        return True
    except Exception as e:
        print(f"[Evaluation] Failed to save config: {e}")
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
