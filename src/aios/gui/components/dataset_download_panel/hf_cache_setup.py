"""
HuggingFace Cache Setup

IMPORTANT: This module must be imported BEFORE any HuggingFace imports
to properly set environment variables for cache directories.

Sets up smart default cache directories on non-C drives when possible.
"""

import os
from pathlib import Path


def setup_hf_cache_env():
    """
    Set Hugging Face cache environment variables if not already set.
    
    Uses smart defaults that work on fresh install:
    1. Check config file for user preference
    2. Try non-C data drives (D, E, F, Z)
    3. Fall back to workspace training_data/hf_cache
    
    Environment variables set:
    - HF_HOME: Base HuggingFace directory
    - HF_DATASETS_CACHE: Datasets cache directory
    - HF_HUB_CACHE: Model hub cache directory
    - HF_HUB_DISABLE_SYMLINKS_WARNING: Disable symlink warnings
    """
    if os.environ.get("HF_HOME"):
        return  # Already configured
    
    # Read from config file if exists
    _config_file = Path.home() / ".config" / "aios" / "hf_cache_path.txt"
    if _config_file.exists():
        try:
            _hf_cache_base = Path(_config_file.read_text().strip())
        except Exception:
            _hf_cache_base = None
    else:
        _hf_cache_base = None
    
    # Use smart defaults if no config
    if not _hf_cache_base or not (_hf_cache_base.exists() or _hf_cache_base.parent.exists()):
        # Try non-C data drives first
        for _drive in ["D", "E", "F", "Z"]:
            try:
                _drive_path = Path(f"{_drive}:/")
                if _drive_path.exists():
                    _hf_cache_base = _drive_path / "AI-OS-Data" / "hf_cache"
                    break
            except Exception:
                continue
        else:
            _hf_cache_base = Path.cwd() / "training_data" / "hf_cache"
    
    try:
        _hf_cache_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        _hf_cache_base = Path.cwd() / "training_data" / "hf_cache"
        _hf_cache_base.mkdir(parents=True, exist_ok=True)
    
    os.environ["HF_HOME"] = str(_hf_cache_base.resolve())
    os.environ["HF_DATASETS_CACHE"] = str((_hf_cache_base / "datasets").resolve())
    os.environ["HF_HUB_CACHE"] = str((_hf_cache_base / "hub").resolve())
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Run setup on import
setup_hf_cache_env()
