"""
HuggingFace Cache Setup

IMPORTANT: This module must be imported BEFORE any HuggingFace imports
to properly set environment variables for cache directories.

Sets up smart default cache directories on non-C drives when possible.
"""

import os
import logging
from pathlib import Path

# Get logger for warnings (may not be configured yet during early import)
logger = logging.getLogger(__name__)


def setup_hf_cache_env():
    """
    Set Hugging Face cache environment variables if not already set.
    
    Uses smart defaults that work on fresh install:
    1. Check config file for user preference
    2. Try non-C data drives (D, E, F, Z)
    3. Fall back to workspace training_datasets/hf_cache
    
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
    _configured_path = None
    if _config_file.exists():
        try:
            _configured_path = Path(_config_file.read_text().strip())
            _hf_cache_base = _configured_path
        except Exception as e:
            logger.warning(f"Failed to read HF cache config from {_config_file}: {e}")
            _hf_cache_base = None
    else:
        _hf_cache_base = None
    
    # Use install root default if no config
    if not _hf_cache_base or not (_hf_cache_base.exists() or _hf_cache_base.parent.exists()):
        # Default to install root location
        _hf_cache_base = Path.cwd() / "training_datasets" / "hf_cache"
        _configured_path = None  # Clear since we're using default
    
    # Try to create the directory
    try:
        _hf_cache_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using HF cache directory: {_hf_cache_base}")
    except Exception as e:
        # Only fall back if we weren't using a user-configured path
        if _configured_path:
            # User explicitly configured a path that failed - warn them loudly
            logger.error(f"⚠️  CRITICAL: Cannot access configured HF cache location: {_configured_path}")
            logger.error(f"   Error: {e}")
            logger.error(f"   The configured path may be on an unmounted drive or have permission issues.")
            logger.error(f"   Please check your configuration at: {_config_file}")
            logger.error(f"   Datasets will NOT be downloaded until this is resolved.")
            # Don't fall back silently - let it fail visibly
            raise RuntimeError(
                f"Cannot access configured HF cache at {_configured_path}. "
                f"Check if drive is mounted and path is writable. Config: {_config_file}"
            ) from e
        else:
            # Default path failed, try one more fallback
            logger.warning(f"Failed to create HF cache at {_hf_cache_base}: {e}")
            _hf_cache_base = Path.cwd() / "training_datasets" / "hf_cache"
            try:
                _hf_cache_base.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using fallback HF cache directory: {_hf_cache_base}")
            except Exception as e2:
                logger.error(f"Cannot create HF cache directory at {_hf_cache_base}: {e2}")
                raise
    
    os.environ["HF_HOME"] = str(_hf_cache_base.resolve())
    os.environ["HF_DATASETS_CACHE"] = str((_hf_cache_base / "datasets").resolve())
    os.environ["HF_HUB_CACHE"] = str((_hf_cache_base / "hub").resolve())
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Set HF_XET_CACHE to avoid permission errors with xet-core library
    # xet-core is used by huggingface_hub for fast downloads but has strict cache requirements
    _xet_cache = _hf_cache_base / "xet"
    try:
        _xet_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_XET_CACHE"] = str(_xet_cache.resolve())
        logger.debug(f"Set HF_XET_CACHE to: {_xet_cache}")
    except PermissionError:
        # Fall back to user home directory if download location not writable
        _xet_fallback = Path.home() / ".cache" / "huggingface" / "xet"
        try:
            _xet_fallback.mkdir(parents=True, exist_ok=True)
            os.environ["HF_XET_CACHE"] = str(_xet_fallback.resolve())
            logger.warning(f"Using fallback XET cache at: {_xet_fallback}")
        except Exception as e3:
            logger.warning(f"Could not set up XET cache: {e3}")
    except Exception as e2:
        logger.warning(f"Could not create XET cache at {_xet_cache}: {e2}")


# Run setup on import
setup_hf_cache_env()
