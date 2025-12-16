"""Dataset storage management and configuration."""

from __future__ import annotations

import os
from pathlib import Path
from .constants import _DATASETS_CAP_GB


def _cap_config_path() -> Path:
    """Return path to the dataset configuration file."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    base = Path(home) if home else Path.home()
    p = base / ".config" / "aios" / "datasets.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p


def _get_hf_cache_config_path() -> Path:
    """Return path to the HF cache config file."""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    base = Path(home) if home else Path.home()
    return base / ".config" / "aios" / "hf_cache_path.txt"


def _get_configured_datasets_root() -> Path | None:
    """Get the user-configured datasets root from HF_HOME or saved config.
    
    The datasets root is the parent of HF_HOME (which typically points to .hf_cache).
    For example, if HF_HOME is Z:/training_datasets/.hf_cache, returns Z:/training_datasets.
    """
    # Check HF_HOME environment variable first
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_path = Path(hf_home).expanduser().resolve()
        # Parent of .hf_cache is the datasets root
        if hf_path.name in (".hf_cache", "hf_cache"):
            return hf_path.parent
        # If HF_HOME doesn't end in hf_cache, use it directly
        return hf_path.parent if hf_path.exists() else hf_path
    
    # Check saved config file
    config_file = _get_hf_cache_config_path()
    if config_file.exists():
        try:
            saved_path = config_file.read_text().strip()
            if saved_path:
                saved = Path(saved_path).expanduser().resolve()
                # Return parent (datasets root)
                if saved.name in (".hf_cache", "hf_cache"):
                    return saved.parent
                return saved.parent if saved.exists() else saved
        except Exception:
            pass
    
    return None


def datasets_base_dir() -> Path:
    """Return the base directory for storing curated datasets.

    Resolution order (first match wins):
    1) Environment override via `AIOS_DATASETS_DIR`
    2) User-configured datasets root (HF_HOME parent or saved config) → `{root}/curated_datasets`
    3) Project root detected from CWD (pyproject.toml/.git) → `{project}/training_datasets/curated_datasets`
    4) Fallback: `~/.local/share/aios/datasets`
    
    Curated datasets are placed in a 'curated_datasets' subfolder alongside
    HuggingFace datasets to keep all training data together.
    """
    # 1) Explicit override
    override = os.environ.get("AIOS_DATASETS_DIR")
    if override:
        p = Path(override).expanduser().resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    # 2) Use user-configured datasets root (from HF_HOME or saved config)
    configured_root = _get_configured_datasets_root()
    if configured_root is not None:
        base = configured_root / "curated_datasets"
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base
        except Exception:
            pass  # Fall through to next option

    # 3) Try to detect project root from CWD and place under training_datasets/curated_datasets
    def _find_project_root(start: Path) -> Path | None:
        cur = start
        # search up to filesystem root
        while True:
            if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
                return cur
            parent = cur.parent
            if parent == cur:
                return None
            cur = parent

    cwd = Path.cwd()
    root = _find_project_root(cwd)
    if root is not None:
        # Use training_datasets (not training_data) for consistency
        base = root / "training_datasets" / "curated_datasets"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return base

    # 4) Fallback to user-local data dir
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        base = Path(home) / ".local" / "share" / "aios" / "datasets"
    else:
        base = Path.home() / ".local" / "share" / "aios" / "datasets"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base


def _dir_size_bytes(path: Path) -> int:
    """Calculate total size of all files in a directory tree."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                continue
    except Exception:
        pass
    return int(total)


def datasets_storage_usage_gb() -> float:
    """Compute total size of dataset storage directory in GB.
    
    Includes both curated datasets and HuggingFace datasets.
    When HF_HOME is set, scans the parent directory (which contains both
    the .hf_cache and downloaded dataset directories).
    """
    total_bytes = 0
    
    # Add curated datasets directory
    base = datasets_base_dir()
    total_bytes += _dir_size_bytes(base)
    
    # Add HuggingFace datasets directory if HF_HOME is set
    # Use parent directory since datasets are stored alongside .hf_cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_path = Path(hf_home).expanduser().resolve()
        # Get parent directory (e.g., Z:\training_datasets instead of Z:\training_datasets\.hf_cache)
        hf_parent = hf_path.parent
        if hf_parent.exists() and hf_parent != base and hf_parent != base.parent:
            total_bytes += _dir_size_bytes(hf_parent)
    
    return total_bytes / (1024 ** 3)


def datasets_storage_cap_gb() -> float:
    """Return the configured cap (GB) for dataset storage.

    Reads ~/.config/aios/datasets.json {"cap_gb": float} when present; falls back to default.
    """
    try:
        p = _cap_config_path()
        if p.exists():
            import json as _json
            with p.open("r", encoding="utf-8") as f:
                data = _json.load(f) or {}
            cap = float(data.get("cap_gb", _DATASETS_CAP_GB))
            if cap > 0:
                return cap
    except Exception:
        pass
    return _DATASETS_CAP_GB


def set_datasets_storage_cap_gb(cap_gb: float) -> bool:
    """Persistently set the dataset storage cap in GB.

    Stores value under ~/.config/aios/datasets.json.
    A value of 0 means unlimited storage.
    """
    try:
        cap = float(cap_gb)
        if cap < 0:
            return False  # Negative values are invalid
        import json as _json
        p = _cap_config_path()
        with p.open("w", encoding="utf-8") as f:
            _json.dump({"cap_gb": cap}, f)
        return True
    except Exception:
        return False


def can_store_additional_gb(required_gb: float) -> bool:
    """Check if storage has capacity for additional data."""
    try:
        used = datasets_storage_usage_gb()
        return (used + float(required_gb)) <= datasets_storage_cap_gb()
    except Exception:
        return False
