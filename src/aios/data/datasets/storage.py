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


def datasets_base_dir() -> Path:
    """Return the base directory for storing datasets.

    Resolution order (first match wins):
    1) Environment override via `AIOS_DATASETS_DIR`
    2) Project root detected from CWD (pyproject.toml/.git) → `training_data/curated_datasets`
    3) Fallback: `~/.local/share/aios/datasets`
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

    # 2) Try to detect project root from CWD and place under training_data/curated_datasets
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
        base = root / "training_data" / "curated_datasets"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return base

    # 3) Fallback to user-local data dir
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
    """
    try:
        cap = float(cap_gb)
        if not (cap > 0):
            return False
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
