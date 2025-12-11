"""Persistent GPU detection cache used by the resources panel."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from aios.system import paths as system_paths
except ImportError:
    system_paths = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CACHE_RELATIVE_DIR = Path("artifacts") / "diagnostics"
_CACHE_FILENAME = "device_info_cache.json"
_DEFAULT_MAX_AGE_SECONDS = 12 * 3600  # 12 hours


def _detect_repository_root() -> Path:
    """Best-effort detection of the repository root for cache placement."""
    # Try centralized paths first
    if system_paths is not None:
        try:
            return system_paths.get_install_root()
        except Exception:
            logger.debug("Failed to get install root from system paths", exc_info=True)
    
    # Fallback to module-based detection
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback: go up to \"src\" parent if present, else use the highest known parent
    for candidate in current.parents:
        if candidate.name.lower() == "src":
            return candidate.parent
    return current.parents[-1]


def _normalize_root(project_root: Path | str | None) -> Path:
    """Return the repository root where cache files should be stored."""
    if project_root is None:
        return _detect_repository_root()

    try:
        root = Path(project_root).resolve()
    except Exception:
        logger.debug("Failed to resolve project_root=%s, falling back to module root", project_root)
        return _detect_repository_root()

    if root.name.lower() == "src" and root.parent.exists():
        return root.parent
    return root


def _cache_path(project_root: Path | str | None) -> Path:
    """Return the filesystem path to the device cache file."""
    # Use centralized paths if available and no explicit root provided
    if project_root is None and system_paths is not None:
        try:
            return system_paths.get_artifacts_root() / "diagnostics" / _CACHE_FILENAME
        except Exception:
            logger.debug("Failed to get artifacts root from system paths", exc_info=True)
    
    root = _normalize_root(project_root)
    return root / _CACHE_RELATIVE_DIR / _CACHE_FILENAME


def load_device_cache(
    project_root: Path | str | None,
    *,
    max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS,
) -> Optional[Dict[str, Any]]:
    """Load cached GPU detection results if they are still fresh."""
    cache_file = _cache_path(project_root)
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.debug("Failed to read device cache %s: %s", cache_file, exc)
        return None

    timestamp = payload.get("timestamp")
    data = payload.get("data")
    if not isinstance(timestamp, (int, float)) or not isinstance(data, dict):
        logger.debug("Device cache payload invalid; ignoring")
        return None

    age = time.time() - float(timestamp)
    if max_age_seconds > 0 and age > max_age_seconds:
        logger.debug("Device cache expired (age %.1fs, max %.1fs)", age, max_age_seconds)
        return None

    info: Dict[str, Any] = dict(data)
    info.setdefault("source", "cache")
    info["_cache_timestamp"] = float(timestamp)
    info["_cache_age_seconds"] = float(age)
    info["_from_cache"] = True
    return info


def save_device_cache(project_root: Path | str | None, info: Dict[str, Any]) -> None:
    """Persist GPU detection results for future startups."""
    cache_file = _cache_path(project_root)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("Could not create cache directory %s: %s", cache_file.parent, exc)
        return

    sanitized = {k: v for k, v in info.items() if not str(k).startswith("_")}
    payload = {
        "timestamp": time.time(),
        "data": sanitized,
    }

    try:
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        logger.debug("Failed to write device cache %s: %s", cache_file, exc)


__all__ = ["load_device_cache", "save_device_cache"]
