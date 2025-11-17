"""Lightweight helpers for reading brain registry state without spawning CLI processes.

These utilities mirror the information surfaced by the ``brains`` CLI commands but run
in-process so that GUI panels can refresh instantly. Results are cached briefly to
avoid excessive disk I/O while staying responsive to user actions.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

from aios.core.brains.registry_core import BrainRegistry

__all__ = [
    "list_brains",
    "get_brain_stats",
    "invalidate_brain_cache",
]

# Default cache TTL keeps UI snappy while reducing redundant disk reads.
_DEFAULT_TTL = 3.0  # seconds

# Keys are (section, store_dir); values are (timestamp, payload)
_Cache = Dict[Tuple[str, str], Tuple[float, Any]]
_cache: _Cache = {}
_cache_lock = threading.Lock()

# Maintain reusable registry instances keyed by store directory so that we
# avoid re-reading pinned/master metadata (which is verbose at INFO level).
_registries: Dict[str, BrainRegistry] = {}
_registry_lock = threading.Lock()

# Temporary/ephemeral brains that should not surface in UI lists.
_TEMP_PATTERN = re.compile(r"^brain-[a-z]+-[0-9a-f]{8}$")
_SYSTEM_NAMES = {
    "parallel_checkpoints",
    "checkpoints",
    "temp",
    "tmp",
    ".git",
    "ddp_logs",
}


def _normalize_store_dir(store_dir: str | None) -> str:
    base = store_dir or "artifacts/brains"
    return os.path.abspath(base)


def _build_registry(store_dir: str) -> BrainRegistry:
    with _registry_lock:
        registry = _registries.get(store_dir)
        if registry is None:
            registry = BrainRegistry()
            registry.store_dir = store_dir
            try:
                registry.load_pinned()
            except Exception:
                pass
            try:
                registry.load_masters()
            except Exception:
                pass
            _registries[store_dir] = registry
    return registry


def _get_cached(key: Tuple[str, str], ttl: float) -> Any | None:
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (now - entry[0]) < ttl:
            return entry[1]
    return None


def _set_cached(key: Tuple[str, str], payload: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.time(), payload)


def _filter_visible(names: Iterable[str]) -> List[str]:
    visible: List[str] = []
    for name in names:
        if not name:
            continue
        if name.startswith("_"):
            continue
        if name in _SYSTEM_NAMES:
            continue
        if _TEMP_PATTERN.match(name):
            continue
        visible.append(name)
    return sorted(visible)


def list_brains(store_dir: str | None, *, ttl: float = _DEFAULT_TTL) -> List[str]:
    """Return the list of available brains without spawning the CLI."""
    path = _normalize_store_dir(store_dir)
    key = ("list", path)
    cached = _get_cached(key, ttl)
    if cached is not None:
        return list(cached)

    registry = _build_registry(path)
    names = _filter_visible(registry.list())
    _set_cached(key, names)
    return names


def get_brain_stats(store_dir: str | None, *, ttl: float = _DEFAULT_TTL) -> Dict[str, Any]:
    """Return registry statistics matching ``brains stats`` output."""
    path = _normalize_store_dir(store_dir)
    key = ("stats", path)
    cached = _get_cached(key, ttl)
    if cached is not None:
        return dict(cached)

    registry = _build_registry(path)
    stats = registry.stats()
    _set_cached(key, stats)
    return stats


def invalidate_brain_cache(store_dir: str | None = None) -> None:
    """Invalidate cached registry data for the provided store directory."""
    path = _normalize_store_dir(store_dir)
    with _cache_lock:
        for k in list(_cache.keys()):
            if k[1] == path:
                del _cache[k]
    with _registry_lock:
        _registries.pop(path, None)
