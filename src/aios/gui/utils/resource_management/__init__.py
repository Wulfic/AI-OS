"""Resource management utilities for long-running stability.

This module provides utilities for managing threads, processes, and other
resources to prevent leaks and ensure clean shutdown during extended runs.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Optional

from .thread_management import ManagedThread
from .process_cleanup import ProcessReaper
from .timer_utils import TimerManager
from .monitoring import ResourceMonitor
from .async_pool import AsyncWorkerPool
from .async_loop import AsyncEventLoop

# Shared worker pool registry -------------------------------------------------

_worker_pool_lock: RLock = RLock()
_worker_pool_ref: Optional[Any] = None
_async_loop_lock: RLock = RLock()
_async_loop_ref: Optional[Any] = None


def set_worker_pool(pool: Optional[Any]) -> None:
    """Register the shared AsyncWorkerPool instance for global access."""
    global _worker_pool_ref
    with _worker_pool_lock:
        _worker_pool_ref = pool


def clear_worker_pool(pool: Optional[Any] = None) -> None:
    """Clear the registered worker pool if it matches the provided instance."""
    global _worker_pool_ref
    with _worker_pool_lock:
        if pool is None or _worker_pool_ref is pool:
            _worker_pool_ref = None


def get_worker_pool() -> Optional[Any]:
    """Return the shared AsyncWorkerPool if one has been registered."""
    with _worker_pool_lock:
        return _worker_pool_ref


def set_async_loop(loop: Optional[Any]) -> None:
    """Register the shared AsyncEventLoop instance for global access."""
    global _async_loop_ref
    with _async_loop_lock:
        _async_loop_ref = loop


def get_async_loop() -> Optional[Any]:
    """Return the shared AsyncEventLoop if one has been registered."""
    with _async_loop_lock:
        return _async_loop_ref


from .task_dispatcher import submit_background, snapshot_metrics as get_dispatcher_metrics  # noqa: E402


__all__ = [
    "ManagedThread",
    "ProcessReaper",
    "TimerManager",
    "ResourceMonitor",
    "AsyncWorkerPool",
    "AsyncEventLoop",
    "set_worker_pool",
    "clear_worker_pool",
    "get_worker_pool",
    "set_async_loop",
    "get_async_loop",
    "submit_background",
    "get_dispatcher_metrics",
]
