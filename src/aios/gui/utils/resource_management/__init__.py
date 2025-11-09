"""Resource management utilities for long-running stability.

This module provides utilities for managing threads, processes, and other
resources to prevent leaks and ensure clean shutdown during extended runs.
"""

from __future__ import annotations

from .thread_management import ManagedThread
from .process_cleanup import ProcessReaper
from .timer_utils import TimerManager
from .monitoring import ResourceMonitor
from .async_pool import AsyncWorkerPool
from .async_loop import AsyncEventLoop


__all__ = [
    "ManagedThread",
    "ProcessReaper",
    "TimerManager",
    "ResourceMonitor",
    "AsyncWorkerPool",
    "AsyncEventLoop",
]
