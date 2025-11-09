"""Resource management setup for AI-OS GUI application.

This module handles initialization of all resource management utilities:
- Managed thread pools
- Process reaper for subprocess cleanup
- Timer management
- Resource monitoring
- Async worker pools
- Async event loop
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk

from ..utils.resource_management import (
    AsyncEventLoop,
    AsyncWorkerPool,
    ManagedThread,
    ProcessReaper,
    ResourceMonitor,
    TimerManager,
)

logger = logging.getLogger(__name__)


def setup_resources(app: Any, root: "tk.Tk", start_minimized: bool) -> None:  # type: ignore[name-defined]
    """Initialize all resource management utilities for the application.
    
    Args:
        app: The AiosTkApp instance
        root: The Tkinter root window
        start_minimized: Whether to start with window minimized to tray
    
    Sets up:
        - _managed_threads: List of managed threads for clean shutdown
        - _process_reaper: Process cleanup utility
        - _timer_manager: Debounced timer management
        - _resource_monitor: System resource monitoring
        - _worker_pool: Async worker pool for background tasks
        - _async_loop: Async event loop for GUI responsiveness
    """
    # Store minimized flag
    app._start_minimized = start_minimized
    
    # Resource management utilities
    app._managed_threads: list[ManagedThread] = []
    app._process_reaper = ProcessReaper()
    app._timer_manager = TimerManager(root)
    app._resource_monitor = ResourceMonitor()
    
    # Async worker pool (configurable via environment variable)
    # Default: (cpu_count * 2) + 1 for I/O-bound GUI operations
    worker_count = None
    try:
        env_workers = os.environ.get("AIOS_WORKER_THREADS")
        if env_workers and env_workers.isdigit():
            worker_count = int(env_workers)
            logger.info(f"Using custom worker count from AIOS_WORKER_THREADS: {worker_count}")
    except Exception:
        pass
    
    app._worker_pool = AsyncWorkerPool(max_workers=worker_count)
    logger.info(f"Initialized worker pool with {app._worker_pool.max_workers} workers")
    
    # Async event loop for GUI responsiveness
    app._async_loop = AsyncEventLoop()
    app._async_loop.start()
    logger.info("Async event loop started for GUI responsiveness")
    
    # Register emergency cleanup
    import atexit
    atexit.register(app._emergency_cleanup)
    logger.info("Emergency cleanup registered with atexit")
