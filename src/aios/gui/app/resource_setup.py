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
    get_dispatcher_metrics,
    submit_background,
    set_async_loop,
    set_worker_pool,
)
from ..services import TkUiDispatcher
from aios.utils.diagnostics import enable_asyncio_diagnostics

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
    logger.info("Initializing resource management subsystems...")
    
    # Store minimized flag
    app._start_minimized = start_minimized
    
    logger.debug("Creating resource manager helpers")
    # Resource management utilities
    app._managed_threads: list[ManagedThread] = []
    app._process_reaper = ProcessReaper()
    app._timer_manager = TimerManager(root)
    app._resource_monitor = ResourceMonitor()
    logger.debug("Basic resource managers initialized (threads, process reaper, timers, monitor)")

    # Thread-safe UI dispatcher for cross-thread updates
    logger.info("Starting Tk UI dispatcher")
    app._ui_dispatcher = TkUiDispatcher(root)
    logger.info("Tk UI dispatcher ready")
    
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
    
    logger.info("Creating async worker pool (requested_workers=%s)", worker_count)
    app._worker_pool = AsyncWorkerPool(max_workers=worker_count)
    set_worker_pool(app._worker_pool)
    logger.info("Initialized worker pool with %s workers", app._worker_pool.max_workers)
    app.dispatch_background = lambda label, fn, *a, **kw: submit_background(
        label,
        fn,
        *a,
        pool=app._worker_pool,
        **kw,
    )
    app.get_background_metrics = get_dispatcher_metrics
    
    try:
        enable_asyncio_diagnostics()
        logger.debug("Asyncio diagnostics enabled for GUI event loops")
    except Exception:
        logger.debug("Failed to enable asyncio diagnostics", exc_info=True)

    # Async event loop for GUI responsiveness
    logger.info("Starting async event loop thread")
    app._async_loop = AsyncEventLoop()
    app._async_loop.start()
    set_async_loop(app._async_loop)
    logger.info("Async event loop started for GUI responsiveness")
    
    # Register emergency cleanup
    import atexit
    atexit.register(app._emergency_cleanup)
    logger.debug("Emergency cleanup registered with atexit")
    
    logger.info("Resource management subsystems initialized successfully")
