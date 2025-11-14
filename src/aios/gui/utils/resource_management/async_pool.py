"""Async worker pool for background task execution."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from typing import Callable, Dict, Optional, Set


logger = logging.getLogger(__name__)


class AsyncWorkerPool:
    """Thread pool executor for running background tasks with CPU-aware worker count.
    
    Creates at least 1 worker per CPU thread for optimal responsiveness.
    Manages async task submission and graceful shutdown.
    
    Usage:
        pool = AsyncWorkerPool(max_workers=None)  # Auto-detects CPU count
        
        # Submit work
        future = pool.submit(my_function, arg1, arg2)
        result = future.result()  # Blocking
        
        # Submit multiple tasks
        futures = [pool.submit(work, i) for i in range(10)]
        
        # On shutdown
        pool.shutdown(wait=True, timeout=5.0)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize worker pool.
        
        Args:
            max_workers: Number of workers. If None, uses (cpu_count * 4) + 1
                        This provides better responsiveness for I/O-bound GUI operations.
        """
        if max_workers is None:
            # For GUI apps with async I/O operations, use more workers than CPUs.
            cpu_count = os.cpu_count() or 4
            max_workers = max((cpu_count * 4) + 1, 12)

            # Windows thread scheduling struggles with very large pools; cap more
            # aggressively to keep the UI thread responsive.
            if platform.system() == "Windows":
                windows_cap = max((cpu_count * 2) + 4, 12)
                max_workers = min(max_workers, windows_cap)
        
        self._max_workers = max_workers
        logger.debug(
            "Creating ThreadPoolExecutor with max_workers=%s, thread_name_prefix='AsyncWorker'",
            max_workers,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncWorker"
        )
        self._shutdown = False
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._queued = 0
        self._in_flight: Set[Future] = set()
        self._future_labels: Dict[Future, str] = {}
        self._future_started: Dict[Future, float] = {}
        logger.info(f"AsyncWorkerPool initialized with {max_workers} workers (CPUs: {os.cpu_count() or 'unknown'})")
        logger.debug(f"ThreadPoolExecutor created successfully: {max_workers} worker threads available")
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a callable to be executed in the worker pool.
        
        Args:
            fn: Callable to execute
            *args: Positional arguments to pass to fn
            **kwargs: Keyword arguments to pass to fn
            
        Returns:
            Future object representing the pending execution
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot submit to shutdown worker pool")
        
        task_label = getattr(fn, "_aios_task_label", None)
        task_name = task_label or (fn.__name__ if hasattr(fn, "__name__") else type(fn).__name__)
        logger.debug("Submitting task to AsyncWorkerPool: %s", task_name)
        try:
            future: Future = self._executor.submit(fn, *args, **kwargs)
        except RuntimeError as exc:
            with self._lock:
                self._shutdown = True
            raise RuntimeError("Worker pool is shutting down") from exc
        submitted_at = time.monotonic()
        if task_label is None:
            task_label = getattr(future, "_aios_task_label", None)
        if task_label is None:
            task_label = task_name
        setattr(future, "_aios_task_label", task_label)
        setattr(future, "_aios_submitted_at", submitted_at)
        with self._lock:
            self._submitted += 1
            self._queued += 1
            self._in_flight.add(future)
            self._future_labels[future] = task_label
            self._future_started[future] = submitted_at

        def _on_done(done_future: Future) -> None:
            with self._lock:
                self._completed += 1
                self._queued = max(0, self._queued - 1)
                self._in_flight.discard(done_future)
                self._future_labels.pop(done_future, None)
                self._future_started.pop(done_future, None)

        future.add_done_callback(_on_done)
        return future
    
    def map(self, fn: Callable, *iterables, timeout: Optional[float] = None, chunksize: int = 1):
        """Map a function over iterables using the worker pool.
        
        Args:
            fn: Callable to map
            *iterables: One or more iterables to map over
            timeout: Maximum seconds to wait for each call
            chunksize: Size of chunks to submit (larger = better for many small tasks)
            
        Returns:
            Iterator over results
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Cannot map on shutdown worker pool")
        
        logger.debug(f"Mapping function to AsyncWorkerPool: {fn.__name__ if hasattr(fn, '__name__') else type(fn).__name__} (chunksize={chunksize})")
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the worker pool.
        
        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Maximum seconds to wait (only used if wait=True)
        """
        with self._lock:
            if self._shutdown:
                logger.debug("AsyncWorkerPool already shutdown, skipping")
                return
            self._shutdown = True

        logger.info(
            "Shutting down AsyncWorkerPool with %s workers (wait=%s, timeout=%s)",
            self._max_workers,
            wait,
            timeout,
        )

        # Initiate shutdown without blocking to allow custom wait handling below.
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Older Python versions may not support cancel_futures (defensive). Fall back gracefully.
            self._executor.shutdown(wait=False)
        logger.debug("Executor shutdown initiated")

        if not wait:
            logger.info("AsyncWorkerPool shutdown initiated (non-blocking)")
            return

        deadline = None if timeout is None else (time.monotonic() + timeout)

        while True:
            with self._lock:
                pending = {f for f in self._in_flight if not f.done()}

            if not pending:
                break

            pending_info: list[str] = []
            with self._lock:
                now = time.monotonic()
                for future in pending:
                    label = self._future_labels.get(future, getattr(future, "_aios_task_label", "<unknown>"))
                    started = self._future_started.get(future, getattr(future, "_aios_submitted_at", now))
                    pending_info.append(f"{label} ({now - started:.2f}s)")

            if pending_info:
                logger.debug("Waiting for %d background task(s) to finish: %s", len(pending), ", ".join(pending_info))
            else:
                logger.debug("Waiting for %d background task(s) to finish", len(pending))
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            wait_timeout = None if remaining is None else max(0.0, remaining)
            done, not_done = wait_futures(pending, timeout=wait_timeout)

            if not not_done:
                break

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    extra = ", ".join(pending_info) if pending_info else ""
                    if extra:
                        logger.warning(
                            "Worker pool shutdown timed out with %d task(s) still running: %s",
                            len(not_done),
                            extra,
                        )
                    else:
                        logger.warning(
                            "Worker pool shutdown timed out with %d task(s) still running",
                            len(not_done),
                        )
                    break

        # Join worker threads to avoid leaks.
        threads = getattr(self._executor, "_threads", set())
        for thread in list(threads):
            join_timeout = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                join_timeout = remaining
            thread.join(join_timeout)
            if thread.is_alive():
                logger.warning("Worker thread %s did not exit before timeout", thread.name)

        with self._lock:
            self._in_flight.clear()
            self._queued = 0
            self._future_labels.clear()
            self._future_started.clear()

        logger.info("AsyncWorkerPool shutdown complete")
    
    @property
    def max_workers(self) -> int:
        """Get the number of workers in the pool."""
        return self._max_workers
    
    @property
    def is_shutdown(self) -> bool:
        """Check if pool is shutdown."""
        return self._shutdown

    @property
    def pending_tasks(self) -> int:
        """Return number of queued or running tasks."""
        with self._lock:
            return self._queued

    @property
    def submitted_tasks(self) -> int:
        """Return total submitted tasks since startup."""
        with self._lock:
            return self._submitted

    @property
    def completed_tasks(self) -> int:
        """Return total completed tasks since startup."""
        with self._lock:
            return self._completed
