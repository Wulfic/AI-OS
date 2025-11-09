"""Async worker pool for background task execution."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional


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
            # For GUI apps with async I/O operations, use more workers than CPUs
            # Formula: (cpu_count * 4) + 1 provides better balance for I/O-bound work
            # This allows multiple concurrent subprocess calls without blocking
            # Minimum of 12 workers to ensure responsiveness even on low-core systems
            cpu_count = os.cpu_count() or 4
            max_workers = max((cpu_count * 4) + 1, 12)
        
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncWorker"
        )
        self._shutdown = False
        logger.info(f"AsyncWorkerPool initialized with {max_workers} workers (CPUs: {os.cpu_count() or 'unknown'})")
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a callable to be executed in the worker pool.
        
        Args:
            fn: Callable to execute
            *args: Positional arguments to pass to fn
            **kwargs: Keyword arguments to pass to fn
            
        Returns:
            Future object representing the pending execution
        """
        if self._shutdown:
            raise RuntimeError("Cannot submit to shutdown worker pool")
        
        return self._executor.submit(fn, *args, **kwargs)
    
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
        if self._shutdown:
            raise RuntimeError("Cannot map on shutdown worker pool")
        
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the worker pool.
        
        Args:
            wait: If True, wait for pending tasks to complete
            timeout: Maximum seconds to wait (only used if wait=True)
        """
        if self._shutdown:
            return
        
        logger.info(f"Shutting down AsyncWorkerPool (wait={wait}, timeout={timeout})")
        self._shutdown = True
        
        if timeout is not None and wait:
            # Implement timeout-based shutdown
            import concurrent.futures
            start = time.time()
            
            try:
                self._executor.shutdown(wait=False)
                # Wait up to timeout for completion
                while time.time() - start < timeout:
                    # Check if all workers are done by attempting a quick submit
                    try:
                        future = self._executor.submit(lambda: None)
                        future.result(timeout=0.1)
                        break
                    except (concurrent.futures.BrokenExecutor, RuntimeError):
                        break
                    except Exception:
                        time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error during timed shutdown: {e}")
        else:
            self._executor.shutdown(wait=wait)
        
        logger.info("AsyncWorkerPool shutdown complete")
    
    @property
    def max_workers(self) -> int:
        """Get the number of workers in the pool."""
        return self._max_workers
    
    @property
    def is_shutdown(self) -> bool:
        """Check if pool is shutdown."""
        return self._shutdown
