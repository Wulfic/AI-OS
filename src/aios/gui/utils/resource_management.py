"""Resource management utilities for long-running stability.

This module provides utilities for managing threads, processes, and other
resources to prevent leaks and ensure clean shutdown during extended runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any


logger = logging.getLogger(__name__)


class ManagedThread:
    """Base class for managed threads with graceful shutdown.
    
    Provides:
    - Non-daemon threads that can be gracefully stopped
    - Stop event for signaling shutdown
    - Timeout-based join for cleanup
    - Automatic error handling and logging
    
    Usage:
        class MyWorker(ManagedThread):
            def _do_work(self):
                # Your work here
                time.sleep(0.1)
        
        worker = MyWorker()
        worker.start()
        # ... later ...
        worker.stop(timeout=5.0)
    """
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
    
    def start(self):
        """Start the thread."""
        if self._running:
            logger.warning(f"{self.__class__.__name__} already running")
            return
        
        self._stop_event.clear()
        self._running = True
        # Non-daemon thread for graceful shutdown
        self._thread = threading.Thread(target=self._run, daemon=False, name=self.__class__.__name__)
        self._thread.start()
        logger.debug(f"{self.__class__.__name__} started")
    
    def stop(self, timeout: float = 5.0):
        """Stop the thread gracefully.
        
        Args:
            timeout: Maximum seconds to wait for thread to stop
        """
        if not self._running:
            return
        
        logger.debug(f"{self.__class__.__name__} stopping...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f"{self.__class__.__name__} did not stop within {timeout}s. "
                    f"It will continue running but should exit soon."
                )
        
        self._running = False
        logger.debug(f"{self.__class__.__name__} stopped")
    
    def is_running(self) -> bool:
        """Check if thread is currently running."""
        return self._running
    
    def _run(self):
        """Internal run loop. Override _do_work() instead."""
        logger.debug(f"{self.__class__.__name__} worker started")
        try:
            while not self._stop_event.is_set():
                try:
                    self._do_work()
                except Exception as e:
                    logger.exception(f"{self.__class__.__name__} error: {e}")
                    # Continue running unless subclass wants to stop
                    if not self._should_continue_on_error():
                        break
        except Exception as e:
            logger.exception(f"{self.__class__.__name__} fatal error: {e}")
        finally:
            logger.debug(f"{self.__class__.__name__} worker exiting")
    
    def _do_work(self):
        """Override this method with your thread logic.
        
        This method is called repeatedly in a loop. It should:
        - Do a small unit of work
        - Return quickly (ideally < 1 second)
        - Check self._stop_event.is_set() for long operations
        """
        raise NotImplementedError(f"{self.__class__.__name__}._do_work() must be implemented")
    
    def _should_continue_on_error(self) -> bool:
        """Override to control error handling behavior.
        
        Returns:
            True to continue running after error (default)
            False to stop thread on error
        """
        return True


class ProcessReaper:
    """Ensures all child processes are cleaned up.
    
    Tracks processes and provides aggressive cleanup to prevent zombies.
    
    Usage:
        reaper = ProcessReaper()
        
        # Register processes
        proc = subprocess.Popen(...)
        reaper.register(proc)
        
        # On shutdown
        reaper.cleanup_all(timeout=5.0)
    """
    
    def __init__(self):
        self.processes: list[subprocess.Popen] = []
        self._lock = threading.Lock()
    
    def register(self, proc: subprocess.Popen):
        """Register a process for cleanup."""
        with self._lock:
            self.processes.append(proc)
            logger.debug(f"Registered process {proc.pid}")
    
    def unregister(self, proc: subprocess.Popen):
        """Unregister a process (e.g., after manual cleanup)."""
        with self._lock:
            if proc in self.processes:
                self.processes.remove(proc)
                logger.debug(f"Unregistered process {proc.pid}")
    
    def cleanup_all(self, timeout: float = 5.0):
        """Terminate all registered processes.
        
        Args:
            timeout: Maximum seconds to wait for graceful termination
        """
        with self._lock:
            if not self.processes:
                return
            
            logger.info(f"Cleaning up {len(self.processes)} processes...")
            
            # Step 1: Send terminate signal to all processes
            for proc in self.processes[:]:  # Copy list
                if proc.poll() is None:  # Still running
                    try:
                        proc.terminate()
                        logger.debug(f"Terminated process {proc.pid}")
                    except Exception as e:
                        logger.warning(f"Failed to terminate process {proc.pid}: {e}")
            
            # Step 2: Wait for graceful termination
            start = time.time()
            while time.time() - start < timeout:
                all_dead = True
                for proc in self.processes:
                    if proc.poll() is None:
                        all_dead = False
                        break
                
                if all_dead:
                    logger.info("All processes terminated gracefully")
                    break
                
                time.sleep(0.1)
            
            # Step 3: Force kill any survivors
            for proc in self.processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                        logger.warning(f"Force killed process {proc.pid}")
                    except Exception as e:
                        logger.error(f"Failed to kill process {proc.pid}: {e}")
            
            self.processes.clear()
            logger.info("Process cleanup complete")


class TimerManager:
    """Manages Tkinter after() callbacks to prevent accumulation.
    
    Ensures only one timer exists per named operation, preventing
    callback accumulation that can cause memory leaks and slowdowns.
    
    Usage:
        timer_mgr = TimerManager(root)
        
        # Set a named timer (cancels previous if exists)
        timer_mgr.set_timer("save_state", 300, self._save_state)
        
        # On shutdown
        timer_mgr.cancel_all()
    """
    
    def __init__(self, root: Any):
        self.root = root
        self._timers: dict[str, Optional[str]] = {}  # name -> timer_id
        self._lock = threading.Lock()
    
    def set_timer(self, name: str, delay_ms: int, callback: Callable):
        """Set a named timer, cancelling previous if exists.
        
        Args:
            name: Unique name for this timer
            delay_ms: Delay in milliseconds
            callback: Function to call after delay
        """
        with self._lock:
            # Cancel existing timer with this name
            self.cancel_timer(name)
            
            # Set new timer
            try:
                timer_id = self.root.after(delay_ms, callback)
                self._timers[name] = timer_id
                logger.debug(f"Set timer '{name}' (delay={delay_ms}ms)")
            except Exception as e:
                logger.error(f"Failed to set timer '{name}': {e}")
    
    def cancel_timer(self, name: str):
        """Cancel a named timer.
        
        Args:
            name: Name of timer to cancel
        """
        timer_id = self._timers.get(name)
        if timer_id:
            try:
                self.root.after_cancel(timer_id)
                logger.debug(f"Cancelled timer '{name}'")
            except Exception as e:
                logger.debug(f"Failed to cancel timer '{name}': {e}")
            self._timers[name] = None
    
    def cancel_all(self):
        """Cancel all timers (call on shutdown)."""
        with self._lock:
            logger.info(f"Cancelling {len(self._timers)} timers...")
            for name in list(self._timers.keys()):
                self.cancel_timer(name)
            logger.info("All timers cancelled")


class ResourceMonitor:
    """Monitors application resource usage for diagnostics.
    
    Provides health metrics like memory, thread count, CPU usage.
    
    Usage:
        monitor = ResourceMonitor()
        health = monitor.get_health_report()
        print(f"Memory: {health['memory_mb']} MB")
    """
    
    def __init__(self):
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not available - resource monitoring limited")
    
    def get_health_report(self) -> dict[str, Any]:
        """Get current health metrics.
        
        Returns:
            Dict with keys: memory_mb, threads, cpu_percent, open_files (if available)
        """
        report: dict[str, Any] = {
            "memory_mb": 0,
            "threads": threading.active_count(),
            "cpu_percent": 0,
            "open_files": 0,
        }
        
        if not self._psutil:
            return report
        
        try:
            import platform
            process = self._psutil.Process()
            
            # Memory in MB
            report["memory_mb"] = process.memory_info().rss / 1024**2
            
            # CPU percent (non-blocking)
            report["cpu_percent"] = process.cpu_percent(interval=0)
            
            # Open file handles (skip on Windows - very slow with many handles)
            if platform.system() != "Windows":
                try:
                    report["open_files"] = len(process.open_files())
                except Exception:
                    pass  # May not be available on all platforms
        
        except Exception as e:
            logger.warning(f"Failed to get health metrics: {e}")
        
        return report
    
    def format_health_status(self) -> str:
        """Get formatted health status string for display.
        
        Returns:
            Human-readable status string like "Mem: 245MB | Threads: 12"
        """
        health = self.get_health_report()
        return f"Mem: {health['memory_mb']:.0f}MB | Threads: {health['threads']}"


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
            max_workers: Number of workers. If None, uses (cpu_count * 2) + 1
                        This provides better responsiveness for I/O-bound GUI operations.
        """
        if max_workers is None:
            # For GUI apps with async I/O operations, use more workers than CPUs
            # Formula: (cpu_count * 2) + 1 provides good balance for I/O-bound work
            # Minimum of 8 workers to ensure responsiveness even on low-core systems
            cpu_count = os.cpu_count() or 4
            max_workers = max((cpu_count * 2) + 1, 8)
        
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


class AsyncEventLoop:
    """Manages an asyncio event loop that runs alongside Tkinter.
    
    Enables async/await patterns in GUI code for better responsiveness.
    The loop runs in a background thread and can be accessed safely.
    
    Usage:
        loop = AsyncEventLoop()
        loop.start()
        
        # Submit async work
        future = loop.run_coroutine(async_function())
        
        # On shutdown
        loop.stop()
    """
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._started = threading.Event()
        self._running = False
    
    def start(self):
        """Start the async event loop in a background thread."""
        if self._running:
            logger.warning("AsyncEventLoop already running")
            return
        
        self._stop_event.clear()
        self._started.clear()
        self._running = True
        
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=False,
            name="AsyncEventLoop"
        )
        self._thread.start()
        
        # Wait for loop to be ready
        self._started.wait(timeout=5.0)
        logger.info("AsyncEventLoop started")
    
    def _run_loop(self):
        """Internal method that runs the event loop."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            
            logger.debug("AsyncEventLoop worker started")
            
            # Run until stop is requested
            while not self._stop_event.is_set():
                try:
                    self._loop.run_until_complete(asyncio.sleep(0.1))
                except Exception as e:
                    logger.error(f"AsyncEventLoop error: {e}")
            
            logger.debug("AsyncEventLoop stopping...")
            
            # Cancel pending tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            
            # Wait for tasks to complete
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
        except Exception as e:
            logger.exception(f"AsyncEventLoop fatal error: {e}")
        finally:
            if self._loop:
                self._loop.close()
            logger.debug("AsyncEventLoop stopped")
    
    def stop(self, timeout: float = 5.0):
        """Stop the event loop.
        
        Args:
            timeout: Maximum seconds to wait for loop to stop
        """
        if not self._running:
            return
        
        logger.info("Stopping AsyncEventLoop...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"AsyncEventLoop did not stop within {timeout}s")
        
        self._running = False
        logger.info("AsyncEventLoop stopped")
    
    def run_coroutine(self, coro):
        """Schedule a coroutine to run on the event loop.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            concurrent.futures.Future that can be used to get the result
        """
        if not self._loop or not self._running:
            raise RuntimeError("AsyncEventLoop is not running")
        
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def create_task(self, coro):
        """Create a task from a coroutine on the event loop.
        
        Args:
            coro: Coroutine to wrap in a task
            
        Returns:
            Task object
        """
        if not self._loop or not self._running:
            raise RuntimeError("AsyncEventLoop is not running")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future
    
    @property
    def is_running(self) -> bool:
        """Check if the event loop is running."""
        return self._running
