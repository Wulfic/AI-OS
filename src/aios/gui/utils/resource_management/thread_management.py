"""Thread management utilities with graceful shutdown support."""

from __future__ import annotations

import logging
import threading
from typing import Optional


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
