"""Async event loop management for Tkinter integration."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional


logger = logging.getLogger(__name__)


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
