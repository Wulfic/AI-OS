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
        self._closing = threading.Event()
    
    def start(self):
        """Start the async event loop in a background thread."""
        if self._running:
            logger.warning("AsyncEventLoop already running")
            return
        
        logger.debug("Starting AsyncEventLoop background thread...")
        self._stop_event.clear()
        self._started.clear()
        self._closing.clear()
        self._running = True
        
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=False,
            name="AsyncEventLoop"
        )
        logger.debug("AsyncEventLoop thread created: AsyncEventLoop")
        self._thread.start()
        logger.debug("AsyncEventLoop thread started, waiting for initialization...")
        
        # Wait for loop to be ready
        if not self._started.wait(timeout=5.0):
            self._running = False
            self._stop_event.set()
            if self._thread:
                self._thread.join(timeout=1.0)
            raise RuntimeError("AsyncEventLoop failed to start within timeout")
        logger.info("AsyncEventLoop started")
        logger.debug("AsyncEventLoop ready (event loop running)")
    
    def _run_loop(self):
        """Internal method that runs the event loop."""
        try:
            logger.debug("AsyncEventLoop thread running, creating event loop...")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()

            logger.debug("Event loop created and set as current")
            logger.debug("AsyncEventLoop entering run loop")

            try:
                self._loop.run_forever()
            finally:
                logger.debug("AsyncEventLoop exiting run loop, beginning shutdown")
                self._closing.set()
                pending = asyncio.all_tasks(self._loop)
                if pending:
                    logger.debug(f"Cancelling {len(pending)} pending async tasks")
                    for task in pending:
                        task.cancel()
                    try:
                        self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        logger.debug("Pending async tasks cancelled")
                    except Exception:
                        logger.exception("Error while awaiting pending tasks during shutdown")
                try:
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                except Exception:
                    logger.debug("Async generator shutdown raised", exc_info=True)
        except Exception as e:
            logger.exception(f"AsyncEventLoop fatal error: {e}")
        finally:
            if self._loop:
                self._loop.close()
                self._loop = None
            self._running = False
            logger.debug("AsyncEventLoop thread stopped")
    
    def stop(self, timeout: float = 5.0):
        """Stop the event loop.
        
        Args:
            timeout: Maximum seconds to wait for loop to stop
        """
        if not self._running:
            return
        
        logger.info("Stopping AsyncEventLoop...")
        logger.debug("Signaling AsyncEventLoop thread to stop")
        self._stop_event.set()

        if self._loop and not self._closing.is_set():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                # Loop might already be closing
                pass

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"AsyncEventLoop did not stop within {timeout}s")
            else:
                logger.debug("AsyncEventLoop thread joined successfully")

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

        coro_name = getattr(coro, "__name__", repr(coro))
        logger.debug(f"Submitting async coroutine to loop: {coro_name}")
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

        coro_name = getattr(coro, "__name__", repr(coro))
        logger.debug(f"Creating async task on loop: {coro_name}")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    @property
    def is_running(self) -> bool:
        """Check if the event loop is running."""
        return self._running

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Return the underlying event loop instance."""
        return self._loop
