"""
Async Bridge for Tkinter GUI

This module provides a bridge between Tkinter's synchronous event loop
and Python's asyncio, enabling non-blocking I/O operations in the GUI.

Usage:
    # In your main App class
    self.async_bridge = AsyncBridge(self.root)
    self.async_bridge.start()
    
    # Run async operation
    async def fetch_data():
        return await some_async_operation()
    
    def update_ui(result):
        self.label.config(text=result)
    
    self.async_bridge.run_async(fetch_data(), update_ui)

Author: AI-OS Development Team
Date: October 12, 2025
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Any, Callable, Coroutine, Optional
import logging

try:
    import tkinter as tk
except ImportError:
    tk = None  # type: ignore


logger = logging.getLogger(__name__)


class AsyncBridge:
    """
    Bridge between Tkinter and asyncio event loops.
    
    Allows running async operations without blocking the Tkinter GUI,
    and safely calling back to update GUI components from async code.
    """
    
    def __init__(self, root: tk.Tk):  # type: ignore[valid-type]
        """
        Initialize the async bridge.
        
        Args:
            root: The Tkinter root window
        """
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        self.root = root
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks_pending = 0
    
    def start(self) -> None:
        """
        Start the asyncio event loop in a background thread.
        
        This should be called once during application initialization.
        """
        if self._running:
            logger.warning("AsyncBridge already running")
            return
        
        self._running = True
        self.loop = asyncio.new_event_loop()
        
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AsyncBridge-EventLoop"
        )
        self._thread.start()
        logger.info("AsyncBridge started")
    
    def _run_loop(self) -> None:
        """Run the asyncio event loop (internal use)."""
        if self.loop is None:
            logger.error("Event loop not initialized")
            return
        
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            logger.info("Event loop stopped")
    
    def run_async(
        self,
        coro: Coroutine[Any, Any, Any],
        callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ) -> None:
        """
        Run a coroutine and call callback with result on GUI thread.
        
        Args:
            coro: The coroutine to execute
            callback: Called with result on success (on GUI thread)
            error_callback: Called with exception on error (on GUI thread)
        
        Example:
            async def fetch():
                await asyncio.sleep(1)
                return "Done"
            
            def on_result(result):
                print(f"Got: {result}")
            
            bridge.run_async(fetch(), on_result)
        """
        if not self._running or self.loop is None:
            logger.error("AsyncBridge not running, call start() first")
            if error_callback:
                error_callback(RuntimeError("AsyncBridge not running"))
            return
        
        # Submit coroutine to asyncio loop
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        self._callbacks_pending += 1
        
        # Poll for completion and invoke callback on GUI thread
        def _check() -> None:
            if not future.done():
                # Check again in 50ms
                if self._running:
                    self.root.after(50, _check)
                return
            
            self._callbacks_pending -= 1
            
            try:
                result = future.result(timeout=0)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"Async operation failed: {e}", exc_info=True)
                if error_callback:
                    error_callback(e)
        
        # Start checking after 50ms
        self.root.after(50, _check)
    
    def run_async_task(
        self,
        coro: Coroutine[Any, Any, Any]
    ) -> asyncio.Task[Any]:
        """
        Run a coroutine as a Task (fire and forget).
        
        Use this for operations that don't need a callback.
        The task runs in the background asyncio loop.
        
        Args:
            coro: The coroutine to execute
        
        Returns:
            The Task object (for cancellation if needed)
        
        Example:
            async def background_work():
                await asyncio.sleep(10)
                await save_data()
            
            task = bridge.run_async_task(background_work())
            # Later: task.cancel() if needed
        """
        if not self._running or self.loop is None:
            raise RuntimeError("AsyncBridge not running")
        
        return asyncio.run_coroutine_threadsafe(coro, self.loop)  # type: ignore[return-value]
    
    def stop(self) -> None:
        """
        Stop the asyncio event loop.
        
        Should be called during application shutdown.
        Waits for pending callbacks to complete (up to 5 seconds).
        """
        if not self._running:
            return
        
        self._running = False
        
        # Wait for pending callbacks (max 5 seconds)
        for _ in range(100):  # 100 * 50ms = 5 seconds
            if self._callbacks_pending == 0:
                break
            import time
            time.sleep(0.05)
        
        if self._callbacks_pending > 0:
            logger.warning(f"{self._callbacks_pending} callbacks still pending on shutdown")
        
        # Stop event loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        logger.info("AsyncBridge stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if the bridge is currently running."""
        return self._running and self.loop is not None
    
    @property
    def pending_count(self) -> int:
        """Get number of pending callbacks."""
        return self._callbacks_pending


# Convenience function for global bridge instance
_global_bridge: Optional[AsyncBridge] = None


def get_async_bridge() -> Optional[AsyncBridge]:
    """Get the global AsyncBridge instance."""
    return _global_bridge


def set_async_bridge(bridge: AsyncBridge) -> None:
    """Set the global AsyncBridge instance."""
    global _global_bridge
    _global_bridge = bridge
