"""Timer management utilities for Tkinter after() callbacks."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


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
        logger.debug(f"Acquiring timer lock for operation: set_timer('{name}')")
        with self._lock:
            logger.debug("Timer lock acquired")
            # Cancel existing timer with this name
            self.cancel_timer(name)
            
            # Set new timer
            try:
                timer_id = self.root.after(delay_ms, callback)
                self._timers[name] = timer_id
                logger.debug(f"Set timer '{name}' (delay={delay_ms}ms)")
            except Exception as e:
                logger.error(f"Failed to set timer '{name}': {e}")
            logger.debug("Timer lock released")
    
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
        logger.debug("Acquiring timer lock for operation: cancel_all")
        with self._lock:
            logger.debug("Timer lock acquired")
            logger.info(f"Cancelling {len(self._timers)} timers...")
            for name in list(self._timers.keys()):
                self.cancel_timer(name)
            logger.info("All timers cancelled")
            logger.debug("Timer lock released")
