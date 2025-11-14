"""Debug utility for monitoring tkinter Variable lifecycle and threading issues.

This module provides utilities to detect and log potential issues with tkinter
Variables, particularly related to threading and main loop lifecycle.

Usage:
    from aios.gui.utils import variable_monitor
    
    # Enable monitoring at startup
    variable_monitor.enable()
    
    # Check for issues
    variable_monitor.report()
"""

import logging
import threading
import weakref
from typing import Any, Dict, List, Set
import tkinter as tk

logger = logging.getLogger(__name__)


class VariableMonitor:
    """Monitor tkinter Variable lifecycle and detect threading issues."""
    
    def __init__(self):
        """Initialize the monitor."""
        self._enabled = False
        self._variables: Set[weakref.ref] = set()
        self._creation_threads: Dict[int, str] = {}
        self._access_threads: Dict[int, Set[str]] = {}
        self._lock = threading.Lock()
    
    def enable(self):
        """Enable variable monitoring."""
        if self._enabled:
            return
        
        self._enabled = True
        logger.info("[VariableMonitor] Monitoring enabled")
        
        # Monkey-patch tk.Variable to track creation
        original_init = tk.Variable.__init__
        monitor = self
        
        def tracked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            
            # Track this variable
            with monitor._lock:
                var_id = id(self)
                monitor._variables.add(weakref.ref(self))
                monitor._creation_threads[var_id] = threading.current_thread().name
                monitor._access_threads[var_id] = {threading.current_thread().name}
        
        tk.Variable.__init__ = tracked_init
        
        # Track get() and set() calls
        original_get = tk.Variable.get
        original_set = tk.Variable.set
        
        def tracked_get(self):
            with monitor._lock:
                var_id = id(self)
                if var_id in monitor._access_threads:
                    monitor._access_threads[var_id].add(threading.current_thread().name)
            return original_get(self)
        
        def tracked_set(self, value):
            with monitor._lock:
                var_id = id(self)
                if var_id in monitor._access_threads:
                    monitor._access_threads[var_id].add(threading.current_thread().name)
            return original_set(self, value)
        
        tk.Variable.get = tracked_get
        tk.Variable.set = tracked_set
    
    def disable(self):
        """Disable variable monitoring."""
        self._enabled = False
        logger.info("[VariableMonitor] Monitoring disabled")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            # Count live variables
            live_vars = [ref for ref in self._variables if ref() is not None]
            
            # Find variables accessed from multiple threads
            multi_thread_vars = [
                (var_id, threads) 
                for var_id, threads in self._access_threads.items()
                if len(threads) > 1
            ]
            
            # Find variables created off main thread
            non_main_vars = [
                (var_id, thread) 
                for var_id, thread in self._creation_threads.items()
                if thread != 'MainThread'
            ]
            
            return {
                'enabled': self._enabled,
                'total_variables': len(self._variables),
                'live_variables': len(live_vars),
                'multi_thread_accessed': len(multi_thread_vars),
                'created_off_main_thread': len(non_main_vars),
                'multi_thread_details': multi_thread_vars[:10],  # First 10
                'non_main_details': non_main_vars[:10],  # First 10
            }
    
    def report(self):
        """Log a monitoring report."""
        stats = self.get_stats()
        
        logger.info("[VariableMonitor] === Variable Lifecycle Report ===")
        logger.info(f"  Enabled: {stats['enabled']}")
        logger.info(f"  Total variables created: {stats['total_variables']}")
        logger.info(f"  Live variables: {stats['live_variables']}")
        logger.info(f"  Variables accessed from multiple threads: {stats['multi_thread_accessed']}")
        logger.info(f"  Variables created off main thread: {stats['created_off_main_thread']}")
        
        if stats['multi_thread_details']:
            logger.warning("[VariableMonitor] Multi-thread accessed variables (potential issues):")
            for var_id, threads in stats['multi_thread_details']:
                logger.warning(f"  Variable {var_id}: accessed from {threads}")
        
        if stats['non_main_details']:
            logger.warning("[VariableMonitor] Variables created off main thread (potential issues):")
            for var_id, thread in stats['non_main_details']:
                logger.warning(f"  Variable {var_id}: created in {thread}")
    
    def check_main_loop_running(self) -> bool:
        """Check if the main loop is running.
        
        Returns:
            True if main loop is running, False otherwise
        """
        try:
            # Try to get the default root
            root = tk._default_root
            if root is None:
                return False
            
            # Try a simple Tcl call
            root.tk.call("info", "exists", "tcl_platform")
            return True
        except Exception:
            return False


# Global monitor instance
_monitor = VariableMonitor()


def enable():
    """Enable variable monitoring."""
    _monitor.enable()


def disable():
    """Disable variable monitoring."""
    _monitor.disable()


def report():
    """Log a monitoring report."""
    _monitor.report()


def get_stats() -> Dict[str, Any]:
    """Get monitoring statistics.
    
    Returns:
        Dictionary with statistics
    """
    return _monitor.get_stats()


def check_main_loop_running() -> bool:
    """Check if the main loop is running.
    
    Returns:
        True if main loop is running, False otherwise
    """
    return _monitor.check_main_loop_running()
