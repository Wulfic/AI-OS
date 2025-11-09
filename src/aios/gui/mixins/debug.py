"""Debug mixin for AI-OS GUI application.

Provides debugging utilities and methods for the main application class.
"""

from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


class DebugMixin:
    """Mixin class providing debug utilities for the GUI application."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the debug mixin."""
        super().__init__(*args, **kwargs)
        self._debug_mode = False
    
    def enable_debug_mode(self) -> None:
        """Enable debug mode with verbose logging."""
        self._debug_mode = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self._debug_mode = False
        logger.setLevel(logging.INFO)
        logger.info("Debug mode disabled")
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self._debug_mode
    
    def debug_log(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message if debug mode is enabled."""
        if self._debug_mode:
            logger.debug(message, *args, **kwargs)
    
    def dump_state(self) -> dict[str, Any]:
        """Dump current application state for debugging.
        
        Returns:
            Dictionary containing application state information
        """
        state = {}
        
        # Collect panel states
        if hasattr(self, 'chat_panel') and self.chat_panel:
            state['chat_panel'] = 'initialized'
        
        if hasattr(self, 'brains_panel') and self.brains_panel:
            state['brains_panel'] = 'initialized'
        
        if hasattr(self, 'hrm_training_panel') and self.hrm_training_panel:
            state['hrm_training_panel'] = 'initialized'
        
        # Collect resource states
        if hasattr(self, '_worker_pool') and self._worker_pool:
            state['worker_pool'] = 'active'
        
        if hasattr(self, '_async_loop') and self._async_loop:
            state['async_loop'] = 'running' if self._async_loop.is_running else 'stopped'
        
        return state
    
    def print_debug_info(self) -> None:
        """Print debug information to console."""
        if not self._debug_mode:
            logger.info("Debug mode is not enabled")
            return
        
        state = self.dump_state()
        logger.debug("=== Application Debug Info ===")
        for key, value in state.items():
            logger.debug(f"  {key}: {value}")
        logger.debug("=" * 30)
