"""Cleanup module for AI-OS GUI.

This module handles:
- Resource cleanup on shutdown
- Thread pool shutdown
- Timer cancellation
- Process cleanup
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def cleanup(app: Any) -> None:
    """
    Clean up application resources on shutdown.
    
    Shuts down:
    - Worker thread pools
    - Background timers
    - Process reaper
    - Async event loop
    - Logging handlers
    
    Args:
        app: AiosTkApp instance
    """
    
    # Shutdown worker pool
    if hasattr(app, '_worker_pool') and app._worker_pool:
        try:
            logger.info("Shutting down worker pool...")
            app._worker_pool.shutdown(wait=True, timeout=5.0)
            logger.info("Worker pool shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down worker pool: {e}")
    
    # Cancel timers
    if hasattr(app, '_timers'):
        for timer_id in app._timers:
            try:
                app.root.after_cancel(timer_id)
            except Exception:
                pass
    
    # Shutdown process reaper
    if hasattr(app, '_process_reaper') and app._process_reaper:
        try:
            logger.info("Shutting down process reaper...")
            app._process_reaper.cleanup_all(timeout=5.0)
            logger.info("Process reaper shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down process reaper: {e}")
    
    # Stop async event loop
    if hasattr(app, '_async_loop') and app._async_loop:
        try:
            logger.info("Stopping async event loop...")
            if app._async_loop.is_running:
                app._async_loop.stop()
            logger.info("Async event loop stopped")
        except Exception as e:
            logger.warning(f"Error stopping async loop: {e}")
    
    # Remove logging handlers
    if hasattr(app, '_debug_log_handler') and app._debug_log_handler:
        try:
            logging.getLogger("aios").removeHandler(app._debug_log_handler)
            logging.getLogger().removeHandler(app._debug_log_handler)
            app._debug_log_handler.close()
        except Exception as e:
            logger.warning(f"Error cleaning up log handler: {e}")
    
    # Clean up panels
    try:
        if hasattr(app, 'chat_panel') and app.chat_panel:
            if hasattr(app.chat_panel, 'cleanup'):
                app.chat_panel.cleanup()
        
        if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
            if hasattr(app.hrm_training_panel, 'cleanup'):
                app.hrm_training_panel.cleanup()
    except Exception as e:
        logger.warning(f"Error cleaning up panels: {e}")
    
    logger.info("Cleanup complete")
