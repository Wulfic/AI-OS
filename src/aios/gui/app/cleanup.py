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
import time

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

from ..utils.resource_management import set_async_loop, set_worker_pool


def cleanup(app: Any) -> None:
    """
    Clean up application resources on shutdown.
    
    Shuts down:
    - Worker thread pools
    - Background timers
    - Process reaper
    - Async event loop
    - Logging handlers
    - Tkinter variables (to prevent threading errors)
    
    Args:
        app: AiosTkApp instance
    """
    logger.info("=" * 60)
    logger.info("AI-OS GUI Application Shutting Down")
    logger.info("=" * 60)
    
    # Stop tray manager first to prevent tray thread from blocking shutdown
    if hasattr(app, '_tray_manager') and app._tray_manager:
        try:
            logger.info("Stopping system tray manager...")
            app._tray_manager.destroy()
            logger.info("System tray manager stopped")
        except Exception as e:
            logger.warning(f"Error stopping tray manager: {e}")
    
    # Clean up Tkinter variables first to prevent "main thread is not in main loop" errors
    try:
        logger.debug("Cleaning up Tkinter variables...")
        # Clear references to all Tk variables to prevent garbage collection issues
        for attr_name in dir(app):
            if attr_name.endswith('_var'):
                try:
                    attr = getattr(app, attr_name, None)
                    if attr is not None and hasattr(attr, '_tk'):
                        # Set to None to break the reference cycle
                        setattr(app, attr_name, None)
                except Exception:
                    pass
        
        # Also clean up panel variables
        for panel_name in ['chat_panel', 'hrm_training_panel', 'resources_panel', 'settings_panel', 
                          'subbrains_manager_panel', 'evaluation_panel', 'dataset_download_panel']:
            try:
                panel = getattr(app, panel_name, None)
                if panel:
                    for attr_name in dir(panel):
                        if attr_name.endswith('_var'):
                            try:
                                attr = getattr(panel, attr_name, None)
                                if attr is not None and hasattr(attr, '_tk'):
                                    setattr(panel, attr_name, None)
                            except Exception:
                                pass
            except Exception:
                pass
        logger.debug("Tkinter variables cleanup complete")
    except Exception as e:
        logger.warning(f"Error cleaning up Tkinter variables: {e}")
    
    # Ask panels to wind down background work FIRST (before stopping dispatcher/worker pool)
    # This prevents panels from submitting new tasks while we're shutting down
    try:
        for panel_attr, label in (
            ("resources_panel", "resources"),
            ("dataset_download_panel", "dataset download"),
            ("evaluation_panel", "evaluation"),
            ("hrm_training_panel", "hrm training"),
            ("chat_panel", "chat"),
            ("brains_panel", "brains"),
        ):
            panel = getattr(app, panel_attr, None)
            if panel and hasattr(panel, "cleanup"):
                logger.debug("Pre-shutdown cleanup for %s panel...", label)
                try:
                    panel.cleanup()
                    logger.debug("%s panel pre-shutdown cleanup: success", label.capitalize())
                except Exception as exc:
                    logger.debug("%s panel pre-shutdown cleanup failed: %s", label.capitalize(), exc)
    except Exception as exc:
        logger.debug("Panel pre-shutdown coordination error: %s", exc)
    
    # Small delay to let pending UI callbacks drain naturally
    import time as time_module
    time_module.sleep(0.1)
    
    # Disconnect LogRouter from dispatcher BEFORE stopping it to prevent
    # the flood of "dropping task" messages during shutdown logging
    if hasattr(app, '_log_router') and app._log_router:
        try:
            logger.debug("Disconnecting log router from UI dispatcher...")
            app._log_router.set_dispatcher(None)
        except Exception as e:
            logger.debug(f"Error disconnecting log router: {e}")
    
    # Stop UI dispatcher to prevent further queued UI work
    if hasattr(app, '_ui_dispatcher') and app._ui_dispatcher:
        try:
            logger.debug("Stopping UI dispatcher...")
            # Flush any remaining callbacks first
            try:
                app._ui_dispatcher.flush()
            except Exception:
                pass
            app._ui_dispatcher.stop()
        except Exception as e:
            logger.debug(f"Error stopping UI dispatcher: {e}")

    # Shutdown worker pool - reduced timeout for faster shutdown
    if hasattr(app, '_worker_pool') and app._worker_pool:
        try:
            logger.info("Shutting down worker pool...")
            start_time = time.time()
            app._worker_pool.shutdown(wait=True, timeout=3.0)  # Reduced from 5.0 to 3.0
            duration = time.time() - start_time
            logger.info(f"Worker pool shutdown complete (took {duration:.2f}s)")
        except Exception as e:
            logger.warning(f"Error shutting down worker pool: {e}")
        finally:
            set_worker_pool(None)
    
    # Cancel timers
    if hasattr(app, '_timers'):
        timer_count = len(app._timers)
        if timer_count > 0:
            logger.info(f"Cancelling {timer_count} active timers...")
            cancelled = 0
            for timer_id in app._timers:
                try:
                    app.root.after_cancel(timer_id)
                    cancelled += 1
                except Exception:
                    pass
            logger.debug(f"Cancelled {cancelled}/{timer_count} timers")
    
    # Shutdown process reaper - reduced timeout for faster shutdown
    if hasattr(app, '_process_reaper') and app._process_reaper:
        try:
            # Get process count before cleanup if available
            process_count = 0
            if hasattr(app._process_reaper, '_processes'):
                process_count = len(app._process_reaper._processes)
            
            logger.info(f"Shutting down process reaper (cleaning up {process_count} processes)...")
            start_time = time.time()
            app._process_reaper.cleanup_all(timeout=3.0)  # Reduced from 5.0 to 3.0
            duration = time.time() - start_time
            logger.info(f"Process reaper shutdown complete (took {duration:.2f}s)")
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
        finally:
            set_async_loop(None)
    
    # Remove logging handlers
    if hasattr(app, '_debug_log_handler') and app._debug_log_handler:
        try:
            logger.debug("Cleaning up logging handlers...")
            logging.getLogger("aios").removeHandler(app._debug_log_handler)
            logging.getLogger().removeHandler(app._debug_log_handler)
            app._debug_log_handler.close()
        except Exception as e:
            logger.warning(f"Error cleaning up log handler: {e}")
    
    # Shutdown async logging
    try:
        from aios.cli.utils import shutdown_logging
        logger.info("Shutting down async logging...")
        start_time = time.time()
        shutdown_logging()
        duration = time.time() - start_time
        logger.info(f"Async logging shutdown complete (took {duration:.2f}s)")
    except Exception as e:
        logger.warning(f"Error shutting down async logging: {e}")
    
    # NOTE: Panel cleanup is now done ONCE at the start of shutdown (see "Pre-shutdown cleanup" above)
    # Previously there was duplicate cleanup here which caused issues and doubled shutdown time
    
    logger.info("Cleanup complete - Application terminated")
    logger.info("=" * 60)
