"""Event handlers module for AI-OS GUI.

This module handles:
- Window events (close, configure)
- Keyboard events
- Timer events
- Background task events
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import time

if TYPE_CHECKING:
    import tkinter as tk

logger = logging.getLogger(__name__)


def setup_event_handlers(app: Any) -> None:
    """
    Set up event handlers for the application.
    
    Args:
        app: AiosTkApp instance
    """
    
    # Window close handler
    def _on_close() -> None:
        """Handle window close event."""
        import threading
        import sys
        import os
        
        logger.info("User action: Closing application window")
        
        # Set a forced exit timeout in case cleanup hangs
        # Reduced from 10s to 5s for faster forced exit
        def _force_exit() -> None:
            logger.warning("Cleanup taking too long (5s), forcing application exit")
            try:
                app.root.destroy()
            except Exception:
                pass
            # Force exit immediately - don't give Tk more time
            logger.warning("Force exiting process now")
            os._exit(0)
        
        force_timer = threading.Timer(5.0, _force_exit)
        force_timer.daemon = True
        force_timer.start()
        
        try:
            # Save state before closing
            app._save_state(sync=True)
            logger.debug("State saved before window close")
            
            # Clean up resources
            app._cleanup()
            logger.debug("Resources cleaned up")
            
            # Cancel force timer since cleanup succeeded
            force_timer.cancel()
            
            # Destroy window
            app.root.destroy()
            logger.info("Application window closed successfully")
            
            # Force exit to ensure no lingering threads keep the process alive
            # This is necessary because some non-daemon threads (like multiprocessing Manager)
            # may still be running even after cleanup
            os._exit(0)
        except Exception as e:
            logger.error(f"Error during close: {e}")
            # Cancel force timer and force close anyway
            force_timer.cancel()
            try:
                app.root.destroy()
            except Exception:
                pass
            # If destroy fails, force exit
            logger.warning("Clean shutdown failed, forcing exit")
            os._exit(0)
    
    app.root.protocol("WM_DELETE_WINDOW", _on_close)
    
    # DISABLED: Window configure auto-save (too aggressive, causes hangs)
    # State is now only saved:
    # 1. On window close
    # 2. On Ctrl+S
    # 3. Periodically every 5 minutes
    # 4. When user clicks Save in settings
    
    # # Window configure handler (resize, move)
    # def _on_configure(event: tk.Event) -> None:
    #     """Handle window configure event."""
    #     # Debounce rapid configure events
    #     if not hasattr(app, '_configure_timer'):
    #         app._configure_timer = None
    #     
    #     # Cancel previous timer
    #     if app._configure_timer:
    #         app.root.after_cancel(app._configure_timer)
    #     
    #     # Set new timer to save state after 1 second
    #     def _delayed_save() -> None:
    #         try:
    #             app._save_state()
    #         except Exception as e:
    #             logger.warning(f"Failed to save state on configure: {e}")
    #     
    #     app._configure_timer = app.root.after(1000, _delayed_save)
    # 
    # app.root.bind("<Configure>", _on_configure)
    
    # Keyboard shortcuts
    def _on_ctrl_s(event: tk.Event) -> str:
        """Handle Ctrl+S (save state)."""
        logger.info("User action: Ctrl+S pressed - saving state")
        try:
            app._save_state()
            logger.info("State saved via Ctrl+S")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        return "break"
    
    def _on_ctrl_q(event: tk.Event) -> str:
        """Handle Ctrl+Q (quit)."""
        logger.info("User action: Ctrl+Q pressed - quitting application")
        _on_close()
        return "break"
    
    def _on_f5(event: tk.Event) -> str:
        """Handle F5 (refresh)."""
        logger.info("User action: F5 pressed - refreshing panels")
        try:
            # Refresh all panels
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
                logger.debug("Brains panel refreshed")
            if hasattr(app, 'mcp_panel') and app.mcp_panel:
                app.mcp_panel.refresh()
                logger.debug("MCP panel refreshed")
            logger.info("Panels refreshed via F5")
        except Exception as e:
            logger.error(f"Failed to refresh: {e}")
        return "break"
    
    app.root.bind("<Control-s>", _on_ctrl_s)
    app.root.bind("<Control-q>", _on_ctrl_q)
    app.root.bind("<F5>", _on_f5)


def setup_periodic_tasks(app: Any) -> None:
    """
    Set up periodic background tasks.
    
    Args:
        app: AiosTkApp instance
    """
    
    # Auto-save state every 5 minutes
    def _periodic_save() -> None:
        """Periodic state save."""
        try:
            app._save_state()
            logger.debug("Periodic state save")
        except Exception as e:
            logger.warning(f"Periodic save failed: {e}")
        
        # Schedule next save
        app.root.after(300000, _periodic_save)  # 5 minutes
    
    # Start periodic save
    app.root.after(300000, _periodic_save)
    
    # Periodic resource monitoring (every 10 seconds)
    def _periodic_resource_check() -> None:
        """Periodic resource monitoring."""
        try:
            if hasattr(app, 'resources_panel') and app.resources_panel:
                # Update resource usage display
                pass  # ResourcesPanel handles its own updates
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
        
        # Schedule next check
        app.root.after(10000, _periodic_resource_check)
    
    # Start resource check
    app.root.after(10000, _periodic_resource_check)

    # UI watchdog to detect long-running callbacks on the Tk thread
    watchdog_interval_ms = 250
    watchdog_threshold = 0.35  # seconds
    watchdog_startup_grace = 30.0  # seconds - suppress warnings during startup
    app._ui_watchdog_last_tick = time.perf_counter()
    app._ui_watchdog_start_time = time.perf_counter()
    app._ui_watchdog_max_delta = 0.0

    def _ui_watchdog() -> None:
        """Detect prolonged UI thread stalls and log diagnostic details."""
        try:
            now = time.perf_counter()
            last_tick = getattr(app, "_ui_watchdog_last_tick", now)
            start_time = getattr(app, "_ui_watchdog_start_time", now)
            delta = now - last_tick
            app._ui_watchdog_last_tick = now

            # Suppress warnings during startup grace period
            in_startup = (now - start_time) < watchdog_startup_grace
            
            if delta > watchdog_threshold:
                app._ui_watchdog_max_delta = max(getattr(app, "_ui_watchdog_max_delta", 0.0), delta)
                if not in_startup:
                    logger.warning(
                        "UI watchdog detected %.1f ms pause (threshold %.0f ms)",
                        delta * 1000.0,
                        watchdog_threshold * 1000.0,
                    )
        except Exception:
            logger.debug("UI watchdog encountered an error", exc_info=True)
        finally:
            try:
                app.root.after(watchdog_interval_ms, _ui_watchdog)
            except Exception:
                logger.debug("Failed to reschedule UI watchdog", exc_info=True)

    app.root.after(watchdog_interval_ms, _ui_watchdog)
