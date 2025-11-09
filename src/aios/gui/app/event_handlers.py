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
        try:
            # Save state before closing
            app._save_state()
            
            # Clean up resources
            app._cleanup()
            
            # Destroy window
            app.root.destroy()
        except Exception as e:
            logger.error(f"Error during close: {e}")
            # Force close anyway
            try:
                app.root.destroy()
            except Exception:
                pass
    
    app.root.protocol("WM_DELETE_WINDOW", _on_close)
    
    # Window configure handler (resize, move)
    def _on_configure(event: tk.Event) -> None:
        """Handle window configure event."""
        # Debounce rapid configure events
        if not hasattr(app, '_configure_timer'):
            app._configure_timer = None
        
        # Cancel previous timer
        if app._configure_timer:
            app.root.after_cancel(app._configure_timer)
        
        # Set new timer to save state after 1 second
        def _delayed_save() -> None:
            try:
                app._save_state()
            except Exception as e:
                logger.warning(f"Failed to save state on configure: {e}")
        
        app._configure_timer = app.root.after(1000, _delayed_save)
    
    app.root.bind("<Configure>", _on_configure)
    
    # Keyboard shortcuts
    def _on_ctrl_s(event: tk.Event) -> str:
        """Handle Ctrl+S (save state)."""
        try:
            app._save_state()
            logger.info("State saved via Ctrl+S")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        return "break"
    
    def _on_ctrl_q(event: tk.Event) -> str:
        """Handle Ctrl+Q (quit)."""
        _on_close()
        return "break"
    
    def _on_f5(event: tk.Event) -> str:
        """Handle F5 (refresh)."""
        try:
            # Refresh all panels
            if hasattr(app, 'brains_panel') and app.brains_panel:
                app.brains_panel.refresh()
            if hasattr(app, 'mcp_panel') and app.mcp_panel:
                app.mcp_panel.refresh()
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
