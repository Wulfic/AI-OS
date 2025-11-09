"""Main application entry point for AI-OS GUI.

This module orchestrates application initialization by:
1. Setting up resources (threads, timers, async loop)
2. Initializing logging
3. Creating UI structure
4. Initializing all panels
5. Loading saved state
6. Setting up event handlers
7. Running main loop
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import time
from pathlib import Path

if TYPE_CHECKING:
    import tkinter as tk

from .resource_setup import setup_resources
from .logging_setup import initialize_logging, configure_log_levels
from .ui_setup import create_ui_structure
from .panel_setup import initialize_panels, deferred_panel_initialization
from .state_management import initialize_state, load_state, restore_state, save_state
from .chat_operations import setup_chat_operations
from .brain_operations import setup_brain_operations
from .goal_operations import setup_goal_operations
from .event_handlers import setup_event_handlers, setup_periodic_tasks
from .cleanup import cleanup
from .tray_management import init_tray, on_tray_settings

logger = logging.getLogger(__name__)


def run_app(app_instance: Any) -> None:
    """
    Main application initialization and run loop.
    
    This is the orchestrator function that initializes all components
    in the correct order and starts the Tkinter main loop.
    
    Args:
        app_instance: AiosTkApp instance (must have root, _project_root, _run_cli)
    """
    
    app = app_instance
    start_time = time.time()
    last_time = start_time
    
    def log_timing(step_name: str) -> None:
        """Log timing for each initialization step."""
        nonlocal last_time
        current = time.time()
        step_duration = current - last_time
        total_duration = current - start_time
        msg = f"[TIMING] {step_name}: {step_duration:.3f}s (total: {total_duration:.3f}s)"
        logger.info(msg)
        print(msg)  # Also print to console for visibility
        last_time = current
    
    try:
        # 1. Initialize resources (threads, timers, async loop)
        logger.info("Initializing resources...")
        setup_resources(app, app.root, False)  # start_minimized handled elsewhere
        log_timing("Resources initialized")
        
        # 2. Initialize logging system
        logger.info("Initializing logging...")
        initialize_logging(app, app._project_root)
        log_timing("Logging initialized")
        
        # 3. Initialize state management
        logger.info("Initializing state management...")
        initialize_state(app, app._project_root)
        log_timing("State management initialized")
        
        # 4. Create UI structure (notebook with tabs)
        logger.info("Creating UI structure...")
        create_ui_structure(app, app.root)
        log_timing("UI structure created")
        
        # 5. Set up operation handlers (BEFORE panels - they depend on these)
        logger.info("Setting up operations...")
        setup_chat_operations(app)
        setup_brain_operations(app)
        setup_goal_operations(app)
        log_timing("Operations set up")
        
        # 5b. Attach cleanup and save functions to app (BEFORE panels - they need these)
        app._cleanup = lambda: cleanup(app)
        app._save_state = lambda: save_state(app)
        
        # 6. Initialize all panels (AFTER operations - they need the callbacks)
        logger.info("Initializing panels...")
        initialize_panels(app)
        log_timing("Panels initialized")
        
        # 7. Set up event handlers
        logger.info("Setting up event handlers...")
        setup_event_handlers(app)
        setup_periodic_tasks(app)
        log_timing("Event handlers set up")
        
        # 8. Set up system tray (optional)
        try:
            init_tray(app)
        except Exception as e:
            logger.warning(f"Failed to set up system tray: {e}")
        log_timing("System tray initialized")
        
        # 9. Load and restore saved state
        logger.info("Loading saved state...")
        state = load_state(app)
        if state:
            restore_state(app, state)
        log_timing("State restored")
        
        # 10. Configure log levels based on settings
        debug_enabled = False
        if hasattr(app, 'settings_panel') and app.settings_panel:
            try:
                settings_state = app.settings_panel.get_state()
                debug_enabled = settings_state.get('debug_enabled', False)
            except Exception:
                pass
        configure_log_levels(app, debug_enabled)
        log_timing("Log levels configured")
        
        logger.info("AI-OS GUI initialized successfully")
        
        # 11. Finalize window display
        logger.info("Finalizing window display...")
        # Force window to update and render all components
        app.root.update_idletasks()
        log_timing("Window updated")
        
        # Show the window (unless starting minimized)
        if not app._start_minimized:
            app.root.deiconify()
            # Maximize window for better user experience
            try:
                app.root.state('zoomed')  # Windows/Linux
            except Exception:
                try:
                    # macOS alternative
                    app.root.attributes('-zoomed', True)
                except Exception:
                    # Fallback: just show normally
                    logger.debug("Could not maximize window, showing normal size")
        else:
            logger.info("Starting minimized to tray")
        log_timing("Window displayed")
        
        total_startup = time.time() - start_time
        msg = f"[TIMING] Total startup time: {total_startup:.3f}s"
        logger.info(msg)
        print(msg)  # Also print to console for visibility
        
        # 12. Schedule deferred panel initialization (after window is shown)
        # This runs data loading operations that make blocking subprocess calls
        app.root.after(300, lambda: deferred_panel_initialization(app))
        
        # 13. Start main loop
        logger.info("Starting main loop...")
        app.root.mainloop()
        
    except Exception as e:
        logger.critical(f"Fatal error during app initialization: {e}", exc_info=True)
        raise
