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
        # Use existing loading screen (created in __init__)
        def update_status(text):
            """Update loading status message (safe against window state changes)."""
            try:
                # Only update if loading is still active
                if not hasattr(app, '_loading_active') or not app._loading_active:
                    return
                
                if hasattr(app, '_loading_canvas') and app._loading_canvas:
                    # Get the status text ID from canvas (set by update_loading_canvas)
                    if hasattr(app._loading_canvas, '_status_text_id'):
                        try:
                            app._loading_canvas.itemconfig(app._loading_canvas._status_text_id, text=text)
                        except Exception:
                            # Canvas may be in bad state during window switching
                            pass
                    
                    # NO update_idletasks here - causes hangs during panel creation
                    # The canvas updates itself via Configure events
            except Exception:
                # Silently fail - loading screen updates are not critical
                pass
        
        # 1. Initialize resources (threads, timers, async loop)
        logger.info("Initializing resources...")
        update_status("Initializing resources...")
        setup_resources(app, app.root, False)  # start_minimized handled elsewhere
        log_timing("Resources initialized")
        
        # 2. Initialize logging system
        logger.info("Initializing logging...")
        update_status("Initializing logging...")
        initialize_logging(app, app._project_root)
        log_timing("Logging initialized")
        
        # 3. Initialize state management
        logger.info("Initializing state management...")
        update_status("Initializing state management...")
        initialize_state(app, app._project_root)
        log_timing("State management initialized")
        
        # 4. Create UI structure (notebook with tabs)
        logger.info("Creating UI structure...")
        update_status("Creating interface layout...")
        create_ui_structure(app, app.root)
        
        # Ensure loading screen stays on top
        if hasattr(app, '_loading_frame') and app._loading_frame:
            app._loading_frame.lift()
        
        log_timing("UI structure created")
        
        # 5. Set up operation handlers (BEFORE panels - they depend on these)
        logger.info("Setting up operations...")
        update_status("Initializing core systems...")
        setup_chat_operations(app)
        setup_brain_operations(app)
        setup_goal_operations(app)
        log_timing("Operations set up")
        
        # 5b. Attach cleanup and save functions to app (BEFORE panels - they need these)
        app._cleanup = lambda: cleanup(app)
        app._save_state = lambda: save_state(app)
        
        # 6. Set up event handlers
        logger.info("Setting up event handlers...")
        update_status("Connecting event handlers...")
        setup_event_handlers(app)
        log_timing("Event handlers set up")
        
        # 7. Set up system tray (optional)
        update_status("Initializing system tray...")
        try:
            init_tray(app)
        except Exception as e:
            logger.warning(f"Failed to set up system tray: {e}")
        log_timing("System tray initialized")
        
        # 8. Initialize all panels (with progress updates)
        logger.info("Initializing panels...")
        update_status("Building interface panels...")
        initialize_panels(app)
        log_timing("Panels initialized")
        
        # 9. Load panel data synchronously while keeping loading screen visible
        # This ensures all data is loaded before user can interact
        logger.info("Loading panel data...")
        update_status("Loading application data...")
        from .panel_setup import load_all_panel_data
        load_all_panel_data(app, update_status)
        log_timing("Panel data loaded")
        
        # 10. Load and restore saved state
        logger.info("Loading saved state...")
        update_status("Restoring your preferences...")
        state = load_state(app)
        if state:
            restore_state(app, state)
        log_timing("State restored")
        
        # 10b. Re-apply theme after all panels are fully initialized
        # This ensures theme is applied to all components including late-loaded ones
        if hasattr(app, 'settings_panel') and app.settings_panel:
            try:
                current_theme = app.settings_panel.theme_var.get()
                logger.info(f"Current theme from settings panel: '{current_theme}'")
                
                # Only re-apply if we have a valid theme (not just the default)
                if current_theme and current_theme in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                    logger.info(f"Re-applying theme after panel initialization: {current_theme}")
                    update_status(f"Applying theme: {current_theme}...")
                    app.settings_panel._apply_theme(current_theme)
                    logger.info(f"Theme '{current_theme}' applied successfully")
                else:
                    logger.warning(f"Invalid or missing theme during re-application: '{current_theme}' - applying Dark Mode as default")
                    # Apply default dark mode if no valid theme
                    app.settings_panel.theme_var.set("Dark Mode")
                    app.settings_panel._apply_theme("Dark Mode")
                    logger.info("Default Dark Mode theme applied")
            except Exception as e:
                logger.error(f"Failed to re-apply theme: {e}", exc_info=True)
        
        # 11. Configure log levels based on settings
        update_status("Finalizing configuration...")
        debug_enabled = False
        if hasattr(app, 'settings_panel') and app.settings_panel:
            try:
                settings_state = app.settings_panel.get_state()
                debug_enabled = settings_state.get('debug_enabled', False)
            except Exception:
                pass
        configure_log_levels(app, debug_enabled)
        log_timing("Log levels configured")
        
        # 12. Setup periodic tasks
        update_status("Starting background tasks...")
        setup_periodic_tasks(app)
        
        logger.info("AI-OS GUI initialized successfully")
        
        total_startup = time.time() - start_time
        msg = f"[TIMING] Total startup time: {total_startup:.3f}s"
        logger.info(msg)
        print(msg)
        
        # Ensure all UI elements are fully rendered before removing loading screen
        update_status("Ready!")
        app.root.update_idletasks()
        app.root.update()
        
        # Remove loading screen with smooth transition
        if hasattr(app, '_loading_frame') and app._loading_frame:
            def _remove_loading():
                try:
                    # Mark loading as no longer active
                    if hasattr(app, '_loading_active'):
                        app._loading_active = False
                    
                    # Unbind visibility handler
                    try:
                        app.root.unbind('<Visibility>')
                    except Exception:
                        pass
                    
                    app._loading_frame.destroy()
                    # Force final update to ensure smooth transition
                    app.root.update_idletasks()
                except Exception as e:
                    logger.warning(f"Error removing loading screen: {e}")
            
            # Schedule removal after a brief delay to ensure UI is ready
            app.root.after(100, _remove_loading)
        
        # Start main event loop
        logger.info("Starting main loop...")
        app.root.mainloop()
        
    except Exception as e:
        logger.critical(f"Fatal error during app initialization: {e}", exc_info=True)
        raise
