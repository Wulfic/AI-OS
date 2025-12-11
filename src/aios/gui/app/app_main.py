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
from typing import TYPE_CHECKING, Any
import faulthandler
import sys
import logging
import time
import os
import traceback
from datetime import datetime
from pathlib import Path

# ============================================================================
# EARLY CRASH PROTECTION - Enable before any heavy imports
# This helps capture crashes that happen during startup before logging is ready
# ============================================================================
_CRASH_LOG_PATH: Path | None = None


def _get_install_root() -> Path:
    """Get the AI-OS install root directory.
    
    This is a simplified version for early crash protection,
    before we can import the full paths module.
    """
    # Walk up from this file to find install root markers
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "launcher.bat").exists():
            return parent
        if (parent / ".venv").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: 4 levels up from this file
    # (app_main.py -> app -> gui -> aios -> src -> root)
    return current.parents[4]


def _get_crash_log_path() -> Path:
    """Get path for early crash log file.
    
    On Windows, uses the install location's logs folder.
    This keeps all logs together and avoids scattered files in AppData/Desktop.
    """
    global _CRASH_LOG_PATH
    if _CRASH_LOG_PATH is not None:
        return _CRASH_LOG_PATH
    
    # Try multiple locations in order of preference
    candidates = []
    
    # 1. Install location's logs folder (preferred on Windows)
    try:
        install_root = _get_install_root()
        logs_dir = install_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        candidates.append(logs_dir / "crash.log")
    except Exception:
        pass
    
    # 2. Current working directory's logs folder
    try:
        cwd_logs = Path.cwd() / "logs"
        cwd_logs.mkdir(parents=True, exist_ok=True)
        candidates.append(cwd_logs / "crash.log")
    except Exception:
        pass
    
    # 3. Temp directory as last resort
    try:
        import tempfile
        candidates.append(Path(tempfile.gettempdir()) / "aios_crash.log")
    except Exception:
        pass
    
    # Try to create and write to each candidate
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            # Test write
            with open(candidate, "a", encoding="utf-8") as f:
                f.write("")
            _CRASH_LOG_PATH = candidate
            return candidate
        except Exception:
            continue
    
    # Absolute fallback
    _CRASH_LOG_PATH = Path("aios_crash.log")
    return _CRASH_LOG_PATH


def _write_crash_log(message: str, exc: Exception | None = None) -> None:
    """Write to crash log file immediately (synchronous, no buffering)."""
    try:
        crash_path = _get_crash_log_path()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        with open(crash_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] {message}\n")
            if exc:
                f.write(f"Exception: {type(exc).__name__}: {exc}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            f.write(f"{'='*80}\n")
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    except Exception:
        pass  # Can't do anything if crash log fails


def _setup_early_crash_protection() -> None:
    """Set up early crash protection before heavy imports."""
    try:
        # Enable faulthandler immediately for native crashes
        faulthandler.enable()
        
        # Try to write to a crash log file
        crash_file = _get_crash_log_path()
        try:
            crash_file_handle = open(crash_file, "a", encoding="utf-8")
            faulthandler.enable(file=crash_file_handle)
        except Exception:
            pass
        
        # Install global exception handler
        _original_excepthook = sys.excepthook
        
        def _crash_excepthook(exc_type, exc_value, exc_tb):
            _write_crash_log(
                f"Unhandled exception in main thread: {exc_type.__name__}",
                exc_value
            )
            _original_excepthook(exc_type, exc_value, exc_tb)
        
        sys.excepthook = _crash_excepthook
        
        # Install threading exception handler (Python 3.8+)
        import threading
        _original_threading_excepthook = getattr(threading, 'excepthook', None)
        
        def _threading_excepthook(args):
            _write_crash_log(
                f"Unhandled exception in thread '{args.thread.name if args.thread else 'unknown'}': {args.exc_type.__name__}",
                args.exc_value
            )
            if _original_threading_excepthook:
                _original_threading_excepthook(args)
        
        threading.excepthook = _threading_excepthook
        
        _write_crash_log("AI-OS GUI starting - crash protection enabled")
        
    except Exception as e:
        # Even crash protection setup can fail - log it
        try:
            print(f"[CRASH PROTECTION] Setup failed: {e}", file=sys.stderr)
        except Exception:
            pass


# Enable crash protection immediately
_setup_early_crash_protection()

try:  # pragma: no cover - bootstrap guard
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None

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


def _apply_saved_artifacts_override() -> None:
    """Load the artifacts directory override from config and apply it early."""

    if system_paths is None:
        return

    try:
        from ..components.resources_panel import config_persistence
    except Exception:
        return

    try:
        resources_cfg = config_persistence.load_resources_from_config()
    except Exception:
        return

    override = str(resources_cfg.get("artifacts_dir") or "").strip()
    if not override:
        # Ensure we reset to defaults if the override was removed
        system_paths.set_artifacts_root_override(None)
        os.environ.pop("AIOS_ARTIFACTS_DIR", None)
        return

    candidate = Path(override).expanduser()
    error = None
    try:
        error = system_paths.test_directory_writable(candidate)
    except Exception as exc:
        error = str(exc)

    if error:
        logger.warning("Artifacts override '%s' ignored: %s", candidate, error)
        return

    system_paths.set_artifacts_root_override(candidate)
    os.environ["AIOS_ARTIFACTS_DIR"] = str(candidate)


def run_app(app_instance: Any) -> None:
    """
    Main application initialization and run loop.
    
    This is the orchestrator function that initializes all components
    in the correct order and starts the Tkinter main loop.
    
    Args:
        app_instance: AiosTkApp instance (must have root, _project_root, _run_cli)
    """
    
    app = app_instance
    _write_crash_log("run_app: Starting application initialization")
    logger.info("Starting application initialization")
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
        _write_crash_log(f"run_app: {step_name} ({step_duration:.3f}s)")
        last_time = current
    
    _faulthandler_timer_active = False
    try:
        _write_crash_log("run_app: Applying artifacts override")
        _apply_saved_artifacts_override()
        # Use existing loading screen (created in __init__)
        def update_status(text):
            """Update startup status messaging without blocking the Tk loop."""
            try:
                logger.debug(f"Loading status: {text}")

                loading_active = getattr(app, '_loading_active', False)

                if loading_active and hasattr(app, '_loading_canvas') and app._loading_canvas:
                    if hasattr(app._loading_canvas, '_status_text_id'):
                        try:
                            app._loading_canvas.itemconfig(app._loading_canvas._status_text_id, text=text)
                            # Queue a no-op idle callback to ensure Tk processes canvas updates asynchronously
                            app.root.after_idle(lambda: None)
                        except Exception:
                            pass

                if hasattr(app, 'status_bar') and getattr(app, 'status_bar', None):
                    try:
                        app.status_bar.set(text)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # 1. Initialize resources (threads, timers, async loop)
        _write_crash_log("run_app: Step 1 - Initializing resources")
        logger.info("Initializing resources...")
        update_status("Initializing resources...")
        setup_resources(app, app.root, False)  # start_minimized handled elsewhere
        log_timing("Resources initialized")
        
        # 2. Initialize logging system
        _write_crash_log("run_app: Step 2 - Initializing logging")
        logger.info("Initializing logging...")
        update_status("Initializing logging...")
        initialize_logging(app, app._project_root)
        log_timing("Logging initialized")
        
        # 3. Initialize state management
        _write_crash_log("run_app: Step 3 - Initializing state management")
        logger.info("Initializing state management...")
        update_status("Initializing state management...")
        initialize_state(app, app._project_root)
        log_timing("State management initialized")
        
        # 4. Create UI structure (notebook with tabs)
        _write_crash_log("run_app: Step 4 - Creating UI structure")
        logger.info("Creating UI structure...")
        update_status("Creating interface layout...")
        create_ui_structure(app, app.root)
        
        # Keep loading screen on top after UI structure is created
        if hasattr(app, '_loading_frame') and app._loading_frame:
            try:
                app._loading_frame.lift()
            except Exception:
                pass
        
        log_timing("UI structure created")
        
        # 5. Set up operation handlers (BEFORE panels - they depend on these)
        _write_crash_log("run_app: Step 5 - Setting up operations")
        logger.info("Setting up operations...")
        update_status("Initializing core systems...")
        setup_chat_operations(app)
        setup_brain_operations(app)
        setup_goal_operations(app)
        log_timing("Operations set up")
        
        # 5b. Attach cleanup and save functions to app (BEFORE panels - they need these)
        app._cleanup = lambda: cleanup(app)
        app._save_state = lambda sync=False: save_state(app, sync=sync)
        
        # Add a scheduled state save mechanism to batch saves
        app._state_save_timer = None

        def schedule_state_save(delay_ms: int = 500) -> None:
            """Schedule a state save after a delay to batch multiple changes."""
            try:
                if hasattr(app, '_state_save_timer') and app._state_save_timer:
                    try:
                        app.root.after_cancel(app._state_save_timer)
                    except Exception:
                        pass

                def _run_save() -> None:
                    app._state_save_timer = None
                    try:
                        app._save_state()
                    except Exception:
                        # Best-effort; errors are already logged inside save_state
                        pass

                app._state_save_timer = app.root.after(delay_ms, _run_save)
            except Exception:
                # Fallback to immediate save if scheduling fails
                try:
                    app._save_state()
                except Exception:
                    pass

        app.schedule_state_save = schedule_state_save
        
        # 6. Set up event handlers
        _write_crash_log("run_app: Step 6 - Setting up event handlers")
        logger.info("Setting up event handlers...")
        update_status("Connecting event handlers...")
        setup_event_handlers(app)
        log_timing("Event handlers set up")
        
        # 7. Set up system tray (optional)
        _write_crash_log("run_app: Step 7 - Initializing system tray")
        update_status("Initializing system tray...")
        try:
            init_tray(app)
        except Exception as e:
            logger.warning(f"Failed to set up system tray: {e}")
        log_timing("System tray initialized")
        
        # 8. Initialize all panels (with progress updates)
        _write_crash_log("run_app: Step 8 - Initializing panels")
        logger.info("Initializing panels...")
        update_status("Building interface panels...")
        initialize_panels(app)
        
        # Keep loading screen on top after panels are created
        if hasattr(app, '_loading_frame') and app._loading_frame:
            try:
                app._loading_frame.lift()
            except Exception:
                pass
        
        log_timing("Panels initialized")
        
        # 9. Load panel data synchronously while keeping loading screen visible
        # This ensures all data is loaded before user can interact
        _write_crash_log("run_app: Step 9 - Loading panel data (parallel)")
        logger.info("Loading panel data...")
        update_status("Loading application data...")
        from .panel_setup import load_all_panel_data
        load_all_panel_data(app, update_status)
        log_timing("Panel data loaded")
        
        # 10. Load and restore saved state
        _write_crash_log("run_app: Step 10 - Restoring saved state")
        logger.info("Loading saved state...")
        update_status("Restoring your preferences...")
        state = load_state(app)
        if state:
            restore_state(app, state)
        else:
            # No state to restore, but enable saving for future
            logger.info("No saved state found - starting with defaults")
            app._state_restored = True
        log_timing("State restored")
        
        # 10b. Re-apply theme after all panels are fully initialized
        # This ensures theme is applied to all components including late-loaded ones
        _write_crash_log("run_app: Step 10b - Applying theme")
        if hasattr(app, 'settings_panel') and app.settings_panel:
            try:
                current_theme = app.settings_panel.theme_var.get()
                logger.info(f"Current theme from settings panel: '{current_theme}'")
                
                # Only re-apply if we have a valid theme (not just the default)
                if current_theme and current_theme in ("Light Mode", "Dark Mode", "Matrix Mode", "Barbie Mode", "Halloween Mode"):
                    last_theme = getattr(app.settings_panel, "_last_theme_applied", None)
                    last_applied_at = getattr(app.settings_panel, "_last_theme_applied_at", 0.0)
                    elapsed_since_apply = time.perf_counter() - last_applied_at if last_applied_at else None

                    if last_theme == current_theme and elapsed_since_apply is not None and elapsed_since_apply < 2.0:
                        logger.info(
                            "Skipping theme re-application; '%s' was applied %.3fs ago",
                            current_theme,
                            elapsed_since_apply,
                        )
                    else:
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
        _write_crash_log("run_app: Step 11 - Configuring log levels")
        update_status("Finalizing configuration...")
        log_level_setting = "Normal"  # Default
        if hasattr(app, 'settings_panel') and app.settings_panel:
            try:
                settings_state = app.settings_panel.get_state()
                log_level_setting = settings_state.get('log_level', 'Normal')
            except Exception:
                pass
        configure_log_levels(app, log_level_setting)
        log_timing("Log levels configured")
        
        # 12. Setup periodic tasks
        _write_crash_log("run_app: Step 12 - Setting up periodic tasks")
        update_status("Starting background tasks...")
        setup_periodic_tasks(app)
        
        # NOTE: Help index building is now handled by HelpPanel itself during initialization
        # No need to build it here - removed to prevent race condition with HelpPanel's own index build
        
        _write_crash_log("run_app: Initialization complete, entering main loop")
        logger.info("AI-OS GUI initialized successfully")
        
        total_startup = time.time() - start_time
        msg = f"[TIMING] Total startup time: {total_startup:.3f}s"
        logger.info(msg)

        if os.environ.get("AIOS_ENABLE_FAULTHANDLER", "0") == "1":
            try:
                faulthandler.enable()
                logger.debug("Faulthandler enabled for post-start diagnostics")

                if sys.platform.startswith("win"):
                    logger.info(
                        "Skipping faulthandler.dump_traceback_later repeating timer on Windows (stability workaround)"
                    )
                else:
                    faulthandler.dump_traceback_later(15.0, repeat=True)
                    _faulthandler_timer_active = True
                    logger.debug("Scheduled faulthandler traceback dumps every 15s")
            except Exception:
                logger.debug("Failed to enable faulthandler diagnostics", exc_info=True)
        else:
            logger.debug(
                "Faulthandler diagnostics disabled by default; set AIOS_ENABLE_FAULTHANDLER=1 to enable periodic stack dumps."
            )
        
        # Final status update
        update_status("Ready!")
        
        # Single final UI update
        app.root.update_idletasks()

        if not getattr(app, "_start_minimized", False) and hasattr(app, "_schedule_foreground_boost"):
            try:
                app.root.after(600, lambda: app._schedule_foreground_boost(initial_delay=0, attempts=3, interval=400))
            except Exception:
                logger.debug("Failed to schedule post-start foreground boost", exc_info=True)
        
        # Remove loading screen with smooth transition
        if hasattr(app, '_loading_frame') and app._loading_frame:
            logger.debug("Preparing loading overlay removal")

            def _remove_loading(source: str = "unknown") -> None:
                try:
                    if getattr(app, '_loading_removed', False):
                        logger.debug("Loading overlay already removed (source=%s)", source)
                        return

                    logger.debug("Removing loading overlay (source=%s)", source)

                    app._loading_removed = True

                    if getattr(app, '_loading_active', False):
                        app._loading_active = False

                    try:
                        app._loading_frame.destroy()
                    finally:
                        app._loading_frame = None

                    app.root.update_idletasks()
                except Exception as e:
                    logger.warning(f"Error removing loading screen: {e}")

            def _ensure_overlay_removed(attempt: int = 1) -> None:
                try:
                    if getattr(app, '_loading_removed', False):
                        logger.debug("Loading overlay removal confirmed (attempt=%s)", attempt)
                        return

                    logger.warning("Loading overlay still present after %s attempt(s); forcing removal", attempt)
                    _remove_loading(f"watchdog-{attempt}")

                    if not getattr(app, '_loading_removed', False) and attempt < 5:
                        app.root.after(500, lambda: _ensure_overlay_removed(attempt + 1))
                except Exception:
                    logger.debug("Loading overlay watchdog failed", exc_info=True)
            
            # Best-effort immediate removal now that startup is complete
            try:
                _remove_loading("pre-loop")
            except Exception:
                logger.debug("Immediate loading overlay removal failed; will rely on scheduled callbacks", exc_info=True)

            # Schedule removal as soon as Tk is idle; fall back to a short delay if idle scheduling fails
            try:
                logger.debug("Scheduling loading overlay removal via after_idle")
                app.root.after_idle(lambda: _remove_loading("after_idle"))
            except Exception:
                app.root.after(100, lambda: _remove_loading("after_idle-fallback"))

            # Safety fallback in case Tk never reaches idle during busy startup
            try:
                logger.debug("Scheduling loading overlay removal fallback in 250ms")
                app.root.after(250, lambda: _remove_loading("delay-250ms"))
            except Exception:
                pass

            # Additional watchdog to ensure overlay cannot linger indefinitely
            try:
                logger.debug("Scheduling loading overlay watchdog in 750ms")
                app.root.after(750, lambda: _ensure_overlay_removed(1))
            except Exception:
                pass
        
        # Start deferred HelpPanel initialization after main loop is running
        # This prevents "main thread is not in main loop" errors
        if hasattr(app, 'help_panel') and app.help_panel:
            logger.info("Scheduling HelpPanel deferred initialization...")
            app.root.after_idle(app.help_panel.start_deferred_initialization)
        
        # Start main event loop
        logger.info("Starting main loop...")
        app.root.mainloop()
        
    except Exception as e:
        logger.critical(f"Fatal error during app initialization: {e}", exc_info=True)
        raise
    finally:
        if _faulthandler_timer_active:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                logger.debug("Failed to cancel faulthandler timer", exc_info=True)
