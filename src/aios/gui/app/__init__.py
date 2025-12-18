"""AI-OS GUI Application Package - Refactored Modular Structure.

This package contains the refactored application modules and main AiosTkApp class.

Modules:
- resource_setup: Thread pools, timers, async loop
- ui_setup: Notebook structure and tabs
- panel_setup: All panel initialization
- logging_setup: Logging configuration
- state_management: State persistence
- chat_operations: Chat routing and execution
- brain_operations: Brain management
- goal_operations: Goal management
- event_handlers: Window and keyboard events
- cleanup: Resource cleanup
- tray_management: System tray integration
- app_main: Main orchestrator

Public API:
- AiosTkApp: Main application class
- run: Entry point function
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast, TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk

# Optional imports: allow module import without Tk installed (for CI/tests)
try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)

# Mixins for CLI bridge and debug functionality
from ..mixins.cli_bridge import CliBridgeMixin
from ..mixins.debug import DebugMixin
from ..utils.resource_management import set_worker_pool

# Main orchestrator function
from .app_main import run_app

logger = logging.getLogger(__name__)


class AiosTkApp(DebugMixin, CliBridgeMixin):
    """
    AI-OS GUI Application - Thin Coordinator.
    
    This class serves as the main entry point for the GUI application.
    It inherits CLI bridging and debug capabilities from mixins, then
    delegates all initialization and operations to specialized modules.
    
    The __init__ method:
    1. Sets up the Tkinter root window
    2. Determines the project root directory
    3. Calls run_app(self) to initialize all components
    
    All panels, callbacks, and attributes are populated by the modular
    initialization functions.
    """
    
    # Predeclare dynamic UI attributes for static analyzers
    # These are populated by the initialization modules
    root: Any
    notebook: Any
    chat_tab: Any
    brains_tab: Any
    datasets_tab: Any
    training_tab: Any
    evaluation_tab: Any
    resources_tab: Any
    debug_tab: Any
    settings_tab: Any
    mcp_tab: Any
    
    chat_panel: Any
    brains_panel: Any
    dataset_download_panel: Any
    dataset_builder_panel: Any
    hrm_training_panel: Any
    evaluation_panel: Any
    resources_panel: Any
    debug_panel: Any
    settings_panel: Any
    mcp_panel: Any
    status_bar: Any
    
    _worker_pool: Any
    _process_reaper: Any
    _timer_manager: Any
    _resource_monitor: Any
    _async_loop: Any
    _log_router: Any
    _ui_dispatcher: Any
    _state_file: Any
    _state: Any
    _tray_manager: Any
    
    # Operation callbacks (populated by setup functions)
    _on_chat_route_and_run: Any
    _on_load_brain: Any
    _on_unload_model: Any
    _on_list_brains: Any
    _on_goal_add_for_brain: Any
    _on_goals_list_for_brain: Any
    _on_goal_remove: Any
    _save_state: Any
    _cleanup: Any
    _set_error: Any
    _append_out: Any
    
    def __init__(self, root: "tk.Tk | None" = None, start_minimized: bool = False) -> None:
        """
        Initialize the AI-OS GUI application.
        
        Args:
            root: Tkinter root window (if None, creates new one)
            start_minimized: If True, start with window minimized to tray
        
        Raises:
            RuntimeError: If Tkinter is not available in this environment
        """
        import time
        import sys
        import platform
        from datetime import datetime
        
        init_start = time.time()
        
        # Log clear session separator for distinguishing between runs
        session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("=" * 80)
        logger.info(f"NEW SESSION STARTED: {session_time}")
        logger.info("=" * 80)
        
        logger.info("[INIT] Starting __init__...")
        
        # Log startup information
        logger.info("=" * 60)
        logger.info("AI-OS GUI Application Starting")
        logger.info("=" * 60)
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Start minimized: {start_minimized}")
        
        # Initialize parent mixins (CliBridgeMixin, DebugMixin)
        super().__init__()
        logger.info(f"[INIT] Mixins initialized: {time.time() - init_start:.3f}s")
        
        if tk is None:
            raise RuntimeError("Tkinter is not available in this environment")
        
        # Create or use provided root window
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        
        logger.info(f"[INIT] Root window created: {time.time() - init_start:.3f}s")

        # Placeholder until resource setup wires the dispatcher
        self._ui_dispatcher = None
        
        self.root.title("AI-OS Control Panel")
        self._start_minimized = start_minimized
        
        # Set default window size for better initial display
        self.root.geometry("1400x900")
        
        # CRITICAL: Withdraw window initially so we can set up loading screen before showing
        self.root.withdraw()
        
        # Ensure window decorations are enabled on all platforms
        # This ensures minimize/maximize/close buttons are present on the window frame
        try:
            # Explicitly disable overrideredirect to ensure decorations are shown
            self.root.overrideredirect(False)
            
            # Make window resizable (required for window manager to add decorations)
            self.root.resizable(True, True)
            
            # On Linux/GNOME: minimize/maximize buttons may be disabled by default
            # Users can enable them via: gsettings set org.gnome.desktop.wm.preferences button-layout ":minimize,maximize,close"
            # Or through GNOME Tweaks -> Window Titlebars -> Titlebar Buttons
            if sys.platform.startswith("linux"):
                logger.debug("Running on Linux - window decorations enabled. "
                           "Note: GNOME disables minimize/maximize buttons by default.")
        except Exception:
            logger.debug("Failed to configure window decorations", exc_info=True)
        
        # Set window icon if available
        try:
            icon_path = Path(__file__).parent.parent.parent.parent / "installers" / "AI-OS.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except Exception:
            pass
        
        # Create loading screen FIRST - before any heavy initialization
        try:
            from tkinter import ttk
            from PIL import Image, ImageTk
            
            # Configure background color
            self.root.configure(bg='#1e1e1e')  # Dark background
            
            # Create loading overlay frame that fills the entire window
            loading_frame = tk.Frame(self.root, bg='#1e1e1e')
            loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            
            # Create canvas for background images - will resize with window
            canvas = tk.Canvas(loading_frame, bg='#1e1e1e', highlightthickness=0)
            canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
            
            # Function to update canvas when window resizes
            # Image cache to prevent repeated loading/resizing
            _image_cache = {}
            _last_canvas_size = [0, 0]
            
            def update_loading_canvas(event=None):
                """Redraw canvas content when window size changes (optimized with caching)."""
                # Skip if loading is no longer active
                if hasattr(self, '_loading_active') and not self._loading_active:
                    return
                    
                # Get actual window dimensions
                try:
                    w = canvas.winfo_width()
                    h = canvas.winfo_height()
                    if w <= 1 or h <= 1:  # Not yet rendered
                        return
                    
                    # Skip if size hasn't changed significantly (prevent micro-updates)
                    if abs(w - _last_canvas_size[0]) < 10 and abs(h - _last_canvas_size[1]) < 10:
                        return
                    
                    _last_canvas_size[0] = w
                    _last_canvas_size[1] = h
                    
                    # Clear canvas
                    canvas.delete('all')
                    
                    # Use cached images if available for this size
                    cache_key = f"{w}x{h}"
                    
                    # Load and scale background image (with caching)
                    if hasattr(canvas, '_bg_image_path') and canvas._bg_image_path:
                        try:
                            if cache_key not in _image_cache:
                                bg_img = Image.open(canvas._bg_image_path)
                                # Scale to FIT INSIDE window (maintain aspect ratio)
                                img_w, img_h = bg_img.size
                                scale = min(w / img_w, h / img_h)
                                new_w = int(img_w * scale)
                                new_h = int(img_h * scale)
                                bg_img = bg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                bg_photo = ImageTk.PhotoImage(bg_img)
                                _image_cache[cache_key] = {'bg': bg_photo}
                            else:
                                bg_photo = _image_cache[cache_key]['bg']
                            
                            canvas._bg_photo = bg_photo  # Keep reference
                            canvas.create_image(w // 2, h // 2, image=bg_photo, anchor='center', tags='bg')
                        except Exception as e:
                            logger.error(f"[CANVAS] Failed to redraw background: {e}")
                    
                    # Semi-transparent overlay
                    canvas.create_rectangle(0, 0, w, h, fill='black', stipple='gray50', tags='overlay')
                    
                    # Load and position logo (logo doesn't need size-specific caching)
                    if hasattr(canvas, '_logo_image_path') and canvas._logo_image_path:
                        try:
                            if 'logo' not in _image_cache:
                                logo_img = Image.open(canvas._logo_image_path)
                                logo_img.thumbnail((100, 100), Image.Resampling.LANCZOS)
                                logo_photo = ImageTk.PhotoImage(logo_img)
                                _image_cache['logo'] = {'photo': logo_photo, 'size': logo_img.size}
                            
                            logo_photo = _image_cache['logo']['photo']
                            logo_w, logo_h = _image_cache['logo']['size']
                            canvas._logo_photo = logo_photo  # Keep reference
                            canvas.create_image(20 + logo_w // 2, h - 20 - logo_h // 2,
                                              image=logo_photo, anchor='center', tags='logo')
                        except Exception as e:
                            logger.error(f"[CANVAS] Failed to redraw logo: {e}")
                    
                    # Text elements at very bottom
                    loading_y = h - 80  # 80px from bottom
                    canvas.create_text(w // 2, loading_y, text="⏳ Loading...",
                                     font=("Arial", 16), fill='#cccccc', tags='loading')
                    
                    # Store status text ID for updates
                    status_id = canvas.create_text(w // 2, loading_y + 30, text="Starting up...",
                                                   font=("Arial", 11), fill='#999999', tags='status')
                    canvas._status_text_id = status_id
                    
                except Exception as e:
                    logger.error(f"[CANVAS] Error updating canvas: {e}")
            
            # Store image paths for reloading
            bg_image_path = Path(__file__).parent.parent.parent.parent.parent / "AI-OS.png"
            logo_image_path = Path(__file__).parent.parent.parent.parent.parent / "WulfNet_Designs.jpg"
            
            canvas._bg_image_path = bg_image_path if bg_image_path.exists() else None
            canvas._logo_image_path = logo_image_path if logo_image_path.exists() else None
            
            # Bind canvas resize with aggressive debouncing to prevent UI hangs
            _resize_scheduled = [False]  # Mutable flag for closure
            _last_resize_time = [0.0]  # Track last resize to prevent rapid updates
            _pending_resize_timer = [None]  # Track pending timer
            
            def _safe_canvas_update(event=None):
                """Canvas update with aggressive debouncing to prevent UI hangs."""
                try:
                    # Skip if loading is done
                    if hasattr(self, '_loading_active') and not self._loading_active:
                        return
                    
                    # Skip if canvas or frame no longer exists
                    if not hasattr(self, '_loading_canvas') or not self._loading_canvas:
                        return
                    if not hasattr(self, '_loading_frame') or not self._loading_frame:
                        return
                    
                    # Cancel any pending update
                    if _pending_resize_timer[0] is not None:
                        try:
                            self.root.after_cancel(_pending_resize_timer[0])
                        except Exception:
                            pass
                        _pending_resize_timer[0] = None
                    
                    # Debounce: only schedule one update at a time
                    if _resize_scheduled[0]:
                        return
                    
                    # Rate limit: don't update more than once every 200ms
                    import time
                    now = time.time()
                    if now - _last_resize_time[0] < 0.2:
                        # Schedule for later
                        _pending_resize_timer[0] = self.root.after(200, lambda: _safe_canvas_update(event))
                        return
                    
                    _resize_scheduled[0] = True
                    
                    def _do_update():
                        _resize_scheduled[0] = False
                        _pending_resize_timer[0] = None
                        try:
                            # Only update if window is viewable and loading is still active
                            if (hasattr(self, '_loading_active') and self._loading_active and
                                self.root.winfo_viewable() and 
                                hasattr(self, '_loading_canvas') and self._loading_canvas):
                                update_loading_canvas(event)
                                _last_resize_time[0] = time.time()
                        except Exception:
                            pass
                    
                    # Debounce by 250ms to batch rapid resize events
                    _pending_resize_timer[0] = self.root.after(250, _do_update)
                except Exception:
                    _resize_scheduled[0] = False
            
            # Store the handler and bind it
            self._canvas_update_handler = _safe_canvas_update
            canvas.bind('<Configure>', _safe_canvas_update)
            
            # Store references for updating during initialization
            self._loading_frame = loading_frame
            self._loading_canvas = canvas
            self._loading_active = True  # Flag to track if loading screen is active

            
            # CRITICAL: Show window FIRST, then update to render the loading screen
            self.root.deiconify()
            if not start_minimized:
                self._maximize_startup_window()
            
            # NOW update to render the loading screen (window must be visible first)
            self.root.update_idletasks()
            self.root.update()
            
            # Trigger initial canvas draw NOW (not after() - those won't run until mainloop starts)
            update_loading_canvas()
            
            # Force another update to ensure canvas is rendered
            self.root.update()

            logger.info(f"[INIT] Loading screen displayed: {time.time() - init_start:.3f}s")
            
        except Exception as e:
            # Log error but don't fall back - let it fail visibly so we can fix it
            logger.error(f"CRITICAL: Failed to create loading screen: {e}", exc_info=True)
            raise  # Re-raise to make failures obvious
        
        # Determine project root
        # Navigate up from src/aios/gui/app/__init__.py to project root
        self._project_root = Path(__file__).parent.parent.parent.parent
        logger.info(f"Project root: {self._project_root}")
        logger.info(f"[INIT] About to call run_app: {time.time() - init_start:.3f}s")
        
        # Delegate all initialization to run_app orchestrator
        # This will:
        # 1. Initialize resources (threads, timers, async loop)
        # 2. Set up logging system
        # 3. Initialize state management
        # 4. Create UI structure (notebook with tabs)
        # 5. Initialize all panels
        # 6. Set up operation handlers
        # 7. Set up event handlers
        # 8. Load saved state
        # 9. Start main loop
        run_app(self)
    
    def run(self) -> None:
        """
        Start the Tkinter main loop.
        
        Note: This method is typically not called directly. The run_app()
        orchestrator handles starting the main loop.
        """
        if not hasattr(self, '_main_loop_started'):
            self.root.mainloop()
    
    def _maximize_startup_window(self, attempt: int = 0) -> None:
        """Best-effort maximize to keep the loading overlay full screen."""

        if getattr(self, "_start_minimized", False):
            return

        # Platform-specific maximization strategies
        import sys
        
        if sys.platform.startswith("win"):
            # Windows: Use zoomed state for proper window maximize
            try:
                self.root.state("zoomed")
            except Exception:
                pass
        elif sys.platform.startswith("linux"):
            # Linux: Avoid fullscreen-like behavior that removes decorations
            # Instead, set geometry to leave room for taskbar/panel and window decorations
            try:
                self.root.update_idletasks()
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()
                
                # Leave space for typical Ubuntu panel (28px top) and decorations
                # This ensures window decorations (close/minimize/maximize) remain visible
                window_w = screen_w - 4  # Small margin to avoid borderless fullscreen
                window_h = screen_h - 60  # Space for panel and decorations
                pos_x = 0
                pos_y = 28  # Typical Ubuntu top panel height
                
                geometry = f"{window_w}x{window_h}+{pos_x}+{pos_y}"
                self.root.geometry(geometry)
                
                # Ensure the window is not in a fullscreen-like state
                try:
                    self.root.attributes("-fullscreen", False)
                except Exception:
                    pass
            except Exception:
                logger.debug("Failed to set Linux window geometry", exc_info=True)
        else:
            # macOS and other platforms: Use zoomed attribute
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

        try:
            self.root.update_idletasks()
        except Exception:
            pass

        # Fallback geometry adjustment for Windows and other platforms
        if not sys.platform.startswith("linux"):
            try:
                screen_w = self.root.winfo_screenwidth()
                screen_h = self.root.winfo_screenheight()

                current_w = self.root.winfo_width()
                current_h = self.root.winfo_height()

                if current_w <= 1 or current_h <= 1:
                    self.root.update_idletasks()
                    current_w = self.root.winfo_width()
                    current_h = self.root.winfo_height()

                slack_w = max(48, int(screen_w * 0.05))
                slack_h = max(48, int(screen_h * 0.05))

                if current_w < (screen_w - slack_w) or current_h < (screen_h - slack_h):
                    geometry = f"{screen_w}x{screen_h}+0+0"
                    self.root.geometry(geometry)
            except Exception:
                pass

        max_attempts = 2
        if attempt < max_attempts:
            try:
                self.root.after(180, lambda: self._maximize_startup_window(attempt + 1))
            except Exception:
                pass

    def _bring_to_front_windows(self) -> None:
        """Best-effort Windows-specific foreground call using Win32 APIs."""

        try:
            import sys

            if not sys.platform.startswith("win"):
                return

            import ctypes

            hwnd = int(self.root.winfo_id())
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
        except Exception:
            logger.debug("Win32 foreground request failed", exc_info=True)

    def _schedule_foreground_boost(self, initial_delay: int = 0, attempts: int = 3, interval: int = 400) -> None:
        """Temporarily mark the window topmost to regain focus during startup."""

        if getattr(self, "_start_minimized", False):
            return

        def _boost(attempt: int = 0) -> None:
            max_attempts = max(0, attempts)
            if attempt >= max_attempts:
                return

            try:
                if not self.root.winfo_exists():
                    return

                # Skip until window is mapped to avoid raising a withdrawn window
                if not self.root.winfo_viewable():
                    self.root.after(interval, lambda: _boost(attempt))
                    return

                self.root.lift()
                self.root.attributes('-topmost', True)
                if attempt == 0:
                    self._bring_to_front_windows()
                try:
                    self.root.focus_force()
                except Exception:
                    try:
                        self.root.focus_set()
                    except Exception:
                        pass

                def _release() -> None:
                    try:
                        self.root.attributes('-topmost', False)
                    except Exception:
                        pass

                self.root.after(250, _release)
            except Exception:
                logger.debug("Foreground boost attempt failed", exc_info=True)

            next_attempt = attempt + 1
            if next_attempt < max_attempts:
                self.root.after(interval, lambda: _boost(next_attempt))

        self.root.after(max(0, initial_delay), _boost)

    def post_to_ui(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Execute ``func`` on the Tk UI thread.

        Background threads should use this helper instead of touching Tk widgets
        directly. When already on the UI thread the callable executes
        immediately to preserve ordering.
        """

        def _invoke() -> None:
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
            except Exception:
                logger.exception("UI callback failed")
            finally:
                duration = time.perf_counter() - start
                threshold = getattr(self, "_ui_callback_warn_threshold", 0.2)
                if duration > threshold:
                    callback_name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
                    logger.warning(
                        "UI callback '%s' executed in %.1f ms (threshold %.0f ms)",
                        callback_name,
                        duration * 1000.0,
                        threshold * 1000.0,
                    )

        dispatcher = getattr(self, '_ui_dispatcher', None)
        if dispatcher is not None:
            try:
                is_thread_check = getattr(dispatcher, 'is_ui_thread', None)
                if callable(is_thread_check) and is_thread_check():
                    _invoke()
                    return
                dispatcher.dispatch(_invoke)
                return
            except Exception:
                logger.exception("Failed to dispatch UI callback via dispatcher")

        try:
            if hasattr(self, 'root') and self.root:
                self.root.after(0, _invoke)
                return
        except Exception:
            pass

        _invoke()

    def _emergency_cleanup(self) -> None:
        """
        Last-resort cleanup if normal shutdown fails.
        
        This method is registered with atexit and will be called when the
        Python interpreter exits, even if the normal cleanup process fails.
        It performs minimal critical cleanup operations.
        """
        logger.info("Emergency cleanup triggered")
        
        # Shutdown worker pool (critical for thread cleanup)
        if hasattr(self, '_ui_dispatcher') and self._ui_dispatcher:
            try:
                self._ui_dispatcher.stop()
            except Exception:
                pass

        if hasattr(self, '_worker_pool') and self._worker_pool:
            try:
                self._worker_pool.shutdown(wait=False)
            except Exception:
                pass
        set_worker_pool(None)
        
        # Stop async event loop
        if hasattr(self, '_async_loop') and self._async_loop:
            try:
                if self._async_loop.is_running:
                    self._async_loop.stop()
            except Exception:
                pass
        
        # Shutdown process reaper
        if hasattr(self, '_process_reaper') and self._process_reaper:
            try:
                self._process_reaper.cleanup_all(timeout=1.0)
            except Exception:
                pass
        
        # Cancel all pending timers
        if hasattr(self, '_timer_manager') and self._timer_manager:
            try:
                self._timer_manager.cancel_all()
            except Exception:
                pass


def run(exit_after: float | None = None, minimized: bool = False):
    """
    Start the Tkinter app (module-level entry point).
    
    Args:
        exit_after: if provided (>0), auto-close the window after N seconds (CI/headless smoke).
        minimized: if True, start with window minimized to tray
    """
    import sys
    import os
    
    # Ensure stdout/stderr are valid to prevent crashes in Tkinter
    # When launched via pythonw.exe or certain wrappers, these can be None or broken
    # This is critical for Windows where GUI apps may not have a console
    try:
        if sys.stdout is None or not hasattr(sys.stdout, 'write'):
            sys.stdout = open(os.devnull, 'w')
        if sys.stderr is None or not hasattr(sys.stderr, 'write'):
            sys.stderr = open(os.devnull, 'w')
    except Exception:
        pass
    
    if tk is None:
        logger.warning("Tkinter is not available - GUI cannot start")
        return
    
    try:
        logger.info("Starting AI-OS GUI...")
        root = tk.Tk()
        app = AiosTkApp(root, start_minimized=minimized)
        
        # Auto-close for CI/testing
        if exit_after and exit_after > 0:
            root.after(int(exit_after * 1000), root.destroy)
        
        logger.info("GUI initialized successfully")
        
        # Note: mainloop is already started by run_app() called in __init__
        # No need to call it again here
        
    except Exception as e:
        logger.error(f"Failed to start GUI: {e}", exc_info=True)
        try:
            if 'root' in locals():
                root.destroy()  # type: ignore[name-defined]
        except Exception:
            pass


__all__ = [
    # Main API
    'AiosTkApp',
    'run',
    # Setup functions (for advanced usage)
    'setup_resources',
    'create_ui_structure',
    'initialize_panels',
    'initialize_logging',
    'configure_log_levels',
    'initialize_state',
    'load_state',
    'save_state',
    'restore_state',
    'setup_chat_operations',
    'setup_brain_operations',
    'setup_goal_operations',
    'setup_event_handlers',
    'setup_periodic_tasks',
    'cleanup',
    'init_tray',
    'on_tray_settings',
    'run_app',
]

# Re-export module functions for advanced usage
from .resource_setup import setup_resources
from .ui_setup import create_ui_structure
from .panel_setup import initialize_panels
from .logging_setup import initialize_logging, configure_log_levels
from .state_management import (
    initialize_state,
    load_state,
    save_state,
    restore_state,
)
from .chat_operations import setup_chat_operations
from .brain_operations import setup_brain_operations
from .goal_operations import setup_goal_operations
from .event_handlers import setup_event_handlers, setup_periodic_tasks
from .cleanup import cleanup
from .tray_management import init_tray, on_tray_settings
from .app_main import run_app
