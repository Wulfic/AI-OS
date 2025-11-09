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
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

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
        # Initialize parent mixins (CliBridgeMixin, DebugMixin)
        super().__init__()
        
        if tk is None:
            raise RuntimeError("Tkinter is not available in this environment")
        
        # Create or use provided root window
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        
        self.root.title("AI-OS Control Panel")
        self._start_minimized = start_minimized
        
        # Set default window size for better initial display
        self.root.geometry("1400x900")
        
        # Hide window during initialization to prevent blank white screen
        self.root.withdraw()
        
        # Determine project root
        # Navigate up from src/aios/gui/app/__init__.py to project root
        self._project_root = Path(__file__).parent.parent.parent.parent
        logger.info(f"Project root: {self._project_root}")
        
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
    
    def _emergency_cleanup(self) -> None:
        """
        Last-resort cleanup if normal shutdown fails.
        
        This method is registered with atexit and will be called when the
        Python interpreter exits, even if the normal cleanup process fails.
        It performs minimal critical cleanup operations.
        """
        logger.info("Emergency cleanup triggered")
        
        # Shutdown worker pool (critical for thread cleanup)
        if hasattr(self, '_worker_pool') and self._worker_pool:
            try:
                self._worker_pool.shutdown(wait=False)
            except Exception:
                pass
        
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
