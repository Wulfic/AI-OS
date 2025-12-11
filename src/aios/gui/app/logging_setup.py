"""Logging setup module for AI-OS GUI.

This module initializes the logging system including:
- Log router for categorized logging
- Log file configuration
- Error callback setup
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import sys
import time
from pathlib import Path

if TYPE_CHECKING:
    pass

from ..services import LogRouter, LogCategory

logger = logging.getLogger(__name__)


def initialize_logging(app: Any, project_root: Path) -> None:
    """
    Initialize logging system for the application.
    
    Sets up:
    - LogRouter for categorized routing (chat, training, debug, dataset, error)
    - Error callback for UI display
    
    Note: Main logging configuration should be set up via setup_logging() from cli.utils
    before the GUI starts. This function only sets up GUI-specific logging components.
    
    Args:
        app: AiosTkApp instance
        project_root: Path to project root directory
    """
    logger.info(f"Initializing GUI logging components (project_root={project_root})")
    
    # Create log router with optional UI dispatcher support
    dispatcher = getattr(app, "_ui_dispatcher", None)
    app._log_router = LogRouter(dispatcher=dispatcher)
    if dispatcher is None:
        logger.debug("Created LogRouter instance without dispatcher (UI updates will execute inline)")
    else:
        logger.debug("Created LogRouter instance with UI dispatcher")
    
    # NOTE: Do not create additional file handlers here - they should be configured
    # via logging.yaml and setup_logging() in cli.utils. This prevents duplicate
    # handlers and ensures consistent logging configuration.
    
    # Error callback (will be connected to debug panel later)
    app._error_callback = None
    
    def _set_error(msg: str) -> None:
        """Set error message in debug panel (if available)."""
        def _apply() -> None:
            if hasattr(app, 'debug_panel') and app.debug_panel:
                try:
                    app.debug_panel.set_error(msg)
                except Exception as exc:  # pragma: no cover - guard rails
                    logger.debug(f"Could not set error in debug panel: {exc}")

        try:
            app.post_to_ui(_apply)
        except Exception:
            _apply()

        # Also log to stderr
        logger.error(msg)
    
    app._set_error = _set_error
    logger.debug("Configured error callback")
    
    # Output append callback (will be connected to debug panel later)
    def _append_out(msg: str) -> None:
        """Append output message to debug panel (if available)."""
        def _apply() -> None:
            if hasattr(app, 'debug_panel') and app.debug_panel:
                try:
                    app.debug_panel.write(msg)
                except Exception as exc:
                    logger.debug(f"Could not write to debug panel: {exc}")

        try:
            app.post_to_ui(_apply)
        except Exception:
            _apply()

        # Also log to logger
        logger.info(msg)
    
    app._append_out = _append_out
    logger.debug("Configured output append callback")
    logger.info("GUI logging components initialized successfully")


def configure_log_levels(app: Any, log_level_setting: str = "Normal") -> None:
    """
    Configure logging levels based on the settings panel log level.
    
    This affects both the Python logging system AND the debug panel display filter.
    
    Log Level Meanings:
    - "Normal": INFO and above with UI filtering that highlights essential categories
    - "Advanced": INFO and above across all categories for deeper troubleshooting  
    - "DEBUG": Everything including debug messages - for developers/troubleshooting
    
    Args:
        app: AiosTkApp instance
        log_level_setting: One of "Normal", "Advanced", or "DEBUG" (from settings panel)
    """
    start_time = time.perf_counter()
    logger.info(f"configure_log_levels called (log_level_setting={log_level_setting})")
    
    # Map settings dropdown to Python logging levels
    level_map = {
        "Normal": logging.INFO,       # Essential info and above
        "Advanced": logging.INFO,     # Info and above
        "DEBUG": logging.DEBUG,       # Everything
    }
    
    python_level = level_map.get(log_level_setting, logging.INFO)
    
    # Update Python logging levels for aios logger and root logger
    level_set_start = time.perf_counter()
    logging.getLogger("aios").setLevel(python_level)
    logging.getLogger().setLevel(python_level)
    level_duration = time.perf_counter() - level_set_start
    
    level_name = logging.getLevelName(python_level)
    logger.info(
        f"Python logging level set to {level_name} (from setting: {log_level_setting}) "
        f"[{level_duration:.3f}s]"
    )
    
    # Also update console handler level if it exists
    handler_start = time.perf_counter()
    handler_count = 0
    updated_handlers = 0
    for handler in logging.getLogger().handlers:
        handler_count += 1
        # Check that handler.stream exists and has a name attribute before accessing
        # This handles edge cases on fresh installs where streams may be None
        if (
            isinstance(handler, logging.StreamHandler)
            and handler.stream is not None
            and getattr(handler.stream, 'name', None) in ('<stderr>', '<stdout>')
        ):
            handler.setLevel(python_level)
            updated_handlers += 1
    handler_duration = time.perf_counter() - handler_start
    if handler_count:
        logger.debug(
            f"configure_log_levels handler sweep: checked {handler_count} handler(s), "
            f"updated {updated_handlers} in {handler_duration:.3f}s"
        )
    
    # The debug panel display filter is managed separately by the settings panel's
    # _apply_log_level() method, which calls debug_panel.set_global_log_level()
    total_duration = time.perf_counter() - start_time
    logger.info(f"configure_log_levels completed in {total_duration:.3f}s")
