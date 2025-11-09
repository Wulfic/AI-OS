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
    - Log file in project root
    - Error callback for UI display
    
    Args:
        app: AiosTkApp instance
        project_root: Path to project root directory
    """
    
    # Create log router
    app._log_router = LogRouter()
    
    # Set up log file
    log_file = project_root / "aios_gui.log"
    try:
        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        ))
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    except Exception as e:
        logger.warning(f"Failed to create log file handler: {e}")
    
    # Error callback (will be connected to debug panel later)
    app._error_callback = None
    
    def _set_error(msg: str) -> None:
        """Set error message in debug panel (if available)."""
        if hasattr(app, 'debug_panel') and app.debug_panel:
            try:
                app.debug_panel.set_error(msg)
            except Exception:
                pass
        
        # Also print to stderr
        print(f"ERROR: {msg}", file=sys.stderr)
    
    app._set_error = _set_error
    
    # Output append callback (will be connected to debug panel later)
    def _append_out(msg: str) -> None:
        """Append output message to debug panel (if available)."""
        if hasattr(app, 'debug_panel') and app.debug_panel:
            try:
                app.debug_panel.append(msg)
            except Exception:
                pass
        
        # Also log to logger
        logger.info(msg)
    
    app._append_out = _append_out


def configure_log_levels(app: Any, debug_enabled: bool = False) -> None:
    """
    Configure logging levels based on debug mode.
    
    Args:
        app: AiosTkApp instance
        debug_enabled: Whether debug mode is enabled
    """
    if debug_enabled:
        logging.getLogger("aios").setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    else:
        logging.getLogger("aios").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Info logging enabled")
