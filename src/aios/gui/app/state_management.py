"""State management module for AI-OS GUI.

This module handles:
- Application state persistence (window geometry, panel settings)
- State loading and saving
- State file management
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import json
from pathlib import Path

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def initialize_state(app: Any, project_root: Path) -> None:
    """
    Initialize state management system.
    
    Args:
        app: AiosTkApp instance
        project_root: Path to project root directory
    """
    app._state_file = project_root / "gui_state.json"
    app._state = {}


def load_state(app: Any) -> dict:
    """
    Load application state from JSON file.
    
    Returns:
        dict: Loaded state or empty dict if file doesn't exist
    """
    if not app._state_file.exists():
        logger.info("No state file found, using defaults")
        return {}
    
    try:
        with open(app._state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        logger.info(f"Loaded state from {app._state_file}")
        return state
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return {}


def save_state(app: Any) -> None:
    """
    Save application state to JSON file.
    
    Collects state from:
    - Window geometry
    - Resources panel (device selections, caps)
    - Settings panel (theme, model settings)
    - MCP panel (server configurations)
    - HRM Training panel (training parameters)
    """
    state = {}
    
    # Window geometry
    try:
        state['window_geometry'] = app.root.geometry()
    except Exception as e:
        logger.warning(f"Failed to save window geometry: {e}")
    
    # Resources panel
    try:
        if hasattr(app, 'resources_panel') and app.resources_panel:
            if hasattr(app.resources_panel, 'get_state'):
                state['resources'] = app.resources_panel.get_state()
    except Exception as e:
        logger.debug(f"Failed to save resources panel state: {e}")
    
    # Settings panel
    try:
        if hasattr(app, 'settings_panel') and app.settings_panel:
            state['settings'] = app.settings_panel.get_state()
    except Exception as e:
        logger.warning(f"Failed to save settings panel state: {e}")
    
    # MCP panel
    try:
        if hasattr(app, 'mcp_panel') and app.mcp_panel:
            if hasattr(app.mcp_panel, 'get_state'):
                state['mcp'] = app.mcp_panel.get_state()
    except Exception as e:
        logger.debug(f"Failed to save MCP panel state: {e}")
    
    # HRM Training panel
    try:
        if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
            state['hrm_training'] = app.hrm_training_panel.get_state()
    except Exception as e:
        logger.warning(f"Failed to save HRM training panel state: {e}")
    
    # Debug panel
    try:
        if hasattr(app, 'debug_panel') and app.debug_panel:
            if hasattr(app.debug_panel, 'get_state'):
                state['debug'] = app.debug_panel.get_state()
    except Exception as e:
        logger.debug(f"Failed to save debug panel state: {e}")
    
    # Save to file
    try:
        with open(app._state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved state to {app._state_file}")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def restore_state(app: Any, state: dict) -> None:
    """
    Restore application state from loaded data.
    
    Args:
        app: AiosTkApp instance
        state: State dictionary loaded from file
    """
    # Window geometry (restored but window remains withdrawn until finalization)
    if 'window_geometry' in state:
        try:
            app.root.geometry(state['window_geometry'])
            logger.info(f"Restored window geometry: {state['window_geometry']}")
        except Exception as e:
            logger.warning(f"Failed to restore window geometry: {e}")
    
    # Resources panel
    if 'resources' in state:
        try:
            if hasattr(app, 'resources_panel') and app.resources_panel:
                app.resources_panel.set_state(state['resources'])
        except Exception as e:
            logger.warning(f"Failed to restore resources panel state: {e}")
    
    # Settings panel
    if 'settings' in state:
        try:
            if hasattr(app, 'settings_panel') and app.settings_panel:
                app.settings_panel.set_state(state['settings'])
        except Exception as e:
            logger.warning(f"Failed to restore settings panel state: {e}")
    
    # MCP panel
    if 'mcp' in state:
        try:
            if hasattr(app, 'mcp_panel') and app.mcp_panel:
                app.mcp_panel.set_state(state['mcp'])
        except Exception as e:
            logger.warning(f"Failed to restore MCP panel state: {e}")
    
    # HRM Training panel
    if 'hrm_training' in state:
        try:
            if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
                app.hrm_training_panel.set_state(state['hrm_training'])
        except Exception as e:
            logger.warning(f"Failed to restore HRM training panel state: {e}")
    
    # Debug panel
    if 'debug' in state:
        try:
            if hasattr(app, 'debug_panel') and app.debug_panel:
                if hasattr(app.debug_panel, 'set_state'):
                    app.debug_panel.set_state(state['debug'])
        except Exception as e:
            logger.debug(f"Failed to restore debug panel state: {e}")
