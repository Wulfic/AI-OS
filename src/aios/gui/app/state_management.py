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
import shutil
import threading
from pathlib import Path

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover - fallback for early boot
    system_paths = None


def _legacy_state_candidates(project_root: Path) -> tuple[Path, ...]:
    """Return possible locations where legacy GUI state files may live."""
    return (
        project_root / "gui_state.json",
        project_root / "src" / "gui_state.json",
    )


def _find_legacy_state_file(project_root: Path, target_path: Path) -> Path | None:
    for candidate in _legacy_state_candidates(project_root):
        try:
            if candidate.resolve() == target_path.resolve():
                continue
        except Exception:
            if candidate == target_path:
                continue
        if candidate.exists():
            return candidate
    return None


def _migrate_legacy_state_file(project_root: Path, target_path: Path) -> None:
    """Copy a legacy gui_state.json to the new user data location if needed."""
    legacy_file = _find_legacy_state_file(project_root, target_path)
    if legacy_file is None or target_path.exists():
        return

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_file, target_path)
        logger.info("Migrated legacy GUI state from %s to %s", legacy_file, target_path)

        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        try:
            if not backup_path.exists():
                shutil.copy2(legacy_file, backup_path)
                logger.info("Backed up legacy GUI state to %s", backup_path)
        except Exception:
            logger.warning("Failed to create GUI state backup at %s", backup_path, exc_info=True)
    except Exception:
        logger.error("Failed to migrate legacy GUI state from %s", legacy_file, exc_info=True)


def _project_state_fallback(project_root: Path) -> Path:
    """Return the fallback state path used when system paths are unavailable."""
    return project_root / "gui_state.json"


def _normalize_artifacts_path(raw_value: str) -> str:
    if system_paths is None or not isinstance(raw_value, str):
        return raw_value

    normalized = raw_value.replace("\\", "/")
    marker = "artifacts/"
    idx = normalized.lower().find(marker)
    if idx == -1:
        return raw_value

    rel = normalized[idx + len(marker):]
    resolved = system_paths.get_artifacts_root() / Path(rel)
    return str(resolved)


def _normalize_state_paths(state: dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False

    mutated = False
    hrm_state = state.get("hrm_training")
    if isinstance(hrm_state, dict):
        for key in ("model", "log_file", "student_init", "bundle_dir"):
            value = hrm_state.get(key)
            if not isinstance(value, str):
                continue
            updated = _normalize_artifacts_path(value)
            if updated != value:
                hrm_state[key] = updated
                mutated = True

    return mutated


def _persist_normalized_state(state_file: Path, state: dict[str, Any]) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info("Persisted normalized GUI state to %s", state_file)
    except Exception as err:
        logger.warning("Failed to persist normalized GUI state to %s: %s", state_file, err)


def initialize_state(app: Any, project_root: Path) -> None:
    """
    Initialize state management system.
    
    Args:
        app: AiosTkApp instance
        project_root: Path to project root directory
    """
    logger.info("Initializing state management system")
    state_file = None
    if system_paths is not None:
        try:
            state_file = system_paths.get_state_file_path()
        except Exception:
            logger.debug("Failed to resolve user state path, falling back to project root", exc_info=True)

    if state_file is None:
        state_file = _project_state_fallback(project_root)

    _migrate_legacy_state_file(project_root, state_file)

    state_file.parent.mkdir(parents=True, exist_ok=True)
    app._state_file = state_file
    logger.info(f"State file path: {app._state_file}")
    app._state = {}
    app._state_restored = False  # Flag to prevent saving before restoration
    app._state_save_seq = 0
    app._state_last_persisted_seq = 0
    app._state_write_lock = threading.Lock()


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
        logger.debug(f"State sections loaded: {len(state)}")

        if _normalize_state_paths(state):
            logger.info("Normalizing GUI state paths to new artifacts directory")
            _persist_normalized_state(app._state_file, state)

        logger.info("State validation passed")
        logger.info(f"Loaded state from {app._state_file}")
        return state
    except json.JSONDecodeError as e:
        error_context = f"State file corrupted at {app._state_file}"
        suggestion = "Delete gui_state.json to reset to defaults, or restore from backup"
        logger.error(f"{error_context}: Invalid JSON at line {e.lineno}, column {e.colno}. Suggestion: {suggestion}", exc_info=True)
        return {}
    except PermissionError as e:
        error_context = f"Permission denied reading state file {app._state_file}"
        suggestion = "Check file permissions or run as administrator"
        logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
        return {}
    except Exception as e:
        error_context = f"Failed to load state from {app._state_file}"
        
        # Provide contextual suggestions
        if "no such file" in str(e).lower():
            suggestion = "State file was deleted. A new one will be created"
        elif "permission" in str(e).lower() or "access" in str(e).lower():
            suggestion = "Check file permissions or run as administrator"
        else:
            suggestion = "State file may be corrupted. Consider deleting gui_state.json to reset"
        
        logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
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
    # Don't save state until after initial restoration
    if not getattr(app, '_state_restored', False):
        logger.debug("Skipping save_state - state not yet restored")
        return
    
    panel_count = 0
    if hasattr(app, 'resources_panel') and app.resources_panel:
        panel_count += 1
    if hasattr(app, 'settings_panel') and app.settings_panel:
        panel_count += 1
    if hasattr(app, 'mcp_panel') and app.mcp_panel:
        panel_count += 1
    if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
        panel_count += 1
    
    logger.debug(f"Collecting state from {panel_count} panels")
    logger.debug("Saving application state")
    state = {}
    
    # Window geometry
    try:
        state['window_geometry'] = app.root.geometry()
        logger.debug(f"Saved window geometry: {state['window_geometry']}")
    except Exception as e:
        logger.warning(f"Failed to save window geometry: {e}")
    
    # Resources panel
    try:
        if hasattr(app, 'resources_panel') and app.resources_panel:
            if hasattr(app.resources_panel, 'get_state'):
                state['resources'] = app.resources_panel.get_state()
                state_size = len(str(state['resources']))
                logger.info(f"Saved resources panel state ({state_size} bytes)")
            else:
                logger.warning("Resources panel exists but has no get_state method")
        else:
            logger.debug("Resources panel not available for state saving")
    except Exception as e:
        logger.error(f"Failed to save resources panel state: {e}", exc_info=True)
    
    # Settings panel
    try:
        if hasattr(app, 'settings_panel') and app.settings_panel:
            state['settings'] = app.settings_panel.get_state()
            state_size = len(str(state['settings']))
            logger.info(f"Saved settings panel state ({state_size} bytes)")
        else:
            logger.warning("Settings panel not available for state saving")
    except Exception as e:
        logger.error(f"Failed to save settings panel state: {e}", exc_info=True)
    
    # MCP panel
    try:
        if hasattr(app, 'mcp_panel') and app.mcp_panel:
            if hasattr(app.mcp_panel, 'get_state'):
                state['mcp'] = app.mcp_panel.get_state()
                state_size = len(str(state['mcp']))
                logger.info(f"Saved MCP panel state ({state_size} bytes)")
            else:
                logger.debug("MCP panel exists but has no get_state method")
        else:
            logger.debug("MCP panel not available for state saving")
    except Exception as e:
        logger.error(f"Failed to save MCP panel state: {e}", exc_info=True)
    
    # HRM Training panel
    try:
        if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
            logger.debug("Attempting to get HRM training panel state...")
            state['hrm_training'] = app.hrm_training_panel.get_state()
            state_size = len(str(state['hrm_training']))
            logger.info(f"Saved HRM training panel state ({state_size} bytes)")
        else:
            if not hasattr(app, 'hrm_training_panel'):
                logger.warning("app.hrm_training_panel attribute does not exist")
            elif not app.hrm_training_panel:
                logger.warning("app.hrm_training_panel is None")
    except Exception as e:
        logger.error(f"Failed to save HRM training panel state: {e}", exc_info=True)
    
    # Chat panel
    try:
        if hasattr(app, 'chat_panel') and app.chat_panel:
            if hasattr(app.chat_panel, 'get_state'):
                state['chat'] = app.chat_panel.get_state()
                state_size = len(str(state['chat']))
                logger.info(f"Saved chat panel state ({state_size} bytes)")
            else:
                logger.warning("Chat panel exists but has no get_state method")
        else:
            logger.debug("Chat panel not available for state saving")
    except Exception as e:
        logger.error(f"Failed to save chat panel state: {e}", exc_info=True)
    
    save_seq = getattr(app, "_state_save_seq", 0) + 1
    app._state_save_seq = save_seq
    section_count = len(state)

    def _write_state_snapshot(state_snapshot: dict[str, Any], seq_id: int, sections: int) -> None:
        try:
            lock = getattr(app, "_state_write_lock", None)
        except Exception:
            lock = None

        def _perform_write() -> None:
            try:
                with open(app._state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_snapshot, f, indent=2)
                app._state_last_persisted_seq = seq_id
                logger.info(f"Saved state to {app._state_file} ({sections} sections)")
            except PermissionError as write_err:
                error_context = f"Permission denied writing state file {app._state_file}"
                suggestion = "Check file permissions or run as administrator. State will not be saved"
                logger.error(f"{error_context}: {write_err}. Suggestion: {suggestion}", exc_info=True)
            except OSError as write_err:
                error_context = f"Failed to write state file {app._state_file}"
                if "disk" in str(write_err).lower() or "space" in str(write_err).lower():
                    suggestion = "Insufficient disk space. Free up space to save application state"
                elif "read-only" in str(write_err).lower():
                    suggestion = "File system is read-only. Check disk permissions"
                else:
                    suggestion = "Check disk space and file permissions"
                logger.error(f"{error_context}: {write_err}. Suggestion: {suggestion}", exc_info=True)
            except Exception as write_err:
                error_context = f"Unexpected error saving state to {app._state_file}"
                suggestion = "Check file system integrity and permissions. State may not be preserved"
                logger.error(f"{error_context}: {write_err}. Suggestion: {suggestion}", exc_info=True)

        if lock is None:
            _perform_write()
            return

        with lock:
            last_seq = getattr(app, "_state_last_persisted_seq", 0)
            if last_seq >= seq_id:
                logger.debug("Skipping state write seq=%s (already persisted seq=%s)", seq_id, last_seq)
                return
            _perform_write()

    worker_pool = getattr(app, "_worker_pool", None)
    if worker_pool is not None:
        try:
            worker_pool.submit(_write_state_snapshot, state, save_seq, section_count)
            return
        except Exception as submit_err:
            logger.debug("Worker pool submit failed, writing state synchronously: %s", submit_err)

    _write_state_snapshot(state, save_seq, section_count)


def restore_state(app: Any, state: dict) -> None:
    """
    Restore application state from loaded data.
    
    Args:
        app: AiosTkApp instance
        state: State dictionary loaded from file
    """
    component_count = len(state)
    logger.info(f"Restoring state for {component_count} components")
    logger.info(f"State keys: {list(state.keys())}")
    
    # Window geometry (restored but window remains withdrawn until finalization)
    if 'window_geometry' in state:
        try:
            app.root.geometry(state['window_geometry'])
            logger.info(f"Restored window geometry: {state['window_geometry']}")
            logger.debug("Window geometry restoration: success")
        except Exception as e:
            logger.warning(f"Failed to restore window geometry: {e}")
            logger.debug("Window geometry restoration: failed")
    
    # Resources panel
    if 'resources' in state:
        try:
            if hasattr(app, 'resources_panel') and app.resources_panel:
                app.resources_panel.set_state(state['resources'])
                logger.debug("Resources panel restoration: success")
        except Exception as e:
            logger.warning(f"Failed to restore resources panel state: {e}")
            logger.debug("Resources panel restoration: failed")
    
    # Settings panel
    if 'settings' in state:
        try:
            if hasattr(app, 'settings_panel') and app.settings_panel:
                app.settings_panel.set_state(state['settings'])
                logger.debug("Settings panel restoration: success")
        except Exception as e:
            logger.warning(f"Failed to restore settings panel state: {e}")
            logger.debug("Settings panel restoration: failed")
    
    # MCP panel
    if 'mcp' in state:
        try:
            if hasattr(app, 'mcp_panel') and app.mcp_panel:
                app.mcp_panel.set_state(state['mcp'])
                logger.debug("MCP panel restoration: success")
        except Exception as e:
            logger.warning(f"Failed to restore MCP panel state: {e}")
            logger.debug("MCP panel restoration: failed")
    
    # HRM Training panel
    if 'hrm_training' in state:
        try:
            if hasattr(app, 'hrm_training_panel') and app.hrm_training_panel:
                hrm_state = state['hrm_training']
                logger.info(f"Restoring HRM training panel state with {len(hrm_state)} parameters")
                app.hrm_training_panel.set_state(hrm_state)
                logger.info("HRM training panel restoration: success")
        except Exception as e:
            logger.error(f"Failed to restore HRM training panel state: {e}", exc_info=True)
    else:
        logger.info("No HRM training state found in gui_state.json")
    
    # Chat panel
    if 'chat' in state:
        try:
            if hasattr(app, 'chat_panel') and app.chat_panel:
                chat_state = state['chat']
                logger.info(f"Restoring chat panel state with {len(chat_state)} parameters")
                app.chat_panel.set_state(chat_state)
                logger.info("Chat panel restoration: success")
        except Exception as e:
            logger.error(f"Failed to restore chat panel state: {e}", exc_info=True)
    else:
        logger.debug("No chat panel state found in gui_state.json")
    
    # Mark state as restored - now safe to save on events
    app._state_restored = True
    logger.info("State restoration complete - saving now enabled")
