"""Centralized path resolution helpers for AI-OS."""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path

_WINDOWS = os.name == "nt"
_DARWIN = platform.system() == "Darwin"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env_path(var: str) -> Path | None:
    value = os.environ.get(var)
    if not value:
        return None
    return Path(value).expanduser()


@lru_cache(maxsize=1)
def get_program_data_root() -> Path:
    """Return the root directory for shared, machine-wide data."""
    override = _env_path("AIOS_PROGRAM_DATA")
    if override:
        return _ensure_dir(override)

    if _WINDOWS:
        base = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    elif _DARWIN:
        base = Path("/Library/Application Support")
    else:
        base = Path(os.environ.get("AIOS_LINUX_GLOBAL", "/var/lib"))

    return _ensure_dir(base / "AI-OS")


@lru_cache(maxsize=1)
def get_user_data_root() -> Path:
    """Return the per-user data directory used for logs and caches."""
    override = _env_path("AIOS_USER_DATA")
    if override:
        return _ensure_dir(override)

    if _WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif _DARWIN:
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    return _ensure_dir(base / "aios")


@lru_cache(maxsize=1)
def get_user_cache_root() -> Path:
    override = _env_path("AIOS_CACHE_DIR")
    if override:
        return _ensure_dir(override)

    if _WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif _DARWIN:
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return _ensure_dir(base / "aios")


@lru_cache(maxsize=1)
def get_user_config_dir() -> Path:
    override = _env_path("AIOS_CONFIG_DIR")
    if override:
        return _ensure_dir(override)

    if _WINDOWS:
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif _DARWIN:
        base = Path.home() / "Library" / "Preferences"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return _ensure_dir(base / "aios")


def get_logs_dir() -> Path:
    return _ensure_dir(get_user_data_root() / "logs")


def get_state_file_path() -> Path:
    """Return the GUI state file path."""
    override = _env_path("AIOS_STATE_FILE")
    if override:
        override.parent.mkdir(parents=True, exist_ok=True)
        return override
    return get_user_data_root() / "gui_state.json"


def get_artifacts_root() -> Path:
    override = _env_path("AIOS_ARTIFACTS_DIR")
    if override:
        return _ensure_dir(override)
    return _ensure_dir(get_program_data_root() / "artifacts")


def get_session_state_file() -> Path:
    """Return the path used for async logging session tracking."""
    lock_root = _ensure_dir(get_user_cache_root() / "logging")
    return lock_root / ".session_state.json"


def get_session_lock_dir() -> Path:
    return _ensure_dir(get_user_cache_root() / "logging" / "locks")


def get_brains_root() -> Path:
    return _ensure_dir(get_artifacts_root() / "brains")


def get_brain_family_dir(family: str = "actv1") -> Path:
    return _ensure_dir(get_brains_root() / family)


def resolve_artifact_path(relative: str | Path) -> Path:
    rel_path = Path(relative)
    if rel_path.is_absolute():
        return rel_path
    return get_artifacts_root() / rel_path
