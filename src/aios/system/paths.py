"""Centralized path resolution helpers for AI-OS."""

from __future__ import annotations

import os
import platform
import uuid
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

_WINDOWS = os.name == "nt"
_DARWIN = platform.system() == "Darwin"

_ARTIFACTS_OVERRIDE: Path | None = None


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


def set_artifacts_root_override(path: str | Path | None) -> None:
    """Set a runtime override for the artifacts directory and clear cached paths."""

    global _ARTIFACTS_OVERRIDE

    if path is None:
        _ARTIFACTS_OVERRIDE = None
    else:
        candidate = Path(path)
        if str(candidate).strip():
            _ARTIFACTS_OVERRIDE = candidate.expanduser()
        else:
            _ARTIFACTS_OVERRIDE = None

    try:
        get_artifacts_root.cache_clear()
    except Exception:
        pass


def get_state_file_path() -> Path:
    """Return the GUI state file path."""
    override = _env_path("AIOS_STATE_FILE")
    if override:
        override.parent.mkdir(parents=True, exist_ok=True)
        return override
    return get_user_data_root() / "gui_state.json"


def get_artifacts_root() -> Path:
    if _ARTIFACTS_OVERRIDE is not None:
        return _ensure_dir(_ARTIFACTS_OVERRIDE)

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


def test_directory_writable(path: Path) -> str | None:
    """Attempt to create the directory and a temporary file to verify access."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - platform specific
        return f"Failed to create directory: {exc}"

    probe_name = f".aios-write-test-{os.getpid()}-{uuid.uuid4().hex}"
    probe_path = path / probe_name

    try:
        probe_path.write_text("ok", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - platform specific
        try:
            probe_path.unlink(missing_ok=True)
        except Exception:
            pass
        return f"Failed to write probe file: {exc}"

    try:
        probe_path.unlink(missing_ok=True)
    except Exception:
        pass

    return None


def check_core_paths_writable() -> list[tuple[str, Path | None, str]]:
    """Return a list of (label, path, error) for unwritable core directories."""

    checks: list[tuple[str, Callable[[], Path]]] = [
        ("Program data root", get_program_data_root),
        ("Artifacts root", get_artifacts_root),
        ("Brains root", get_brains_root),
        ("User data root", get_user_data_root),
        ("User cache root", get_user_cache_root),
        ("User config directory", get_user_config_dir),
        ("Logs directory", get_logs_dir),
        ("GUI state directory", lambda: get_state_file_path().parent),
        ("Logging state directory", lambda: get_session_state_file().parent),
        ("Logging lock directory", get_session_lock_dir),
    ]

    issues: list[tuple[str, Path | None, str]] = []

    for label, resolver in checks:
        try:
            resolved = resolver()
        except Exception as exc:  # pragma: no cover - platform specific
            issues.append((label, None, f"Failed to resolve path: {exc}"))
            continue

        error = test_directory_writable(resolved)
        if error:
            issues.append((label, resolved, error))

    return issues
