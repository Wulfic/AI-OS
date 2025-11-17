"""Startup utilities for AI-OS (Windows and Linux).

This module provides functionality to enable/disable AI-OS starting
automatically when the desktop session begins. On Windows it manages the
registry entry in HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run,
while on Linux it writes a freedesktop.org autostart entry under
~/.config/autostart.

Example:
    >>> from aios.gui.utils.startup import set_startup_enabled, is_startup_enabled
    >>> 
    >>> # Enable startup
    >>> success = set_startup_enabled(True)
    >>> print(f"Startup enabled: {success}")
    >>> 
    >>> # Check if startup is enabled
    >>> enabled = is_startup_enabled()
    >>> print(f"Current status: {enabled}")
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Windows-specific registry module
try:
    import winreg  # type: ignore
    HAS_WINREG = True
except ImportError:
    winreg = None  # type: ignore
    HAS_WINREG = False
    logger.debug("winreg module not available (non-Windows platform)")


# Constants
APP_NAME = "AI-OS"
REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
LINUX_AUTOSTART_DIR = Path.home() / ".config" / "autostart"
LINUX_AUTOSTART_FILENAME = "ai-os.desktop"


def is_windows() -> bool:
    """Check if running on Windows.
    
    Returns:
        True if platform is Windows, False otherwise
    """
    return sys.platform == "win32"


def is_linux() -> bool:
    """Check if running on a Linux platform."""
    return sys.platform.startswith("linux")


def _linux_autostart_file() -> Path:
    """Return the path to the Linux autostart .desktop file."""
    return LINUX_AUTOSTART_DIR / LINUX_AUTOSTART_FILENAME


def _detect_project_root() -> Optional[Path]:
    """Best effort detection of the project root containing pyproject.toml."""
    try:
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent
    except Exception:
        logger.debug("Unable to detect project root for autostart", exc_info=True)
    return None


def _resolve_icon_path() -> Optional[Path]:
    """Locate the application icon if packaged with the repository."""
    try:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "installers" / "AI-OS.png"
            if candidate.exists():
                return candidate
    except Exception:
        logger.debug("Unable to resolve icon path for autostart", exc_info=True)
    return None


def _render_linux_desktop_entry(command: str, working_dir: Optional[Path]) -> str:
    """Create the contents of the autostart .desktop file."""
    icon_path = _resolve_icon_path()
    lines: list[str] = [
        "[Desktop Entry]",
        "Type=Application",
        "Version=1.0",
        "Name=AI-OS",
        "Comment=Launch AI-OS Control Panel on login",
    ]

    if icon_path:
        lines.append(f"Icon={icon_path}")

    lines.extend(
        [
            f"Exec={command}",
            "Terminal=false",
            "Categories=Development;Utility;",
            "X-GNOME-Autostart-enabled=true",
            "StartupNotify=false",
            "Hidden=false",
        ]
    )

    if working_dir:
        lines.append(f"Path={working_dir}")

    return "\n".join(lines) + "\n"


def get_startup_command(minimized: bool = False) -> str:
    """Return the platform-specific command used for autostart."""
    if is_windows():
        return _get_windows_startup_command(minimized)

    minimized_flag = " --minimized" if minimized else ""
    python_exe = sys.executable
    logger.debug("Using Python module invocation for autostart: %s", python_exe)
    return f'"{python_exe}" -m aios.cli.aios gui{minimized_flag}'


def _get_windows_startup_command(minimized: bool) -> str:
    """Determine the best command to use for Windows startup."""
    minimized_flag = " --minimized" if minimized else ""
    
    # Option 1: Check for installed executable
    # Use environment variables for cross-machine compatibility
    program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
    program_files_x86 = os.environ.get("PROGRAMW6432", program_files)
    local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    
    possible_exe_paths = [
        Path(program_files) / "AI-OS" / "aios.exe",
        Path(program_files_x86) / "AI-OS" / "aios.exe",
        Path(local_appdata) / "Programs" / "AI-OS" / "aios.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "AI-OS" / "aios.exe",
    ]
    
    for exe_path in possible_exe_paths:
        try:
            if exe_path.exists() and exe_path.is_file():
                logger.debug(f"Found installed executable: {exe_path}")
                return f'"{exe_path}" gui{minimized_flag}'
        except Exception as e:
            logger.debug(f"Failed to check exe path {exe_path}: {e}")
            continue
    
    # Option 2: Check for batch script in current directory or parent
    try:
        # Check if aios.bat exists in project root
        current_dir = Path(__file__).parent.parent.parent.parent.parent
        batch_path = current_dir / "aios.bat"
        if batch_path.exists() and batch_path.is_file():
            logger.debug(f"Found batch script: {batch_path}")
            return f'"{batch_path}" gui{minimized_flag}'
    except Exception as e:
        logger.debug(f"Failed to check batch script: {e}")
    
    # Option 3: Fall back to Python module invocation
    python_exe = sys.executable
    logger.debug("Using Python module invocation for Windows autostart: %s", python_exe)
    return f'"{python_exe}" -m aios.gui{minimized_flag}'


def set_startup_enabled(enabled: bool, minimized: bool = False) -> bool:
    """Enable or disable AI-OS autostart on supported platforms."""
    if is_windows() and HAS_WINREG:
        return _set_windows_startup_enabled(enabled, minimized)
    if is_linux():
        return _set_linux_startup_enabled(enabled, minimized)

    logger.debug("Startup configuration not available on this platform")
    return False


def _set_windows_startup_enabled(enabled: bool, minimized: bool) -> bool:
    """Windows-specific implementation for enabling autostart via registry."""
    key = None
    access_flags = (  # type: ignore[attr-defined]
        winreg.KEY_SET_VALUE  # type: ignore[union-attr]
        | getattr(winreg, "KEY_WRITE", 0)  # Some Python builds omit KEY_WRITE
    )
    try:
        try:
            # Prefer opening existing key so we do not create it when disabling
            key = winreg.OpenKey(  # type: ignore[union-attr]
                winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
                REGISTRY_KEY,
                0,
                access_flags,
            )
        except FileNotFoundError:
            if not enabled:
                logger.debug("Startup key missing while disabling; nothing to remove")
                return True
            # Ensure the Run key exists before writing the value
            key = winreg.CreateKeyEx(  # type: ignore[union-attr]
                winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
                REGISTRY_KEY,
                0,
                access_flags,
            )
        
        if enabled:
            # Add startup entry with optional minimize flag
            command = get_startup_command(minimized=minimized)
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)  # type: ignore[union-attr]
            logger.info(f"Startup enabled: {command}")
        else:
            # Remove startup entry
            try:
                winreg.DeleteValue(key, APP_NAME)  # type: ignore[union-attr]
                logger.info("Startup disabled")
            except FileNotFoundError:
                # Already not set - this is fine
                logger.debug("Startup entry already absent")
        
        return True
        
    except PermissionError as e:
        # User doesn't have permission to modify registry
        logger.error(
            "Permission denied to modify startup registry entry. "
            "Try running AI-OS once with elevated privileges or disable corporate policy blocks.",
            exc_info=True,
        )
        return False
    except Exception as e:
        # Other registry errors
        logger.error(f"Failed to modify startup configuration: {e}", exc_info=True)
        return False
    finally:
        if key is not None:
            winreg.CloseKey(key)  # type: ignore[union-attr]


def _set_linux_startup_enabled(enabled: bool, minimized: bool) -> bool:
    """Linux-specific implementation writing a .desktop autostart entry."""
    desktop_file = _linux_autostart_file()
    try:
        if enabled:
            command = get_startup_command(minimized=minimized)
            working_dir = _detect_project_root()
            content = _render_linux_desktop_entry(command, working_dir)
            LINUX_AUTOSTART_DIR.mkdir(parents=True, exist_ok=True)
            desktop_file.write_text(content, encoding="utf-8")
            desktop_file.chmod(0o755)
            logger.info("Startup enabled via %s", desktop_file)
        else:
            if desktop_file.exists():
                desktop_file.unlink()
                logger.info("Startup disabled (removed %s)", desktop_file)
            else:
                logger.debug("No autostart desktop file found to remove")
        return True
    except PermissionError as exc:
        logger.error("Insufficient permissions to manage autostart file: %s", exc, exc_info=True)
        return False
    except Exception as exc:
        logger.error("Failed to configure Linux autostart: %s", exc, exc_info=True)
        return False


def is_startup_enabled() -> bool:
    """Return True if autostart is currently enabled on this system."""
    if is_windows() and HAS_WINREG:
        return _is_windows_startup_enabled()
    if is_linux():
        return _is_linux_startup_enabled()
    return False


def _is_windows_startup_enabled() -> bool:
    """Check startup status via the Windows registry."""
    try:
        # Open registry key for reading
        key = winreg.OpenKey(  # type: ignore[union-attr]
            winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
            REGISTRY_KEY,
            0,
            winreg.KEY_READ  # type: ignore[union-attr]
        )
        
        # Read the value
        value, reg_type = winreg.QueryValueEx(key, APP_NAME)  # type: ignore[union-attr]
        winreg.CloseKey(key)  # type: ignore[union-attr]
        
        # Check if value is valid (non-empty string)
        is_enabled = value is not None and len(str(value)) > 0
        logger.debug(f"Startup {'enabled' if is_enabled else 'disabled'}")
        return is_enabled
        
    except FileNotFoundError:
        # Key or value doesn't exist
        logger.debug("Startup entry not found in registry")
        return False


def _is_linux_startup_enabled() -> bool:
    """Check whether the Linux autostart .desktop file exists and is active."""
    desktop_file = _linux_autostart_file()
    if not desktop_file.exists():
        return False
    try:
        for line in desktop_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("Hidden=") and line.split("=", 1)[1].strip().lower() == "true":
                return False
            if line.startswith("Exec="):
                return bool(line.split("=", 1)[1].strip())
    except Exception as exc:
        logger.error("Failed to read autostart desktop file: %s", exc, exc_info=True)
        return False
    return False


def get_startup_path() -> Optional[str]:
    """Return the stored autostart command, if any."""
    if is_windows() and HAS_WINREG:
        return _get_windows_startup_path()
    if is_linux():
        return _get_linux_startup_path()
    return None


def _get_windows_startup_path() -> Optional[str]:
    try:
        key = winreg.OpenKey(  # type: ignore[union-attr]
            winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
            REGISTRY_KEY,
            0,
            winreg.KEY_READ  # type: ignore[union-attr]
        )
        value, reg_type = winreg.QueryValueEx(key, APP_NAME)  # type: ignore[union-attr]
        winreg.CloseKey(key)  # type: ignore[union-attr]
        path = str(value) if value else None
        logger.debug("Startup path: %s", path)
        return path
    except FileNotFoundError:
        logger.debug("Startup path not found in registry")
        return None
    except Exception as exc:
        logger.error("Failed to get startup path: %s", exc, exc_info=True)
        return None


def _get_linux_startup_path() -> Optional[str]:
    desktop_file = _linux_autostart_file()
    if not desktop_file.exists():
        return None
    try:
        for line in desktop_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("Exec="):
                return line.split("=", 1)[1].strip() or None
    except Exception as exc:
        logger.error("Failed to read Linux autostart command: %s", exc, exc_info=True)
        return None
    return None


def verify_startup_command() -> tuple[bool, str]:
    """Verify that the startup command is valid and executable.
    
    This function checks if the current startup command points to
    a valid file that can be executed.
    
    Returns:
        Tuple of (is_valid, message) where:
        - is_valid: True if command is valid
        - message: Description of status
    
    Example:
        >>> is_valid, message = verify_startup_command()
        >>> print(f"Valid: {is_valid}, Message: {message}")
    """
    if not is_startup_enabled():
        return False, "Startup not enabled"
    
    path = get_startup_path()
    if not path:
        return False, "No startup path found"
    
    # Try to extract executable path from command
    # Handle quoted paths like: "C:\path\to\python.exe" -m aios.gui
    import shlex
    try:
        parts = shlex.split(path)
        if not parts:
            logger.warning("Empty startup command")
            return False, "Empty command"
        
        exe_path = Path(parts[0])
        if exe_path.exists() and exe_path.is_file():
            logger.debug(f"Startup command valid: {exe_path.name}")
            return True, f"Valid: {exe_path.name}"
        else:
            logger.warning(f"Startup executable not found: {exe_path}")
            return False, f"File not found: {exe_path}"
    except Exception as e:
        logger.error(f"Failed to verify startup command: {e}", exc_info=True)
        return False, f"Invalid command: {str(e)}"


# Summary for module
r"""
Module Functions:
-----------------
- is_windows() -> bool
  Check if running on Windows platform

- get_startup_command() -> str
    Get the command to use for startup (auto-detects best option per platform)

- set_startup_enabled(enabled: bool) -> bool
    Enable/disable startup on login (registry on Windows, autostart entry on Linux)

- is_startup_enabled() -> bool
  Check if startup is currently enabled

- get_startup_path() -> Optional[str]
    Get current startup command from registry or .desktop entry

- verify_startup_command() -> tuple[bool, str]
  Verify startup command is valid and executable

Registry Details:
-----------------
Key: HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run
Value Name: AI-OS
Value Type: REG_SZ (string)
Value Data: Full command to execute (e.g., "C:\Python\python.exe" -m aios.gui)

Cross-Platform:
---------------
- Windows: Full functionality using winreg
- Linux: Writes ~/.config/autostart/ai-os.desktop
- Other platforms: Functions return False/None (no-op)

Security:
---------
- Uses HKEY_CURRENT_USER (no admin privileges required)
- User-level startup only (not system-wide)
- No elevation prompts
"""
