"""Windows startup utilities for AI-OS.

This module provides functionality to enable/disable AI-OS starting
automatically when Windows boots. It manages the Windows registry
entry in HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run.

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


def is_windows() -> bool:
    """Check if running on Windows.
    
    Returns:
        True if platform is Windows, False otherwise
    """
    return sys.platform == "win32"


def get_startup_command(minimized: bool = False) -> str:
    """Get the command to use for Windows startup.
    
    This function determines the best command to use for starting AI-OS
    on Windows boot. It tries several approaches in order of preference:
    
    1. Installed executable (if found in Program Files or AppData)
    2. Batch script (if aios.bat is in PATH)
    3. Python module invocation (fallback for development)
    
    Args:
        minimized: If True, add --minimized flag to start minimized to tray
    
    Returns:
        Command string to execute on Windows startup
    
    Example:
        >>> cmd = get_startup_command()
        >>> print(cmd)
        "C:\\Python\\python.exe" -m aios.gui
        
        >>> cmd = get_startup_command(minimized=True)
        >>> print(cmd)
        "C:\\Python\\python.exe" -m aios.gui --minimized
    """
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
    logger.debug(f"Using Python module invocation: {python_exe}")
    return f'"{python_exe}" -m aios.gui{minimized_flag}'


def set_startup_enabled(enabled: bool, minimized: bool = False) -> bool:
    """Enable or disable AI-OS starting on Windows boot.
    
    This function modifies the Windows registry to add or remove
    the AI-OS startup entry. The entry is added to:
    HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
    
    Args:
        enabled: True to enable startup, False to disable
        minimized: If True and enabled, start minimized to tray
    
    Returns:
        True if operation succeeded, False if failed or not on Windows
    
    Example:
        >>> # Enable startup
        >>> success = set_startup_enabled(True)
        >>> if success:
        ...     print("Startup enabled successfully")
        
        >>> # Enable startup with minimize
        >>> success = set_startup_enabled(True, minimized=True)
        
        >>> # Disable startup
        >>> success = set_startup_enabled(False)
        >>> if success:
        ...     print("Startup disabled successfully")
    
    Note:
        - Only works on Windows platform
        - Requires winreg module (standard on Windows Python)
        - Does not require admin privileges (uses HKEY_CURRENT_USER)
        - On non-Windows platforms, returns False silently
    """
    if not is_windows() or not HAS_WINREG:
        logger.debug("Startup configuration not available (non-Windows or winreg missing)")
        return False
    
    try:
        # Open registry key for writing
        key = winreg.OpenKey(  # type: ignore[union-attr]
            winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
            REGISTRY_KEY,
            0,
            winreg.KEY_SET_VALUE  # type: ignore[union-attr]
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
        
        winreg.CloseKey(key)  # type: ignore[union-attr]
        return True
        
    except PermissionError as e:
        # User doesn't have permission to modify registry
        logger.error(f"Permission denied to modify registry: {e}")
        return False
    except Exception as e:
        # Other registry errors
        logger.error(f"Failed to modify startup configuration: {e}", exc_info=True)
        return False


def is_startup_enabled() -> bool:
    """Check if AI-OS is currently configured to start on Windows boot.
    
    This function queries the Windows registry to check if the
    AI-OS startup entry exists.
    
    Returns:
        True if startup is enabled, False otherwise
    
    Example:
        >>> if is_startup_enabled():
        ...     print("AI-OS will start on boot")
        ... else:
        ...     print("AI-OS will not start on boot")
    
    Note:
        - Only works on Windows platform
        - Returns False on non-Windows platforms
        - Returns False if registry access fails
    """
    if not is_windows() or not HAS_WINREG:
        return False
    
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
    except Exception as e:
        # Other registry errors
        logger.error(f"Failed to check startup status: {e}", exc_info=True)
        return False


def get_startup_path() -> Optional[str]:
    """Get the current startup command if enabled.
    
    This function retrieves the actual command string stored in the
    Windows registry for AI-OS startup.
    
    Returns:
        Command string if startup is enabled, None otherwise
    
    Example:
        >>> path = get_startup_path()
        >>> if path:
        ...     print(f"Startup command: {path}")
        ... else:
        ...     print("Startup not enabled")
    """
    if not is_windows() or not HAS_WINREG:
        return None
    
    try:
        # Open registry key for reading
        key = winreg.OpenKey(  # type: ignore[union-attr]
            winreg.HKEY_CURRENT_USER,  # type: ignore[union-attr]
            REGISTRY_KEY,
            0,
            winreg.KEY_READ  # type: ignore[union-attr]
        )
        
        # Try to read the value
        value, reg_type = winreg.QueryValueEx(key, APP_NAME)  # type: ignore[union-attr]
        winreg.CloseKey(key)  # type: ignore[union-attr]
        
        path = str(value) if value else None
        logger.debug(f"Startup path: {path}")
        return path
        
    except FileNotFoundError:
        # Key or value doesn't exist
        logger.debug("Startup path not found in registry")
        return None
    except Exception as e:
        # Other registry errors
        logger.error(f"Failed to get startup path: {e}", exc_info=True)
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
  Get the command to use for startup (auto-detects best option)

- set_startup_enabled(enabled: bool) -> bool
  Enable/disable startup on Windows boot via registry

- is_startup_enabled() -> bool
  Check if startup is currently enabled

- get_startup_path() -> Optional[str]
  Get current startup command from registry

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
- Linux/macOS: Functions return False/None (no-op)
- No exceptions raised on non-Windows platforms

Security:
---------
- Uses HKEY_CURRENT_USER (no admin privileges required)
- User-level startup only (not system-wide)
- No elevation prompts
"""
