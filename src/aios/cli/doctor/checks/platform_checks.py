"""Platform and elevation checks for AI-OS Doctor."""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional

from ..runner import DiagnosticResult, DiagnosticSeverity


def check_platform(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check platform information and Python version."""
    results = []
    
    # Python version check
    py_version = sys.version_info
    py_version_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    
    if py_version < (3, 10):
        results.append(DiagnosticResult(
            name="Python Version",
            severity=DiagnosticSeverity.ERROR,
            message=f"Python {py_version_str} is not supported",
            details={"version": py_version_str, "minimum_required": "3.10"},
            suggestion="Upgrade to Python 3.10 or newer",
        ))
    elif py_version < (3, 11):
        results.append(DiagnosticResult(
            name="Python Version",
            severity=DiagnosticSeverity.WARNING,
            message=f"Python {py_version_str} is supported but 3.11+ recommended",
            details={"version": py_version_str, "recommended": "3.11+"},
        ))
    else:
        results.append(DiagnosticResult(
            name="Python Version",
            severity=DiagnosticSeverity.OK,
            message=f"Python {py_version_str}",
            details={"version": py_version_str},
        ))
    
    # Platform info
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    
    results.append(DiagnosticResult(
        name="Operating System",
        severity=DiagnosticSeverity.INFO,
        message=f"{system} {release} ({machine})",
        details={
            "system": system,
            "release": release,
            "machine": machine,
            "platform": platform.platform(),
        },
    ))
    
    # AI-OS version
    try:
        from aios import __version__
        results.append(DiagnosticResult(
            name="AI-OS Version",
            severity=DiagnosticSeverity.INFO,
            message=f"v{__version__}",
            details={"version": __version__},
        ))
    except ImportError:
        results.append(DiagnosticResult(
            name="AI-OS Version",
            severity=DiagnosticSeverity.WARNING,
            message="Could not determine AI-OS version",
            suggestion="Ensure AI-OS is properly installed",
        ))
    
    return results


def check_elevation(*, auto_repair: bool = False) -> DiagnosticResult:
    """Check if running with elevated privileges."""
    system = platform.system()
    
    if system == "Windows":
        return _check_windows_admin()
    elif system == "Linux":
        return _check_linux_root()
    elif system == "Darwin":
        return _check_macos_root()
    else:
        return DiagnosticResult(
            name="Elevation Check",
            severity=DiagnosticSeverity.INFO,
            message=f"Elevation check not implemented for {system}",
        )


def _check_windows_admin() -> DiagnosticResult:
    """Check Windows Administrator status."""
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        is_admin = False
    
    if is_admin:
        return DiagnosticResult(
            name="Windows Administrator",
            severity=DiagnosticSeverity.OK,
            message="Running as Administrator",
            details={"elevated": True},
        )
    else:
        return DiagnosticResult(
            name="Windows Administrator",
            severity=DiagnosticSeverity.INFO,
            message="Running as standard user",
            details={"elevated": False},
            suggestion="Admin rights needed for ProgramData access and GPU scheduling. Run as Administrator for full functionality.",
        )


def _check_linux_root() -> DiagnosticResult:
    """Check Linux root/sudo status."""
    is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
    
    # Check if user has sudo capability
    has_sudo = False
    try:
        import subprocess
        result = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=2,
        )
        has_sudo = result.returncode == 0
    except Exception:
        pass
    
    if is_root:
        return DiagnosticResult(
            name="Linux Privileges",
            severity=DiagnosticSeverity.INFO,
            message="Running as root",
            details={"root": True, "sudo_available": True},
        )
    elif has_sudo:
        return DiagnosticResult(
            name="Linux Privileges",
            severity=DiagnosticSeverity.OK,
            message="Running as user with sudo access",
            details={"root": False, "sudo_available": True},
        )
    else:
        return DiagnosticResult(
            name="Linux Privileges",
            severity=DiagnosticSeverity.INFO,
            message="Running as standard user (no sudo)",
            details={"root": False, "sudo_available": False},
            suggestion="Some features may require sudo access",
        )


def _check_macos_root() -> DiagnosticResult:
    """Check macOS root/sudo status."""
    is_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
    
    if is_root:
        return DiagnosticResult(
            name="macOS Privileges",
            severity=DiagnosticSeverity.INFO,
            message="Running as root",
            details={"root": True},
        )
    else:
        return DiagnosticResult(
            name="macOS Privileges",
            severity=DiagnosticSeverity.OK,
            message="Running as standard user",
            details={"root": False},
        )
