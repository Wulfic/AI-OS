"""Directory permissions check for AI-OS Doctor."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)


def check_permissions(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check directory permissions for AI-OS paths."""
    results = []
    
    try:
        from aios.system import paths as system_paths
    except ImportError:
        return [DiagnosticResult(
            name="Path System",
            severity=DiagnosticSeverity.ERROR,
            message="Could not import aios.system.paths",
            suggestion="Ensure AI-OS is properly installed",
        )]
    
    # Define paths to check
    paths_to_check = [
        ("Logs Directory", system_paths.get_logs_dir),
        ("Config Directory", system_paths.get_user_config_dir),
        ("State Directory", lambda: system_paths.get_state_file_path().parent),
        ("Artifacts Root", system_paths.get_artifacts_root),
        ("Cache Directory", system_paths.get_user_cache_root),
        ("Brains Directory", system_paths.get_brains_root),
        ("Program Data Root", system_paths.get_program_data_root),
        ("User Data Root", system_paths.get_user_data_root),
    ]
    
    for name, path_func in paths_to_check:
        try:
            path = path_func()
            result = _check_path_permissions(name, path, auto_repair=auto_repair)
            results.append(result)
        except Exception as e:
            results.append(DiagnosticResult(
                name=name,
                severity=DiagnosticSeverity.ERROR,
                message=f"Failed to resolve path: {e}",
            ))
    
    return results


def _check_path_permissions(
    name: str,
    path: Path,
    *,
    auto_repair: bool = False,
) -> DiagnosticResult:
    """Check if a path exists and is writable."""
    details = {"path": str(path)}
    
    # Check existence
    if not path.exists():
        if auto_repair:
            try:
                path.mkdir(parents=True, exist_ok=True)
                details["action"] = "created"
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.OK,
                    message=f"Created directory: {path}",
                    details=details,
                )
            except PermissionError:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Cannot create directory (permission denied): {path}",
                    details=details,
                    suggestion="Run as Administrator or adjust folder permissions",
                    auto_fixable=True,
                )
            except Exception as e:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Cannot create directory: {e}",
                    details=details,
                )
        else:
            # Try to create anyway
            try:
                path.mkdir(parents=True, exist_ok=True)
                details["action"] = "created"
            except Exception:
                return DiagnosticResult(
                    name=name,
                    severity=DiagnosticSeverity.WARNING,
                    message=f"Directory does not exist: {path}",
                    details=details,
                    suggestion="Run with --repair to create missing directories",
                    auto_fixable=True,
                )
    
    # Test write access
    test_file = path / f".aios_write_test_{os.getpid()}"
    try:
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
        details["writable"] = True
        
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.OK,
            message=f"Writable: {path}",
            details=details,
        )
    except PermissionError:
        details["writable"] = False
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Not writable (permission denied): {path}",
            details=details,
            suggestion="Run as Administrator or adjust folder permissions",
        )
    except Exception as e:
        details["writable"] = False
        return DiagnosticResult(
            name=name,
            severity=DiagnosticSeverity.ERROR,
            message=f"Write test failed: {e}",
            details=details,
        )
    finally:
        # Cleanup
        try:
            if test_file.exists():
                test_file.unlink()
        except Exception:
            pass
