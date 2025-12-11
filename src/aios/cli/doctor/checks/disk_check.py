"""Disk space check for AI-OS Doctor."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Optional

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)

# Minimum recommended disk space in GB
MIN_CACHE_SPACE_GB = 10
MIN_ARTIFACTS_SPACE_GB = 20
WARNING_SPACE_GB = 5


def check_disk_space(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check disk space for critical directories."""
    results = []
    
    # Get paths to check
    paths_to_check = _get_paths_to_check()
    
    # Group paths by drive/mount point to avoid duplicate checks
    checked_drives = set()
    
    for name, path in paths_to_check:
        try:
            # Get the mount point / drive root
            drive = _get_drive_root(path)
            
            # Only check each drive once
            if drive in checked_drives:
                continue
            checked_drives.add(drive)
            
            result = _check_drive_space(name, path, drive)
            results.append(result)
            
        except Exception as e:
            results.append(DiagnosticResult(
                name=f"Disk Space ({name})",
                severity=DiagnosticSeverity.WARNING,
                message=f"Could not check: {e}",
            ))
    
    # Check training data directory size
    training_data_result = _check_training_data_size()
    if training_data_result:
        results.append(training_data_result)
    
    return results


def _get_paths_to_check() -> list[tuple[str, Path]]:
    """Get list of paths to check disk space for."""
    paths = []
    
    try:
        from aios.system import paths as system_paths
        
        paths.append(("Cache", system_paths.get_user_cache_root()))
        paths.append(("Artifacts", system_paths.get_artifacts_root()))
        paths.append(("Logs", system_paths.get_logs_dir()))
        
    except ImportError:
        # Fallback paths
        if platform.system() == "Windows":
            paths.append(("User Home", Path.home()))
            paths.append(("System Drive", Path("C:/")))
        else:
            paths.append(("Home", Path.home()))
            paths.append(("Root", Path("/")))
    
    # HuggingFace cache
    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        paths.append(("HF Cache", Path(hf_cache)))
    else:
        default_hf = Path.home() / ".cache" / "huggingface"
        if default_hf.exists():
            paths.append(("HF Cache", default_hf))
    
    return paths


def _get_drive_root(path: Path) -> str:
    """Get the drive root or mount point for a path."""
    try:
        path = path.resolve()
        
        if platform.system() == "Windows":
            # Windows: return drive letter
            return path.drive or "C:"
        else:
            # Unix: find mount point
            while not path.is_mount() and path != path.parent:
                path = path.parent
            return str(path)
    except Exception:
        return str(path.root) if path.root else "/"


def _check_drive_space(name: str, path: Path, drive: str) -> DiagnosticResult:
    """Check disk space for a drive/mount point."""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(drive)
        
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        usage_pct = (used / total) * 100
        
        details = {
            "path": str(path),
            "drive": drive,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "usage_percent": round(usage_pct, 1),
        }
        
        if free_gb < WARNING_SPACE_GB:
            return DiagnosticResult(
                name=f"Disk Space ({drive})",
                severity=DiagnosticSeverity.ERROR,
                message=f"CRITICALLY LOW: {free_gb:.1f} GB free ({usage_pct:.0f}% used)",
                details=details,
                suggestion=f"Free up disk space on {drive}. AI-OS may fail to download models.",
            )
        elif free_gb < MIN_CACHE_SPACE_GB:
            return DiagnosticResult(
                name=f"Disk Space ({drive})",
                severity=DiagnosticSeverity.WARNING,
                message=f"Low: {free_gb:.1f} GB free ({usage_pct:.0f}% used)",
                details=details,
                suggestion=f"Consider freeing space on {drive} for HF model cache.",
            )
        else:
            return DiagnosticResult(
                name=f"Disk Space ({drive})",
                severity=DiagnosticSeverity.OK,
                message=f"{free_gb:.1f} GB free ({usage_pct:.0f}% used)",
                details=details,
            )
            
    except Exception as e:
        return DiagnosticResult(
            name=f"Disk Space ({drive})",
            severity=DiagnosticSeverity.WARNING,
            message=f"Could not check: {e}",
        )


def _check_training_data_size() -> Optional[DiagnosticResult]:
    """Check the size of training data directory."""
    try:
        # Find training_data directory
        training_data = None
        
        try:
            from aios.system import paths as system_paths
            install_root = system_paths.get_install_root()
            training_data = install_root / "training_data"
        except ImportError:
            pass
        
        if not training_data or not training_data.exists():
            # Try current working directory
            cwd = Path.cwd()
            for parent in [cwd] + list(cwd.parents)[:5]:
                candidate = parent / "training_data"
                if candidate.exists():
                    training_data = candidate
                    break
        
        if not training_data or not training_data.exists():
            return None
        
        # Calculate directory size
        total_size = 0
        file_count = 0
        
        for dirpath, dirnames, filenames in os.walk(training_data):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except Exception:
                    pass
        
        size_gb = total_size / (1024**3)
        
        return DiagnosticResult(
            name="Training Data",
            severity=DiagnosticSeverity.INFO,
            message=f"{size_gb:.2f} GB ({file_count} files)",
            details={
                "path": str(training_data),
                "size_gb": round(size_gb, 2),
                "file_count": file_count,
            },
        )
        
    except Exception as e:
        logger.debug(f"Failed to check training data size: {e}")
        return None
