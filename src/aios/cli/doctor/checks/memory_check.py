"""Memory check for AI-OS Doctor."""

from __future__ import annotations

import logging
import platform
from typing import Optional

from ..runner import DiagnosticResult, DiagnosticSeverity

logger = logging.getLogger(__name__)

# Minimum RAM requirements
MIN_RAM_GB = 8
RECOMMENDED_RAM_GB = 16
MIN_VRAM_GB = 6


def check_memory(*, auto_repair: bool = False) -> list[DiagnosticResult]:
    """Check system memory availability."""
    results = []
    
    # System RAM check
    ram_result = _check_system_ram()
    results.append(ram_result)
    
    # Process memory usage
    process_result = _check_process_memory()
    if process_result:
        results.append(process_result)
    
    # GPU memory summary
    gpu_mem_result = _check_gpu_memory_summary()
    if gpu_mem_result:
        results.append(gpu_mem_result)
    
    return results


def _check_system_ram() -> DiagnosticResult:
    """Check total and available system RAM."""
    try:
        import psutil
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        percent_used = mem.percent
        
        details = {
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "used_gb": round(used_gb, 2),
            "percent_used": percent_used,
        }
        
        # Determine severity
        if total_gb < MIN_RAM_GB:
            return DiagnosticResult(
                name="System RAM",
                severity=DiagnosticSeverity.ERROR,
                message=f"{total_gb:.1f} GB total (below minimum {MIN_RAM_GB} GB)",
                details=details,
                suggestion=f"AI-OS requires at least {MIN_RAM_GB} GB RAM. Consider upgrading.",
            )
        elif available_gb < 2:
            return DiagnosticResult(
                name="System RAM",
                severity=DiagnosticSeverity.WARNING,
                message=f"{available_gb:.1f} GB available ({percent_used:.0f}% used)",
                details=details,
                suggestion="Low available memory. Close other applications.",
            )
        elif total_gb < RECOMMENDED_RAM_GB:
            return DiagnosticResult(
                name="System RAM",
                severity=DiagnosticSeverity.OK,
                message=f"{total_gb:.1f} GB total, {available_gb:.1f} GB available",
                details=details,
                suggestion=f"{RECOMMENDED_RAM_GB} GB RAM recommended for larger models",
            )
        else:
            return DiagnosticResult(
                name="System RAM",
                severity=DiagnosticSeverity.OK,
                message=f"{total_gb:.1f} GB total, {available_gb:.1f} GB available ({percent_used:.0f}% used)",
                details=details,
            )
            
    except ImportError:
        return DiagnosticResult(
            name="System RAM",
            severity=DiagnosticSeverity.WARNING,
            message="Could not check (psutil not installed)",
            suggestion="Install psutil: pip install psutil",
        )
    except Exception as e:
        return DiagnosticResult(
            name="System RAM",
            severity=DiagnosticSeverity.WARNING,
            message=f"Could not check: {e}",
        )


def _check_process_memory() -> Optional[DiagnosticResult]:
    """Check current process memory usage."""
    try:
        import psutil
        
        process = psutil.Process()
        mem_info = process.memory_info()
        
        rss_mb = mem_info.rss / (1024**2)
        vms_mb = mem_info.vms / (1024**2)
        
        details = {
            "rss_mb": round(rss_mb, 2),
            "vms_mb": round(vms_mb, 2),
            "pid": process.pid,
        }
        
        # Add memory percent if available
        try:
            details["percent"] = round(process.memory_percent(), 2)
        except Exception:
            pass
        
        return DiagnosticResult(
            name="Process Memory",
            severity=DiagnosticSeverity.INFO,
            message=f"{rss_mb:.0f} MB RSS, {vms_mb:.0f} MB VMS",
            details=details,
        )
        
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Failed to check process memory: {e}")
        return None


def _check_gpu_memory_summary() -> Optional[DiagnosticResult]:
    """Check GPU memory availability summary."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        device_count = torch.cuda.device_count()
        total_vram_gb = 0
        total_free_gb = 0
        gpu_summaries = []
        
        for i in range(device_count):
            try:
                free, total = torch.cuda.mem_get_info(i)
                free_gb = free / (1024**3)
                total_gb = total / (1024**3)
                total_vram_gb += total_gb
                total_free_gb += free_gb
                
                gpu_summaries.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                })
            except Exception:
                continue
        
        if not gpu_summaries:
            return None
        
        details = {
            "device_count": device_count,
            "total_vram_gb": round(total_vram_gb, 2),
            "total_free_gb": round(total_free_gb, 2),
            "gpus": gpu_summaries,
        }
        
        if total_vram_gb < MIN_VRAM_GB:
            return DiagnosticResult(
                name="GPU Memory Total",
                severity=DiagnosticSeverity.WARNING,
                message=f"{total_vram_gb:.1f} GB total VRAM (below recommended {MIN_VRAM_GB} GB)",
                details=details,
                suggestion="Consider using smaller models or CPU inference",
            )
        elif total_free_gb < 2:
            return DiagnosticResult(
                name="GPU Memory Total",
                severity=DiagnosticSeverity.WARNING,
                message=f"{total_free_gb:.1f} GB free of {total_vram_gb:.1f} GB",
                details=details,
                suggestion="GPU memory is low. Clear unused models or restart.",
            )
        else:
            return DiagnosticResult(
                name="GPU Memory Total",
                severity=DiagnosticSeverity.OK,
                message=f"{total_free_gb:.1f} GB free of {total_vram_gb:.1f} GB ({device_count} GPU(s))",
                details=details,
            )
            
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Failed to check GPU memory: {e}")
        return None
