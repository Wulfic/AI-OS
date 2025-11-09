"""GPU monitoring integration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional


def create_gpu_monitor_safe(device_ids: List[int], log_file: Path, monitor_interval: float):
    """Safely create GPU monitor with error handling.
    
    Args:
        device_ids: List of CUDA device IDs to monitor
        log_file: Path to GPU metrics log file
        monitor_interval: Monitoring interval in seconds
        
    Returns:
        GPU monitor instance or None if creation failed
    """
    if not device_ids:
        return None
    
    try:
        from ..gpu_monitor import create_gpu_monitor
        monitor = create_gpu_monitor(device_ids, str(log_file))
        monitor.start_monitoring(interval=monitor_interval)
        return monitor
    except Exception:
        return None


def stop_gpu_monitor(monitor) -> Dict[str, Any]:
    """Stop GPU monitor and return summary.
    
    Args:
        monitor: GPU monitor instance or None
        
    Returns:
        Summary dictionary (empty if monitor is None or error occurs)
    """
    if not monitor:
        return {}
    try:
        monitor.stop_monitoring()
        return monitor.get_summary() or {}
    except Exception:
        return {}


def extract_utilization(summary: Dict[str, Any]) -> float:
    """Extract maximum GPU utilization from summary.
    
    Args:
        summary: GPU monitoring summary dictionary
        
    Returns:
        Maximum utilization percentage (0.0 if unavailable)
    """
    if not summary or "error" in summary:
        return 0.0
    utils: List[float] = []
    for metrics in summary.values():
        if isinstance(metrics, dict):
            util = metrics.get("utilization_avg")
            if util is None:
                util = metrics.get("utilization_max")
            if util is not None:
                try:
                    utils.append(float(util))
                except Exception:
                    continue
    return max(utils) if utils else 0.0


def extract_memory(summary: Dict[str, Any]) -> float:
    """Extract maximum GPU memory usage from summary.
    
    Args:
        summary: GPU monitoring summary dictionary
        
    Returns:
        Maximum memory percentage (0.0 if unavailable)
    """
    if not summary or "error" in summary:
        return 0.0
    mems: List[float] = []
    for metrics in summary.values():
        if isinstance(metrics, dict):
            mem = metrics.get("memory_avg")
            if mem is None:
                mem = metrics.get("memory_max")
            if mem is not None:
                try:
                    mems.append(float(mem))
                except Exception:
                    continue
    return max(mems) if mems else 0.0
