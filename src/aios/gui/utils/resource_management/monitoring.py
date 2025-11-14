"""Resource monitoring utilities for diagnostics and health checks."""

from __future__ import annotations

import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitors application resource usage for diagnostics.
    
    Provides health metrics like memory, thread count, CPU usage.
    
    Usage:
        monitor = ResourceMonitor()
        health = monitor.get_health_report()
        print(f"Memory: {health['memory_mb']} MB")
    """
    
    def __init__(self):
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not available - resource monitoring limited")
    
    def get_health_report(self) -> dict[str, Any]:
        """Get current health metrics.
        
        Returns:
            Dict with keys: memory_mb, threads, cpu_percent, open_files (if available)
        """
        report: dict[str, Any] = {
            "memory_mb": 0,
            "threads": threading.active_count(),
            "cpu_percent": 0,
            "open_files": 0,
        }
        
        if not self._psutil:
            logger.warning("Metrics collection unavailable: psutil not installed")
            return report
        
        try:
            import platform
            process = self._psutil.Process()
            
            # Memory in MB
            report["memory_mb"] = process.memory_info().rss / 1024**2
            
            # CPU percent (non-blocking)
            report["cpu_percent"] = process.cpu_percent(interval=0)
            
            # Open file handles (skip on Windows - very slow with many handles)
            if platform.system() != "Windows":
                try:
                    report["open_files"] = len(process.open_files())
                except Exception as e:
                    logger.debug(f"Could not get open files count: {e}")
        
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
        
        return report
    
    def format_health_status(self) -> str:
        """Get formatted health status string for display.
        
        Returns:
            Human-readable status string like "Mem: 245MB | Threads: 12"
        """
        health = self.get_health_report()
        return f"Mem: {health['memory_mb']:.0f}MB | Threads: {health['threads']}"
