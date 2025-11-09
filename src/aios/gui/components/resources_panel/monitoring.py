"""Resource monitoring data management and update scheduling."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from .constants import MATPLOTLIB_AVAILABLE
from . import stats_collectors
from . import chart_widgets
from . import fallback_widgets

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel


def init_monitoring_data(panel: "ResourcesPanel") -> None:
    """Initialize data structures for historical monitoring.
    
    Args:
        panel: ResourcesPanel instance
    """
    # Timeline options and current selection
    panel._timeline_options = {
        "1 minute": 60,
        "5 minutes": 300,
        "15 minutes": 900,
        "1 hour": 3600,
    }
    panel._current_timeline = "1 minute"
    
    # Historical data storage (max 1 hour at 1.5s intervals = ~2400 points)
    max_points = 2400
    panel._history = {
        "timestamps": deque(maxlen=max_points),
        "cpu_util": deque(maxlen=max_points),
        "cpu_temp": deque(maxlen=max_points),
        "ram_used": deque(maxlen=max_points),
        "ram_total": deque(maxlen=max_points),
        "net_upload": deque(maxlen=max_points),
        "net_download": deque(maxlen=max_points),
        "disk_read": deque(maxlen=max_points),
        "disk_write": deque(maxlen=max_points),
        "gpu": {}  # {idx: {util: deque, mem_used: deque, mem_total: deque, temp: deque}}
    }
    
    # Previous values for rate calculations
    panel._last_net_io = None
    panel._last_disk_io = None
    panel._last_io_time = time.time()


def update_storage_usage(panel: "ResourcesPanel") -> None:
    """Update dataset storage usage display.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        # Import storage functions
        from aios.data.datasets.storage import datasets_storage_usage_gb, datasets_storage_cap_gb
        
        usage_gb = datasets_storage_usage_gb()
        cap_gb = datasets_storage_cap_gb()
        
        # Calculate percentage if cap is set
        if cap_gb > 0:
            pct = (usage_gb / cap_gb) * 100
            usage_text = f"Usage: {usage_gb:.2f} GB / {cap_gb:.2f} GB ({pct:.1f}%)"
        else:
            usage_text = f"Usage: {usage_gb:.2f} GB (no cap set)"
        
        # Update label if it exists
        if hasattr(panel, 'dataset_usage_label'):
            panel.dataset_usage_label.config(text=usage_text)
    except Exception:
        # Silently fail if there's an error calculating storage
        if hasattr(panel, 'dataset_usage_label'):
            panel.dataset_usage_label.config(text="Usage: unavailable")


def update_monitor(panel: "ResourcesPanel") -> None:
    """Update all monitoring displays.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        timestamp = datetime.now()
        panel._history["timestamps"].append(timestamp)

        # Update CPU
        cpu_usage, cpu_temp = stats_collectors.get_cpu_stats()
        panel._history["cpu_util"].append(cpu_usage)
        panel._history["cpu_temp"].append(cpu_temp)

        # Update RAM
        ram_usage, ram_used, ram_total = stats_collectors.get_ram_stats()
        panel._history["ram_used"].append(ram_used)
        panel._history["ram_total"].append(ram_total)

        # Update Network
        net_up, net_down, panel._last_net_io = stats_collectors.get_network_stats(
            panel._last_net_io, panel._last_io_time
        )
        panel._history["net_upload"].append(net_up)
        panel._history["net_download"].append(net_down)

        # Update Disk
        disk_read, disk_write, panel._last_disk_io, panel._last_io_time = stats_collectors.get_disk_stats(
            panel._last_disk_io, panel._last_io_time
        )
        panel._history["disk_read"].append(disk_read)
        panel._history["disk_write"].append(disk_write)

        # Update GPUs (with timeout protection to avoid blocking GUI)
        try:
            gpu_stats = stats_collectors.get_gpu_stats()
            for gpu in gpu_stats:
                idx = gpu["index"]
                if idx not in panel._history["gpu"]:
                    # Initialize deques for new GPU
                    max_points = 2400
                    panel._history["gpu"][idx] = {
                        "util": deque(maxlen=max_points),
                        "mem_used": deque(maxlen=max_points),
                        "mem_total": deque(maxlen=max_points),
                        "temp": deque(maxlen=max_points),
                        "name": gpu.get("name", f"GPU{idx}"),
                    }
                
                panel._history["gpu"][idx]["util"].append(gpu.get("util", 0.0))
                panel._history["gpu"][idx]["mem_used"].append(gpu.get("mem_used_mb", 0.0) / 1024)  # Convert to GB
                panel._history["gpu"][idx]["mem_total"].append(gpu.get("mem_total_mb", 1.0) / 1024)  # Convert to GB
                panel._history["gpu"][idx]["temp"].append(gpu.get("temp"))
                panel._history["gpu"][idx]["name"] = gpu.get("name", f"GPU{idx}")
        except Exception:
            # GPU stats collection failed (nvidia-smi timeout or error) - skip this update
            pass

        # Update storage usage display (only every 10th update to reduce I/O overhead)
        if not hasattr(panel, '_storage_update_counter'):
            panel._storage_update_counter = 0
        panel._storage_update_counter = (panel._storage_update_counter + 1) % 10
        if panel._storage_update_counter == 0:
            update_storage_usage(panel)

        # Update charts or fallback UI
        if MATPLOTLIB_AVAILABLE:
            chart_widgets.update_charts(panel)
        else:
            fallback_widgets.update_fallback_ui(panel)

    except Exception:
        pass


def schedule_monitor_update(panel: "ResourcesPanel") -> None:
    """Schedule the next monitoring update.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        update_monitor(panel)
    finally:
        try:
            if panel._root is not None:
                panel._monitor_after_id = panel._root.after(1500, lambda: schedule_monitor_update(panel))
        except Exception:
            pass


__all__ = [
    "init_monitoring_data",
    "update_monitor",
    "update_storage_usage",
    "schedule_monitor_update",
]
