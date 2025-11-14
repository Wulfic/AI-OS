"""Fallback progress bar UI for when matplotlib is not available."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .constants import tk, ttk
from ...utils import safe_variables

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel


def create_fallback_ui(panel: "ResourcesPanel", parent: Any) -> None:
    """Create fallback progress bar UI when matplotlib is not available.
    
    Args:
        panel: ResourcesPanel instance
        parent: Parent widget to contain the fallback UI
    """
    # CPU row
    cpu_frame = ttk.Frame(parent)
    cpu_frame.pack(fill="x", padx=4, pady=2)
    ttk.Label(cpu_frame, text="CPU:", width=6).pack(side="left")
    panel._cpu_label_var = safe_variables.StringVar(parent, value="-- %")
    ttk.Label(cpu_frame, textvariable=panel._cpu_label_var, width=35).pack(side="left")
    panel._cpu_progress_var = safe_variables.DoubleVar(parent, value=0.0)
    ttk.Progressbar(cpu_frame, variable=panel._cpu_progress_var, maximum=100, length=200).pack(side="left", padx=(4, 0))

    # RAM row
    ram_frame = ttk.Frame(parent)
    ram_frame.pack(fill="x", padx=4, pady=2)
    ttk.Label(ram_frame, text="RAM:", width=6).pack(side="left")
    panel._ram_label_var = safe_variables.StringVar(parent, value="-- %")
    ttk.Label(ram_frame, textvariable=panel._ram_label_var, width=35).pack(side="left")
    panel._ram_progress_var = safe_variables.DoubleVar(parent, value=0.0)
    ttk.Progressbar(ram_frame, variable=panel._ram_progress_var, maximum=100, length=200).pack(side="left", padx=(4, 0))

    # Network row
    net_frame = ttk.Frame(parent)
    net_frame.pack(fill="x", padx=4, pady=2)
    ttk.Label(net_frame, text="NET:", width=6).pack(side="left")
    panel._net_label_var = safe_variables.StringVar(parent, value="-- MB/s")
    ttk.Label(net_frame, textvariable=panel._net_label_var, width=35).pack(side="left")

    # Disk row
    disk_frame = ttk.Frame(parent)
    disk_frame.pack(fill="x", padx=4, pady=2)
    ttk.Label(disk_frame, text="DISK:", width=6).pack(side="left")
    panel._disk_label_var = safe_variables.StringVar(parent, value="-- MB/s")
    ttk.Label(disk_frame, textvariable=panel._disk_label_var, width=35).pack(side="left")

    # GPU rows container
    panel._gpu_monitor_frame = ttk.Frame(parent)
    panel._gpu_monitor_frame.pack(fill="x", padx=4, pady=2)
    panel._gpu_monitors = {}


def update_fallback_ui(panel: "ResourcesPanel") -> None:
    """Update fallback progress bar UI.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        if len(panel._history["cpu_util"]) > 0:
            cpu_usage = panel._history["cpu_util"][-1]
            cpu_temp = panel._history["cpu_temp"][-1]
            temp_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A"
            panel._cpu_label_var.set(f"{cpu_usage:.0f}% | Temp: {temp_str}")
            panel._cpu_progress_var.set(cpu_usage)

        if len(panel._history["ram_used"]) > 0:
            ram_used = panel._history["ram_used"][-1]
            ram_total = panel._history["ram_total"][-1]
            ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0
            panel._ram_label_var.set(f"{ram_pct:.0f}% ({ram_used:.1f}/{ram_total:.1f} GB)")
            panel._ram_progress_var.set(ram_pct)

        if len(panel._history["net_upload"]) > 0:
            net_up = panel._history["net_upload"][-1]
            net_down = panel._history["net_download"][-1]
            panel._net_label_var.set(f"↑{net_up:.2f} MB/s | ↓{net_down:.2f} MB/s")

        if len(panel._history["disk_read"]) > 0:
            disk_read = panel._history["disk_read"][-1]
            disk_write = panel._history["disk_write"][-1]
            panel._disk_label_var.set(f"R:{disk_read:.2f} MB/s | W:{disk_write:.2f} MB/s")

        # Update GPU monitors
        _update_gpu_monitor_rows(panel)
    except Exception:
        pass


def _update_gpu_monitor_rows(panel: "ResourcesPanel") -> None:
    """Update GPU monitor rows for fallback UI.
    
    Args:
        panel: ResourcesPanel instance
    """
    current_gpu_indices = set(panel._history["gpu"].keys())
    existing_indices = set(panel._gpu_monitors.keys())

    # Remove monitors for GPUs that no longer exist
    for idx in existing_indices - current_gpu_indices:
        try:
            panel._gpu_monitors[idx]["frame"].destroy()
            del panel._gpu_monitors[idx]
        except Exception:
            pass

    # Create or update monitors
    for idx in current_gpu_indices:
        if idx not in panel._gpu_monitors:
            # Create new monitor row
            frame = ttk.Frame(panel._gpu_monitor_frame)
            frame.pack(fill="x", pady=1)
            ttk.Label(frame, text=f"GPU{idx}:", width=6).pack(side="left")
            label_var = safe_variables.StringVar(panel._gpu_monitor_frame, value="--")
            ttk.Label(frame, textvariable=label_var, width=35).pack(side="left")
            progress_var = safe_variables.DoubleVar(panel._gpu_monitor_frame, value=0.0)
            ttk.Progressbar(frame, variable=progress_var, maximum=100, length=200).pack(side="left", padx=(4, 0))
            
            panel._gpu_monitors[idx] = {
                "frame": frame,
                "label_var": label_var,
                "progress_var": progress_var,
            }

        # Update values
        gpu_data = panel._history["gpu"][idx]
        if len(gpu_data["util"]) > 0:
            util = gpu_data["util"][-1]
            mem_used = gpu_data["mem_used"][-1]
            mem_total = gpu_data["mem_total"][-1]
            temp = gpu_data["temp"][-1]
            name = gpu_data["name"]

            label_text = f"{name[:15]:15s} | {util:.0f}% | {mem_used:.1f}/{mem_total:.1f} GB"
            if temp is not None:
                label_text += f" | {temp:.0f}°C"
            
            panel._gpu_monitors[idx]["label_var"].set(label_text)
            panel._gpu_monitors[idx]["progress_var"].set(util)


__all__ = [
    "create_fallback_ui",
    "update_fallback_ui",
]
