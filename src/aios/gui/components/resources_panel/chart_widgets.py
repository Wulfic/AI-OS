"""Matplotlib chart widgets for system monitoring."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, TYPE_CHECKING
import warnings

from .constants import MATPLOTLIB_AVAILABLE, Figure, FigureCanvasTkAgg, mdates, ttk

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel

logger = logging.getLogger(__name__)

WIDE_FIGSIZE = (8, 3.4)
NARROW_FIGSIZE = (3, 1.9)
DEFAULT_FIGSIZE = (5, 4)
NARROW_ROW_MIN = 90
WIDE_COLUMN_MIN = 400
NARROW_COLUMN_MIN = 200


def create_charts(panel: "ResourcesPanel") -> None:
    """Create matplotlib charts for system monitoring.
    
    Args:
        panel: ResourcesPanel instance
    """
    logger.debug("Creating system monitoring charts (4 charts: processor, memory, network, disk)")
    
    # Clear existing charts
    for widget in panel._charts_container.winfo_children():
        widget.destroy()
    panel._charts.clear()

    # Configure grid to keep wide charts side by side while narrow charts stack on the right
    for row_idx in range(2):
        panel._charts_container.grid_rowconfigure(row_idx, weight=1, minsize=NARROW_ROW_MIN, uniform="monitor_rows")
    panel._charts_container.grid_columnconfigure(0, weight=2, minsize=WIDE_COLUMN_MIN, uniform="wide")
    panel._charts_container.grid_columnconfigure(1, weight=2, minsize=WIDE_COLUMN_MIN, uniform="wide")
    panel._charts_container.grid_columnconfigure(2, weight=1, minsize=NARROW_COLUMN_MIN, uniform="narrow")

    # Layout:
    # Row 0-1, Col 0: Processor (tall, spans both narrow rows)
    # Row 0-1, Col 1: Memory (tall, spans both narrow rows)
    # Row 0,    Col 2: Network (reduced height)
    # Row 1,    Col 2: Disk I/O (reduced height)
    _create_chart_widget(panel, "processor", "Processor (%)", row=0, col=0, ylabel="%", ylim=(0, 100), multi_line=True, max_lines=10, colspan=1, rowspan=2, is_wide=True)
    _create_chart_widget(panel, "memory", "Memory (GB)", row=0, col=1, ylabel="GB", multi_line=True, max_lines=10, colspan=1, rowspan=2, is_wide=True)
    _create_chart_widget(panel, "network", "Network (MB/s)", row=0, col=2, ylabel="MB/s", multi_line=True, is_narrow=True)
    _create_chart_widget(panel, "disk", "Disk I/O (MB/s)", row=1, col=2, ylabel="MB/s", multi_line=True, is_narrow=True)
    
    logger.debug(f"Created {len(panel._charts)} charts successfully")


def _create_chart_widget(
    panel: "ResourcesPanel",
    name: str,
    title: str,
    row: int,
    col: int,
    ylabel: str = "",
    ylim: tuple | None = None,
    multi_line: bool = False,
    max_lines: int = 2,
    colspan: int = 1,
    rowspan: int = 1,
    is_wide: bool = False,
    is_narrow: bool = False
) -> None:
    """Create a single chart widget.
    
    Args:
        panel: ResourcesPanel instance
        name: Chart identifier key
        title: Chart title
        row: Grid row position
        col: Grid column position
        ylabel: Y-axis label
        ylim: Y-axis limits tuple (min, max)
        multi_line: Whether to support multiple lines
        max_lines: Maximum number of lines to create
        colspan: Number of columns to span
        is_wide: Whether this is a wide chart (Processor/Memory)
        is_narrow: Whether this is a narrow chart (Network/Disk)
    """
    logger.debug(f"Creating chart: {name} (title: {title}, lines: {max_lines if multi_line else 1})")
    
    # Create frame for this chart with padding
    frame = ttk.Frame(panel._charts_container, relief="solid", borderwidth=1)
    frame.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan, padx=4, pady=4, sticky="nsew")

    # Create matplotlib figure with appropriate size
    if is_wide:
        fig = Figure(figsize=WIDE_FIGSIZE, dpi=95, facecolor='white')  # type: ignore[misc]
    elif is_narrow:
        fig = Figure(figsize=NARROW_FIGSIZE, dpi=95, facecolor='white')  # type: ignore[misc]
    else:
        fig = Figure(figsize=DEFAULT_FIGSIZE, dpi=95, facecolor='white')  # type: ignore[misc]
        
    ax = fig.add_subplot(111)
    ax.set_facecolor('#fafafa')  # Light gray background for contrast
    ax.set_title(title, fontsize=10 if is_narrow else 12, pad=8, weight='bold')
    ax.set_ylabel(ylabel, fontsize=9 if is_narrow else 10, weight='bold')
    ax.tick_params(labelsize=8 if is_narrow else 9, width=1.2)
    ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7, color='gray')
    if is_narrow:
        try:
            locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.tick_params(axis="x", labelrotation=0)
        except Exception:
            pass
    
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    # Create line(s) with color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lines = []
    if multi_line:
        for i in range(max_lines):
            line, = ax.plot([], [], color=colors[i % len(colors)], linewidth=2, label=f'Line {i+1}')
            lines.append(line)
    else:
        line, = ax.plot([], [], 'b-', linewidth=2)
        lines = [line]

    fig.tight_layout(pad=1.0)

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)  # type: ignore[misc]
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Store references
    panel._charts[name] = {
        "figure": fig,
        "ax": ax,
        "lines": lines,
        "canvas": canvas,
        "frame": frame,
        "labels": [],  # Will store dynamic labels
    }


def update_charts(panel: "ResourcesPanel") -> None:
    """Update all matplotlib charts with current data.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        # Get timeline in seconds
        timeline_seconds = panel._timeline_options.get(panel._current_timeline, 300)
        
        # Filter data based on timeline
        if len(panel._history["timestamps"]) > 0:
            from datetime import datetime
            cutoff_time = datetime.now().timestamp() - timeline_seconds
            visible_indices = [
                i for i, ts in enumerate(panel._history["timestamps"])
                if ts.timestamp() >= cutoff_time
            ]
        else:
            visible_indices = []

        if visible_indices:
            times = [panel._history["timestamps"][i] for i in visible_indices]
            num_points = len(times)
            logger.debug(f"Updating charts with {num_points} data points (timeline: {timeline_seconds}s)")
            
            # Update unified Processor chart (CPU + all GPUs)
            if "processor" in panel._charts:
                # CPU utilization
                cpu_data = [panel._history["cpu_util"][i] for i in visible_indices]
                _update_chart_line(panel, "processor", 0, times, cpu_data, label="CPU")
                
                # GPU utilizations
                line_idx = 1
                labels = ["CPU"]
                for gpu_idx in sorted(panel._history["gpu"].keys()):
                    gpu_data = panel._history["gpu"][gpu_idx]
                    gpu_name = gpu_data.get("name", f"GPU{gpu_idx}")
                    # Short name for legend
                    short_name = f"GPU{gpu_idx}" if "GPU" in gpu_name else gpu_name.split()[-1]
                    
                    if len(gpu_data["util"]) > 0:
                        # Get the most recent N entries from GPU deque where N = len(visible_indices)
                        gpu_util_list = list(gpu_data["util"])
                        num_points = len(visible_indices)
                        if len(gpu_util_list) >= num_points:
                            util_data = gpu_util_list[-num_points:]
                        else:
                            # Pad with zeros if GPU has fewer entries
                            util_data = [0.0] * (num_points - len(gpu_util_list)) + gpu_util_list
                        
                        if line_idx < len(panel._charts["processor"]["lines"]):
                            _update_chart_line(panel, "processor", line_idx, times, util_data, label=short_name)
                            labels.append(short_name)
                            line_idx += 1
                
                # Update legend with actual labels
                _update_chart_legend(panel, "processor", labels)
            
            # Update unified Memory chart (RAM + all GPU memory)
            if "memory" in panel._charts:
                # RAM usage
                ram_data = [panel._history["ram_used"][i] for i in visible_indices]
                _update_chart_line(panel, "memory", 0, times, ram_data, label="RAM")
                
                # GPU memory
                line_idx = 1
                labels = ["RAM"]
                for gpu_idx in sorted(panel._history["gpu"].keys()):
                    gpu_data = panel._history["gpu"][gpu_idx]
                    gpu_name = gpu_data.get("name", f"GPU{gpu_idx}")
                    short_name = f"GPU{gpu_idx}" if "GPU" in gpu_name else gpu_name.split()[-1]
                    
                    if len(gpu_data["mem_used"]) > 0:
                        # Get the most recent N entries from GPU deque where N = len(visible_indices)
                        gpu_mem_list = list(gpu_data["mem_used"])
                        num_points = len(visible_indices)
                        if len(gpu_mem_list) >= num_points:
                            mem_data = gpu_mem_list[-num_points:]
                        else:
                            # Pad with zeros if GPU has fewer entries
                            mem_data = [0.0] * (num_points - len(gpu_mem_list)) + gpu_mem_list
                        
                        if line_idx < len(panel._charts["memory"]["lines"]):
                            _update_chart_line(panel, "memory", line_idx, times, mem_data, label=short_name)
                            labels.append(short_name)
                            line_idx += 1
                
                # Update legend with actual labels
                _update_chart_legend(panel, "memory", labels)
            
            # Update Network chart (upload and download)
            if "network" in panel._charts:
                net_up = [panel._history["net_upload"][i] for i in visible_indices]
                net_down = [panel._history["net_download"][i] for i in visible_indices]
                _update_chart_line(panel, "network", 0, times, net_up, label="↑ Upload")
                _update_chart_line(panel, "network", 1, times, net_down, label="↓ Download")
                _update_chart_legend(panel, "network", ["↑ Upload", "↓ Download"])
            
            # Update Disk chart (read and write)
            if "disk" in panel._charts:
                disk_r = [panel._history["disk_read"][i] for i in visible_indices]
                disk_w = [panel._history["disk_write"][i] for i in visible_indices]
                _update_chart_line(panel, "disk", 0, times, disk_r, label="Read")
                _update_chart_line(panel, "disk", 1, times, disk_w, label="Write")
                _update_chart_legend(panel, "disk", ["Read", "Write"])
    
    except Exception:
        pass


def _update_chart_line(panel: "ResourcesPanel", chart_name: str, line_idx: int, times: list, data: list, label: str | None = None) -> None:
    """Update a specific line in a chart.
    
    Args:
        panel: ResourcesPanel instance
        chart_name: Chart identifier
        line_idx: Line index within chart
        times: X-axis data (timestamps)
        data: Y-axis data
        label: Optional legend label
    """
    try:
        if chart_name not in panel._charts:
            return
        
        chart = panel._charts[chart_name]
        if line_idx >= len(chart["lines"]):
            return
            
        line = chart["lines"][line_idx]
        ax = chart["ax"]
        
        # Update line data
        line.set_data(times, data)
        
        if label:
            line.set_label(label)
        
        # Auto-scale axes
        if times:
            # Avoid warning when setting identical xlim values (single data point)
            if len(times) > 1 and times[0] != times[-1]:
                ax.set_xlim(times[0], times[-1])
            elif len(times) == 1:
                # Single point: set a small range around it
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    try:
                        # Expand by 30 seconds on each side
                        margin = timedelta(seconds=30)
                        ax.set_xlim(times[0] - margin, times[0] + margin)
                    except Exception:
                        pass
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # type: ignore[attr-defined]
            ax.figure.autofmt_xdate(rotation=25)
            # Make x-axis labels readable
            try:
                for label in ax.get_xticklabels():
                    label.set_fontsize(8)  # type: ignore[attr-defined]
            except Exception:
                pass
        
        if data:
            # Add some padding to y-axis
            data_min = min(data)
            data_max = max(data)
            padding = (data_max - data_min) * 0.1 if data_max > data_min else 1.0
            ax.set_ylim(max(0, data_min - padding), data_max + padding)
        
        # Redraw
        chart["canvas"].draw_idle()
    
    except Exception:
        pass


def _update_chart_legend(panel: "ResourcesPanel", chart_name: str, labels: list[str]) -> None:
    """Update chart legend with active labels.
    
    Args:
        panel: ResourcesPanel instance
        chart_name: Chart identifier
        labels: List of legend labels
    """
    try:
        if chart_name not in panel._charts:
            return
        
        chart = panel._charts[chart_name]
        ax = chart["ax"]
        
        # Update legend with only active lines
        handles = []
        legend_labels = []
        for i, label in enumerate(labels):
            if i < len(chart["lines"]):
                handles.append(chart["lines"][i])
                legend_labels.append(label)
        
        if handles:
            ax.legend(handles, legend_labels, fontsize=9, loc='upper left', framealpha=0.95, 
                     edgecolor='gray', fancybox=True, shadow=True)
            chart["canvas"].draw_idle()
    
    except Exception:
        pass


def on_timeline_changed(panel: "ResourcesPanel", event: Any = None) -> None:
    """Handle timeline dropdown change.
    
    Args:
        panel: ResourcesPanel instance
        event: Tkinter event (unused)
    """
    try:
        panel._current_timeline = panel._timeline_var.get()
        # Charts will update on next refresh cycle
    except Exception:
        pass


__all__ = [
    "create_charts",
    "update_charts",
    "on_timeline_changed",
]
