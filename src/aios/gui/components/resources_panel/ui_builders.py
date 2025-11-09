"""UI building functions for resources panel sections."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .constants import MATPLOTLIB_AVAILABLE, tk, ttk
from . import chart_widgets
from . import fallback_widgets

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel


def build_limits_ui(panel: "ResourcesPanel") -> None:
    """Build CPU/GPU/RAM limits section.
    
    Args:
        panel: ResourcesPanel instance
    """
    r = ttk.Frame(panel)
    r.pack(fill="x")
    
    lbl_cpu_threads = ttk.Label(r, text="CPU threads (logical):")
    lbl_cpu_threads.pack(side="left")
    ent_cpu_threads = ttk.Entry(r, textvariable=panel.cpu_threads_var, width=6)
    ent_cpu_threads.pack(side="left", padx=(4, 8))
    
    lbl_cpu_util = ttk.Label(r, text="CPU util %:")
    lbl_cpu_util.pack(side="left")
    ent_cpu_util = ttk.Entry(r, textvariable=panel.cpu_util_pct_var, width=4)
    ent_cpu_util.pack(side="left", padx=(0, 12))
    
    lbl_sys_mem = ttk.Label(r, text="System RAM limit (GB):")
    lbl_sys_mem.pack(side="left")
    ent_sys_mem = ttk.Entry(r, textvariable=panel.system_mem_limit_gb_var, width=8)
    ent_sys_mem.pack(side="left", padx=(4, 12))
    
    # Add validation callback for RAM limit
    def _validate_ram_limit(*args):
        try:
            import psutil
            val = panel.system_mem_limit_gb_var.get().strip()
            if not val or val == "0":  # Empty or 0 = system limit
                return
            
            limit_gb = float(val)
            if limit_gb < 0:
                # Negative not allowed
                panel.system_mem_limit_gb_var.set("0")
                return
            
            # Get actual system RAM
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            if limit_gb > system_ram_gb:
                # Cap at system RAM
                panel.system_mem_limit_gb_var.set(str(int(system_ram_gb)))
        except Exception:
            pass
    
    panel.system_mem_limit_gb_var.trace_add("write", _validate_ram_limit)
    
    # Multi-GPU Training Mode Toggle (moved here from devices section)
    mode_label = ttk.Label(r, text="Multi-GPU Mode:")
    mode_label.pack(side="left", padx=(12, 0))
    
    # Create toggle switch effect using radiobuttons
    toggle_frame = ttk.Frame(r, relief="solid", borderwidth=1)
    toggle_frame.pack(side="left", padx=(8, 0))
    
    ddp_btn = ttk.Radiobutton(
        toggle_frame, 
        text="DDP", 
        variable=panel.training_mode_var, 
        value="ddp",
        state="disabled"  # Will be enabled/disabled dynamically
    )
    ddp_btn.pack(side="left", padx=2, pady=2)
    
    parallel_btn = ttk.Radiobutton(
        toggle_frame, 
        text="Parallel", 
        variable=panel.training_mode_var, 
        value="parallel",
        state="disabled"  # Will be enabled/disabled dynamically
    )
    parallel_btn.pack(side="left", padx=2, pady=2)
    
    # Lock indicator label (shown conditionally)
    lock_label = ttk.Label(r, text="", foreground="gray")
    lock_label.pack(side="left", padx=(8, 0))
    
    # Store references for dynamic updates
    panel._training_mode_toggle_widgets = {
        "label": mode_label,
        "ddp_btn": ddp_btn,
        "parallel_btn": parallel_btn,
        "lock_label": lock_label,
    }
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(mode_label, "Training mode for multi-GPU setups. Only enabled when 2+ GPUs are selected.")
        add_tooltip(ddp_btn, "DistributedDataParallel: Each GPU trains on different data batches (faster, recommended)")
        add_tooltip(parallel_btn, "DataParallel: Legacy mode, splits batch across GPUs (slower but more compatible)")
    except Exception:
        pass
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(mode_label, "Training mode for multi-GPU setups. Only enabled when 2+ GPUs are selected.")
        add_tooltip(ddp_btn, "DistributedDataParallel: Each GPU trains on different data batches (faster, recommended)")
        add_tooltip(parallel_btn, "DataParallel: Legacy mode, splits batch across GPUs (slower but more compatible)")
    except Exception:
        pass
    
    # Add tooltips
    _add_limits_tooltips(lbl_cpu_threads, ent_cpu_threads, lbl_cpu_util, ent_cpu_util, lbl_sys_mem, ent_sys_mem)


def build_devices_ui(panel: "ResourcesPanel") -> Any:
    """Build device selection section.
    
    Args:
        panel: ResourcesPanel instance
        
    Returns:
        The devices frame widget
    """
    devs = ttk.LabelFrame(panel, text="Devices")
    devs.pack(fill="x", pady=(8, 0))
    
    # Train device selection
    train_box = ttk.LabelFrame(devs, text="Training Device")
    train_box.pack(fill="x", padx=0, pady=2)
    train_auto_radio = ttk.Radiobutton(train_box, text="Auto", variable=panel.train_device_var, value="auto")
    train_auto_radio.pack(side="left")
    train_cpu_radio = ttk.Radiobutton(train_box, text="CPU", variable=panel.train_device_var, value="cpu")
    train_cpu_radio.pack(side="left", padx=(6,0))
    train_cuda_radio = ttk.Radiobutton(train_box, text="CUDA GPU(s)", variable=panel.train_device_var, value="cuda")
    train_cuda_radio.pack(side="left", padx=(6,0))
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(train_auto_radio, "Automatically select best available device for training (prefers CUDA if available)")
        add_tooltip(train_cpu_radio, "Force training to use CPU only (slower but more compatible)")
        add_tooltip(train_cuda_radio, "Use CUDA GPU(s) for training (fastest if available)")
    except Exception:
        pass
    
    # CUDA Train selection list
    panel.cuda_train_group = ttk.Frame(train_box)
    panel.cuda_train_group.pack(fill="x", padx=(8, 0), pady=(4, 0))

    # Run device selection
    run_box = ttk.LabelFrame(devs, text="Inference/Run Device")
    run_box.pack(fill="x", padx=0, pady=2)
    run_auto_radio = ttk.Radiobutton(run_box, text="Auto", variable=panel.run_device_var, value="auto")
    run_auto_radio.pack(side="left")
    run_cpu_radio = ttk.Radiobutton(run_box, text="CPU", variable=panel.run_device_var, value="cpu")
    run_cpu_radio.pack(side="left", padx=(6,0))
    run_cuda_radio = ttk.Radiobutton(run_box, text="CUDA GPU(s)", variable=panel.run_device_var, value="cuda")
    run_cuda_radio.pack(side="left", padx=(6,0))
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(run_auto_radio, "Automatically select best available device for inference (prefers CUDA if available)")
        add_tooltip(run_cpu_radio, "Force inference to use CPU only (lower memory usage)")
        add_tooltip(run_cuda_radio, "Use CUDA GPU(s) for inference (faster responses)")
    except Exception:
        pass
    
    # CUDA Run selection list
    panel.cuda_run_group = ttk.Frame(run_box)
    panel.cuda_run_group.pack(fill="x", padx=(8, 0), pady=(4, 0))
    
    # Add tooltip
    _add_devices_tooltip(devs)
    
    return devs


def build_status_ui(panel: "ResourcesPanel") -> None:
    """Build status message section.
    
    Args:
        panel: ResourcesPanel instance
    """
    panel._status_frame = ttk.Frame(panel)
    panel._status_frame.pack(fill="x", pady=(4, 0))
    panel._status_label = ttk.Label(
        panel._status_frame, 
        text="", 
        foreground="gray",
        font=("TkDefaultFont", 8, "italic")
    )
    panel._status_label.pack(side="left")


def build_detect_button_ui(panel: "ResourcesPanel") -> Any:
    """Build device detection button.
    
    Args:
        panel: ResourcesPanel instance
        
    Returns:
        The detect button widget or None
    """
    btn_detect = None
    if panel._detect_fn is not None:
        btn_detect = ttk.Button(panel, text="Detect devices", command=panel._detect_and_update)
        btn_detect.pack(pady=(8,0))
        _add_detect_button_tooltip(btn_detect)
    return btn_detect


def build_storage_caps_ui(panel: "ResourcesPanel") -> Any:
    """Build storage caps section.
    
    Args:
        panel: ResourcesPanel instance
        
    Returns:
        The storage caps frame widget
    """
    caps = ttk.LabelFrame(panel, text="Storage Caps")
    caps.pack(fill="x", pady=(8, 0))
    
    # Dataset cap only (set on Datasets page, enforced for download folder)
    row1 = ttk.Frame(caps)
    row1.pack(fill="x", padx=0, pady=2)
    ttk.Label(row1, text="Dataset cap (GB):").pack(side="left")
    dataset_cap_entry = ttk.Entry(row1, width=8, textvariable=panel.dataset_cap_var)
    dataset_cap_entry.pack(side="left", padx=(6, 12))
    ttk.Label(row1, text="(enforced for dataset download folder)", font=("TkDefaultFont", 8, "italic")).pack(side="left")
    
    # Add usage display label
    row2 = ttk.Frame(caps)
    row2.pack(fill="x", padx=0, pady=2)
    panel.dataset_usage_label = ttk.Label(row2, text="Usage: calculating...", font=("TkDefaultFont", 8))
    panel.dataset_usage_label.pack(side="left", padx=(6, 0))
    
    # Add tooltip
    try:
        from ..tooltips import add_tooltip
        add_tooltip(dataset_cap_entry, "Maximum disk space for dataset downloads (GB). Empty = unlimited. Enforced in training_data folder.")
        add_tooltip(panel.dataset_usage_label, "Current dataset storage usage (excludes cached data, only counts downloaded datasets)")
    except Exception:
        pass
    
    _add_caps_tooltip(caps)
    
    return caps


def build_apply_button_ui(panel: "ResourcesPanel") -> None:
    """Build bottom action bar with Apply button.
    
    Args:
        panel: ResourcesPanel instance
    """
    bottom = ttk.Frame(panel)
    bottom.pack(fill="x", pady=(10, 0))
    btn_apply_all = ttk.Button(bottom, text="Apply", command=panel._on_apply_all)
    btn_apply_all.pack(side="left")
    _add_apply_button_tooltip(btn_apply_all)


def _add_tooltip(widget: Any, text: str) -> None:
    """Add a tooltip to a widget.
    
    Args:
        widget: Widget to add tooltip to
        text: Tooltip text
    """
    def on_enter(event: Any) -> None:
        try:
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                tooltip, 
                text=text, 
                background="#ffffe0", 
                relief="solid", 
                borderwidth=1,
                font=("TkDefaultFont", 8)
            )
            label.pack()
            widget._tooltip = tooltip
        except Exception:
            pass
    
    def on_leave(event: Any) -> None:
        try:
            if hasattr(widget, '_tooltip'):
                widget._tooltip.destroy()
                delattr(widget, '_tooltip')
        except Exception:
            pass
    
    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)


def build_monitor_ui(panel: "ResourcesPanel") -> None:
    """Build the Live Resource Monitor UI section with charts.
    
    Args:
        panel: ResourcesPanel instance
    """
    monitor = ttk.LabelFrame(panel, text="Live Resource Monitor")
    monitor.pack(fill="both", expand=True, pady=(8, 0))

    # Timeline selector
    timeline_frame = ttk.Frame(monitor)
    timeline_frame.pack(fill="x", padx=4, pady=4)
    ttk.Label(timeline_frame, text="Timeline:").pack(side="left", padx=(0, 4))
    panel._timeline_var = tk.StringVar(value="1 minute")
    timeline_combo = ttk.Combobox(
        timeline_frame,
        textvariable=panel._timeline_var,
        values=list(["1 minute", "5 minutes", "15 minutes", "1 hour"]),
        state="readonly",
        width=15
    )
    timeline_combo.pack(side="left")
    timeline_combo.bind("<<ComboboxSelected>>", lambda e: chart_widgets.on_timeline_changed(panel, e))

    if MATPLOTLIB_AVAILABLE:
        # Create scrollable charts container for better space management
        canvas_frame = ttk.Frame(monitor)
        canvas_frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        panel._charts_container = ttk.Frame(canvas_frame)
        panel._charts_container.pack(fill="both", expand=True)
        
        # Configure grid for 4 columns
        for i in range(4):
            panel._charts_container.grid_columnconfigure(i, weight=1, minsize=300)
        
        # Create initial charts
        chart_widgets.create_charts(panel)
    else:
        # Fallback to progress bars if matplotlib not available
        fallback_widgets.create_fallback_ui(panel, monitor)


# Tooltip helpers

def _add_limits_tooltips(lbl_cpu_threads: Any, ent_cpu_threads: Any, lbl_cpu_util: Any, ent_cpu_util: Any, lbl_sys_mem: Any, ent_sys_mem: Any) -> None:
    """Add tooltips to limits section widgets."""
    try:  # pragma: no cover - UI enhancement only
        from ..tooltips import add_tooltip
        add_tooltip(lbl_cpu_threads, "Logical CPU threads for PyTorch operations (not physical cores). Your CPU: 8 cores = 16 threads.")
        add_tooltip(ent_cpu_threads, "Number of logical threads for inference + data loading. Lower values reduce CPU contention.")
        add_tooltip(lbl_cpu_util, "Target max aggregate CPU utilization (0 disables limiter).")
        add_tooltip(ent_cpu_util, "Optional CPU utilization target; 0 means no limit.")
        add_tooltip(lbl_sys_mem, "Maximum system RAM to use in GB. 0 = system limit (no restriction). Auto-capped at system max.")
        add_tooltip(ent_sys_mem, "Limit RAM usage to prevent OOM. 0 = system limit. Values > system RAM are auto-capped.")
    except Exception:
        pass


def _add_devices_tooltip(devs: Any) -> None:
    """Add tooltip to devices section."""
    try:  # pragma: no cover - UI enhancement only
        from ..tooltips import add_tooltip
        add_tooltip(devs, "Configure which devices are used for training vs inference.")
    except Exception:
        pass


def _add_detect_button_tooltip(btn_detect: Any) -> None:
    """Add tooltip to detect button."""
    try:  # pragma: no cover - UI enhancement only
        from ..tooltips import add_tooltip
        add_tooltip(btn_detect, "Probe system for CUDA devices and capabilities.")
    except Exception:
        pass


def _add_caps_tooltip(caps: Any) -> None:
    """Add tooltip to storage caps section."""
    try:  # pragma: no cover - UI enhancement only
        from ..tooltips import add_tooltip
        add_tooltip(caps, "Set disk usage cap for dataset downloads (GB). Configured per dataset on Datasets page.")
    except Exception:
        pass


def _add_apply_button_tooltip(btn_apply_all: Any) -> None:
    """Add tooltip to apply button."""
    try:  # pragma: no cover - UI enhancement only
        from ..tooltips import add_tooltip
        add_tooltip(btn_apply_all, "Persist all resource settings (threads, devices, dataset cap).")
    except Exception:
        pass


__all__ = [
    "build_limits_ui",
    "build_devices_ui",
    "build_status_ui",
    "build_detect_button_ui",
    "build_storage_caps_ui",
    "build_apply_button_ui",
    "build_monitor_ui",
]
