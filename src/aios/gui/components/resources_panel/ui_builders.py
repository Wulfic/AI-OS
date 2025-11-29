"""UI building functions for resources panel sections."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .constants import MATPLOTLIB_AVAILABLE, tk, ttk
from . import chart_widgets
from . import fallback_widgets

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel

logger = logging.getLogger(__name__)


def build_limits_ui(panel: "ResourcesPanel") -> None:
    """Build CPU/GPU/RAM limits section.
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        r = ttk.Frame(panel)
        r.pack(fill="x")
    except Exception as e:
        logger.error(f"Failed to create Frame widget: {e}")
        logger.warning("Using fallback UI element")
        return
    
    try:
        lbl_cpu_threads = ttk.Label(r, text="CPU threads (logical):")
        lbl_cpu_threads.pack(side="left")
        ent_cpu_threads = ttk.Entry(r, textvariable=panel.cpu_threads_var, width=6)
        ent_cpu_threads.pack(side="left", padx=(4, 8))
    except Exception as e:
        logger.error(f"Failed to create CPU threads widgets: {e}")
        logger.warning("Using fallback UI element")
    
    try:
        lbl_cpu_util = ttk.Label(r, text="CPU util %:")
        lbl_cpu_util.pack(side="left")
        ent_cpu_util = ttk.Entry(r, textvariable=panel.cpu_util_pct_var, width=4)
        ent_cpu_util.pack(side="left", padx=(0, 12))
    except Exception as e:
        logger.error(f"Failed to create CPU util widgets: {e}")
        logger.warning("Using fallback UI element")
    
    try:
        lbl_sys_mem = ttk.Label(r, text="System RAM limit (GB):")
        lbl_sys_mem.pack(side="left")
        ent_sys_mem = ttk.Entry(r, textvariable=panel.system_mem_limit_gb_var, width=8)
        ent_sys_mem.pack(side="left", padx=(4, 12))
    except Exception as e:
        logger.error(f"Failed to create System RAM widgets: {e}")
        logger.warning("Using fallback UI element")
    
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
        state="disabled",  # Will be enabled/disabled dynamically
    )
    ddp_btn.pack(side="left", padx=2, pady=2)

    zero3_btn = ttk.Radiobutton(
        toggle_frame,
        text="Zero3",
        variable=panel.training_mode_var,
        value="zero3",
        state="disabled",  # Enabled only on Linux with 2+ GPUs
    )
    zero3_btn.pack(side="left", padx=2, pady=2)

    parallel_btn = ttk.Radiobutton(
        toggle_frame,
        text="Parallel",
        variable=panel.training_mode_var,
        value="parallel",
        state="disabled",  # Will be enabled/disabled dynamically
    )
    parallel_btn.pack(side="left", padx=2, pady=2)

    none_btn = ttk.Radiobutton(
        toggle_frame,
        text="None",
        variable=panel.training_mode_var,
        value="none",
        state="disabled",  # Enabled when single-GPU mode is active
    )
    none_btn.pack(side="left", padx=2, pady=2)
    
    # Lock indicator label (shown conditionally)
    lock_label = ttk.Label(r, text="", foreground="gray")
    lock_label.pack(side="left", padx=(8, 0))
    
    # Add "Detect devices" button next to Multi-GPU Mode
    if panel._detect_fn is not None:
        btn_detect = ttk.Button(r, text="Detect devices", command=panel._detect_and_update)
        btn_detect.pack(side="left", padx=(12, 0))
        panel.detect_button = btn_detect
        _add_detect_button_tooltip(btn_detect)
    
    # Add "Max Performance" checkbox
    max_perf_cb = ttk.Checkbutton(
        r, 
        text="Max Performance", 
        variable=panel.max_performance_var,
        command=lambda: _on_max_performance_toggle(panel)
    )
    max_perf_cb.pack(side="left", padx=(12, 0))
    
    # Add tooltip for Max Performance
    try:
        from ..tooltips import add_tooltip
        add_tooltip(max_perf_cb, "Enable maximum performance mode: Sets all GPUs to 100% memory and 0% utilization limit (no throttling)")
    except Exception:
        pass
    
    # Add Storage Caps inline on the same row
    ttk.Label(r, text="Dataset cap (GB):").pack(side="left", padx=(24, 4))
    dataset_cap_entry = ttk.Entry(r, width=8, textvariable=panel.dataset_cap_var)
    dataset_cap_entry.pack(side="left")
    panel.dataset_cap_entry = dataset_cap_entry
    
    # Add usage display label inline
    panel.dataset_usage_label = ttk.Label(r, text="", font=("TkDefaultFont", 8), foreground="gray")
    panel.dataset_usage_label.pack(side="left", padx=(6, 0))
    
    # Add tooltip
    try:
        from ..tooltips import add_tooltip
        add_tooltip(dataset_cap_entry, "Maximum disk space for dataset downloads (GB). Empty = unlimited. Enforced in training_data folder.")
    except Exception:
        pass
    
    # Store references for dynamic updates
    panel._training_mode_toggle_widgets = {
        "label": mode_label,
        "ddp_btn": ddp_btn,
        "parallel_btn": parallel_btn,
        "zero3_btn": zero3_btn,
        "none_btn": none_btn,
        "lock_label": lock_label,
    }
    
    # Add tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(mode_label, "Training mode for multi-GPU setups. Only enabled when 2+ GPUs are selected.")
        add_tooltip(ddp_btn, "DistributedDataParallel: Each GPU trains on different data batches (faster, recommended)")
        add_tooltip(parallel_btn, "DataParallel: Legacy mode, splits batch across GPUs (slower but more compatible)")
        add_tooltip(none_btn, "Single-GPU mode: disable multi-GPU orchestration and use the selected device only.")
        add_tooltip(zero3_btn, "Enable DeepSpeed ZeRO Stage 3 with dedicated inference shard (Linux only, requires 2+ GPUs).")
        add_tooltip(none_btn, "Single-GPU mode: disable multi-GPU orchestration and use the selected device only.")
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
    devs.pack(fill="both", expand=True, pady=(8, 0))

    # Track canvas heights so inference list stays aligned with training list
    panel._train_canvas_height = 0
    panel._ensure_run_canvas_min_height = lambda: None
    
    # Train device selection
    train_box = ttk.LabelFrame(devs, text="Training Device")
    train_box.pack(fill="both", expand=True, padx=0, pady=2)
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
    
    # CUDA Train selection list - now with scrollable canvas and grid layout
    train_scroll_frame = ttk.Frame(train_box)
    train_scroll_frame.pack(fill="both", expand=True, padx=(8, 0), pady=(4, 0))
    
    # Create canvas and scrollbar for train GPUs (dynamic height, max 140px)
    train_canvas = tk.Canvas(train_scroll_frame, borderwidth=0, highlightthickness=0)
    train_scrollbar = ttk.Scrollbar(train_scroll_frame, orient="vertical", command=train_canvas.yview)
    train_scrollable_frame = ttk.Frame(train_canvas)
    
    def _update_train_scroll():
        """Update scroll region and canvas height based on content."""
        train_canvas.update_idletasks()
        bbox = train_canvas.bbox("all")
        if bbox:
            # Calculate content height
            content_height = bbox[3]
            # Set canvas height to content height, but max 140px
            canvas_height = min(content_height, 140)
            train_canvas.configure(height=canvas_height)
            # Only set vertical scrolling, keep width at canvas width
            train_canvas.configure(scrollregion=(0, 0, train_canvas.winfo_width(), bbox[3]))
            panel._train_canvas_height = canvas_height
            try:
                panel._ensure_run_canvas_min_height()
            except Exception:
                pass
    
    train_scrollable_frame.bind("<Configure>", lambda e: _update_train_scroll())
    
    train_canvas_window = train_canvas.create_window((0, 0), window=train_scrollable_frame, anchor="nw")
    train_canvas.configure(yscrollcommand=train_scrollbar.set)
    
    # Make the canvas window fill the width
    def _configure_train_canvas(event):
        train_canvas.itemconfig(train_canvas_window, width=event.width)
        _update_train_scroll()
    train_canvas.bind("<Configure>", _configure_train_canvas)
    
    panel._train_canvas = train_canvas
    train_canvas.pack(side="left", fill="both", expand=True)
    train_scrollbar.pack(side="right", fill="y")
    
    # Mouse wheel scrolling
    def _on_train_mousewheel(event):
        train_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    train_canvas.bind_all("<MouseWheel>", _on_train_mousewheel)
    
    # Configure grid layout for GPU cards (4 columns)
    for i in range(4):
        train_scrollable_frame.grid_columnconfigure(i, weight=1, uniform="gpu_cards")
    
    panel.cuda_train_group = train_scrollable_frame
    panel._cuda_train_grid_col = 0  # Track current grid column
    panel._cuda_train_grid_row = 0  # Track current grid row

    # Run device selection
    run_box = ttk.LabelFrame(devs, text="Inference/Run Device")
    run_box.pack(fill="both", expand=True, padx=0, pady=2)
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

    # CUDA Run selection list - now with scrollable canvas and grid layout
    run_scroll_frame = ttk.Frame(run_box)
    run_scroll_frame.pack(fill="both", expand=True, padx=(8, 0), pady=(4, 0))
    
    # Create canvas and scrollbar for run GPUs (dynamic height, max 140px)
    run_canvas = tk.Canvas(run_scroll_frame, borderwidth=0, highlightthickness=0)
    run_scrollbar = ttk.Scrollbar(run_scroll_frame, orient="vertical", command=run_canvas.yview)
    run_scrollable_frame = ttk.Frame(run_canvas)
    
    def _update_run_scroll():
        """Update scroll region and canvas height based on content."""
        run_canvas.update_idletasks()
        bbox = run_canvas.bbox("all")
        if bbox:
            # Calculate content height
            content_height = bbox[3]
            # Set canvas height to content height, but max 140px
            canvas_height = min(content_height, 140)
            canvas_height = max(canvas_height, getattr(panel, "_train_canvas_height", 0))
            run_canvas.configure(height=canvas_height)
            # Only set vertical scrolling, keep width at canvas width
            run_canvas.configure(scrollregion=(0, 0, run_canvas.winfo_width(), bbox[3]))
        else:
            baseline = getattr(panel, "_train_canvas_height", 0)
            if baseline:
                run_canvas.configure(height=baseline)
    
    run_scrollable_frame.bind("<Configure>", lambda e: _update_run_scroll())
    
    run_canvas_window = run_canvas.create_window((0, 0), window=run_scrollable_frame, anchor="nw")
    run_canvas.configure(yscrollcommand=run_scrollbar.set)
    
    # Make the canvas window fill the width
    def _configure_run_canvas(event):
        run_canvas.itemconfig(run_canvas_window, width=event.width)
        _update_run_scroll()
    run_canvas.bind("<Configure>", _configure_run_canvas)
    
    panel._run_canvas = run_canvas
    run_canvas.pack(side="left", fill="both", expand=True)
    run_scrollbar.pack(side="right", fill="y")
    
    # Mouse wheel scrolling
    def _on_run_mousewheel(event):
        run_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    run_canvas.bind_all("<MouseWheel>", _on_run_mousewheel)

    def _ensure_run_canvas_min_height():
        try:
            _update_run_scroll()
        except Exception:
            pass

    panel._ensure_run_canvas_min_height = _ensure_run_canvas_min_height
    
    # Configure grid layout for GPU cards (4 columns)
    for i in range(4):
        run_scrollable_frame.grid_columnconfigure(i, weight=1, uniform="gpu_cards")
    
    panel.cuda_run_group = run_scrollable_frame
    panel._cuda_run_grid_col = 0  # Track current grid column
    panel._cuda_run_grid_row = 0  # Track current grid row
    
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


def build_storage_caps_ui(panel: "ResourcesPanel") -> Any:
    """Build storage caps section (now integrated into limits row).
    
    Args:
        panel: ResourcesPanel instance
        
    Returns:
        None
    """
    try:
        frame = ttk.LabelFrame(panel, text="Storage Paths")
        frame.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(frame, text="Artifacts directory:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(frame, textvariable=panel.artifacts_dir_var, width=60)
        entry.grid(row=0, column=1, sticky="ew", padx=(6, 4))
        entry.bind("<FocusOut>", lambda _e: panel._validate_artifacts_dir())

        browse_btn = ttk.Button(frame, text="Browse…", command=panel._browse_artifacts_dir)
        browse_btn.grid(row=0, column=2, padx=(4, 0))

        reset_btn = ttk.Button(frame, text="Use Default", command=panel._reset_artifacts_dir)
        reset_btn.grid(row=0, column=3, padx=(4, 0))

        frame.columnconfigure(1, weight=1)

        status = ttk.Label(
            frame,
            textvariable=panel._artifacts_status_var,
            font=("TkDefaultFont", 8),
            foreground="gray",
        )
        status.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        panel._artifacts_status_label = status

        try:
            from ..tooltips import add_tooltip

            add_tooltip(
                entry,
                "Optional custom location for artifacts and brains. Leave blank to use ProgramData/AI-OS/artifacts.",
            )
            add_tooltip(reset_btn, "Revert to the default ProgramData path.")
        except Exception:
            pass

        panel._validate_artifacts_dir(apply_override=False)
    except Exception as exc:
        logger.error(f"Failed to build storage caps UI: {exc}", exc_info=True)


def build_apply_button_ui(panel: "ResourcesPanel") -> None:
    """Build bottom action bar with Apply button (deprecated - now auto-saves).
    
    Args:
        panel: ResourcesPanel instance
    """
    # Apply button removed - settings now auto-save on change
    # This function kept for compatibility but does nothing
    pass


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
        
        # Configure grid for 3 columns, 2 rows
        # Columns 0-1: Wide charts (Processor, Memory)
        # Column 2: Narrow charts (Network, Disk stacked)
        # Grid weights are set in create_charts based on chart type
        
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


def _on_max_performance_toggle(panel: "ResourcesPanel") -> None:
    """Handle Max Performance checkbox toggle.
    
    When enabled: Sets all GPU memory to 100% and utilization to 0% (no limits) and disables inputs
    When disabled: Restores previous values and enables inputs
    
    Args:
        panel: ResourcesPanel instance
    """
    is_max_perf = panel.max_performance_var.get()
    
    if is_max_perf:
        logger.info("Max Performance mode enabled - setting all GPUs to 100% memory, 0% utilization")
        # Apply to training GPUs
        for row in panel._cuda_train_rows:
            try:
                row["mem_pct"].set("100")
                row["util_pct"].set("0")
                # Disable the input fields
                for widget in row.get("widgets", []):
                    if hasattr(widget, 'configure') and isinstance(widget, ttk.Entry):
                        widget.configure(state="disabled")
            except Exception:
                pass
        
        # Apply to inference GPUs
        for row in panel._cuda_run_rows:
            try:
                row["mem_pct"].set("100")
                row["util_pct"].set("0")
                # Disable the input fields
                for widget in row.get("widgets", []):
                    if hasattr(widget, 'configure') and isinstance(widget, ttk.Entry):
                        widget.configure(state="disabled")
            except Exception:
                pass
    else:
        logger.info("Max Performance mode disabled - restoring default values")
        # Restore to default values
        default_mem = panel.gpu_mem_pct_var.get() or "90"
        default_util = panel.gpu_util_pct_var.get() or "0"
        
        # Apply to training GPUs
        for row in panel._cuda_train_rows:
            try:
                row["mem_pct"].set(default_mem)
                row["util_pct"].set(default_util)
                # Enable the input fields
                for widget in row.get("widgets", []):
                    if hasattr(widget, 'configure') and isinstance(widget, ttk.Entry):
                        widget.configure(state="normal")
            except Exception:
                pass
        
        # Apply to inference GPUs
        for row in panel._cuda_run_rows:
            try:
                row["mem_pct"].set(default_mem)
                row["util_pct"].set(default_util)
                # Enable the input fields
                for widget in row.get("widgets", []):
                    if hasattr(widget, 'configure') and isinstance(widget, ttk.Entry):
                        widget.configure(state="normal")
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
    "build_storage_caps_ui",
    "build_apply_button_ui",
    "build_monitor_ui",
]
