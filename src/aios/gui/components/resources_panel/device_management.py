"""Device detection and GPU row management for resources panel."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .constants import tk, ttk

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel


__all__ = ["clear_cuda_rows", "build_cuda_rows", "update_training_mode_toggle_state", "set_detected", "refresh_detected"]


def clear_cuda_rows(panel: "ResourcesPanel", which: str) -> None:
    """Destroy dynamic CUDA row widgets for 'train' or 'run'.
    
    Args:
        panel: ResourcesPanel instance
        which: Either "train" or "run"
    """
    rows = panel._cuda_train_rows if which == "train" else panel._cuda_run_rows
    for row in rows:
        try:
            for w in row.get("widgets", []):
                try:
                    w.destroy()
                except Exception:
                    pass
        except Exception:
            pass
    rows.clear()


def build_cuda_rows(panel: "ResourcesPanel", devices: list[dict], which: str) -> None:
    """Build GPU selection rows for training or inference.
    
    Args:
        panel: ResourcesPanel instance
        devices: List of GPU device dicts with keys: id, name, total_mem_mb
        which: Either "train" or "run"
    """
    clear_cuda_rows(panel, which)
    if not isinstance(devices, list) or not devices:
        return
        
    for dev in devices:
        try:
            raw_id = dev.get("id")
            if raw_id is None:
                continue
            did = int(raw_id)
        except Exception:
            continue
            
        name = str(dev.get("name") or f"CUDA {did}")
        mem_mb = dev.get("total_mem_mb")
        
        # Row widgets
        container = panel.cuda_train_group if which == "train" else panel.cuda_run_group
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        
        en_var = tk.BooleanVar(value=True)
        pct_var = tk.StringVar(value=panel.gpu_mem_pct_var.get() or "90")
        util_var = tk.StringVar(value=panel.gpu_util_pct_var.get() or "0")
        
        cb = ttk.Checkbutton(row, text=f"GPU{did}: {name}", variable=en_var)
        cb.pack(side="left")
        
        ttk.Label(row, text="Mem %:").pack(side="left", padx=(6, 2))
        ent = ttk.Entry(row, width=4, textvariable=pct_var)
        ent.pack(side="left")
        
        ttk.Label(row, text=" Util %:").pack(side="left", padx=(6, 2))
        entu = ttk.Entry(row, width=4, textvariable=util_var)
        entu.pack(side="left")
        
        if isinstance(mem_mb, int):
            ttk.Label(row, text=f"({mem_mb} MB)").pack(side="left", padx=(6, 0))
            
        target = panel._cuda_train_rows if which == "train" else panel._cuda_run_rows
        target.append({
            "id": did,
            "name": name,
            "enabled": en_var,
            "mem_pct": pct_var,
            "util_pct": util_var,
            "widgets": [row, cb, ent, entu],
        })
        
        # Dynamic tooltips for each GPU row
        try:  # pragma: no cover
            from ..tooltips import add_tooltip
            add_tooltip(cb, f"Toggle use of GPU {did} for {('training' if which=='train' else 'inference')}.")
            add_tooltip(ent, "Allowed memory percentage for this GPU.")
            add_tooltip(entu, "Target utilization percentage (0 disables).")
        except Exception:
            pass
    
    # Apply pending GPU settings if they exist
    _apply_pending_gpu_settings(panel, which)
    
    # Update training mode toggle state based on GPU count
    if which == "train":
        update_training_mode_toggle_state(panel)


def _apply_pending_gpu_settings(panel: "ResourcesPanel", which: str) -> None:
    """Apply pending GPU settings to newly created rows.
    
    Args:
        panel: ResourcesPanel instance
        which: Either "train" or "run"
    """
    try:
        if which == "train" and "train_cuda_selected" in panel._pending_gpu_settings:
            ids = panel._pending_gpu_settings.get("train_cuda_selected", set())
            mem = panel._pending_gpu_settings.get("train_cuda_mem_pct", {})
            util = panel._pending_gpu_settings.get("train_cuda_util_pct", {})
            
            for row in panel._cuda_train_rows:
                try:
                    did = int(row.get("id"))
                    row["enabled"].set(did in ids)
                    if isinstance(mem, dict) and did in mem:
                        row["mem_pct"].set(str(int(mem[did])))
                    if isinstance(util, dict) and did in util:
                        row["util_pct"].set(str(int(util[did])))
                except Exception:
                    continue
                    
            # Clear applied settings
            panel._pending_gpu_settings.pop("train_cuda_selected", None)
            panel._pending_gpu_settings.pop("train_cuda_mem_pct", None)
            panel._pending_gpu_settings.pop("train_cuda_util_pct", None)
            
        elif which == "run" and "run_cuda_selected" in panel._pending_gpu_settings:
            ids = panel._pending_gpu_settings.get("run_cuda_selected", set())
            mem = panel._pending_gpu_settings.get("run_cuda_mem_pct", {})
            util = panel._pending_gpu_settings.get("run_cuda_util_pct", {})
            
            for row in panel._cuda_run_rows:
                try:
                    did = int(row.get("id"))
                    row["enabled"].set(did in ids)
                    if isinstance(mem, dict) and did in mem:
                        row["mem_pct"].set(str(int(mem[did])))
                    if isinstance(util, dict) and did in util:
                        row["util_pct"].set(str(int(util[did])))
                except Exception:
                    continue
                    
            # Clear applied settings
            panel._pending_gpu_settings.pop("run_cuda_selected", None)
            panel._pending_gpu_settings.pop("run_cuda_mem_pct", None)
            panel._pending_gpu_settings.pop("run_cuda_util_pct", None)
    except Exception:
        pass


def update_training_mode_toggle_state(panel: "ResourcesPanel") -> None:
    """Update training mode toggle state based on GPU count.
    
    Enables toggle only when multiple GPUs detected, unless on Windows (always parallel).
    
    Args:
        panel: ResourcesPanel instance
    """
    try:
        widgets = getattr(panel, "_training_mode_toggle_widgets", None)
        if not widgets:
            return
        
        # Count detected training GPUs
        gpu_count = len(getattr(panel, "_cuda_train_rows", []))
        has_multi_gpu = gpu_count > 1
        
        # Determine if toggle should be enabled
        if panel.is_windows:
            # Windows: Always locked to Parallel
            state = "disabled"
            lock_text = "🔒 (Windows: Parallel only)"
            tooltip = "Windows DDP is broken in PyTorch. Parallel Independent Training is automatically used."
        elif has_multi_gpu:
            # Linux with multiple GPUs: Enable toggle
            state = "normal"
            lock_text = ""
            tooltip = "DDP: Standard distributed training (Linux only)\nParallel: Independent block training (Windows-compatible)"
        else:
            # No multiple GPUs: Grey out toggle
            state = "disabled"
            lock_text = "(Requires 2+ GPUs)"
            tooltip = "Multi-GPU training mode selection. Enable when 2 or more GPUs are detected."
        
        # Update button states
        widgets["ddp_btn"].config(state=state)
        widgets["parallel_btn"].config(state=state)
        
        # Update lock label
        widgets["lock_label"].config(text=lock_text)
        
        # Update tooltips
        try:
            from ..tooltips import add_tooltip
            add_tooltip(widgets["label"], tooltip)
            if lock_text:
                add_tooltip(widgets["lock_label"], tooltip)
        except Exception:
            pass
            
    except Exception:
        pass


def set_detected(panel: "ResourcesPanel", info: dict) -> None:
    """Update device availability based on detection info.
    
    Called on initial startup to create GPU rows.
    Settings will be applied from pending state if available.

    Args:
        panel: ResourcesPanel instance
        info: Detection dict with keys:
            - cuda_available, xpu_available, mps_available, directml_available (bool)
            - directml_python (str)
            - cuda_devices: list of {id,name,total_mem_mb}
            - nvidia_smi_devices: list of {id,name,total_mem_mb}
    """
    # Update status
    try:
        panel._status_label.config(text="Detecting devices...")
        panel._status_label.update()
    except Exception:
        pass
    
    try:
        nvsmi_list = info.get("nvidia_smi_devices")
        nvsmi_has = isinstance(nvsmi_list, list) and len(nvsmi_list) > 0
        cuda_ok = bool(info.get("cuda_available")) or nvsmi_has
    except Exception:
        cuda_ok = False
    
    # Build CUDA device rows for train and run
    device_count = 0
    try:
        devs = info.get("cuda_devices")
        if not devs or not isinstance(devs, list):
            devs = info.get("nvidia_smi_devices") or []
        if isinstance(devs, list):
            device_count = len(devs)
            build_cuda_rows(panel, devs, "train")
            build_cuda_rows(panel, devs, "run")
    except Exception:
        build_cuda_rows(panel, [], "train")
        build_cuda_rows(panel, [], "run")
    
    # Update status
    try:
        if device_count > 0:
            panel._status_label.config(
                text=f"✓ {device_count} GPU(s) detected",
                foreground="green"
            )
        else:
            panel._status_label.config(
                text="No GPUs detected (CPU mode)",
                foreground="gray"
            )
    except Exception:
        pass


def refresh_detected(panel: "ResourcesPanel", info: dict) -> None:
    """Refresh device detection without resetting settings.
    
    Called periodically to update device info,
    but preserves user settings and GPU row configurations.

    Args:
        panel: ResourcesPanel instance
        info: Detection dict (same format as set_detected)
    """
    # For now, just update the status without rebuilding rows
    # Future: smart refresh that updates device info without destroying rows
    try:
        devs = info.get("cuda_devices") or info.get("nvidia_smi_devices") or []
        device_count = len(devs) if isinstance(devs, list) else 0
        if device_count > 0:
            panel._status_label.config(
                text=f"✓ {device_count} GPU(s) detected (refreshed)",
                foreground="green"
            )
    except Exception:
        pass


__all__ = [
    "clear_cuda_rows",
    "build_cuda_rows",
    "set_detected",
    "refresh_detected",
]
