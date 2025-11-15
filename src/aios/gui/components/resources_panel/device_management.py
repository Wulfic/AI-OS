"""Device detection and GPU row management for resources panel."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, TYPE_CHECKING

from .constants import tk, ttk, safe_variables

if TYPE_CHECKING:
    from .panel_main import ResourcesPanel

logger = logging.getLogger(__name__)


__all__ = [
    "clear_cuda_rows",
    "build_cuda_rows",
    "update_training_mode_toggle_state",
    "set_detected",
    "refresh_detected",
]


def _snapshot_devices(info: dict) -> tuple:
    """Return a stable snapshot of detected devices for change detection."""

    try:
        devices = info.get("cuda_devices") or info.get("nvidia_smi_devices") or []
        snapshot = []
        for dev in devices if isinstance(devices, list) else []:
            try:
                dev_id = int(dev.get("id"))
            except Exception:
                continue
            snapshot.append(
                (
                    dev_id,
                    str(dev.get("name", "")),
                    int(dev.get("total_mem_mb") or 0),
                )
            )
        snapshot.sort(key=lambda item: item[0])
        return (
            bool(info.get("cuda_available")),
            tuple(snapshot),
        )
    except Exception:
        return (False, tuple())


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
    
    # Reset grid position counters
    if which == "train":
        panel._cuda_train_grid_col = 0
        panel._cuda_train_grid_row = 0
    else:
        panel._cuda_run_grid_col = 0
        panel._cuda_run_grid_row = 0


def build_cuda_rows(panel: "ResourcesPanel", devices: list[dict], which: str) -> None:
    """Build GPU selection rows for training or inference without freezing UI."""
    logger.debug(f"Building CUDA rows for {which} mode with {len(devices) if devices else 0} devices")
    clear_cuda_rows(panel, which)
    if not isinstance(devices, list) or not devices:
        logger.info(f"No CUDA devices to display for {which} mode")
        return

    queue = deque()
    for dev in devices:
        try:
            raw_id = dev.get("id")
            if raw_id is None:
                continue
            did = int(raw_id)
            queue.append((did, dev))
        except Exception as e:
            logger.warning(f"Invalid device ID in CUDA row: {e}")

    if not queue:
        logger.info(f"No valid CUDA devices to display for {which} mode")
        return

    batch_size = 2  # Two devices per iteration keeps UI responsive even with many GPUs
    start = time.perf_counter()

    def _create_row(device_id: int, dev_data: dict) -> None:
        name = str(dev_data.get("name") or f"CUDA {device_id}")
        mem_mb = dev_data.get("total_mem_mb")

        container = panel.cuda_train_group if which == "train" else panel.cuda_run_group

        if which == "train":
            col = panel._cuda_train_grid_col
            row = panel._cuda_train_grid_row
            panel._cuda_train_grid_col += 1
            if panel._cuda_train_grid_col >= 4:
                panel._cuda_train_grid_col = 0
                panel._cuda_train_grid_row += 1
        else:
            col = panel._cuda_run_grid_col
            row = panel._cuda_run_grid_row
            panel._cuda_run_grid_col += 1
            if panel._cuda_run_grid_col >= 4:
                panel._cuda_run_grid_col = 0
                panel._cuda_run_grid_row += 1

        card = ttk.Frame(container, relief="solid", borderwidth=1)
        card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        en_var = safe_variables.BooleanVar(value=True)
        pct_var = safe_variables.StringVar(value=panel.gpu_mem_pct_var.get() or "90")
        util_var = safe_variables.StringVar(value=panel.gpu_util_pct_var.get() or "0")

        row1 = ttk.Frame(card)
        row1.pack(fill="x", padx=4, pady=(4, 2))

        cb = ttk.Checkbutton(row1, text=f"GPU{device_id}", variable=en_var)
        cb.pack(side="left")

        name_label = ttk.Label(row1, text=name, font=("TkDefaultFont", 8))
        name_label.pack(side="left", padx=(4, 0))

        row2 = ttk.Frame(card)
        row2.pack(fill="x", padx=4, pady=(2, 4))

        ttk.Label(row2, text="Mem %:", font=("TkDefaultFont", 8)).pack(side="left")
        ent = ttk.Entry(row2, width=5, textvariable=pct_var)
        ent.pack(side="left", padx=(2, 8))

        ttk.Label(row2, text="Util %:", font=("TkDefaultFont", 8)).pack(side="left")
        entu = ttk.Entry(row2, width=5, textvariable=util_var)
        entu.pack(side="left", padx=(2, 8))

        if isinstance(mem_mb, int):
            ttk.Label(row2, text=f"({mem_mb} MB)", font=("TkDefaultFont", 8), foreground="gray").pack(side="left")

        def _on_toggle(*args):
            if en_var.get():
                row2.pack(fill="x", padx=4, pady=(2, 4))
            else:
                row2.pack_forget()
            try:
                panel._on_apply_all()
            except Exception:
                pass

        def _on_gpu_setting_change(*args):
            try:
                panel._on_apply_all()
            except Exception:
                pass

        en_var.trace_add("write", _on_toggle)
        pct_var.trace_add("write", _on_gpu_setting_change)
        util_var.trace_add("write", _on_gpu_setting_change)

        target = panel._cuda_train_rows if which == "train" else panel._cuda_run_rows
        target.append({
            "id": device_id,
            "name": name,
            "enabled": en_var,
            "mem_pct": pct_var,
            "util_pct": util_var,
            "widgets": [card, cb, ent, entu, row2],
        })

        try:  # pragma: no cover
            from ..tooltips import add_tooltip
            add_tooltip(cb, f"Toggle use of GPU {device_id} for {('training' if which=='train' else 'inference')}.")
            add_tooltip(ent, "Allowed memory percentage for this GPU.")
            add_tooltip(entu, "Target utilization percentage (0 disables).")
        except Exception:
            pass

    def _process_queue() -> None:
        processed = 0
        while queue and processed < batch_size:
            did, data = queue.popleft()
            logger.debug(f"Creating {which} row for GPU{did}: {data.get('name', 'Unknown')} ({data.get('total_mem_mb')}MB)")
            _create_row(did, data)
            processed += 1

        if queue:
            try:
                panel.after(1, _process_queue)
            except Exception:
                pass
        else:
            elapsed = time.perf_counter() - start
            total = len(panel._cuda_train_rows if which == 'train' else panel._cuda_run_rows)
            logger.info(f"Built {total} CUDA rows for {which} mode in {elapsed:.3f}s")
            _apply_pending_gpu_settings(panel, which)
            if which == "train":
                update_training_mode_toggle_state(panel)

    try:
        panel.after_idle(_process_queue)
    except Exception:
        _process_queue()


def _apply_pending_gpu_settings(panel: "ResourcesPanel", which: str) -> None:
    """Apply pending GPU settings to newly created rows.
    
    Args:
        panel: ResourcesPanel instance
        which: Either "train" or "run"
    """
    try:
        with panel.suspend_auto_apply():
            if which == "train" and "train_cuda_selected" in panel._pending_gpu_settings:
                ids = panel._pending_gpu_settings.get("train_cuda_selected", set())
                mem = panel._pending_gpu_settings.get("train_cuda_mem_pct", {})
                util = panel._pending_gpu_settings.get("train_cuda_util_pct", {})
                
                logger.debug(f"Applying pending GPU settings for training: {len(ids)} selected GPUs")
                
                for row in panel._cuda_train_rows:
                    try:
                        did = int(row.get("id"))
                        row["enabled"].set(did in ids)
                        if isinstance(mem, dict) and did in mem:
                            row["mem_pct"].set(str(int(mem[did])))
                        if isinstance(util, dict) and did in util:
                            row["util_pct"].set(str(int(util[did])))
                    except Exception as e:
                        logger.warning(f"Failed to apply settings to GPU row: {e}")
                        continue
                        
                # Clear applied settings
                panel._pending_gpu_settings.pop("train_cuda_selected", None)
                panel._pending_gpu_settings.pop("train_cuda_mem_pct", None)
                panel._pending_gpu_settings.pop("train_cuda_util_pct", None)
                logger.info(f"Applied pending training GPU settings to {len(panel._cuda_train_rows)} rows")
                
            elif which == "run" and "run_cuda_selected" in panel._pending_gpu_settings:
                ids = panel._pending_gpu_settings.get("run_cuda_selected", set())
                mem = panel._pending_gpu_settings.get("run_cuda_mem_pct", {})
                util = panel._pending_gpu_settings.get("run_cuda_util_pct", {})
                
                logger.debug(f"Applying pending GPU settings for inference: {len(ids)} selected GPUs")
                
                for row in panel._cuda_run_rows:
                    try:
                        did = int(row.get("id"))
                        row["enabled"].set(did in ids)
                        if isinstance(mem, dict) and did in mem:
                            row["mem_pct"].set(str(int(mem[did])))
                        if isinstance(util, dict) and did in util:
                            row["util_pct"].set(str(int(util[did])))
                    except Exception as e:
                        logger.warning(f"Failed to apply settings to GPU row: {e}")
                        continue
                        
                # Clear applied settings
                panel._pending_gpu_settings.pop("run_cuda_selected", None)
                panel._pending_gpu_settings.pop("run_cuda_mem_pct", None)
                panel._pending_gpu_settings.pop("run_cuda_util_pct", None)
                logger.info(f"Applied pending inference GPU settings to {len(panel._cuda_run_rows)} rows")
    except Exception as e:
        logger.error(f"Failed to apply pending GPU settings: {e}")
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
        
        logger.debug(f"Updating training mode toggle state: {gpu_count} GPUs, Windows={panel.is_windows}")
        
        # Determine if toggle should be enabled
        if panel.is_windows:
            # Windows: Always locked to Parallel
            state = "disabled"
            lock_text = "🔒 (Windows: Parallel only)"
            tooltip = "Windows DDP is broken in PyTorch. Parallel Independent Training is automatically used."
            logger.info("Training mode locked to Parallel on Windows (DDP not supported)")
        elif has_multi_gpu:
            # Linux with multiple GPUs: Enable toggle
            state = "normal"
            lock_text = ""
            tooltip = "DDP: Standard distributed training (Linux only)\nParallel: Independent block training (Windows-compatible)"
            logger.info(f"Training mode toggle enabled: {gpu_count} GPUs detected")
        else:
            # No multiple GPUs: Grey out toggle
            state = "disabled"
            lock_text = "(Requires 2+ GPUs)"
            tooltip = "Multi-GPU training mode selection. Enable when 2 or more GPUs are detected."
            logger.info("Training mode toggle disabled: insufficient GPUs")
        
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
            
    except Exception as e:
        logger.error(f"Failed to update training mode toggle state: {e}")
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
    # Detect duplicate payloads to avoid redundant rebuilds
    snapshot = _snapshot_devices(info if isinstance(info, dict) else {})
    try:
        last_snapshot = getattr(panel, "_last_detected_snapshot", None)
    except Exception:
        last_snapshot = None

    if snapshot == last_snapshot:
        logger.debug("Device detection unchanged; skipping GPU row rebuild")
        return

    try:
        panel._last_detected_snapshot = snapshot
    except Exception:
        pass

    # Update status without forcing a synchronous UI refresh (avoids blocking)
    try:
        panel._status_label.config(text="Detecting devices...")
        try:
            panel._status_label.after_idle(panel._status_label.update_idletasks)
        except Exception:
            pass
    except Exception:
        pass
    
    try:
        nvsmi_list = info.get("nvidia_smi_devices")
        nvsmi_has = isinstance(nvsmi_list, list) and len(nvsmi_list) > 0
        cuda_ok = bool(info.get("cuda_available")) or nvsmi_has
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")
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
            if device_count > 0:
                logger.info(f"Detected {device_count} GPU(s): {[d.get('name', 'Unknown') for d in devs]}")
            else:
                logger.info("No GPUs detected - CPU mode only")
    except Exception as e:
        error_context = "Failed to build GPU device rows"
        
        # Provide helpful suggestions
        if "cuda" in str(e).lower() or "nvidia" in str(e).lower():
            suggestion = "CUDA/NVIDIA driver error. Update GPU drivers or reinstall CUDA toolkit"
        elif "permission" in str(e).lower():
            suggestion = "Permission error accessing GPU. Run as administrator or check GPU permissions"
        else:
            suggestion = "GPU detection failed. Update GPU drivers and ensure hardware is properly connected"
        
        logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
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
            status_msg = "No GPUs detected (CPU mode)"
            suggestion = "If you have a GPU: Update drivers, check CUDA installation, or verify hardware is connected"
            panel._status_label.config(
                text=status_msg,
                foreground="gray"
            )
            logger.warning(f"{status_msg}. {suggestion}")
    except Exception as e:
        logger.error(f"Failed to update GPU detection status: {e}")


def refresh_detected(panel: "ResourcesPanel", info: dict) -> None:
    """Refresh device detection without resetting settings.
    
    Called periodically to update device info,
    but preserves user settings and GPU row configurations.

    Args:
        panel: ResourcesPanel instance
        info: Detection dict (same format as set_detected)
    """
    payload = info if isinstance(info, dict) else {}
    snapshot = _snapshot_devices(payload)
    last_snapshot = getattr(panel, "_last_detected_snapshot", None)

    devs = payload.get("cuda_devices") or payload.get("nvidia_smi_devices") or []
    device_count = len(devs) if isinstance(devs, list) else 0
    current_train = len(getattr(panel, "_cuda_train_rows", []) or [])
    current_run = len(getattr(panel, "_cuda_run_rows", []) or [])

    rebuild_needed = (
        snapshot != last_snapshot
        or device_count != current_train
        or device_count != current_run
        or (device_count > 0 and current_train == 0 and current_run == 0)
    )

    if rebuild_needed:
        try:
            values = panel.get_values()
        except Exception:
            values = {}

        def _stash(prefix: str, selected_key: str, mem_key: str, util_key: str) -> None:
            sel_raw = values.get(selected_key)
            mem_raw = values.get(mem_key)
            util_raw = values.get(util_key)

            selected = set()
            if isinstance(sel_raw, (list, tuple, set)):
                for item in sel_raw:
                    try:
                        selected.add(int(item))
                    except Exception:
                        continue

            mem: dict[int, int] = {}
            if isinstance(mem_raw, dict):
                for key, value in mem_raw.items():
                    try:
                        mem[int(key)] = int(value)
                    except Exception:
                        continue

            util: dict[int, int] = {}
            if isinstance(util_raw, dict):
                for key, value in util_raw.items():
                    try:
                        util[int(key)] = int(value)
                    except Exception:
                        continue

            target_prefix = f"{prefix}_cuda_"
            panel._pending_gpu_settings[f"{target_prefix}selected"] = selected
            panel._pending_gpu_settings[f"{target_prefix}mem_pct"] = mem
            panel._pending_gpu_settings[f"{target_prefix}util_pct"] = util

        _stash("train", "train_cuda_selected", "train_cuda_mem_pct", "train_cuda_util_pct")
        _stash("run", "run_cuda_selected", "run_cuda_mem_pct", "run_cuda_util_pct")

        try:
            panel._last_detected_snapshot = None
        except Exception:
            pass

        set_detected(panel, payload)
        return

    try:
        panel._last_detected_snapshot = snapshot
    except Exception:
        pass

    try:
        if device_count > 0:
            panel._status_label.config(
                text=f"✓ {device_count} GPU(s) detected (refreshed)",
                foreground="green",
            )
    except Exception:
        pass


__all__ = [
    "clear_cuda_rows",
    "build_cuda_rows",
    "set_detected",
    "refresh_detected",
]
