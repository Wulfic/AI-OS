"""Main ResourcesPanel class - orchestrates UI building and state management."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .constants import tk, ttk
from . import caps_manager
from . import device_management
from . import monitoring
from . import ui_builders


class ResourcesPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Resource limits and monitoring panel for AI-OS GUI."""
    
    def __init__(
        self,
        parent: Any,
        *,
        cores: int,
        detect_fn: Any | None = None,
        apply_caps_fn: Optional[Callable[[float, Optional[float], Optional[float]], dict]] = None,
        fetch_caps_fn: Optional[Callable[[], dict]] = None,
        save_state_fn: Optional[Callable[[], None]] = None,
        root: Any | None = None,
    ) -> None:
        """Initialize ResourcesPanel.
        
        Args:
            parent: Parent widget
            cores: Number of CPU cores
            detect_fn: Callback for device detection (returns dict with cuda_available, etc.)
            apply_caps_fn: Callback to apply storage caps
            fetch_caps_fn: Callback to fetch current caps
            save_state_fn: Callback to save panel state
            root: Root widget for scheduling updates
        """
        super().__init__(parent, text="Resource Limits")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="x", padx=8, pady=8)

        # External callbacks
        self._detect_fn = detect_fn
        self._apply_caps_fn = apply_caps_fn
        self._fetch_caps_fn = fetch_caps_fn
        self._save_state_fn = save_state_fn
        self._root = root
        self._monitor_after_id = None
        
        # Initialize chart references early (before build_monitor_ui)
        self._charts: dict[str, Any] = {}
        self._charts_container: Any = None

        # State variables
        self.cpu_threads_var = tk.StringVar(value=str(max(1, cores // 2)))
        self.gpu_mem_pct_var = tk.StringVar(value="90")
        self.system_mem_limit_gb_var = tk.StringVar(value="0")  # System RAM limit in GB (0 = system limit)
        self.cpu_util_pct_var = tk.StringVar(value="0")
        self.gpu_util_pct_var = tk.StringVar(value="0")
        self.train_device_var = tk.StringVar(value="auto")  # auto|cpu|cuda
        self.run_device_var = tk.StringVar(value="auto")    # auto|cpu|cuda
        
        # Training mode: "ddp" or "parallel" (Windows locked to parallel)
        import platform
        default_mode = "parallel" if platform.system() == "Windows" else "ddp"
        self.training_mode_var = tk.StringVar(value=default_mode)
        self.is_windows = platform.system() == "Windows"
        
        # Dynamic CUDA rows
        self._cuda_train_rows = []  # list of {id, name, enabled_var, mem_var, util_var, row_widgets}
        self._cuda_run_rows = []    # list of {id, name, enabled_var, mem_var, util_var, row_widgets}
        self._pending_gpu_settings: dict[str, Any] = {}
        
        # Storage caps variables
        self.dataset_cap_var = tk.StringVar(value="")
        self.model_cap_var = tk.StringVar(value="")
        self.per_brain_cap_var = tk.StringVar(value="")
        
        # Build UI sections
        ui_builders.build_limits_ui(self)
        ui_builders.build_devices_ui(self)
        ui_builders.build_status_ui(self)
        ui_builders.build_detect_button_ui(self)
        ui_builders.build_storage_caps_ui(self)
        ui_builders.build_apply_button_ui(self)
        ui_builders.build_monitor_ui(self)
        
        # Initialize training mode toggle state (disabled until GPUs detected)
        from . import device_management
        device_management.update_training_mode_toggle_state(self)
        
        # Start monitoring if root is available
        # IMPORTANT: Defer first update to avoid blocking GUI startup with nvidia-smi call
        if self._root is not None:
            monitoring.init_monitoring_data(self)
            # Schedule first update after a delay to avoid blocking startup
            self._root.after(500, lambda: monitoring.schedule_monitor_update(self))
            # Also schedule initial storage usage update
            self._root.after(1000, lambda: monitoring.update_storage_usage(self))

    # Button handlers
    
    def _on_apply_all(self) -> None:
        """Persist current Resources selections to config/default.yaml (source of truth)."""
        try:
            # Get current panel values
            values = self.get_values()
            
            # Save to config file (source of truth)
            from . import config_persistence
            success = config_persistence.save_resources_to_config(values)
            if success:
                # Also trigger GUI state save for backward compatibility
                if callable(self._save_state_fn):
                    self._save_state_fn()  # type: ignore[misc]
        except Exception as e:
            print(f"[Resources] Error saving settings: {e}")

    def _on_apply_caps(self) -> None:
        """Apply storage caps via callback."""
        caps_manager.on_apply_caps(self, self._apply_caps_fn)

    def _on_refresh_caps(self) -> None:
        """Refresh storage caps from config via callback."""
        caps_manager.on_refresh_caps(self, self._fetch_caps_fn)

    # Storage caps methods
    
    def set_caps(self, caps: dict) -> None:
        """Populate storage caps inputs from a dict.

        Args:
            caps: Dict with keys dataset_cap_gb, model_cap_gb, per_brain_cap_gb
        """
        caps_manager.set_caps(self, caps)

    # Device detection methods
    
    def _detect_and_update(self) -> None:
        """Trigger device detection and update UI."""
        try:
            if self._detect_fn is None:
                return
            info = self._detect_fn()
            if not isinstance(info, dict):
                return
            self.set_detected(info)
        except Exception:
            pass

    def set_detected(self, info: dict) -> None:
        """Update availability toggles based on detection info.
        
        Args:
            info: Detection dict with keys cuda_available, cuda_devices, etc.
        """
        device_management.set_detected(self, info)

    def refresh_detected(self, info: dict) -> None:
        """Refresh device detection without resetting settings.
        
        Args:
            info: Detection dict (same format as set_detected)
        """
        device_management.refresh_detected(self, info)
    
    # State persistence methods
    
    def get_values(self) -> dict[str, Any]:
        """Get current panel values for persistence.
        
        Returns:
            Dict with all panel settings
        """
        try:
            th = int(self.cpu_threads_var.get() or 0)
        except Exception:
            th = 0
        try:
            gp = int(self.gpu_mem_pct_var.get() or 90)
        except Exception:
            gp = 90
        try:
            cpuu = int(self.cpu_util_pct_var.get() or 0)
        except Exception:
            cpuu = 0
        try:
            gpuu = int(self.gpu_util_pct_var.get() or 0)
        except Exception:
            gpuu = 0
            
        # CUDA selections (train)
        cuda_train_selected: list[int] = []
        cuda_train_mem_pct: dict[int, int] = {}
        cuda_train_util_pct: dict[int, int] = {}
        print(f"[Resources DEBUG] Reading train GPU rows: {len(self._cuda_train_rows)} rows")
        for row in self._cuda_train_rows:
            try:
                did = int(row["id"])  # type: ignore[index]
                enabled = bool(row["enabled"].get())
                print(f"[Resources DEBUG] GPU {did}: enabled={enabled}")
                if enabled:
                    cuda_train_selected.append(did)
                    try:
                        cuda_train_mem_pct[did] = int(str(row["mem_pct"].get() or gp))  # type: ignore[index]
                    except Exception:
                        cuda_train_mem_pct[did] = gp
                    try:
                        cuda_train_util_pct[did] = int(str(row["util_pct"].get() or 0))  # type: ignore[index]
                    except Exception:
                        cuda_train_util_pct[did] = 0
            except Exception as e:
                print(f"[Resources DEBUG] Error reading GPU row: {e}")
                continue
        print(f"[Resources DEBUG] Final train_cuda_selected: {cuda_train_selected}")
                
        # CUDA selections (run)
        cuda_run_selected: list[int] = []
        cuda_run_mem_pct: dict[int, int] = {}
        cuda_run_util_pct: dict[int, int] = {}
        for row in self._cuda_run_rows:
            try:
                if bool(row["enabled"].get()):
                    did = int(row["id"])  # type: ignore[index]
                    cuda_run_selected.append(did)
                    try:
                        cuda_run_mem_pct[did] = int(str(row["mem_pct"].get() or gp))  # type: ignore[index]
                    except Exception:
                        cuda_run_mem_pct[did] = gp
                    try:
                        cuda_run_util_pct[did] = int(str(row["util_pct"].get() or 0))  # type: ignore[index]
                    except Exception:
                        cuda_run_util_pct[did] = 0
            except Exception:
                continue
                
        vals: dict[str, Any] = {
            "cpu_threads": th,
            "gpu_mem_pct": gp,
            "cpu_util_pct": cpuu,
            "gpu_util_pct": gpuu,
            # Train device selection
            "train_device": self.train_device_var.get(),
            "train_cuda_selected": cuda_train_selected,
            "train_cuda_mem_pct": cuda_train_mem_pct or gp,
            "train_cuda_util_pct": cuda_train_util_pct,
            # Run device selection
            "run_device": self.run_device_var.get(),
            "run_cuda_selected": cuda_run_selected,
            "run_cuda_mem_pct": cuda_run_mem_pct or gp,
            "run_cuda_util_pct": cuda_run_util_pct,
            # Training mode (DDP vs Parallel)
            "training_mode": self.training_mode_var.get(),
            # Storage caps
            "dataset_cap": self.dataset_cap_var.get(),
        }
        return vals

    def set_values(self, vals: dict) -> None:
        """Apply selections into UI from a dict returned by get_values().
        
        Args:
            vals: Dict of panel settings
        """
        try:
            v = int(vals.get("cpu_threads", 0))
            if v > 0:
                self.cpu_threads_var.set(str(v))
        except Exception:
            pass
        try:
            v = int(vals.get("gpu_mem_pct", 0))
            if v > 0:
                self.gpu_mem_pct_var.set(str(v))
        except Exception:
            pass
        try:
            v = int(vals.get("cpu_util_pct", 0))
            if v >= 0:
                self.cpu_util_pct_var.set(str(v))
        except Exception:
            pass
        try:
            v = int(vals.get("gpu_util_pct", 0))
            if v >= 0:
                self.gpu_util_pct_var.set(str(v))
        except Exception:
            pass
        try:
            td = vals.get("train_device")
            if isinstance(td, str) and td in {"auto", "cpu", "cuda"}:
                self.train_device_var.set(td)
        except Exception:
            pass
        try:
            rd = vals.get("run_device")
            if isinstance(rd, str) and rd in {"auto", "cpu", "cuda"}:
                self.run_device_var.set(rd)
        except Exception:
            pass
        
        # Training mode (DDP vs Parallel)
        try:
            tm = vals.get("training_mode")
            if isinstance(tm, str) and tm in {"ddp", "parallel"}:
                self.training_mode_var.set(tm)
        except Exception:
            pass
            
        # Apply CUDA selections (train)
        try:
            sel = vals.get("train_cuda_selected") or []
            if isinstance(sel, list):
                ids = {int(i) for i in sel if isinstance(i, (int, str)) and str(i).isdigit()}
            else:
                ids = set()
            mem_raw = vals.get("train_cuda_mem_pct") or {}
            util_raw = vals.get("train_cuda_util_pct") or {}
            # Coerce possible string keys to ints
            mem = {}
            util = {}
            if isinstance(mem_raw, dict):
                for k, v in mem_raw.items():
                    try:
                        mem[int(k)] = int(v)
                    except Exception:
                        continue
            if isinstance(util_raw, dict):
                for k, v in util_raw.items():
                    try:
                        util[int(k)] = int(v)
                    except Exception:
                        continue
            # Store settings for later application if no GPU rows exist yet
            if not self._cuda_train_rows:
                self._pending_gpu_settings["train_cuda_selected"] = ids
                self._pending_gpu_settings["train_cuda_mem_pct"] = mem
                self._pending_gpu_settings["train_cuda_util_pct"] = util
            else:
                # Apply to existing rows
                for row in self._cuda_train_rows:
                    try:
                        did = int(row.get("id"))
                        row["enabled"].set(did in ids)
                        if isinstance(mem, dict) and did in mem:
                            row["mem_pct"].set(str(int(mem[did])))
                        if isinstance(util, dict) and did in util:
                            row["util_pct"].set(str(int(util[did])))
                    except Exception:
                        continue
        except Exception:
            pass
            
        # Apply CUDA selections (run)
        try:
            sel = vals.get("run_cuda_selected") or []
            if isinstance(sel, list):
                ids = {int(i) for i in sel if isinstance(i, (int, str)) and str(i).isdigit()}
            else:
                ids = set()
            mem_raw = vals.get("run_cuda_mem_pct") or {}
            util_raw = vals.get("run_cuda_util_pct") or {}
            mem = {}
            util = {}
            if isinstance(mem_raw, dict):
                for k, v in mem_raw.items():
                    try:
                        mem[int(k)] = int(v)
                    except Exception:
                        continue
            if isinstance(util_raw, dict):
                for k, v in util_raw.items():
                    try:
                        util[int(k)] = int(v)
                    except Exception:
                        continue
            # Store settings for later application if no GPU rows exist yet
            if not self._cuda_run_rows:
                self._pending_gpu_settings["run_cuda_selected"] = ids
                self._pending_gpu_settings["run_cuda_mem_pct"] = mem
                self._pending_gpu_settings["run_cuda_util_pct"] = util
            else:
                # Apply to existing rows
                for row in self._cuda_run_rows:
                    try:
                        did = int(row.get("id"))
                        row["enabled"].set(did in ids)
                        if isinstance(mem, dict) and did in mem:
                            row["mem_pct"].set(str(int(mem[did])))
                        if isinstance(util, dict) and did in util:
                            row["util_pct"].set(str(int(util[did])))
                    except Exception:
                        continue
        except Exception:
            pass
            
        # Storage caps
        try:
            ds = vals.get("dataset_cap")
            if ds and str(ds).strip():
                self.dataset_cap_var.set(str(ds))
        except Exception:
            pass


__all__ = ["ResourcesPanel"]
