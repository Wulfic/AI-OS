"""Main ResourcesPanel class - orchestrates UI building and state management."""

from __future__ import annotations

# Import safe variable wrappers
from ...utils import safe_variables

import logging
import os
import platform
import threading
import time
from concurrent.futures import CancelledError, TimeoutError, Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

try:  # pragma: no cover - GUI environment dependent
    from tkinter import filedialog
except Exception:  # pragma: no cover - GUI fallback
    filedialog = None

try:  # pragma: no cover - bootstrap guard
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None

from .constants import tk, ttk
from . import caps_manager
from . import device_management
from . import monitoring
from . import ui_builders
from ...utils.resource_management import submit_background

logger = logging.getLogger(__name__)


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
        worker_pool: Any | None = None,
        post_to_ui: Optional[Callable[[Callable[..., None]], None]] = None,
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
            worker_pool: Optional worker pool for running background jobs
        """
        super().__init__(parent, text="Resource Limits")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        # External callbacks
        self._detect_fn = detect_fn
        self._apply_caps_fn = apply_caps_fn
        self._fetch_caps_fn = fetch_caps_fn
        self._save_state_fn = save_state_fn
        # Store explicit root reference without shadowing Tk's internal _root() method
        self._tk_root = root
        if self._tk_root is None:
            try:
                self._tk_root = self.winfo_toplevel()
            except Exception:
                self._tk_root = None
        self._worker_pool = worker_pool
        self._post_to_ui = post_to_ui
        self._monitor_after_id = None
        self._caps_busy_count = 0
        self._detect_inflight = False
        self.detect_button = None
        self._last_detected_snapshot: dict | None = None
        self._last_caps_snapshot: tuple | None = None
        self._caps_refresh_after_id: Any | None = None
        self._last_caps_refresh_source: str | None = None
        self._last_caps_refresh_at: float = 0.0
        self._background_futures_lock = threading.Lock()
        self._background_futures: set[Future] = set()
        self._shutting_down = False
        
        # Initialize chart references early (before build_monitor_ui)
        self._charts: dict[str, Any] = {}
        self._charts_container: Any = None

        # Guard counter used to suppress auto-apply during bulk updates
        self._auto_apply_guard = 0

        # State variables
        self.cpu_threads_var = safe_variables.StringVar(value=str(max(1, cores // 2)))
        self.gpu_mem_pct_var = safe_variables.StringVar(value="90")
        self.system_mem_limit_gb_var = safe_variables.StringVar(value="0")  # System RAM limit in GB (0 = system limit)
        self.cpu_util_pct_var = safe_variables.StringVar(value="0")
        self.gpu_util_pct_var = safe_variables.StringVar(value="0")
        self.train_device_var = safe_variables.StringVar(value="auto")  # auto|cpu|cuda
        self.run_device_var = safe_variables.StringVar(value="auto")    # auto|cpu|cuda
        
        # Training mode: "ddp", "parallel", or "zero3" (Windows locked to parallel)
        default_mode = "parallel" if platform.system() == "Windows" else "ddp"
        self.training_mode_var = safe_variables.StringVar(value=default_mode)
        self.zero_stage_var = safe_variables.StringVar(value="none")
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux"
        self._active_os_label = platform.system().strip().lower()
        self._last_loaded_os: str | None = self._active_os_label
        self._last_selection_warning: str | None = None
        
        # Max Performance mode
        self.max_performance_var = safe_variables.BooleanVar(value=False)
        
        # Add trace to log training mode changes
        def _on_training_mode_change(*args):
            try:
                mode = self.training_mode_var.get()
                logger.info(f"Training mode toggled: {mode}")
                self._sync_zero_stage_with_mode()
                self._notify_zero_stage_consumers()
            except Exception:
                pass
        self.training_mode_var.trace_add("write", _on_training_mode_change)
        self.zero_stage_var.trace_add("write", lambda *args: self._on_zero_stage_change())
        
        # Dynamic CUDA rows
        self._cuda_train_rows = []  # list of {id, name, enabled_var, mem_var, util_var, row_widgets}
        self._cuda_run_rows = []    # list of {id, name, enabled_var, mem_var, util_var, row_widgets}
        self._pending_gpu_settings: dict[str, Any] = {}
        self._cuda_train_build_token: Any | None = None
        self._cuda_run_build_token: Any | None = None
        
        # Storage caps variables
        self.dataset_cap_var = safe_variables.StringVar(value="")
        self.model_cap_var = safe_variables.StringVar(value="")
        self.per_brain_cap_var = safe_variables.StringVar(value="")
        self.artifacts_dir_var = safe_variables.StringVar(value="")
        self._artifacts_status_var = safe_variables.StringVar(value="")
        self._artifacts_status_label: Any | None = None
        self._artifacts_default_dir = self._resolve_default_artifacts_dir()
        self._artifacts_dir_is_valid = True
        
        # Build UI sections
        ui_builders.build_limits_ui(self)
        ui_builders.build_devices_ui(self)
        ui_builders.build_status_ui(self)
        ui_builders.build_storage_caps_ui(self)
        ui_builders.build_apply_button_ui(self)
        ui_builders.build_monitor_ui(self)
        
        # Initialize training mode toggle state (disabled until GPUs detected)
        from . import device_management
        device_management.update_training_mode_toggle_state(self)
        self._sync_zero_stage_with_mode()
        
        # Setup auto-save on variable changes
        self._setup_autosave()

        # Track visibility transitions for diagnostics
        self._map_event_count = 0
        try:
            self.bind("<Map>", self._on_map_event, add="+")
        except Exception:
            logger.debug("Resources panel map binding failed", exc_info=True)
        
        # Start monitoring if root is available
        # IMPORTANT: Defer first update to avoid blocking GUI startup with nvidia-smi call
        if self._tk_root is not None:
            monitoring.init_monitoring_data(self)
            # Schedule first update AFTER loading screen is removed (5 seconds delay)
            # This prevents nvidia-smi from blocking the GUI during startup
            self._tk_root.after(5000, lambda: monitoring.schedule_monitor_update(self))
            # Also schedule initial storage usage update with delay (async-safe)
            monitoring.schedule_storage_update(self, delay_ms=6000)
    
    def _setup_autosave(self) -> None:
        """Setup auto-save callbacks for all input variables."""
        # Debounce timer to prevent save spam during initialization
        self._autosave_timer = None
        
        def _autosave(*args):
            """Auto-save when any setting changes (debounced)."""
            if getattr(self, "_auto_apply_guard", 0) > 0:
                logger.debug("Suppressing autosave scheduling (guard active)")
                return
            # Cancel previous timer if exists
            if self._autosave_timer and self._tk_root:
                try:
                    self._tk_root.after_cancel(self._autosave_timer)
                except Exception:
                    pass
            
            # Schedule save for 1 second from now
            def _do_save():
                try:
                    self._on_apply_all()
                except Exception as e:
                    logger.error(f"Auto-save error: {e}")
                finally:
                    self._autosave_timer = None
            
            # Use the explicit root reference if available, otherwise fall back to toplevel
            if self._tk_root:
                self._autosave_timer = self._tk_root.after(1000, _do_save)
            else:
                try:
                    self._autosave_timer = self.winfo_toplevel().after(1000, _do_save)
                except Exception as e:
                    logger.debug(f"Failed to schedule autosave: {e}")
        
        # Add traces to main variables
        self.cpu_threads_var.trace_add("write", _autosave)
        self.gpu_mem_pct_var.trace_add("write", _autosave)
        self.system_mem_limit_gb_var.trace_add("write", _autosave)
        self.cpu_util_pct_var.trace_add("write", _autosave)
        self.gpu_util_pct_var.trace_add("write", _autosave)
        self.train_device_var.trace_add("write", _autosave)
        self.run_device_var.trace_add("write", _autosave)
        self.training_mode_var.trace_add("write", _autosave)
        self.zero_stage_var.trace_add("write", _autosave)
        self.dataset_cap_var.trace_add("write", _autosave)
        self.max_performance_var.trace_add("write", _autosave)
        self.artifacts_dir_var.trace_add("write", _autosave)

    # ------------------------------------------------------------------
    # Artifacts path overrides
    # ------------------------------------------------------------------

    def _resolve_default_artifacts_dir(self) -> str:
        if system_paths is not None:
            try:
                return str(system_paths.get_artifacts_root())
            except Exception:
                logger.debug("Failed to resolve ProgramData artifacts root", exc_info=True)
        try:
            return str(Path(__file__).resolve().parents[5] / "artifacts")
        except Exception:
            return str(Path.cwd() / "artifacts")

    def _set_artifacts_status(self, message: str, color: str = "gray") -> None:
        try:
            self._artifacts_status_var.set(message)
            if self._artifacts_status_label is not None:
                self._artifacts_status_label.configure(foreground=color)
        except Exception:
            logger.debug("Failed to update artifacts status label", exc_info=True)

    def _probe_directory_writable(self, path: Path) -> str | None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f"Failed to create directory: {exc}"

        probe = path / f".aios-write-test-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
        except Exception as exc:
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass
            return f"Failed to write probe file: {exc}"

        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    def _apply_artifacts_override(self, path: Path | None) -> None:
        if system_paths is not None:
            try:
                system_paths.set_artifacts_root_override(path)
            except Exception:
                logger.debug("Failed to apply artifacts override", exc_info=True)

        if path is None:
            os.environ.pop("AIOS_ARTIFACTS_DIR", None)
        else:
            os.environ["AIOS_ARTIFACTS_DIR"] = str(path)

    def _validate_artifacts_dir(self, value: str | None = None, *, apply_override: bool = True) -> bool:
        raw_value = (value if value is not None else self.artifacts_dir_var.get() or "").strip()
        if not raw_value:
            self._artifacts_dir_is_valid = True
            self._set_artifacts_status(f"Using default path: {self._artifacts_default_dir}", "gray")
            if apply_override:
                self._apply_artifacts_override(None)
            return True

        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            self._artifacts_dir_is_valid = False
            self._set_artifacts_status("Enter an absolute path (e.g., D:\\AI-OS\\artifacts)", "#c0392b")
            return False

        if system_paths is not None:
            try:
                error = system_paths.test_directory_writable(candidate)
            except Exception as exc:
                error = str(exc)
        else:
            error = self._probe_directory_writable(candidate)

        if error:
            self._artifacts_dir_is_valid = False
            self._set_artifacts_status(error, "#c0392b")
            return False

        self._artifacts_dir_is_valid = True
        self._set_artifacts_status(f"Custom path OK: {candidate}", "#1d8348")
        if apply_override:
            self._apply_artifacts_override(candidate)
        return True

    def _browse_artifacts_dir(self) -> None:
        if filedialog is None:
            return

        initial = self.artifacts_dir_var.get().strip() or self._artifacts_default_dir
        try:
            selected = filedialog.askdirectory(initialdir=initial, title="Select artifacts directory")
        except Exception:
            logger.warning("Failed to open artifacts folder picker", exc_info=True)
            return

        if selected:
            self.artifacts_dir_var.set(selected)
            self._validate_artifacts_dir()

    def _reset_artifacts_dir(self) -> None:
        self.artifacts_dir_var.set("")
        self._validate_artifacts_dir()
        
        logger.info("Auto-save enabled for resource settings (debounced 1s)")

    @contextmanager
    def suspend_auto_apply(self):
        """Context manager to suppress autosave and auto-apply triggers."""
        self._auto_apply_guard += 1
        try:
            yield
        finally:
            self._auto_apply_guard = max(0, self._auto_apply_guard - 1)

    def _on_map_event(self, event=None) -> None:
        """Log when the resources panel becomes visible."""
        try:
            count = getattr(self, "_map_event_count", 0) + 1
            self._map_event_count = count
        except Exception:
            self._map_event_count = 1
            count = 1

        log_fn = logger.info if count == 1 else logger.debug
        try:
            log_fn("Resources panel mapped (count=%d)", count)
        except Exception:
            logger.debug("Failed to log resources panel map event", exc_info=True)

    def on_tab_activated(self) -> None:
        """Notebook activation hook for diagnostics."""
        try:
            detect_inflight = getattr(self, "_detect_inflight", False)
            monitor_after = getattr(self, "_monitor_after_id", None)
            monitor_queue = getattr(self, "_monitor_queue", None)
            queue_depth: str | int = "n/a"
            if monitor_queue is not None and hasattr(monitor_queue, "qsize"):
                try:
                    queue_depth = monitor_queue.qsize()
                except Exception:
                    queue_depth = "error"

            storage_queue = getattr(self, "_storage_usage_queue", None)
            storage_depth: str | int = "n/a"
            if storage_queue is not None and hasattr(storage_queue, "qsize"):
                try:
                    storage_depth = storage_queue.qsize()
                except Exception:
                    storage_depth = "error"

            logger.info(
                "Resources panel activated (detect_inflight=%s monitor_after_id=%s monitor_queue=%s storage_queue=%s)",
                detect_inflight,
                monitor_after if monitor_after else "none",
                queue_depth,
                storage_depth,
            )
        except Exception:
            logger.exception("Resources panel activation diagnostics failed")

    # Button handlers
    
    def _set_status(self, message: str) -> None:
        try:
            if hasattr(self, "_status_label") and self._status_label is not None:
                self._status_label.config(text=message)
        except Exception:
            pass

    def _run_in_executor(self, label: str, work: Callable[[], None]) -> bool:
        try:
            future = submit_background(label, work, pool=self._worker_pool)
            self._register_future(future)
            return True
        except RuntimeError as exc:
            logger.error("Failed to queue resources task '%s': %s", label, exc)
            return False

    def _register_future(self, future: Future | None) -> None:
        if future is None:
            return

        try:
            with self._background_futures_lock:
                self._background_futures.add(future)
        except Exception:
            return

        def _on_done(done_future: Future) -> None:
            with self._background_futures_lock:
                self._background_futures.discard(done_future)

        try:
            future.add_done_callback(_on_done)
        except Exception:
            with self._background_futures_lock:
                self._background_futures.discard(future)

    def _cancel_background_futures(self, timeout: float = 0.5) -> None:
        try:
            with self._background_futures_lock:
                pending = list(self._background_futures)
        except Exception:
            pending = []

        if not pending:
            return

        deadline = time.monotonic() + timeout if timeout else None

        for future in pending:
            try:
                if future is None:
                    continue
                if not future.done():
                    future.cancel()
                remaining = None
                if deadline is not None:
                    remaining = max(0.0, deadline - time.monotonic())
                future.result(timeout=remaining)
            except (CancelledError, TimeoutError):
                continue
            except Exception as exc:
                logger.debug("Background future raised during cleanup: %s", exc)

        with self._background_futures_lock:
            self._background_futures.difference_update(pending)

    def _on_apply_all(self) -> None:
        """Persist current Resources selections to config/default.yaml (source of truth)."""
        if getattr(self, "_auto_apply_guard", 0) > 0:
            logger.debug("Skipping resource auto-apply (guard active)")
            return
        if getattr(self, "_save_in_progress", False):
            logger.debug("Resource settings save already in progress; skipping new request")
            return

        try:
            values = self.get_values()
        except Exception as exc:
            logger.error(f"Failed to collect resource settings for save: {exc}", exc_info=True)


        logger.info("User action: Applying resource settings (devices, memory limits, training config)")
        logger.debug("Resource config: %s", values)

        self._save_in_progress = True
        self._set_status("Saving resource settings…")

        def _work() -> None:
            success = False
            error: Exception | None = None
            try:
                from . import config_persistence

                success = config_persistence.save_resources_to_config(values)
            except Exception as err:
                error = err

            def _finish() -> None:
                self._save_in_progress = False
                if error is not None:
                    logger.error("Error saving resource settings: %s", error, exc_info=True)
                    self._set_status("Failed to save settings")
                    return

                if success:
                    logger.info("Resource settings saved successfully to config/default.yaml")
                    self._set_status("Settings saved")
                    if callable(self._save_state_fn):
                        try:
                            self._save_state_fn()  # type: ignore[misc]
                        except Exception as save_err:
                            logger.debug("State save callback failed: %s", save_err, exc_info=True)
                    try:
                        self.after(2500, lambda: self._set_status(""))
                    except Exception:
                        pass
                else:
                    logger.warning("Failed to save resource settings to config")
                    self._set_status("Save failed")

            try:
                self.after(0, _finish)
            except Exception:
                _finish()

        if not self._run_in_executor("resources-save", _work):
            self._save_in_progress = False
            self._set_status("Failed to queue save")
    def _on_apply_caps(self) -> None:
        """Apply storage caps via callback."""
        caps_manager.on_apply_caps(self, self._apply_caps_fn)

    def schedule_caps_refresh(self, *, delay_ms: int = 750, source: str = "user") -> None:
        """Schedule an asynchronous caps refresh with throttling."""
        root = self._tk_root or self.winfo_toplevel()

        if root is None:
            self._on_refresh_caps(source=source)
            return

        if self._caps_refresh_after_id is not None:
            try:
                root.after_cancel(self._caps_refresh_after_id)
            except Exception:
                pass

        def _dispatch() -> None:
            self._caps_refresh_after_id = None
            self._on_refresh_caps(source=source)

        try:
            self._caps_refresh_after_id = root.after(delay_ms, _dispatch)
        except Exception:
            _dispatch()

    def _on_refresh_caps(self, *, source: str = "user") -> None:
        """Refresh storage caps from config via callback."""
        now = time.perf_counter()
        last_source = getattr(self, "_last_caps_refresh_source", None)
        last_at = getattr(self, "_last_caps_refresh_at", 0.0)

        if source != "user" and last_source == source and last_at and (now - last_at) < 10.0:
            logger.debug(
                "Skipping '%s' caps refresh; last run %.3fs ago",
                source,
                now - last_at,
            )
            return

        if source == "user":
            logger.info("User action: Refreshing GPU capabilities")
        else:
            logger.info("Refreshing GPU capabilities (%s)", source)

        self._last_caps_refresh_source = source
        self._last_caps_refresh_at = now
        caps_manager.on_refresh_caps(self, self._fetch_caps_fn, source=source)

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
        if self._detect_fn is None:
            logger.debug("Detect devices requested but no callback configured")
            return
        if self._detect_inflight:
            logger.debug("Device detection already in progress; ignoring duplicate request")
            return

        self._detect_inflight = True
        self._set_status("Detecting devices…")
        detect_button = getattr(self, "detect_button", None)
        if detect_button is not None:
            try:
                detect_button.config(state="disabled")
            except Exception:
                pass

        def _work() -> None:
            try:
                info = self._detect_fn()
            except Exception as exc:
                logger.error(f"Device detection failed: {exc}", exc_info=True)

                def _on_error() -> None:
                    self._set_status("Device detection failed")
                    if detect_button is not None:
                        try:
                            detect_button.config(state="normal")
                        except Exception:
                            pass
                    self._detect_inflight = False

                try:
                    self.after(0, _on_error)
                except Exception:
                    _on_error()
                return

            if not isinstance(info, dict):
                info = {}

            def _on_success() -> None:
                try:
                    self.set_detected(info)
                except Exception as update_exc:
                    logger.error(f"Failed to apply detected device info: {update_exc}")
                self._set_status("Devices updated")
                if detect_button is not None:
                    try:
                        detect_button.config(state="normal")
                    except Exception:
                        pass
                self._detect_inflight = False
                try:
                    self.after(3000, lambda: self._set_status(""))
                except Exception:
                    self._set_status("")

            try:
                self.after(0, _on_success)
            except Exception:
                _on_success()

        if not self._run_in_executor("resources-detect", _work):
            self._set_status("Device detection unavailable")
            if detect_button is not None:
                try:
                    detect_button.config(state="normal")
                except Exception:
                    pass
            self._detect_inflight = False

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

    # Training mode helpers

    def _selected_training_gpu_count(self) -> int:
        """Return the number of GPUs currently enabled for training."""
        rows = getattr(self, "_cuda_train_rows", [])
        if not rows:
            pending = getattr(self, "_pending_gpu_settings", {}).get("train_cuda_selected")
            if isinstance(pending, (set, list, tuple)):
                try:
                    return len(pending)
                except Exception:
                    return 0
            return 0

        count = 0
        for row in rows:
            try:
                enabled = row.get("enabled")
                if enabled is not None and bool(enabled.get()):  # type: ignore[union-attr]
                    count += 1
            except Exception:
                continue
        return count

    def _sync_zero_stage_with_mode(self, zero3_available: bool | None = None) -> None:
        """Keep ZeRO stage selection consistent with training mode and platform."""
        try:
            mode = self.training_mode_var.get()
        except Exception:
            mode = "parallel"

        selected_count = self._selected_training_gpu_count()

        if self.is_windows:
            if selected_count <= 1:
                if mode != "none":
                    with self.suspend_auto_apply():
                        self.training_mode_var.set("none")
                    mode = "none"
            else:
                if mode != "parallel":
                    with self.suspend_auto_apply():
                        self.training_mode_var.set("parallel")
                    mode = "parallel"
            if self.zero_stage_var.get() != "none":
                with self.suspend_auto_apply():
                    self.zero_stage_var.set("none")
            return

        if mode == "none":
            if self.zero_stage_var.get() != "none":
                with self.suspend_auto_apply():
                    self.zero_stage_var.set("none")
            return

        if selected_count <= 1 and mode in {"ddp", "parallel", "zero3"}:
            with self.suspend_auto_apply():
                self.training_mode_var.set("none")
            mode = "none"
            if self.zero_stage_var.get() != "none":
                with self.suspend_auto_apply():
                    self.zero_stage_var.set("none")
            return

        if zero3_available is None:
            zero3_available = self.is_linux and selected_count > 1
        elif not zero3_available and not getattr(self, "_cuda_train_rows", []):
            zero3_available = self.is_linux and selected_count > 1

        zero_stage = self.zero_stage_var.get()

        if zero_stage == "zero3" and mode != "zero3":
            if zero3_available and self.is_linux and selected_count > 1:
                with self.suspend_auto_apply():
                    self.training_mode_var.set("zero3")
                mode = "zero3"
            else:
                with self.suspend_auto_apply():
                    self.zero_stage_var.set("none")
                zero_stage = "none"

        if mode == "zero3":
            if zero3_available and self.is_linux and selected_count > 1:
                if zero_stage != "zero3":
                    with self.suspend_auto_apply():
                        self.zero_stage_var.set("zero3")
            else:
                fallback = "ddp" if (self.is_linux and selected_count > 1) else "none"
                with self.suspend_auto_apply():
                    self.training_mode_var.set(fallback)
                if zero_stage == "zero3":
                    with self.suspend_auto_apply():
                        self.zero_stage_var.set("none")
            return

        if zero_stage == "zero3":
            with self.suspend_auto_apply():
                self.zero_stage_var.set("none")

    def _on_zero_stage_change(self) -> None:
        """Notify dependent panels when ZeRO stage changes."""
        try:
            if self.is_windows and self.zero_stage_var.get() != "none":
                with self.suspend_auto_apply():
                    self.zero_stage_var.set("none")
                return
            if self.zero_stage_var.get() == "zero3" and self.training_mode_var.get() != "zero3":
                self._sync_zero_stage_with_mode()
        except Exception:
            logger.debug("ZeRO stage validation failed", exc_info=True)

        self._notify_zero_stage_consumers()

    def _notify_zero_stage_consumers(self) -> None:
        """Dispatch shared ZeRO stage updates to interested panels."""
        callback = getattr(self, "_hrm_deepspeed_callback", None)
        if not callable(callback):
            return

        def _invoke() -> None:
            try:
                callback()
            except Exception:
                logger.debug("HRM ZeRO callback failed", exc_info=True)

        if self._post_to_ui is not None:
            try:
                self._post_to_ui(_invoke)
                return
            except Exception:
                logger.debug("post_to_ui dispatch failed; running callback inline", exc_info=True)

        _invoke()
    
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
        logger.debug(f"[Resources] Reading train GPU rows: {len(self._cuda_train_rows)} rows")
        for row in self._cuda_train_rows:
            try:
                did = int(row["id"])  # type: ignore[index]
                enabled = bool(row["enabled"].get())
                logger.debug(f"[Resources] GPU {did}: enabled={enabled}")
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
                logger.debug(f"[Resources] Error reading GPU row: {e}")
                continue
        logger.debug(f"[Resources] Final train_cuda_selected: {cuda_train_selected}")
        
        if cuda_train_selected:
            logger.info(f"User selected GPUs for training: {cuda_train_selected}")
                
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
        
        if cuda_run_selected:
            logger.info(f"User selected GPUs for inference: {cuda_run_selected}")
        
        # Log memory cap changes
        if gp != 90:  # 90 is default
            logger.info(f"GPU memory cap set to: {gp}%")
                
        current_os = platform.system().strip().lower()
        self._active_os_label = current_os

        artifacts_override = self.artifacts_dir_var.get().strip()
        if not self._artifacts_dir_is_valid:
            artifacts_override = ""

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
            # Training mode (DDP | Parallel | Zero3 | None)
            "training_mode": self.training_mode_var.get(),
            "zero_stage": self.zero_stage_var.get(),
            # Storage caps
            "dataset_cap": self.dataset_cap_var.get(),
            "artifacts_dir": artifacts_override,
            # Persist OS so we can warn when selections migrate across platforms
            "os_name": current_os,
        }
        return vals

    def get_state(self) -> dict[str, Any]:
        """Get current panel state for persistence (alias for get_values).
        
        Returns:
            Dict with all panel settings
        """
        return self.get_values()

    def set_values(self, vals: dict) -> None:
        """Apply selections into UI from a dict returned by get_values().
        
        Args:
            vals: Dict of panel settings
        """
        logger.info("Setting resources panel values from config")
        logger.debug(f"Values to apply: {vals}")
        previous_os = str(vals.get("os_name") or "").strip().lower()
        self._last_loaded_os = previous_os or self._active_os_label
        with self.suspend_auto_apply():
            try:
                v = int(vals.get("cpu_threads", 0))
                if v > 0:
                    self.cpu_threads_var.set(str(v))
                    logger.debug(f"Set cpu_threads to {v}")
            except Exception as e:
                logger.error(f"Failed to set cpu_threads: {e}")
            
            try:
                v = int(vals.get("gpu_mem_pct", 0))
                if v > 0:
                    self.gpu_mem_pct_var.set(str(v))
                    logger.debug(f"Set gpu_mem_pct to {v}")
            except Exception as e:
                logger.error(f"Failed to set gpu_mem_pct: {e}")
            
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
            
            # Training mode (DDP vs Parallel vs None)
            try:
                tm = vals.get("training_mode")
                if isinstance(tm, str) and tm in {"ddp", "parallel", "zero3", "none"}:
                    self.training_mode_var.set(tm)
            except Exception:
                pass

            try:
                zs = vals.get("zero_stage")
                if isinstance(zs, str) and zs in {"none", "zero1", "zero2", "zero3"}:
                    self.zero_stage_var.set(zs)
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

            try:
                ad = vals.get("artifacts_dir")
                if isinstance(ad, str):
                    self.artifacts_dir_var.set(ad.strip())
                else:
                    self.artifacts_dir_var.set("")
                self._validate_artifacts_dir(apply_override=True)
            except Exception:
                logger.debug("Failed to restore artifacts directory override", exc_info=True)

        try:
            snapshot_items: list[tuple[str, float]] = []
            if self.dataset_cap_var.get().strip():
                try:
                    snapshot_items.append(("dataset_cap_gb", float(self.dataset_cap_var.get())))
                except Exception:
                    pass
            if hasattr(self, "model_cap_var") and self.model_cap_var.get().strip():
                try:
                    snapshot_items.append(("model_cap_gb", float(self.model_cap_var.get())))
                except Exception:
                    pass
            if hasattr(self, "per_brain_cap_var") and self.per_brain_cap_var.get().strip():
                try:
                    snapshot_items.append(("per_brain_cap_gb", float(self.per_brain_cap_var.get())))
                except Exception:
                    pass
            if snapshot_items:
                self._last_caps_snapshot = tuple(sorted(snapshot_items))
        except Exception:
            pass

        try:
            device_management.update_training_mode_toggle_state(self)
        except Exception:
            logger.debug("Failed to refresh training mode toggles after config load", exc_info=True)

        try:
            self._sync_zero_stage_with_mode()
        except Exception:
            logger.debug("Failed to sync ZeRO stage after config load", exc_info=True)

    def set_state(self, state: dict) -> None:
        """Restore panel state from saved data (alias for set_values).
        
        Args:
            state: Dict of panel settings
        """
        self.set_values(state)

    def cleanup(self) -> None:
        """Clean up resources before panel destruction."""
        try:
            self._shutting_down = True

            if hasattr(self, '_autosave_timer') and self._autosave_timer is not None and self._tk_root is not None:
                try:
                    self._tk_root.after_cancel(self._autosave_timer)
                except Exception:
                    pass
                finally:
                    self._autosave_timer = None

            if self._caps_refresh_after_id is not None and self._tk_root is not None:
                try:
                    self._tk_root.after_cancel(self._caps_refresh_after_id)
                except Exception:
                    pass
                finally:
                    self._caps_refresh_after_id = None

            try:
                monitoring.shutdown_monitoring_data(self)
            except Exception as exc:
                logger.debug("Resources panel monitoring shutdown error: %s", exc)

            # Cancel monitoring timer
            if hasattr(self, '_monitor_after_id') and self._monitor_after_id is not None:
                try:
                    if self._tk_root is not None:
                        self._tk_root.after_cancel(self._monitor_after_id)
                    self._monitor_after_id = None
                except Exception:
                    pass
            
            # Clear tkinter Variable references to prevent RuntimeError on cleanup
            # These are created in fallback_widgets.py
            if hasattr(self, '_cpu_label_var'):
                del self._cpu_label_var
            if hasattr(self, '_cpu_progress_var'):
                del self._cpu_progress_var
            if hasattr(self, '_ram_label_var'):
                del self._ram_label_var
            if hasattr(self, '_ram_progress_var'):
                del self._ram_progress_var
            if hasattr(self, '_net_label_var'):
                del self._net_label_var
            if hasattr(self, '_disk_label_var'):
                del self._disk_label_var
            
            # Clear GPU monitor variables
            if hasattr(self, '_gpu_monitors'):
                for gpu_mon in self._gpu_monitors.values():
                    if 'label_var' in gpu_mon:
                        del gpu_mon['label_var']
                    if 'progress_var' in gpu_mon:
                        del gpu_mon['progress_var']
                self._gpu_monitors.clear()
            
            self._cancel_background_futures(timeout=1.0)

            logger.debug("Resources panel cleanup complete")
        except Exception as e:
            logger.debug(f"Resources panel cleanup error: {e}")


__all__ = ["ResourcesPanel"]
