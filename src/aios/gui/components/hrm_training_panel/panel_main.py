"""Main HRM Training Panel class.

Orchestrates all UI builders and provides the main panel interface.
"""

from __future__ import annotations
import logging
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HRMTrainingPanel(ttk.LabelFrame):  # type: ignore[misc]
    """ACTV1 HRM Training panel for configuring and launching training runs.

    This triggers the CLI command `aios hrm-hf train-actv1` with configurable
    dataset path and training knobs.
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Optional[Callable[[str], None]] = None,
        save_state_fn: Optional[Callable[[], None]] = None,
        title: str = "HRM Training",
        worker_pool: Any = None,
        resources_panel: Any = None,
        post_to_ui: Optional[Callable[[Callable[..., None]], None]] = None,
        app: Any = None,
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter not available")
        
        logger.info("Initializing HRM Training Panel")
        
        super().__init__(parent, text=title)
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._save_state_fn = save_state_fn
        self._save_after_id: Optional[str] = None
        self._worker_pool = worker_pool
        self._resources_panel = resources_panel
        self._post_to_ui = post_to_ui
        self._app = app
        if self._resources_panel is not None:
            try:
                self._resources_panel._hrm_deepspeed_callback = self.update_deepspeed_state
            except Exception:
                pass
        
        # Get project root
        from .helpers import project_root
        self._project_root = project_root()
        logger.debug(f"Project root: {self._project_root}")
        
        # Initialize variables
        from .variable_setup import setup_variables, setup_variable_traces
        logger.debug("Setting up HRM training panel variables")
        if self._resources_panel is not None and hasattr(self._resources_panel, "zero_stage_var"):
            try:
                self.zero_stage_var = self._resources_panel.zero_stage_var
            except Exception:
                pass
        setup_variables(self)
        setup_variable_traces(self)
        
        # Initialize state
        self._metrics_polling_active = False
        self._run_in_progress = False
        self._last_gen_total = None
        self._last_steps_total = None
        self._gen_hist = []
        self._step_hist = []
        self._stopped_dialog_shown = False
        self._bg_thread = None
        self._bg_future = None
        self._proc = None
        self._stop_requested = False
        self._graceful_stop_requested = False  # Track graceful stop state for two-stage stop button
        self._last_heartbeat = None
        self._start_time = None  # Track training start time for ETA
        self._last_progress_step = 0  # Track cycle-local step for iterate mode progress bar
        
        # Training progress tracking
        self._total_steps_all_gpus = 0
        self._chunks_completed = 0
        self._current_block_id = 0
        self._blocks_processed = 0
        self._chunks_in_current_block = 0
        self._current_epoch = 0
        self._chunks_per_block = None
        self._total_blocks = None
        self._dataset_name = None
        
        import os
        self._heartbeat_timeout = float(os.environ.get("AIOS_HEARTBEAT_TIMEOUT", "30"))
        self._stop_escalation_timeout = float(os.environ.get("AIOS_STOP_TIMEOUT", "10"))
        self._vram_update_after_id: Optional[str] = None
        self._vram_task_id = 0
        self._vram_estimate_future = None
        self._vram_warned_long_seq: Optional[tuple[int, bool]] = None
        
        # Layout
        g = ttk.Frame(self)
        g.pack(fill="x")
        
        # Build UI sections
        logger.debug("Building HRM training panel UI sections")
        from .ui_core_form import build_core_form
        from .ui_architecture import build_architecture_display
        from .ui_optimizations import build_optimizations_section
        from .ui_controls import build_controls, build_progress_bar, build_log_output
        from .ui_metrics import build_memory_panels, build_epoch_tracking_panel, initialize_epoch_tracking_display
        
        build_core_form(self, g)
        build_architecture_display(self, g)
        build_optimizations_section(self, g)
        build_controls(self, self)
        build_progress_bar(self, self)
        build_log_output(self, self)
        build_epoch_tracking_panel(self, self)
        initialize_epoch_tracking_display(self)
        build_memory_panels(self, self)
        
        logger.info("HRM Training Panel UI constructed successfully")
        
        # Data loading deferred to async initialization during startup
        # (prefill_last_safe_batches and update_vram_estimate will be called
        #  in background thread via _load_hrm_training_panel_sync)
        
        # Bind real-time VRAM estimate updates
        from .variable_setup import _schedule_vram_update
        for v in [
            self.batch_var, self.max_seq_var, self.h_layers_var, self.l_layers_var,
            self.hidden_size_var, self.expansion_var, self.num_heads_var,
            self.gradient_checkpointing_var, self.use_amp_var, self.use_cpu_offload_var,
            self.use_8bit_optimizer_var, self.use_chunked_training_var, self.chunk_size_var,
            self.use_peft_var, self.lora_r_var, self.lora_alpha_var, self.lora_dropout_var,
            self.zero_stage_var,
        ]:
            try:
                v.trace_add("write", lambda *args: _schedule_vram_update(self))
            except Exception:
                pass
        
        # Setup trace on resources panel training mode to update DeepSpeed state
        try:
            if self._resources_panel is not None and hasattr(self._resources_panel, "training_mode_var"):
                self._resources_panel.training_mode_var.trace_add("write", lambda *args: self.update_deepspeed_state())
                # Also trace GPU selection changes
                if hasattr(self._resources_panel, "_cuda_train_rows"):
                    for row in self._resources_panel._cuda_train_rows:
                        var = row.get("enabled")
                        if hasattr(var, "trace_add"):
                            var.trace_add("write", lambda *args: self.update_deepspeed_state())
            if self._resources_panel is not None and hasattr(self._resources_panel, "zero_stage_var"):
                self._resources_panel.zero_stage_var.trace_add("write", lambda *args: self.update_deepspeed_state())
        except Exception:
            pass
        
        # Initial update of DeepSpeed state
        self.update_deepspeed_state()

    def dispatch_to_ui(self, callback: Callable[[], None]) -> bool:
        """Schedule ``callback`` on the Tk UI thread."""
        if self._post_to_ui is not None:
            try:
                self._post_to_ui(callback)
                return True
            except Exception:
                logger.debug("Failed to dispatch via post_to_ui", exc_info=True)

        if threading.current_thread() is threading.main_thread():
            try:
                self.after(0, callback)
                return True
            except Exception:
                logger.debug("Failed to schedule callback with after", exc_info=True)

        return False

    def update_theme(self) -> None:
        """Update Text widget colors when theme changes."""
        from .theme_utils import get_theme_colors
        
        try:
            theme_colors = get_theme_colors()
            logger.debug(f"Applying theme to HRM panel: bg={theme_colors.get('bg')}, fg={theme_colors.get('fg')}")
            self.log.config(
                bg=theme_colors["bg"],
                fg=theme_colors["fg"],
                selectbackground=theme_colors["selectbg"],
                selectforeground=theme_colors["selectfg"],
                insertbackground=theme_colors["insertbg"]
            )
        except Exception as e:
            logger.error(f"Failed to apply theme to HRM panel: {e}")

    def update_deepspeed_state(self) -> None:
        """Update DeepSpeed ZeRO dropdown state based on training mode and platform."""
        try:
            if not hasattr(self, "_resources_panel") or self._resources_panel is None:
                return

            import platform

            is_linux = platform.system() == "Linux"
            is_windows = platform.system() == "Windows"
            rvals = self._resources_panel.get_values()
            training_mode = rvals.get("training_mode", "none")
            selected = rvals.get("train_cuda_selected", [])
            if (not selected) and hasattr(self._resources_panel, "_pending_gpu_settings"):
                try:
                    pending = self._resources_panel._pending_gpu_settings.get("train_cuda_selected", set())
                    if isinstance(pending, (set, list, tuple)) and pending:
                        selected = list(pending)
                except Exception:
                    selected = selected
            num_gpus = len(selected) if isinstance(selected, list) else 0
            zero_stage_resource = rvals.get("zero_stage", "none")

            # ZeRO 1 and 2 are available on:
            # - Linux with 1+ GPUs
            # - Windows with 1 GPU (single GPU mode) or parallel mode
            linux_gpu_available = is_linux and num_gpus >= 1
            windows_single_gpu = is_windows and num_gpus == 1
            windows_parallel_mode = is_windows and num_gpus > 1 and training_mode == "parallel"
            zero12_available = linux_gpu_available or windows_single_gpu or windows_parallel_mode
            
            zero3_active = zero_stage_resource == "zero3" or training_mode == "zero3"

            allowed_values: list[str] = ["none"]
            if zero12_available:
                allowed_values.extend(["zero1", "zero2"])
            if zero3_active and "zero3" not in allowed_values:
                allowed_values.append("zero3")

            logger.debug(
                "DeepSpeed state update (linux=%s, windows=%s, gpus=%d, mode=%s, zero=%s, zero12_avail=%s)",
                is_linux,
                is_windows,
                num_gpus,
                training_mode,
                zero_stage_resource,
                zero12_available,
            )

            if not hasattr(self, "zero_combo"):
                return

            try:
                current_values = tuple(self.zero_combo["values"])
            except Exception:
                current_values = tuple()

            desired_values = tuple(allowed_values)
            if desired_values != current_values:
                self.zero_combo["values"] = desired_values

            if not zero12_available:
                if self.zero_stage_var.get() not in ["none", "zero3"]:
                    logger.debug("ZeRO 1/2 disabled (incompatible platform/mode), resetting to none")
                    self.zero_stage_var.set("none")
                # Only disable if zero3 is also not active
                if not zero3_active:
                    self.zero_combo.config(state="disabled")
                    return

            if zero3_active:
                if self.zero_stage_var.get() != "zero3":
                    logger.debug("ZeRO-3 enforced by Resources panel")
                    self.zero_stage_var.set("zero3")
                self.zero_combo.config(state="disabled")
                return

            if self.zero_stage_var.get() == "zero3":
                logger.debug("Clearing ZeRO-3 selection; Resources mode no longer requires it")
                self.zero_stage_var.set("none")

            self.zero_combo.config(state="readonly")
        except Exception as e:
            logger.error(f"Failed to update DeepSpeed state: {e}")

    def _log(self, msg: str) -> None:
        """Append a line of text to the panel log and external output."""
        from .helpers import log
        log(self, msg)

    def _clear_output(self) -> None:
        """Clear the training output log."""
        from .helpers import clear_output
        clear_output(self)

    def get_state(self) -> dict:
        """Return a dict of current UI settings for persistence."""
        from .state_management import get_state
        return get_state(self)

    def set_state(self, state: dict) -> None:
        """Apply settings from a dict produced by get_state()."""
        from .state_management import set_state
        set_state(self, state)

    def build_training_config(self) -> Any:
        """Build a TrainingConfig object from current GUI state."""
        from .config_builder import build_training_config
        return build_training_config(self)

    def _on_start(self) -> None:
        """Start training."""
        from .actions import on_start_wrapper
        on_start_wrapper(self)

    def _on_stop(self) -> None:
        """Stop training."""
        from .actions import on_stop_wrapper
        on_stop_wrapper(self)

    def _stop_all(self) -> None:
        """Stop training with two-stage behavior (graceful then immediate)."""
        from ..hrm_training.actions import on_stop
        on_stop(self)

    def _on_select_student(self) -> None:
        """Select student brain."""
        from .actions import on_select_student
        on_select_student(self)

    def _open_rank_logs(self) -> None:
        """Open rank logs."""
        from .actions import open_rank_logs
        open_rank_logs(self)

    def _on_optimize(self) -> None:
        """Run optimization."""
        from .actions import on_optimize
        on_optimize(self)

    def _poll_metrics(self) -> None:
        """Poll metrics."""
        from .metrics import poll_metrics_wrapper
        poll_metrics_wrapper(self)

    def _show_stopped_dialog(self) -> None:
        """Show stopped dialog."""
        from .metrics import show_stopped_dialog_wrapper
        show_stopped_dialog_wrapper(self)

    def _set_arch_widgets_state(self, state: str) -> None:
        """Enable/disable architecture widgets.
        
        Args:
            state: Widget state ("normal", "disabled", or "readonly")
        """
        from .ui_architecture import set_arch_widgets_state
        set_arch_widgets_state(self, state)

    def cleanup(self) -> None:
        """Clean up HRM training panel resources on shutdown.
        
        This must be FAST and synchronous - no background threads or async callbacks.
        The application is shutting down, so we force-terminate any running processes.
        """
        logger.info("Cleaning up HRM Training Panel")
        
        # Stop metrics polling FIRST to prevent new UI updates
        if self._metrics_polling_active:
            logger.debug("Stopping metrics polling")
            self._metrics_polling_active = False
        
        # Cancel any scheduled callbacks to prevent them from firing during shutdown
        if self._vram_update_after_id:
            try:
                self.after_cancel(self._vram_update_after_id)
                logger.debug("Cancelled scheduled VRAM update")
            except Exception as e:
                logger.debug(f"Error cancelling VRAM update: {e}")
            self._vram_update_after_id = None
        
        if self._save_after_id:
            try:
                self.after_cancel(self._save_after_id)
                logger.debug("Cancelled scheduled state save")
            except Exception as e:
                logger.debug(f"Error cancelling state save: {e}")
            self._save_after_id = None
        
        # FORCE-TERMINATE any running training process immediately
        # During shutdown we don't have time for graceful stops
        proc = getattr(self, '_proc', None)
        if proc is not None:
            try:
                is_alive = proc.is_alive() if hasattr(proc, 'is_alive') else (proc.poll() is None)
            except Exception:
                is_alive = False
            
            if is_alive:
                logger.info(f"Force-terminating training process during shutdown (PID: {getattr(proc, 'pid', 'unknown')})")
                try:
                    # Try terminate first
                    proc.terminate()
                    # Wait briefly for termination
                    import time
                    for _ in range(10):  # 1 second max
                        time.sleep(0.1)
                        try:
                            still_alive = proc.is_alive() if hasattr(proc, 'is_alive') else (proc.poll() is None)
                        except Exception:
                            still_alive = False
                        if not still_alive:
                            logger.debug("Training process terminated successfully")
                            break
                    else:
                        # Still alive after 1s, force kill
                        logger.warning("Process did not respond to terminate, forcing kill")
                        try:
                            proc.kill()
                        except Exception as kill_err:
                            logger.error(f"Error killing process: {kill_err}")
                except Exception as e:
                    logger.error(f"Error terminating training process: {e}")
            
            self._proc = None
        
        # Clean up multiprocessing manager and events
        try:
            manager = getattr(self, '_training_manager', None)
            if manager is not None:
                logger.debug("Shutting down multiprocessing manager")
                try:
                    manager.shutdown()
                except Exception as e:
                    logger.debug(f"Manager shutdown error (may be expected): {e}")
                self._training_manager = None
        except Exception as e:
            logger.debug(f"Error cleaning up manager: {e}")
        
        # Clear event references
        self._stop_event = None
        self._graceful_stop_event = None
        self._stop_ack_event = None
        self._graceful_stop_ack_event = None
        self._run_in_progress = False
        self._stop_requested = False
        self._graceful_stop_requested = False

        logger.info("HRM Training Panel cleanup complete")
