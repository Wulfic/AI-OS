"""Main HRM Training Panel class.

Orchestrates all UI builders and provides the main panel interface.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional


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
    ) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter not available")
        super().__init__(parent, text=title)
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._save_state_fn = save_state_fn
        self._save_after_id: Optional[str] = None
        self._worker_pool = worker_pool
        self._resources_panel = resources_panel
        
        # Get project root
        from .helpers import project_root
        self._project_root = project_root()
        
        # Initialize variables
        from .variable_setup import setup_variables, setup_variable_traces
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
        
        # Layout
        g = ttk.Frame(self)
        g.pack(fill="x")
        
        # Build UI sections
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
                        if "enabled_var" in row:
                            row["enabled_var"].trace_add("write", lambda *args: self.update_deepspeed_state())
        except Exception:
            pass
        
        # Initial update of DeepSpeed state
        self.update_deepspeed_state()

    def update_theme(self) -> None:
        """Update Text widget colors when theme changes."""
        from .theme_utils import get_theme_colors
        
        try:
            theme_colors = get_theme_colors()
            self.log.config(
                bg=theme_colors["bg"],
                fg=theme_colors["fg"],
                selectbackground=theme_colors["selectbg"],
                selectforeground=theme_colors["selectfg"],
                insertbackground=theme_colors["insertbg"]
            )
        except Exception:
            pass

    def update_deepspeed_state(self) -> None:
        """Update DeepSpeed ZeRO dropdown state based on training mode.
        
        Only enables DeepSpeed ZeRO in Linux environments with multiple GPUs in DDP mode.
        Disables it in all other scenarios (Windows, single GPU, or parallel mode).
        """
        try:
            # Get training mode from resources panel
            if not hasattr(self, "_resources_panel") or self._resources_panel is None:
                return
            
            rvals = self._resources_panel.get_values()
            training_mode = rvals.get("training_mode", "ddp")
            num_gpus = len(rvals.get("train_cuda_selected", []))
            
            # Check if we're on Linux
            import platform
            is_linux = platform.system() == "Linux"
            
            # DeepSpeed ZeRO should only be enabled on Linux with multiple GPUs in DDP mode
            should_enable = is_linux and num_gpus > 1 and training_mode == "ddp"
            
            if hasattr(self, "zero_combo"):
                if should_enable:
                    # Enable ZeRO dropdown
                    self.zero_combo.config(state="readonly")
                else:
                    # Disable ZeRO dropdown and set to "none"
                    self.zero_combo.config(state="disabled")
                    self.zero_stage_var.set("none")
        except Exception:
            pass

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
