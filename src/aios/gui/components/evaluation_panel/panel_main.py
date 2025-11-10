"""Main evaluation panel class."""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Any, Callable, Optional
from pathlib import Path

from aios.core.evaluation import EvaluationResult, HarnessWrapper, EvaluationHistory
from aios.gui.dialogs import EvaluationResultsDialog

from . import ui_builders, event_handlers, tree_management, history_management, export_utils
from .benchmark_data import BENCHMARKS
from .config_persistence import (
    load_evaluation_from_config,
    merge_config_with_defaults,
    save_evaluation_to_config,
)


class EvaluationPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Model Evaluation panel for running standardized benchmarks.
    
    Integrates with lm-evaluation-harness to evaluate models on academic benchmarks
    like MMLU, HumanEval, GSM8K, and more. Provides preset benchmark groups,
    real-time progress tracking, results visualization, and evaluation history.
    """

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Optional[Callable[[str], None]] = None,
        save_state_fn: Optional[Callable[[], None]] = None,
        title: str = "Model Evaluation",
        worker_pool: Any = None,
        on_list_brains: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        import time
        init_start = time.time()
        
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter not available")
        super().__init__(parent, text=title)
        self.pack(fill="both", expand=True, padx=8, pady=8)
        
        step1 = time.time()
        print(f"[EVAL INIT] Superclass init: {step1 - init_start:.3f}s")

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._save_state_fn = save_state_fn
        self._save_after_id: Optional[str] = None
        self._worker_pool = worker_pool
        self._eval_process: Any = None
        self._is_running = False
        self._on_list_brains = on_list_brains
        
        # Get project root first
        try:
            self._project_root = str(Path(__file__).parent.parent.parent.parent.parent.parent)
        except Exception:
            self._project_root = os.getcwd()
        
        step2 = time.time()
        print(f"[EVAL INIT] Basic setup: {step2 - step1:.3f}s")
        
        # Initialize HarnessWrapper
        self._harness = HarnessWrapper(
            log_callback=self._log_threadsafe,
            progress_callback=self._on_progress_update_threadsafe,
        )
        self._current_result: Optional[EvaluationResult] = None
        
        # History will be initialized asynchronously during startup
        # and set via _set_history() on main thread
        self._history: Optional[EvaluationHistory] = None
        self._history_db_path = str(Path(self._project_root) / "artifacts" / "evaluation" / "history.db")

        step3 = time.time()
        print(f"[EVAL INIT] Harness/history setup: {step3 - step2:.3f}s")

        # Schedule save helper
        def _schedule_save(delay_ms: int = 400) -> None:
            try:
                if not callable(self._save_state_fn):
                    return
                if self._save_after_id is not None:
                    try:
                        self.after_cancel(self._save_after_id)
                    except Exception:
                        pass
                self._save_after_id = self.after(
                    delay_ms,
                    lambda: self._save_state_fn() if callable(self._save_state_fn) else None
                )
            except Exception:
                pass
        self._schedule_save = _schedule_save  # type: ignore[attr-defined]

        # State variables
        self.model_source_var = tk.StringVar(value="huggingface")  # huggingface, local, brain
        self.model_name_var = tk.StringVar(value="gpt2")
        
        # Benchmark selection
        self.selected_benchmarks_var = tk.StringVar(value="")
        self.benchmark_preset_var = tk.StringVar(value="")
        
        step4 = time.time()
        print(f"[EVAL INIT] State variables: {step4 - step3:.3f}s")
        
        # Load configuration defaults from config file
        config_values = load_evaluation_from_config()
        defaults = {
            'batch_size': 'auto',
            'limit': '0',
            'num_fewshot': '5',
            'output_path': 'artifacts/evaluation',
            'log_samples': False,
            'cache_requests': True,
            'check_integrity': False,
        }
        merged = merge_config_with_defaults(config_values, defaults)
        
        step5 = time.time()
        print(f"[EVAL INIT] Config loading: {step5 - step4:.3f}s")
        
        # Configuration (from config file or defaults)
        self.batch_size_var = tk.StringVar(value=merged['batch_size'])
        self.limit_var = tk.StringVar(value=merged['limit'])
        self.num_fewshot_var = tk.StringVar(value=merged['num_fewshot'])
        self.output_path_var = tk.StringVar(value=merged['output_path'])
        
        # Advanced options (from config file or defaults)
        self.log_samples_var = tk.BooleanVar(value=merged['log_samples'])
        self.cache_requests_var = tk.BooleanVar(value=merged['cache_requests'])
        self.check_integrity_var = tk.BooleanVar(value=merged['check_integrity'])
        
        # Available benchmarks by category
        self.benchmarks = BENCHMARKS
        
        step6 = time.time()
        print(f"[EVAL INIT] More variables: {step6 - step5:.3f}s")
        
        # Build UI
        self._build_ui()
        
        step7 = time.time()
        print(f"[EVAL INIT] UI building: {step7 - step6:.3f}s")
        
        # Auto-persist on changes
        self._setup_auto_persist()
        
        step8 = time.time()
        print(f"[EVAL INIT] Auto-persist setup: {step8 - step7:.3f}s")
        print(f"[EVAL INIT] TOTAL: {step8 - init_start:.3f}s")

    def _build_ui(self) -> None:
        """Build the evaluation panel UI."""
        import time
        
        t0 = time.time()
        ui_builders.create_model_selection(self)
        t1 = time.time()
        print(f"[EVAL UI] Model selection: {t1 - t0:.3f}s")
        
        ui_builders.create_benchmark_selection(self)
        t2 = time.time()
        print(f"[EVAL UI] Benchmark selection: {t2 - t1:.3f}s")
        
        ui_builders.create_configuration_section(self)
        t3 = time.time()
        print(f"[EVAL UI] Configuration section: {t3 - t2:.3f}s")
        
        ui_builders.create_advanced_options(self)
        t4 = time.time()
        print(f"[EVAL UI] Advanced options: {t4 - t3:.3f}s")
        
        ui_builders.create_control_buttons(self)
        t5 = time.time()
        print(f"[EVAL UI] Control buttons: {t5 - t4:.3f}s")
        
        ui_builders.create_progress_section(self)
        t6 = time.time()
        print(f"[EVAL UI] Progress section: {t6 - t5:.3f}s")
        
        ui_builders.create_output_section(self)
        t7 = time.time()
        print(f"[EVAL UI] Output section: {t7 - t6:.3f}s")
        
        ui_builders.create_results_section(self)
        t8 = time.time()
        print(f"[EVAL UI] Results section: {t8 - t7:.3f}s")
        print(f"[EVAL UI] TOTAL UI BUILD: {t8 - t0:.3f}s")

    def _populate_benchmark_tree(self) -> None:
        """Populate the benchmark tree with available benchmarks."""
        tree_management.populate_benchmark_tree(self)

    def _on_benchmark_click(self, event: Any) -> None:
        """Handle benchmark selection toggle."""
        tree_management.on_benchmark_click(self, event)

    def _update_selected_benchmarks(self) -> None:
        """Update the selected benchmarks list and info label."""
        tree_management.update_selected_benchmarks(self)

    def _select_preset(self, preset: str) -> None:
        """Select a benchmark preset."""
        tree_management.select_preset(self, preset)

    def _refresh_brains(self) -> None:
        """Refresh the brain list from registry."""
        if not self._on_list_brains:
            return
        
        try:
            brains = self._on_list_brains()
            self.brain_combo.config(values=brains)
            
            # Set default if nothing selected
            current = self.model_name_var.get()
            if not current or current not in brains:
                if brains:
                    self.model_name_var.set(brains[0])
            
            self._log(f"[eval] Loaded {len(brains)} brain(s)")
        except Exception as e:
            self._log(f"[eval] Error refreshing brains: {e}")

    def _browse_output(self) -> None:
        """Browse for output directory."""
        event_handlers.browse_output_directory(self)

    def _on_start_evaluation(self) -> None:
        """Start evaluation."""
        event_handlers.start_evaluation(self)

    def _on_progress_update(self, progress: float, status_msg: str) -> None:
        """Handle progress updates from harness."""
        event_handlers.on_progress_update(self, progress, status_msg)
    
    def _on_evaluation_complete(self, result: EvaluationResult) -> None:
        """Handle evaluation completion."""
        event_handlers.on_evaluation_complete(self, result)

    def _on_stop_evaluation(self) -> None:
        """Stop evaluation."""
        event_handlers.on_stop_evaluation(self)

    def _on_view_history(self) -> None:
        """View evaluation history."""
        history_management.view_history(self)

    def _set_history(self, history: EvaluationHistory) -> None:
        """Set history instance (called from main thread after async init)."""
        self._history = history

    def _log_threadsafe(self, msg: str) -> None:
        """Thread-safe wrapper for logging (calls _append_out on GUI thread)."""
        try:
            self.after(0, lambda: self._append_out(msg))
        except Exception:
            # Fallback to direct call if after() fails
            self._append_out(msg)
    
    def _on_progress_update_threadsafe(self, progress: float, status_msg: str) -> None:
        """Thread-safe wrapper for progress updates (calls on GUI thread)."""
        try:
            self.after(0, lambda: self._on_progress_update(progress, status_msg))
        except Exception:
            # Fallback to direct call if after() fails
            self._on_progress_update(progress, status_msg)

    def _clear_output(self) -> None:
        """Clear the output log."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
        self._log("[eval] Output cleared")

    def _export_results(self, format: str) -> None:
        """Export evaluation results."""
        if not self._current_result or self._current_result.status != "completed":
            messagebox.showwarning(
                "No Results",
                "No completed evaluation results to export.\nPlease run an evaluation first."
            )
            return
        
        model_name = self.model_name_var.get() or "unknown"
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        
        if format == "csv":
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"eval_results_{safe_name}.csv",
            )
            
            if not filepath:
                return
            
            try:
                export_utils.export_to_csv(self._current_result, filepath, model_name)
                self._log(f"[eval] Results exported to CSV: {filepath}")
                messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
            except Exception as e:
                self._log(f"[eval] Error exporting CSV: {e}")
                messagebox.showerror("Export Failed", f"Failed to export CSV:\n{e}")
        
        elif format == "json":
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"eval_results_{safe_name}.json",
            )
            
            if not filepath:
                return
            
            try:
                export_utils.export_to_json(self._current_result, filepath, model_name)
                self._log(f"[eval] Results exported to JSON: {filepath}")
                messagebox.showinfo("Export Successful", f"Results exported to:\n{filepath}")
            except Exception as e:
                self._log(f"[eval] Error exporting JSON: {e}")
                messagebox.showerror("Export Failed", f"Failed to export JSON:\n{e}")

    def _view_detailed_results(self) -> None:
        """View detailed evaluation results."""
        if not self._current_result or self._current_result.status != "completed":
            messagebox.showwarning(
                "No Results",
                "No completed evaluation results to view.\nPlease run an evaluation first."
            )
            return
        
        try:
            # Open results dialog
            model_name = self.model_name_var.get() or "Unknown Model"
            dialog = EvaluationResultsDialog(self, self._current_result, model_name)
            self._log("[eval] Opened detailed results viewer")
        except Exception as e:
            self._log(f"[eval] Error opening results viewer: {e}")
            messagebox.showerror("Error", f"Failed to open results viewer:\n{e}")

    def _log(self, message: str) -> None:
        """Log a message to the output log."""
        try:
            # Check if user is at bottom before inserting
            try:
                yview = self.log_text.yview()
                at_bottom = yview[1] >= 0.95  # Within ~5% of bottom
            except Exception:
                at_bottom = True  # Default to scrolling if can't check
            
            self.log_text.config(state="normal")
            self.log_text.insert("end", message + "\n")
            
            # Only scroll if user was at bottom
            if at_bottom:
                self.log_text.see("end")
            
            self.log_text.config(state="disabled")
            
            # Also send to main output if available
            if self._append_out:
                self._append_out(message)
        except Exception:
            pass

    def _setup_auto_persist(self) -> None:
        """Setup auto-persistence for state variables."""
        try:
            # Create a callback that saves to both GUI state AND config file
            def on_config_change(*args: Any) -> None:
                """Called when evaluation configuration changes."""
                # Save to GUI state (for session persistence)
                self._schedule_save()
                
                # Also save to config file (for permanent defaults)
                try:
                    state = self.get_state()
                    save_evaluation_to_config(state)
                except Exception as e:
                    # Don't show error to user, just log it
                    print(f"[Evaluation] Failed to save to config: {e}")
            
            # Watch configuration variables (not model/benchmark selections)
            config_vars = [
                self.batch_size_var,
                self.limit_var,
                self.num_fewshot_var,
                self.output_path_var,
                self.log_samples_var,
                self.cache_requests_var,
                self.check_integrity_var,
            ]
            
            # Watch model/benchmark vars for GUI state only
            gui_state_vars = [
                self.model_source_var,
                self.model_name_var,
                self.selected_benchmarks_var,
            ]
            
            # Add traces for config vars (save to both)
            for v in config_vars:
                try:
                    v.trace_add("write", on_config_change)  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Add traces for GUI state vars (save to GUI state only)
            for v in gui_state_vars:
                try:
                    v.trace_add("write", lambda *args: self._schedule_save())  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    def get_state(self) -> dict[str, Any]:
        """Return current UI state for persistence."""
        return {
            "model_source": self.model_source_var.get(),
            "model_name": self.model_name_var.get(),
            "selected_benchmarks": self.selected_benchmarks_var.get(),
            "batch_size": self.batch_size_var.get(),
            "limit": self.limit_var.get(),
            "num_fewshot": self.num_fewshot_var.get(),
            "output_path": self.output_path_var.get(),
            "log_samples": self.log_samples_var.get(),
            "cache_requests": self.cache_requests_var.get(),
            "check_integrity": self.check_integrity_var.get(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore UI state from dict."""
        try:
            self.model_source_var.set(state.get("model_source", "huggingface"))
            self.model_name_var.set(state.get("model_name", "gpt2"))
            self.selected_benchmarks_var.set(state.get("selected_benchmarks", ""))
            self.batch_size_var.set(state.get("batch_size", "auto"))
            self.limit_var.set(state.get("limit", ""))
            self.num_fewshot_var.set(state.get("num_fewshot", "5"))
            self.output_path_var.set(state.get("output_path", "artifacts/evaluation"))
            self.log_samples_var.set(state.get("log_samples", False))
            self.cache_requests_var.set(state.get("cache_requests", True))
            self.check_integrity_var.set(state.get("check_integrity", False))
            
            # Restore benchmark selections
            selected = state.get("selected_benchmarks", "").split(",")
            if selected and selected[0]:
                for item_id, item_info in self._tree_items.items():
                    if item_info["type"] == "benchmark":
                        if item_info["name"] in selected:
                            item_info["checked"] = True
                            self.bench_tree.item(item_id, text="☑")
                self._update_selected_benchmarks()
        except Exception as e:
            self._log(f"[eval] Error restoring state: {e}")
