"""Event handlers for evaluation operations."""

from __future__ import annotations
import logging
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from aios.core.evaluation import EvaluationResult
    from .panel_main import EvaluationPanel

logger = logging.getLogger(__name__)


def browse_output_directory(panel: "EvaluationPanel") -> None:
    """Browse for output directory.
    
    Args:
        panel: The evaluation panel instance
    """
    path = filedialog.askdirectory(
        initialdir=panel._project_root,
        title="Select output directory"
    )
    if path:
        panel.output_path_var.set(path)


def start_evaluation(panel: "EvaluationPanel") -> None:
    """Start evaluation.
    
    Args:
        panel: The evaluation panel instance
    """
    from aios.core.evaluation import HarnessWrapper
    
    # Validate inputs
    selected = panel.selected_benchmarks_var.get()
    if not selected:
        logger.info("Evaluation start aborted: no benchmarks selected")
        messagebox.showwarning(
            "No Benchmarks Selected",
            "Please select at least one benchmark to evaluate."
        )
        return
    
    model = panel.model_name_var.get().strip()
    if not model:
        logger.info("Evaluation start aborted: no model specified")
        messagebox.showwarning(
            "No Model Selected",
            "Please specify a model to evaluate."
        )
        return
    
    # Check if lm_eval is installed
    if not HarnessWrapper.is_lm_eval_installed():
        logger.error("lm-evaluation-harness not installed")
        messagebox.showerror(
            "lm-evaluation-harness Not Found",
            "The lm-evaluation-harness package is not installed.\n\n"
            "Please install it with:\n"
            "pip install lm-eval[api]"
        )
        return
    
    logger.info(f"Starting evaluation: model={model}, benchmarks={selected}")
    panel._log("[eval] Starting evaluation...")
    panel._log(f"[eval] Model: {model}")
    panel._log(f"[eval] Benchmarks: {selected}")
    
    # Parse selected benchmarks and warn about large benchmark sets
    tasks = [t.strip() for t in selected.split(",") if t.strip()]
    
    # Count subtasks for benchmarks with many subtasks
    large_benchmarks = {"bbh": 27, "mmlu": 57, "minerva_math": 4}
    estimated_subtasks = sum(large_benchmarks.get(t, 1) for t in tasks)
    
    if len(tasks) > 10 or estimated_subtasks > 30:
        panel._log(f"[eval] Warning: {len(tasks)} benchmarks selected ({estimated_subtasks}+ subtasks)")
        panel._log("[eval] First run will download datasets - this may take several minutes")
        panel._log("[eval] Subsequent runs will be faster (cached datasets)")
    
    # Update UI state
    panel._is_running = True
    panel.start_btn.config(state="disabled")
    panel.stop_btn.config(state="normal")
    panel.progress["mode"] = "indeterminate"
    panel.progress.start(10)
    panel.progress_label.config(text="initializing...")
    panel.results_label.config(text="")
    
    # Get configuration
    model_args = "dtype=auto"  # Default model args
    batch_size = panel.batch_size_var.get().strip() or "auto"
    
    # Parse limit as percentage and convert to fraction (0.0-1.0) for lm_eval
    limit_str = panel.limit_var.get().strip()
    limit = None
    if limit_str and limit_str.replace('.', '', 1).isdigit():
        limit_pct = float(limit_str)
        # 0 means unlimited (100%)
        if limit_pct == 0:
            limit = None
        elif limit_pct < 0 or limit_pct > 100:
            logger.warning(f"Invalid limit percentage: {limit_pct}")
            messagebox.showerror("Invalid Limit", "Limit percentage must be between 0-100")
            panel.start_btn.config(state="normal")
            panel.stop_btn.config(state="disabled")
            panel._is_running = False
            return
        else:
            # Convert percentage to fraction for lm_eval (e.g., 10% -> 0.10)
            limit = limit_pct / 100.0
            logger.debug(f"Evaluation limit set to {limit_pct}% ({limit} fraction)")
    
    num_fewshot = int(panel.num_fewshot_var.get() or "5")
    
    # Get device from resources panel (run/inference device)
    device = "cuda:0"  # default
    try:
        rp = getattr(panel, "_resources_panel", None)
        if rp is not None:
            rvals = rp.get_values()
            run_device = str(rvals.get("run_device") or "cuda:0").lower()
            if run_device in {"cpu", "cuda", "xpu", "mps", "dml"} or run_device.startswith("cuda:"):
                device = run_device
                logger.debug(f"Using device from resources panel: {device}")
    except Exception as e:
        logger.warning(f"Failed to get device from resources panel, using default: {e}")
        pass
    
    logger.info(f"Evaluation config: batch_size={batch_size}, limit={limit}, num_fewshot={num_fewshot}, device={device}")
    
    output_path = panel.output_path_var.get() or "artifacts/evaluation"
    log_samples = panel.log_samples_var.get()
    cache_requests = panel.cache_requests_var.get()
    check_integrity = panel.check_integrity_var.get()
    
    # Determine model type - auto-detect if it's a brain or external model
    model_type = "hf"  # Default to HuggingFace
    brain_name = model
    brain_path = Path(panel._project_root) / "artifacts" / "brains" / "actv1" / brain_name
    
    # Debug: Log the paths being checked
    panel._log(f"[eval] Checking for brain: {brain_name}")
    panel._log(f"[eval] Project root: {panel._project_root}")
    panel._log(f"[eval] Brain path: {brain_path}")
    panel._log(f"[eval] Brain path exists: {brain_path.exists()}")
    
    # Check if this looks like a local brain
    if brain_path.exists():
        panel._log(f"[eval] Found brain directory: {brain_path}")
        # Check for brain.json to determine type
        brain_json_path = brain_path / "brain.json"
        if brain_json_path.exists():
            try:
                import json
                with open(brain_json_path, 'r') as f:
                    brain_config = json.load(f)
                
                brain_type = brain_config.get("type", "unknown")
                
                if brain_type == "actv1":
                    # This is a native AI-OS brain - use custom adapter
                    panel._log(f"[eval] Detected AI-OS ACTv1 brain - using native adapter")
                    
                    # Import and register the custom adapter
                    try:
                        from aios.core.evaluation.aios_lm_eval_adapter import register_aios_model
                        register_aios_model()
                    except Exception as e:
                        panel._log(f"[eval] Warning: Could not register AI-OS adapter: {e}")
                    
                    # Use the custom "aios" model type with brain_path parameter
                    model_type = "aios"
                    model = f"brain_path={brain_path}"
                else:
                    # Unknown brain type - try as HF-compatible model
                    panel._log(f"[eval] Unknown brain type '{brain_type}', treating as HF-compatible")
                    model = str(brain_path)
                    model_type = "hf"
                    
            except Exception as e:
                panel._log(f"[eval] Error reading brain config: {e}")
                # Fallback to HF
                model = str(brain_path)
                model_type = "hf"
        else:
            # No brain.json found - treat as HF-compatible model
            panel._log(f"[eval] No brain.json found, treating as HF-compatible")
            model = str(brain_path)
            model_type = "hf"
    else:
        # Brain path doesn't exist - check if it's a file path or HF model
        if Path(model).exists():
            # It's a local file path
            panel._log(f"[eval] Using local model from path: {model}")
            model_type = "hf"
        else:
            # Assume it's a HuggingFace model identifier
            panel._log(f"[eval] Treating as HuggingFace model identifier: {model}")
            model_type = "hf"    # Run evaluation asynchronously
    def on_complete(result: "EvaluationResult") -> None:
        panel._current_result = result
        on_evaluation_complete(panel, result)
    
    try:
        logger.info(f"Launching async evaluation: {len(tasks)} tasks, model_type={model_type}")
        panel._harness.run_evaluation_async(
            model_name=model,
            tasks=tasks,
            model_args=model_args,
            batch_size=batch_size,
            limit=limit,
            num_fewshot=num_fewshot,
            device=device,
            output_path=output_path,
            log_samples=log_samples,
            cache_requests=cache_requests,
            check_integrity=check_integrity,
            model_type=model_type,
            callback=on_complete,
        )
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}", exc_info=True)
        panel._log(f"[eval] Error starting evaluation: {e}")
        panel._is_running = False
        panel.start_btn.config(state="normal")
        panel.stop_btn.config(state="disabled")
        messagebox.showerror("Evaluation Error", f"Failed to start evaluation:\n{e}")


def on_progress_update(panel: "EvaluationPanel", progress: float, status_msg: str) -> None:
    """Handle progress updates from harness.
    
    Args:
        panel: The evaluation panel instance
        progress: Progress value (0.0 to 1.0)
        status_msg: Status message
    """
    try:
        # Stop indeterminate mode if active
        if panel.progress["mode"] == "indeterminate":
            panel.progress.stop()
            panel.progress["mode"] = "determinate"
        
        # Update progress bar
        panel.progress["value"] = min(100, max(0, progress * 100))
        
        # Update status label
        panel.progress_label.config(text=status_msg)
        
        # Force update to ensure GUI responsiveness
        panel.update_idletasks()
    except Exception as e:
        # Silently ignore update errors to prevent crashes
        pass


def on_evaluation_complete(panel: "EvaluationPanel", result: "EvaluationResult") -> None:
    """Handle evaluation completion.
    
    Args:
        panel: The evaluation panel instance
        result: The evaluation result
    """
    logger.info(f"Evaluation complete: status={result.status}, score={result.overall_score:.2%}")
    panel._is_running = False
    panel.start_btn.config(state="normal")
    panel.stop_btn.config(state="disabled")
    
    # Stop progress bar animation
    try:
        if panel.progress["mode"] == "indeterminate":
            panel.progress.stop()
        panel.progress["mode"] = "determinate"
    except Exception:
        pass
    
    if result.status == "completed":
        logger.info(f"Evaluation successful: duration={result.duration_str}, benchmarks={len(result.benchmark_scores)}")
        panel._log(f"[eval] Evaluation completed in {result.duration_str}")
        panel._log(f"[eval] Overall score: {result.overall_score:.2%}")
        
        # Update results display
        if result.benchmark_scores:
            # Show top 3 benchmark scores
            sorted_scores = sorted(
                result.benchmark_scores.items(),
                key=lambda x: list(x[1].get("scores", {}).values())[0] if x[1].get("scores") else 0,
                reverse=True
            )[:3]
            
            result_parts = [f"Overall: {result.overall_score:.1%}"]
            for task_name, task_data in sorted_scores:
                scores = task_data.get("scores", {})
                if scores:
                    score_val = list(scores.values())[0]
                    result_parts.append(f"{task_name}: {score_val:.1%}")
            
            panel.results_label.config(text=" | ".join(result_parts))
        else:
            panel.results_label.config(text=f"Overall: {result.overall_score:.1%}")
        
        panel.progress["value"] = 100
        panel.progress_label.config(text="completed")
        
        # Enable result buttons
        if hasattr(panel, "_result_buttons"):
            for btn in panel._result_buttons:
                btn.config(state="normal")
        
        # Save to history database
        try:
            # Get configuration
            selected_benchmarks = panel.selected_benchmarks_var.get()
            tasks = [t.strip() for t in selected_benchmarks.split(",") if t.strip()]
            
            # Get device from resources panel for history
            history_device = "cuda:0"
            try:
                rp = getattr(panel, "_resources_panel", None)
                if rp is not None:
                    rvals = rp.get_values()
                    run_device = str(rvals.get("run_device") or "cuda:0")
                    history_device = run_device
            except Exception:
                pass
            
            config = {
                "model_source": panel.model_source_var.get(),
                "batch_size": panel.batch_size_var.get(),
                "limit": panel.limit_var.get(),
                "num_fewshot": panel.num_fewshot_var.get(),
                "device": history_device,
                "output_path": panel.output_path_var.get(),
                "log_samples": panel.log_samples_var.get(),
                "cache_requests": panel.cache_requests_var.get(),
                "check_integrity": panel.check_integrity_var.get(),
            }
            
            # Find samples files if log_samples was enabled
            samples_path = ""
            log_samples_enabled = config.get("log_samples", False)
            
            panel._log(f"[eval] Log samples enabled: {log_samples_enabled}")
            panel._log(f"[eval] Result output path: {result.output_path}")
            
            if log_samples_enabled and result.output_path:
                try:
                    from pathlib import Path
                    # Ensure we have an absolute path
                    base_output_dir = Path(result.output_path)
                    if not base_output_dir.is_absolute():
                        # Make it absolute relative to project root
                        base_output_dir = Path(panel._project_root) / base_output_dir
                    
                    panel._log(f"[eval] Checking for samples in: {base_output_dir}")
                    
                    if base_output_dir.exists():
                        # lm-eval creates subdirectories based on model name
                        # Search in the base directory AND subdirectories
                        sample_files = list(base_output_dir.glob("samples_*.jsonl"))
                        sample_files.extend(list(base_output_dir.glob("*/samples_*.jsonl")))
                        
                        if sample_files:
                            # Use the directory containing the samples
                            # (could be base_output_dir or a subdirectory)
                            samples_dir = sample_files[0].parent
                            samples_path = str(samples_dir)
                            panel._log(f"[eval] Found {len(sample_files)} sample file(s) in {samples_dir.name}")
                        else:
                            panel._log(f"[eval] No sample files found (log_samples was enabled)")
                    else:
                        panel._log(f"[eval] Output directory does not exist: {base_output_dir}")
                except Exception as e:
                    panel._log(f"[eval] Error checking for samples: {e}")
            
            history = getattr(panel, "_history", None)
            if history is None:
                panel._log("[eval] Warning: Evaluation history is unavailable; results were not persisted.")
            else:
                eval_id = history.save_evaluation(
                    result=result,
                    model_name=panel.model_name_var.get(),
                    model_source=panel.model_source_var.get(),
                    model_args="dtype=auto",  # Default model args
                    tasks=tasks,
                    config=config,
                    samples_path=samples_path,
                )
                
                logger.info(f"Saved evaluation to history: ID={eval_id}, tasks={len(tasks)}, samples={bool(samples_path)}")
                panel._log(f"[eval] Saved to history (ID: {eval_id})")
            
            # Refresh recent results list
            if hasattr(panel, "recent_tree"):
                from . import ui_builders
                ui_builders._refresh_recent_results(panel)
        except Exception as e:
            logger.error(f"Failed to save evaluation to history: {e}", exc_info=True)
            panel._log(f"[eval] Warning: Failed to save to history: {e}")
        
    elif result.status == "cancelled":
        logger.info("Evaluation cancelled by user")
        panel._log("[eval] Evaluation cancelled by user")
        panel.progress["value"] = 0
        panel.progress_label.config(text="cancelled")
        panel.results_label.config(text="")
        
    elif result.status == "failed":
        logger.error(f"Evaluation failed: {result.error_message}")
        panel._log(f"[eval] Evaluation failed: {result.error_message}")
        panel.progress["value"] = 0
        panel.progress_label.config(text="failed")
        panel.results_label.config(text="")
        messagebox.showerror(
            "Evaluation Failed",
            f"Evaluation failed:\n{result.error_message}"
        )


def on_stop_evaluation(panel: "EvaluationPanel") -> None:
    """Stop evaluation.
    
    Args:
        panel: The evaluation panel instance
    """
    if not panel._is_running:
        logger.debug("Stop evaluation called but no evaluation running")
        return
    
    logger.info("User requested evaluation cancellation")
    panel._log("[eval] Stopping evaluation...")
    panel._harness.cancel()
    
    # UI will be updated by on_evaluation_complete callback
