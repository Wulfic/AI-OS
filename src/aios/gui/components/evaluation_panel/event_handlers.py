"""Event handlers for evaluation operations."""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import TYPE_CHECKING

from aios.gui.services import (
    build_device_message,
    emit_analytics_event,
    resolve_inference_devices,
    resolve_inference_devices_from_state,
    warning_message,
)
from .multi_gpu import MultiGpuEvaluationRunner

try:  # pragma: no cover - runtime dependency
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None

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


def _resolve_brain_path(brain_name: str, project_root: str | Path | None) -> Path:
    """Resolve the on-disk path for an ACTv1 brain bundle."""
    if system_paths is not None:
        try:
            return system_paths.get_brain_family_dir("actv1") / brain_name
        except Exception:
            logger.debug("Failed to resolve ProgramData brain path", exc_info=True)

    base = Path(project_root) if project_root else Path.cwd()
    return base / "artifacts" / "brains" / "actv1" / brain_name


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
            "Dependency Error",
            "The lm-evaluation-harness package is missing.\n\n"
            "This component is bundled with AI-OS but failed to load.\n"
            "Please run 'aios doctor' in a terminal to repair dependencies."
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
    
    # Resolve device selection prior to launching the evaluation so downstream
    # runners can honour the same configuration.
    resources_panel = getattr(panel, "_resources_panel", None)
    try:
        selection = resolve_inference_devices(resources_panel)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to resolve evaluation device selection: %s", exc, exc_info=True)
        selection = resolve_inference_devices_from_state({}, platform.system())

    panel._last_device_selection = selection
    env_overrides = dict(selection.env_overrides)
    if selection.requested_devices:
        env_overrides["AIOS_REQUESTED_DEVICES"] = ",".join(selection.requested_devices)
    physical_visible_devices: list[str] = []
    alias_map: dict[str, str] = {}
    if isinstance(selection.metadata, dict):
        maybe_physical = selection.metadata.get("physical_visible_devices")
        if isinstance(maybe_physical, list):
            physical_visible_devices = [str(dev) for dev in maybe_physical]
        maybe_alias_map = selection.metadata.get("alias_physical_map")
        if isinstance(maybe_alias_map, dict):
            alias_map = {str(k): str(v) for k, v in maybe_alias_map.items()}

    visible_descriptors = physical_visible_devices or selection.visible_devices
    visible_aliases = ",".join(visible_descriptors) if visible_descriptors else selection.primary_device
    env_overrides["AIOS_VISIBLE_DEVICES"] = visible_aliases
    if selection.visible_devices:
        env_overrides.setdefault("AIOS_VISIBLE_ALIAS_DEVICES", ",".join(selection.visible_devices))
    if physical_visible_devices:
        env_overrides.setdefault("AIOS_VISIBLE_PHYSICAL_DEVICES", ",".join(physical_visible_devices))
    env_overrides["AIOS_DEVICE_KIND"] = selection.device_kind
    if selection.warnings:
        env_overrides["AIOS_DEVICE_WARNINGS"] = ",".join(selection.warnings)

    selection_message = build_device_message(selection)
    logger.info("Evaluation device selection resolved: %s", selection.to_log_payload())
    panel._log(f"[eval] Device selection: {selection_message}")

    for token in selection.warnings:
        panel._log(f"[eval] {warning_message(token)}")

    platform_name = selection.metadata.get("os") if isinstance(selection.metadata, dict) else None
    platform_name = platform_name or platform.system().lower()
    try:
        emit_analytics_event(
            "eval.device_selection",
            {
                "platform": platform_name,
                "device_kind": selection.device_kind,
                "requested": ",".join(selection.requested_devices) or "none",
                "effective": visible_aliases or selection.primary_device,
                "warnings": ",".join(selection.warnings) or "none",
                "multi_gpu": 1 if selection.device_kind == "cuda" and len(selection.visible_devices) > 1 else 0,
            },
            context={
                "requested_count": len(selection.requested_devices),
                "effective_count": len(selection.visible_devices),
            },
        )
    except Exception:
        logger.debug("Failed to emit eval.device_selection analytics event", exc_info=True)

    # Update UI state
    if hasattr(panel, "_reset_progress_tracker"):
        try:
            panel._reset_progress_tracker()
        except Exception:
            logger.debug("Failed to reset evaluation progress tracker", exc_info=True)

    panel._is_running = True
    panel.start_btn.config(state="disabled")
    panel.stop_btn.config(state="normal")
    panel.progress["mode"] = "indeterminate"
    panel.progress.start(10)
    panel.progress_label.config(text="initializing...")
    panel.results_label.config(text="")
    panel._active_eval_runner = None
    
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

    def _deduplicate(items):
        ordered: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    shard_devices: list[str] = []
    if selection.device_kind == "cuda":
        phys_candidates = [dev for dev in physical_visible_devices if isinstance(dev, str) and dev.startswith("cuda")]
        shard_devices = _deduplicate(phys_candidates)
        if not shard_devices:
            shard_devices = _deduplicate([dev for dev in selection.requested_devices if dev.startswith("cuda")])
        if not shard_devices and alias_map:
            mapped = [alias_map.get(dev) for dev in selection.visible_devices if dev.startswith("cuda")]
            shard_devices = _deduplicate([dev for dev in mapped if isinstance(dev, str)])
        if not shard_devices and selection.primary_device.startswith("cuda"):
            shard_devices = [alias_map.get(selection.primary_device, selection.primary_device)]
    else:
        shard_devices = [selection.primary_device]

    requested_summary = selection.requested_devices or shard_devices or [selection.primary_device]
    try:
        if selection.device_kind == "cuda":
            panel.progress_label.config(
                text=f"initializing... GPU: {', '.join(shard_devices or [visible_descriptors[0] if visible_descriptors else selection.primary_device])} (requested {', '.join(requested_summary)})"
            )
        else:
            panel.progress_label.config(text=f"initializing... device {selection.primary_device}")
    except Exception:
        panel.progress_label.config(text="initializing...")

    device = selection.primary_device if selection.device_kind == "cuda" else selection.primary_device
    multi_gpu_enabled = selection.device_kind == "cuda" and len(shard_devices) > 1

    logger.info(
        "Evaluation config: batch_size=%s, limit=%s, num_fewshot=%s, devices=%s",
        batch_size,
        limit,
        num_fewshot,
        shard_devices,
    )

    output_path = panel.output_path_var.get() or "artifacts/evaluation"
    log_samples = panel.log_samples_var.get()
    cache_requests = panel.cache_requests_var.get()
    check_integrity = panel.check_integrity_var.get()
    
    # Determine model type - auto-detect if it's a brain or external model
    model_type = "hf"  # Default to HuggingFace
    brain_name = model
    brain_path = _resolve_brain_path(brain_name, panel._project_root)
    
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
                    
                    # Check for checkpoint
                    checkpoint = brain_path / "actv1_student.safetensors"
                    if not checkpoint.exists():
                        msg = f"Missing checkpoint: {checkpoint}\nCannot evaluate brain without model weights."
                        panel._log(f"[eval] Error: {msg}")
                        messagebox.showerror("Missing Checkpoint", msg)
                        panel.start_btn.config(state="normal")
                        panel.stop_btn.config(state="disabled")
                        panel._is_running = False
                        panel.progress.stop()
                        panel.progress_label.config(text="failed")
                        return

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
            model_type = "hf"

    model_kwargs = {
        "model_name": model,
        "model_args": model_args,
        "batch_size": batch_size,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "device": device,
        "output_path": output_path,
        "log_samples": log_samples,
        "cache_requests": cache_requests,
        "check_integrity": check_integrity,
        "model_type": model_type,
    }

    def on_complete(result: "EvaluationResult") -> None:
        panel._current_result = result
        panel._active_eval_runner = None
        on_evaluation_complete(panel, result)

    if multi_gpu_enabled:
        requested_str = ", ".join(requested_summary)
        shard_str = ", ".join(shard_devices)
        panel._log(f"[eval] Requested CUDA devices: {requested_str}")
        panel._log(f"[eval] Launching shard workers on devices {shard_str} (mode=fanout)")
        logger.info(
            "Launching multi-GPU evaluation: devices=%s, tasks=%d, env_overrides=%s",
            shard_devices,
            len(tasks),
            env_overrides,
        )
        try:
            runner = MultiGpuEvaluationRunner(
                panel=panel,
                tasks=tasks,
                devices=shard_devices,
                base_env=env_overrides,
                model_kwargs=model_kwargs,
                on_complete=on_complete,
            )
            panel._active_eval_runner = runner
            runner.start()
            return
        except Exception as exc:
            panel._log(f"[eval] Multi-GPU fan-out failed: {exc}; falling back to single GPU")
            logger.error("Failed to launch multi-GPU evaluation", exc_info=True)
            panel._active_eval_runner = None
            # Fall through to single-device execution

    panel._active_eval_runner = None
    try:
        logger.info(
            "Launching async evaluation: %d tasks, model_type=%s, env_overrides=%s",
            len(tasks),
            model_type,
            env_overrides,
        )
        panel._get_harness().run_evaluation_async(
            tasks=tasks,
            callback=on_complete,
            env_overrides=env_overrides,
            **model_kwargs,
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

        if hasattr(panel, "_record_progress_update"):
            try:
                panel._record_progress_update(progress, status_msg)
            except Exception:
                logger.debug("Failed to log progress update", exc_info=True)
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
            selection = getattr(panel, "_last_device_selection", None)
            if selection is not None:
                history_device = selection.primary_device
            else:
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

            if selection is not None:
                config["device_visible"] = selection.visible_devices
                config["device_requested"] = selection.requested_devices
                config["device_warnings"] = selection.warnings
            
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

    if hasattr(panel, "_reset_progress_tracker"):
        try:
            panel._reset_progress_tracker()
        except Exception:
            logger.debug("Failed to reset progress tracker after evaluation", exc_info=True)


def on_stop_evaluation(panel: "EvaluationPanel") -> None:
    """Stop evaluation.
    
    Args:
        panel: The evaluation panel instance
    """
    if not panel._is_running and not getattr(panel, "_active_eval_runner", None):
        logger.debug("Stop evaluation called but no evaluation running")
        return

    logger.info("User requested evaluation cancellation")
    panel._log("[eval] Stopping evaluation...")

    # Update UI immediately so shutdown paths aren't blocked waiting on callbacks
    panel._is_running = False
    try:
        panel.start_btn.config(state="normal")
        panel.stop_btn.config(state="disabled")
    except Exception:
        pass

    runner = getattr(panel, "_active_eval_runner", None)
    if runner is not None:
        try:
            runner.cancel(wait=False)
        except Exception as exc:
            logger.debug("Failed to cancel multi-GPU runner: %s", exc, exc_info=True)
        else:
            if not runner.wait_for_exit(timeout=10.0):
                logger.warning("Multi-GPU evaluation runner did not terminate within timeout during stop")
        panel._active_eval_runner = None
    else:
        try:
            panel._get_harness().cancel()
        except Exception as exc:
            logger.debug("Failed to cancel evaluation harness: %s", exc, exc_info=True)
        panel._get_harness().wait_for_completion(timeout=10.0)

    # UI will be finalised by on_evaluation_complete if it runs later
