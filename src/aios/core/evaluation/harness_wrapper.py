"""Wrapper for EleutherAI's lm-evaluation-harness.

This module provides a Python interface to run evaluations using the lm_eval CLI,
with support for real-time progress tracking, cancellation, and result parsing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    
    overall_score: float = 0.0
    benchmark_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_results: dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    status: str = "pending"  # pending, running, completed, failed, cancelled
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Calculate evaluation duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def duration_str(self) -> str:
        """Format duration as human-readable string."""
        duration = self.duration
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"


class HarnessWrapper:
    """Wrapper for running lm-evaluation-harness evaluations.
    
    This class handles:
    - Subprocess execution of lm_eval CLI
    - Real-time output parsing and progress tracking
    - Cancellation support
    - Results parsing from results.json
    - Error handling and logging
    """
    
    def __init__(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Initialize the harness wrapper.
        
        Args:
            log_callback: Function to call with log messages
            progress_callback: Function to call with (progress_pct, status_msg)
        """
        self.log_callback = log_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cancelled = False
        self._running = False
        self._start_time = 0.0
        self._last_progress_time = 0.0
        
    def is_running(self) -> bool:
        """Check if an evaluation is currently running."""
        return self._running
    
    def run_evaluation(
        self,
        model_name: str,
        tasks: list[str],
        *,
        model_args: str = "",
        batch_size: str = "auto",
        limit: Optional[float] = None,
        num_fewshot: int = 5,
        device: str = "cuda:0",
        output_path: str = "artifacts/evaluation",
        log_samples: bool = False,
        cache_requests: bool = True,
        check_integrity: bool = False,
        model_type: str = "hf",
        env_overrides: Optional[dict[str, str]] = None,
    ) -> EvaluationResult:
        """Run an evaluation synchronously.
        
        Args:
            model_name: Model name or path
            tasks: List of task names to evaluate
            model_args: Additional model arguments (e.g., "dtype=auto")
            batch_size: Batch size or "auto"
            limit: Limit number of samples per task (for quick testing)
            num_fewshot: Number of few-shot examples
            device: Device to use (cuda:0, cpu, etc.)
            output_path: Directory to save results
            log_samples: Whether to log individual samples
            cache_requests: Whether to cache requests
            check_integrity: Whether to check data integrity
            model_type: Model type (hf, vllm, local-completions, etc.)
            env_overrides: Environment variables to apply for the subprocess
            
        Returns:
            EvaluationResult with scores and metadata
        """
        if self._running:
            raise RuntimeError("An evaluation is already running")
        
        result = EvaluationResult(start_time=time.time(), status="running")
        self._running = True
        self._cancelled = False
        self._start_time = time.time()
        self._last_progress_time = time.time()
        
        try:
            # Platform-specific task filtering
            filtered_tasks = self._filter_tasks(tasks)
            if not filtered_tasks:
                raise RuntimeError("All selected tasks are unsupported in the current environment")

            effective_limit = limit
            if limit is not None:
                try:
                    limit_value = float(limit)
                except (TypeError, ValueError):
                    limit_value = None

                if limit_value is not None and limit_value >= 1 and limit_value < num_fewshot:
                    effective_limit = float(num_fewshot)
                    self.log_callback(
                        f"[eval] Adjusting limit from {limit} to {effective_limit} to satisfy num_fewshot={num_fewshot}"
                    )

            # Build command
            cmd = self._build_command(
                model_name=model_name,
                tasks=filtered_tasks,
                model_args=model_args,
                batch_size=batch_size,
                limit=effective_limit,
                num_fewshot=num_fewshot,
                device=device,
                output_path=output_path,
                log_samples=log_samples,
                cache_requests=cache_requests,
                check_integrity=check_integrity,
                model_type=model_type,
            )
            
            self.log_callback(f"[eval] Running command: {' '.join(cmd)}")
            
            # Create output directory and cache root so lm-eval can persist artifacts
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_root = output_dir / ".lm_eval_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            result.output_path = output_path

            # Ensure lm-eval default cache dir exists so caching never fails on Windows installs
            try:
                import lm_eval  # type: ignore

                package_cache_dir = Path(lm_eval.__file__).resolve().parent / "caching" / ".cache"
                package_cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - best effort
                self.log_callback(f"[eval] Warning: unable to prepare default lm-eval cache dir: {exc}")
            
            # For AIOS models, ensure the adapter is registered by importing it
            # in this process. The subprocess will inherit the registration if
            # it uses the same Python environment.
            if model_type == "aios":
                try:
                    from aios.core.evaluation.aios_lm_eval_adapter import AIOSBrainModel
                    self.log_callback("[eval] AIOS adapter module imported (registration happens in subprocess)")
                except ImportError as e:
                    self.log_callback(f"[eval] Warning: Could not import AIOS adapter: {e}")
            
            # Build a Python command that imports the adapter before running lm_eval
            # This ensures the model is registered in the subprocess
            if model_type == "aios":
                # Wrap the lm_eval command with Python -c that imports the adapter first
                python_exe = sys.executable
                
                # Better: Create a small wrapper script
                wrapper_script = f"""
import sys
# Register the AIOS model
from aios.core.evaluation.aios_lm_eval_adapter import AIOSBrainModel
print('[eval] AIOS adapter registered in subprocess', file=sys.stderr, flush=True)

# Now run lm_eval CLI
from lm_eval.__main__ import cli_evaluate
sys.argv = {cmd}  # Restore original command line
cli_evaluate()
"""
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(wrapper_script)
                    wrapper_file = f.name
                
                # Run the wrapper script instead
                cmd = [python_exe, wrapper_file]
            
            # Set up environment with UTF-8 support for Windows
            env = os.environ.copy()
            if env_overrides:
                env.update({k: v for k, v in env_overrides.items()})
            env['PYTHONIOENCODING'] = 'utf-8'
            # Allow code-eval metrics (HumanEval/MBPP) to execute sandboxed code
            env.setdefault('HF_ALLOW_CODE_EVAL', '1')
            env.setdefault('LM_EVAL_CACHE_PATH', str(cache_root))
            env.setdefault('LM_HARNESS_CACHE_PATH', str(cache_root))
            
            # Set up HF_XET_CACHE to a writable location to avoid permission errors
            # xet-core (used by huggingface_hub) needs a writable cache directory
            if 'HF_HOME' in env:
                hf_home = Path(env['HF_HOME'])
                xet_cache = hf_home / "xet"
            else:
                # Default to user's cache directory
                xet_cache = Path.home() / ".cache" / "huggingface" / "xet"
            try:
                xet_cache.mkdir(parents=True, exist_ok=True)
                env.setdefault('HF_XET_CACHE', str(xet_cache.resolve()))
            except PermissionError:
                # Fallback to home directory
                fallback_xet = Path.home() / ".cache" / "huggingface" / "xet"
                fallback_xet.mkdir(parents=True, exist_ok=True)
                env['HF_XET_CACHE'] = str(fallback_xet.resolve())
                self.log_callback(f"[eval] Using fallback XET cache: {fallback_xet}")
            except Exception as e:
                self.log_callback(f"[eval] Warning: Could not set up XET cache: {e}")
            
            # Run subprocess with explicit UTF-8 encoding
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace problematic characters instead of crashing
                bufsize=1,
                universal_newlines=True,
                env=env,
            )
            
            # Parse output in real-time
            self._parse_output(self._process)
            
            # Wait for completion
            return_code = self._process.wait()
            
            if self._cancelled:
                result.status = "cancelled"
                self.log_callback("[eval] Evaluation cancelled by user")
            elif return_code != 0:
                result.status = "failed"
                result.error_message = f"Process exited with code {return_code}"
                self.log_callback(f"[eval] Evaluation failed: {result.error_message}")
            else:
                result.status = "completed"
                self.log_callback("[eval] Evaluation completed successfully")
                
                # Parse results file
                self._parse_results(result, output_path)
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            self.log_callback(f"[eval] Evaluation error: {e}")
        
        finally:
            result.end_time = time.time()
            self._running = False
            self._process = None
            
        return result

    def _filter_tasks(self, tasks: list[str]) -> list[str]:
        """Filter tasks that are unsupported on the current platform.

        Returns a possibly reduced list of tasks and logs any removals.
        """
        filtered = list(tasks)

        if sys.platform.startswith("win"):
            unsupported = {"humaneval", "mbpp"}
            removed = [task for task in filtered if task in unsupported]
            if removed:
                filtered = [task for task in filtered if task not in unsupported]
                self.log_callback(
                    "[eval] Skipping Windows-unsupported code-eval tasks: " + ", ".join(removed)
                )
                self.log_callback(
                    "[eval] (Set up Linux or WSL to include code generation benchmarks.)"
                )

        return filtered
    
    def run_evaluation_async(
        self,
        model_name: str,
        tasks: list[str],
        callback: Optional[Callable[[EvaluationResult], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Run an evaluation asynchronously in a background thread.
        
        Args:
            model_name: Model name or path
            tasks: List of task names to evaluate
            callback: Function to call when evaluation completes
            **kwargs: Additional arguments passed to run_evaluation
        """
        if self._running:
            raise RuntimeError("An evaluation is already running")
        
        def _run_thread():
            try:
                result = self.run_evaluation(model_name, tasks, **kwargs)
                if callback:
                    callback(result)
            finally:
                self._thread = None
        
        self._thread = threading.Thread(target=_run_thread, daemon=True)
        self._thread.start()
    
    def cancel(self) -> None:
        """Cancel the currently running evaluation."""
        if not self._running or self._process is None:
            return
        
        self._cancelled = True
        self.log_callback("[eval] Cancelling evaluation...")
        
        try:
            self._process.terminate()
            # Give it a moment to terminate gracefully
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        except Exception as e:
            self.log_callback(f"[eval] Error cancelling: {e}")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for the background evaluation thread to finish.

        Args:
            timeout: Maximum time to wait in seconds (None waits indefinitely)

        Returns:
            True if the thread finished, False if it is still running after timeout
        """
        thread = self._thread
        if thread is None:
            return True

        if thread.is_alive():
            thread.join(timeout)

        finished = not thread.is_alive()
        if finished:
            self._thread = None
        return finished
    
    def _build_command(
        self,
        model_name: str,
        tasks: list[str],
        model_args: str,
        batch_size: str,
        limit: Optional[float],
        num_fewshot: int,
        device: str,
        output_path: str,
        log_samples: bool,
        cache_requests: bool,
        check_integrity: bool,
        model_type: str,
    ) -> list[str]:
        """Build the lm_eval command."""
        cmd = ["lm_eval", "--model", model_type]
        
        # Model configuration
        if model_type == "aios":
            # For AI-OS brains, model_name contains the brain_path parameter
            # e.g., "brain_path=/path/to/brain"
            cmd.extend(["--model_args", model_name])
        else:
            # For HF models, build pretrained argument
            pretrained_arg = f"pretrained={model_name}"
            if model_args:
                pretrained_arg += f",{model_args}"
            cmd.extend(["--model_args", pretrained_arg])
        
        # Tasks
        cmd.extend(["--tasks", ",".join(tasks)])
        
        # Batch size
        cmd.extend(["--batch_size", batch_size])
        
        # Limit
        if limit is not None and limit > 0:
            cmd.extend(["--limit", str(limit)])
        
        # Few-shot
        cmd.extend(["--num_fewshot", str(num_fewshot)])
        
        # Device
        if device and device != "auto":
            cmd.extend(["--device", device])
        
        # Output
        cmd.extend(["--output_path", output_path])
        
        # Optional flags
        if log_samples:
            cmd.append("--log_samples")
        
        if cache_requests:
            cmd.extend(["--cache_requests", "true"])
        
        if check_integrity:
            cmd.append("--check_integrity")
        
        return cmd
    
    def _parse_output(self, process: subprocess.Popen) -> None:
        """Parse subprocess output in real-time."""
        if process.stdout is None:
            return
        
        total_tasks = 0
        completed_tasks = 0
        current_task = ""
        last_update_time = time.time()
        dataset_gen_count = 0  # Track dataset generation messages
        last_logged_dataset = None  # Avoid spam
        
        for line in iter(process.stdout.readline, ""):
            if not line or self._cancelled:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Filter out repetitive dataset generation messages to reduce spam
            should_log = True
            if "Generating" in line and "split" in line:
                dataset_gen_count += 1
                # Only log every 5th dataset generation or unique datasets
                if dataset_gen_count % 5 != 0 and line == last_logged_dataset:
                    should_log = False
                else:
                    last_logged_dataset = line
            
            # Log the line (filtered for less spam)
            if should_log:
                self.log_callback(f"[eval] {line}")
            
            # Update last progress time
            current_time = time.time()
            self._last_progress_time = current_time
            
            # Parse progress information
            progress_updated = False
            
            # Pattern 0: Dataset generation (common early step)
            if "Generating" in line and "split" in line:
                # Show cumulative progress
                if dataset_gen_count < 10:
                    progress = 0.02
                elif dataset_gen_count < 30:
                    progress = 0.05
                elif dataset_gen_count < 60:
                    progress = 0.08
                else:
                    progress = 0.10
                self.progress_callback(progress, f"Loading datasets ({dataset_gen_count} loaded)...")
                progress_updated = True
            
            # Pattern 1: "Running loglikelihood requests"
            if "Running" in line and "requests" in line:
                self.progress_callback(0.15, "Running evaluations...")
                progress_updated = True
            
            # Pattern 2: Task name - multiple patterns
            task_patterns = [
                r"Task:\s*(\S+)",
                r"Evaluating\s+(?:on\s+)?(\S+)",
                r"Running\s+task\s+(\S+)",
                r"^\s*(\w+)\s*\|",  # Table-like output with task name
                r"Selected Tasks:.*\[(.*?)\]",  # Matches "Selected Tasks: ['task1', 'task2']"
            ]
            for pattern in task_patterns:
                task_match = re.search(pattern, line, re.IGNORECASE)
                if task_match:
                    current_task = task_match.group(1)
                    if current_task and current_task not in ["", "task", "Task"]:
                        # Clean up task name (remove quotes, brackets, etc)
                        current_task = current_task.strip("'\"[]")
                        self.progress_callback(0.2, f"Evaluating {current_task}...")
                        progress_updated = True
                        break
            
            # Pattern 3: Loading/downloading model
            if any(keyword in line.lower() for keyword in ["loading", "downloading", "fetching"]):
                if not progress_updated and ("model" in line.lower() or "checkpoint" in line.lower() or "tokenizer" in line.lower()):
                    self.progress_callback(0.08, "Loading model...")
                    progress_updated = True
            
            # Pattern 4: Progress bar or percentage
            pct_match = re.search(r"(\d+)%", line)
            if pct_match:
                pct = int(pct_match.group(1))
                status_text = f"Processing... {pct}%"
                if current_task:
                    status_text = f"{current_task}: {pct}%"
                self.progress_callback(pct / 100.0, status_text)
                progress_updated = True
            
            # Pattern 5: AIOS Adapter progress (high priority)
            aios_progress_match = re.search(r"\[AIOS Adapter\] Progress: (\d+)/(\d+)", line)
            if aios_progress_match and not progress_updated:
                current = int(aios_progress_match.group(1))
                total = int(aios_progress_match.group(2))
                if total > 0:
                    # Use full 0-100% range based on actual progress
                    pct = current / total
                    status_text = f"Evaluating: {current}/{total} ({int(pct*100)}%)"
                    if current_task:
                        status_text = f"{current_task}: {current}/{total} ({int(pct*100)}%)"
                    self.progress_callback(pct, status_text)
                    progress_updated = True
            
            # Pattern 6: Batch/iteration indicators (fallback)
            if not progress_updated:
                batch_match = re.search(r"(\d+)/(\d+)", line)
                if batch_match:
                    current = int(batch_match.group(1))
                    total = int(batch_match.group(2))
                    if total > 0:
                        pct = current / total
                        status_text = f"Processing {current}/{total}"
                        if current_task:
                            status_text = f"{current_task}: {current}/{total}"
                        self.progress_callback(pct, status_text)
                        progress_updated = True
            
            # Pattern 7: Completion messages
            if "results" in line.lower() and "saved" in line.lower():
                self.progress_callback(0.95, "Saving results...")
                progress_updated = True
            
            if "evaluation complete" in line.lower():
                self.progress_callback(1.0, "Completed")
                progress_updated = True
            
            # Periodic keepalive: If no progress update for 5 seconds, show elapsed time
            if not progress_updated and (current_time - last_update_time) >= 5.0:
                elapsed = current_time - self._start_time
                elapsed_str = self._format_elapsed(elapsed)
                status_text = f"Running... {elapsed_str}"
                if current_task:
                    status_text = f"{current_task} - {elapsed_str}"
                self.progress_callback(0.5, status_text)
                last_update_time = current_time
    
    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as MM:SS or HH:MM:SS."""
        seconds = int(seconds)
        if seconds < 3600:
            mins, secs = divmod(seconds, 60)
            return f"{mins:02d}:{secs:02d}"
        else:
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
    
    def _parse_results(self, result: EvaluationResult, output_path: str) -> None:
        """Parse the results.json file."""
        try:
            # Look for results.json in output directory
            results_file = Path(output_path) / "results.json"
            
            if not results_file.exists():
                # Try alternative locations - including timestamped files
                for pattern in ["**/results_*.json", "**/results.json", "*_results.json"]:
                    matches = list(Path(output_path).glob(pattern))
                    if matches:
                        # Use the most recent file if multiple matches
                        results_file = max(matches, key=lambda p: p.stat().st_mtime)
                        self.log_callback(f"[eval] Found results file: {results_file.name}")
                        break
            
            if not results_file.exists():
                self.log_callback(f"[eval] Warning: results.json not found in {output_path}")
                return
            
            self.log_callback(f"[eval] Parsing results from: {results_file}")
            
            # Parse JSON
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            result.raw_results = data
            
            # Extract scores
            if "results" in data:
                results_data = data["results"]
                
                total_score = 0.0
                count = 0
                
                for task_name, task_data in results_data.items():
                    if not isinstance(task_data, dict):
                        continue
                    
                    # Extract primary metric (usually accuracy)
                    # Note: lm-eval stores metrics with format "metric,aggregation" (e.g., "acc,none")
                    score_dict = {}
                    
                    # First try standard metric names with aggregation suffix
                    for base_metric in ["acc", "accuracy", "acc_norm", "exact_match", "pass@1"]:
                        for key in task_data.keys():
                            # Match "acc,none", "accuracy,mean", etc.
                            if key.startswith(f"{base_metric},") or key == base_metric:
                                score = task_data[key]
                                if isinstance(score, (int, float)):
                                    # Store with base metric name for consistency
                                    score_dict[base_metric] = float(score)
                                    if "acc" in base_metric or base_metric == "exact_match":
                                        total_score += float(score)
                                        count += 1
                                    break  # Found this metric, move to next
                    
                    # Store all metrics for this task
                    result.benchmark_scores[task_name] = {
                        "scores": score_dict,
                        "raw": task_data,
                    }
                    
                    self.log_callback(f"[eval] Extracted scores for {task_name}: {score_dict}")
                
                # Calculate overall score
                if count > 0:
                    result.overall_score = total_score / count
                    self.log_callback(f"[eval] Calculated overall score from {count} metrics: {result.overall_score:.2%}")
                else:
                    self.log_callback(f"[eval] Warning: No valid metrics found to calculate overall score")
            
            self.log_callback(f"[eval] Parsed results: Overall score = {result.overall_score:.2%}")
            
        except Exception as e:
            self.log_callback(f"[eval] Error parsing results: {e}")
    
    @staticmethod
    def is_lm_eval_installed() -> bool:
        """Check if lm-evaluation-harness is installed.
        
        First tries to import the lm_eval module directly (more reliable),
        then falls back to checking the CLI command (for backwards compatibility).
        """
        # Primary check: try importing the module
        try:
            import lm_eval
            return True
        except ImportError:
            pass
        
        # Fallback check: try CLI (in case installed differently)
        try:
            result = subprocess.run(
                ["lm_eval", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_available_tasks() -> list[str]:
        """Get list of available evaluation tasks.
        
        Returns:
            List of task names, or empty list if unavailable
        """
        try:
            result = subprocess.run(
                ["lm_eval", "--tasks", "list"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode != 0:
                return []
            
            # Parse task names from output
            tasks = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("-"):
                    # Task names are usually listed one per line or comma-separated
                    if "," in line:
                        tasks.extend([t.strip() for t in line.split(",")])
                    else:
                        tasks.append(line)
            
            return sorted(tasks)
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return []
