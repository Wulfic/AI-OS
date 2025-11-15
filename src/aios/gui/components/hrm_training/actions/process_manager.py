"""Training process management for HRM panel.

This module handles:
- Building training config
- Creating multiprocessing.Process for training
- Setting up multiprocessing.Queue for output
- Monitoring process and updating GUI
- Managing stop Events
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any
from multiprocessing import Process, Queue
import queue

from aios.python_exec import get_preferred_python_executable
from ....utils.resource_management import submit_background

logger = logging.getLogger(__name__)


def launch_training_process(panel: Any, args: list[str], config: Any, stop_event: Any, graceful_stop_event: Any) -> None:
    """
    Launch training process using multiprocessing.Process.
    
    Args:
        panel: HRM training panel instance
        args: CLI arguments for training (informational only - not used with Process)
        config: TrainingConfig object
        stop_event: Multiprocessing Event for immediate stop
        graceful_stop_event: Multiprocessing Event for graceful stop
    """
    from aios.cli.hrm_hf.train_actv1 import run_training_multiprocessing_entry
    
    logger.info("Launching training process via multiprocessing.Process")
    logger.debug(f"Config: dataset={config.dataset_file}, batch_size={config.batch_size}, steps={config.steps}")
    
    # Create Queue for output communication
    output_queue = Queue()
    panel._output_queue = output_queue
    logger.debug("Created multiprocessing.Queue for process output")
    
    try:
        # CRITICAL: Set CUDA device IDs in environment BEFORE creating the Process
        # This ensures CUDA_VISIBLE_DEVICES is set before torch is imported in the child process
        cuda_ids_env = {}
        if hasattr(config, 'cuda_ids') and config.cuda_ids:
            cuda_ids_str = str(config.cuda_ids) if isinstance(config.cuda_ids, str) else ','.join(map(str, config.cuda_ids))
            cuda_ids_env['AIOS_CUDA_IDS'] = cuda_ids_str
            panel._log(f"[hrm] Setting CUDA devices for training process: {cuda_ids_str}")
            logger.info(f"Setting CUDA devices for training process: {cuda_ids_str}")
        
        # Ensure CUDA environment variables propagate to spawned process (Windows uses spawn)
        original_env: dict[str, str | None] = {}
        for key, value in cuda_ids_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Create and start multiprocessing.Process
        logger.debug("Creating multiprocessing.Process for training")
        try:
            proc = Process(
                target=_training_process_wrapper,
                args=(run_training_multiprocessing_entry, config, stop_event, graceful_stop_event, output_queue, cuda_ids_env),
                daemon=False
            )
            proc.start()
        finally:
            for key, previous in original_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        panel._proc = proc
        panel._log(f"[hrm] Training process started (PID={proc.pid})")
        logger.info(f"Training process started successfully (PID={proc.pid})")
        
        # Start output monitoring thread
        logger.debug("Starting output queue monitoring thread")
        _start_queue_monitoring(panel, output_queue)
        
        # Update UI state
        _update_ui_start(panel)
        
        # Start a separate thread to wait for process completion and cleanup
        def _wait_and_cleanup():
            logger.debug(f"Starting process wait thread for PID {proc.pid}")
            _wait_for_process(panel, proc)
            
            # Get exit code
            proc.join(timeout=1.0)
            rc = proc.exitcode if proc.exitcode is not None else 1
            panel._log(f"[hrm] Training process exited (rc={rc})")
            logger.info(f"Training process exited with code {rc}")
            logger.debug(f"Process wait thread completed for PID {proc.pid}")
            
            # Cleanup and update UI
            _update_ui_done(panel)
            _cleanup_process_resources(panel)
        
        try:
            submit_background(
                "hrm-process-wait",
                _wait_and_cleanup,
                pool=getattr(panel, "_worker_pool", None),
            )
            logger.debug("Process wait task queued for PID %s", proc.pid)
        except RuntimeError as exc:
            logger.error("Failed to queue process wait task: %s", exc)
            _wait_and_cleanup()
        
    except Exception as e:
        panel._log(f"[hrm] Process launch error: {e}")
        logger.error(f"Failed to launch training process: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        _update_ui_done(panel)
        _cleanup_process_resources(panel)


def _training_process_wrapper(entry_func, config, stop_event, graceful_stop_event, output_queue, cuda_env):
    """Wrapper to redirect output to Queue and call training entry point.
    
    Args:
        entry_func: Training entry point function
        config: Training configuration
        stop_event: Event for immediate stop
        graceful_stop_event: Event for graceful stop
        output_queue: Queue for output communication
        cuda_env: Dict with CUDA environment variables to set (e.g., {'AIOS_CUDA_IDS': '0,1'})
    """
    import sys
    import io
    
    # Set CUDA environment variables BEFORE any imports
    # This ensures they're set before torch.cuda is initialized
    if cuda_env:
        for key, value in cuda_env.items():
            os.environ[key] = value
    
    class QueueWriter(io.TextIOBase):
        """Redirects writes to multiprocessing.Queue."""
        def __init__(self, queue_obj):
            self.queue = queue_obj
            
        def write(self, msg):
            if msg and msg.strip():
                try:
                    self.queue.put(('stdout', msg))
                except:
                    pass
            return len(msg)
        
        def flush(self):
            pass
    
    # Redirect stdout/stderr to Queue
    sys.stdout = QueueWriter(output_queue)
    sys.stderr = QueueWriter(output_queue)
    
    try:
        # Call the actual training function
        entry_func(config, stop_event=stop_event, graceful_stop_event=graceful_stop_event)
        output_queue.put(('exit', 0))
    except KeyboardInterrupt:
        output_queue.put(('exit', 1))
    except Exception as e:
        output_queue.put(('error', str(e)))
        output_queue.put(('exit', 1))
        raise


def _start_queue_monitoring(panel: Any, output_queue: Queue) -> None:
    """Monitor output Queue using the Tkinter event loop."""
    logger.debug("Starting queue monitor via Tk callbacks")

    def _pump() -> None:
        drained = False
        try:
            while True:
                try:
                    item = output_queue.get_nowait()
                except queue.Empty:
                    break

                if item is None:
                    logger.debug("Received stop signal (None), exiting queue monitor")
                    return

                drained = True
                msg_type, content = item
                logger.debug("Queue monitor processing event: type=%s", msg_type)

                if msg_type in ("stdout", "stderr"):
                    _thread_safe_log(panel, content.rstrip())
                    if msg_type == "stdout":
                        _parse_and_update_progress(panel, content.rstrip())
                elif msg_type == "exit":
                    logger.debug("Received exit signal from training process (code=%s)", content)
                    return
                elif msg_type == "error":
                    _thread_safe_log(panel, f"[ERROR] {content}")
                    logger.error("Training process error: %s", content)
        except Exception as exc:
            logger.error("Queue monitoring error: %s", exc, exc_info=True)
            return

        try:
            proc = getattr(panel, "_proc", None)
            if proc is not None and not proc.is_alive():
                logger.debug("Training process ended; stopping queue monitor")
                return
        except Exception:
            pass

        delay = 25 if drained else 100
        try:
            panel.after(delay, _pump)
        except Exception as exc:
            logger.debug("Queue monitor scheduling failed: %s", exc, exc_info=True)

    try:
        panel.after(0, _pump)
    except Exception:
        _pump()
    panel._queue_monitor_thread = None


def _wait_for_process(panel: Any, proc: Process) -> None:
    """Wait for process to complete, checking for stop requests."""
    while proc.is_alive():
        proc.join(timeout=0.5)
        
        # Check for stop request
        if getattr(panel, "_stop_requested", False):
            # Stop already signaled via Events, just wait
            pass
    
    # Final join to ensure cleanup
    proc.join(timeout=2.0)


def _cleanup_process_resources(panel: Any) -> None:
    """Clean up multiprocessing resources."""
    logger.debug("Starting process resource cleanup")
    try:
        # Signal queue monitor to stop
        if hasattr(panel, "_output_queue"):
            try:
                panel._output_queue.put(None)
                logger.debug("Sent stop signal to queue monitor")
            except:
                pass
        
        # Clean up Manager
        if hasattr(panel, "_training_manager"):
            try:
                panel._training_manager.shutdown()
                panel._log("[hrm] Manager shutdown complete")
                logger.info("Training manager shutdown complete")
            except Exception as e:
                logger.warning(f"Error during manager shutdown: {e}")
                pass
            finally:
                panel._training_manager = None
        
        # Clear Events
        panel._stop_event = None
        panel._graceful_stop_event = None
        logger.debug("Process resource cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during process resource cleanup: {e}", exc_info=True)


def _thread_safe_log(panel: Any, msg: str) -> None:
    """Thread-safe logging that schedules log calls on the main thread."""
    try:
        panel.after(0, lambda: panel._log(msg))
    except Exception:
        try:
            panel._log(msg)
        except Exception:
            pass





def _parse_and_update_progress(panel: Any, line: str) -> None:
    """Parse training output and update progress bar.
    
    This handles real-time progress updates during parallel/DDP training.
    Supports both JSON format and plain text format.
    """
    import json
    import re
    
    line = line.strip()
    if not line:
        return
    
    current_step = None
    total_steps = None
    
    # Try to parse as JSON first
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            # Update heartbeat for any training-related event
            if obj.get("event") in ("train", "step") or "LOOP_ITERATION_START" in obj or "step" in obj:
                panel._last_heartbeat = time.time()
            
            # Extract step information from JSON
            if "display_step" in obj:
                current_step = int(obj["display_step"])
            elif "step" in obj:
                current_step = int(obj["step"])
            elif "LOOP_ITERATION_START" in obj:
                current_step = int(obj["LOOP_ITERATION_START"]) + 1
            
            if current_step is not None:
                _update_progress_bar(panel, current_step, total_steps)
            return
            
    except (json.JSONDecodeError, ValueError, KeyError):
        # Not JSON, try plain text parsing
        pass
    
    # Parse plain text formats
    try:
        # Pattern 1: "[GPU X] Step Y/Z: ..." or "Step Y/Z: ..."
        # Example: "[GPU 0] Step 5/100: Block 0 Chunk 2 (step 5/100): Loss=12.4569"
        match = re.search(r'Step\s+(\d+)/(\d+)', line, re.IGNORECASE)
        if match:
            current_step = int(match.group(1))
            total_steps = int(match.group(2))
            panel._last_heartbeat = time.time()
            _update_progress_bar(panel, current_step, total_steps)
            return
        
        # Pattern 2: "step X/Y" (lowercase)
        match = re.search(r'step\s+(\d+)/(\d+)', line, re.IGNORECASE)
        if match:
            current_step = int(match.group(1))
            total_steps = int(match.group(2))
            panel._last_heartbeat = time.time()
            _update_progress_bar(panel, current_step, total_steps)
            return
            
    except (ValueError, AttributeError):
        # Parsing failed - that's ok
        pass
    except Exception:
        # Unexpected error - don't crash the monitor thread
        pass


def _update_progress_bar(panel: Any, current_step: int, total_steps: int | None = None) -> None:
    """Update progress bar with current step.
    
    Thread-safe update that schedules UI changes on the main thread.
    
    In iterate mode, converts cumulative steps to cycle-local steps.
    
    Args:
        panel: HRM training panel instance
        current_step: Current training step (may be cumulative in iterate mode)
        total_steps: Total steps (if None, will read from panel.steps_var)
    """
    def _update():
        try:
            # Get total steps from parameter or panel
            steps_total = total_steps
            if steps_total is None or steps_total <= 0:
                try:
                    steps_total = int(panel.steps_var.get().strip() or "0")
                except Exception:
                    steps_total = 0
            
            if steps_total > 0:
                # In iterate mode, convert cumulative step to cycle-local step
                # Check if iterate mode is enabled
                iterate_enabled = False
                try:
                    iterate_enabled = bool(panel.iterate_var.get())
                except Exception:
                    pass
                
                cycle_local_step = current_step
                if iterate_enabled and steps_total > 0:
                    # Calculate cycle-local step (0 to steps_total)
                    # Current step might be cumulative (e.g., 4001 on cycle 2)
                    cycle_local_step = ((current_step - 1) % steps_total) + 1
                    
                    # Store last step to detect resets/new cycles
                    last_step = getattr(panel, '_last_progress_step', 0)
                    if cycle_local_step < last_step:
                        # New cycle detected - step wrapped around
                        panel._last_progress_step = cycle_local_step
                    else:
                        panel._last_progress_step = cycle_local_step
                else:
                    cycle_local_step = current_step
                
                # Calculate percentage using cycle-local step
                pct = int(max(0, min(100, round((cycle_local_step / max(1, steps_total)) * 100))))
                
                # Calculate time remaining
                eta_str = ""
                try:
                    start_time = getattr(panel, "_start_time", None)
                    if start_time and current_step > 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = current_step / elapsed
                        if steps_per_sec > 0:
                            remaining_steps = steps_total - current_step
                            eta_seconds = remaining_steps / steps_per_sec
                            
                            # Format ETA as HH:MM:SS or MM:SS
                            if eta_seconds < 60:
                                eta_str = f" ETA {int(eta_seconds)}s"
                            elif eta_seconds < 3600:
                                minutes = int(eta_seconds // 60)
                                seconds = int(eta_seconds % 60)
                                eta_str = f" ETA {minutes}m {seconds}s"
                            else:
                                hours = int(eta_seconds // 3600)
                                minutes = int((eta_seconds % 3600) // 60)
                                eta_str = f" ETA {hours}h {minutes}m"
                except Exception:
                    # If ETA calculation fails, continue without it
                    pass
                
                # Update progress bar and label with cycle-local step
                panel.progress.configure(mode="determinate", value=pct)
                # Show cycle-local step for iterate mode, cumulative for regular mode
                display_step = cycle_local_step if iterate_enabled else current_step
                panel.progress_lbl.config(text=f"train {display_step}/{steps_total} ({pct}%){eta_str}")
        except Exception:
            # UI widget might not exist yet or might be destroyed
            pass
    
    try:
        # Schedule on main thread
        panel.after_idle(_update)
    except Exception:
        # Panel might be destroyed
        pass


def _update_ui_start(panel: Any) -> None:
    """Update UI to show training started."""
    def _update():
        try:
            panel._run_in_progress = True
            panel._start_time = time.time()  # Track start time for ETA
            panel._last_progress_step = 0  # Reset cycle-local step tracking
            panel._graceful_stop_requested = False  # Reset graceful stop flag
            panel.start_btn.config(state="disabled")
            # Reset stop button to normal appearance
            try:
                panel.stop_btn.config(text="Stop", style="TButton")
            except Exception:
                pass
            panel.progress_lbl.config(text="starting…")
            panel.progress.configure(value=0, mode="determinate")
        except Exception:
            pass
    
    try:
        # Use after_idle to avoid race conditions
        panel.after_idle(_update)
    except Exception:
        _update()


def _update_ui_done(panel: Any) -> None:
    """Update UI to show training completed."""
    def _done():
        panel._run_in_progress = False
        panel._start_time = None  # Reset start time
        panel._graceful_stop_requested = False  # Clear graceful stop flag
        panel._stop_requested = False  # Clear stop flag
        try:
            panel.start_btn.config(state="normal")
            panel.stop_btn.config(text="Stop", style="TButton", state="normal")  # Reset stop button
            panel.progress.stop()
        except Exception:
            pass
        try:
            panel.progress_lbl.config(text="done")
            panel.progress.configure(value=0, mode="determinate")
        except Exception:
            pass
        # Refresh brains panel to update training steps counter
        try:
            brains_panel = getattr(panel, "_brains_panel", None)
            if brains_panel and hasattr(brains_panel, "refresh"):
                brains_panel.refresh()
        except Exception:
            pass
    
    try:
        panel.after(0, _done)
    except Exception:
        _done()
