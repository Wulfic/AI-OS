"""Training stop handler for HRM panel.

This module handles graceful and forceful termination of training processes.
"""

from __future__ import annotations
import logging
import time
from typing import Any

from ....utils.resource_management import submit_background

logger = logging.getLogger(__name__)


def _thread_safe_log(panel: Any, msg: str) -> None:
    """Thread-safe logging that schedules log calls on the main thread."""
    try:
        # Use after_idle to avoid blocking if event queue is full
        panel.after_idle(lambda: panel._log(msg))
    except Exception:
        # Fallback: try direct call (may fail if on wrong thread)
        try:
            panel._log(msg)
        except Exception:
            # Last resort: log to console
            logger.debug(msg)


def on_stop(panel: Any) -> None:
    """
    Stop current training run with two-stage behavior.
    
    This function runs in the GUI thread and must return IMMEDIATELY.
    All work is done in a background thread.
    """
    # Immediately spawn a background thread to handle everything
    def _handle_stop():
        # Check if there's actually a process running
        proc = getattr(panel, "_proc", None)
        if proc is None:
            _thread_safe_log(panel, "[hrm] No training process is running")
            return
        
        # Check if process is already dead (with timeout to prevent hanging)
        try:
            is_alive = proc.is_alive()
        except Exception as e:
            # If we can't check, assume it's dead
            _thread_safe_log(panel, f"[hrm] Cannot check process status: {e}")
            panel._proc = None
            panel._graceful_stop_requested = False
            panel._stop_requested = False
            return
        
        if not is_alive:
            _thread_safe_log(panel, "[hrm] Training process has already finished")
            # Clear the process reference
            panel._proc = None
            panel._graceful_stop_requested = False
            panel._stop_requested = False
            return
        
        # Check if this is the second click (graceful stop already in progress)
        already_graceful_stopping = getattr(panel, "_graceful_stop_requested", False)
        
        # Optimization always stops immediately (no graceful mode)
        is_optimizing = getattr(panel, "_active_optimizer", None) is not None
        
        if already_graceful_stopping or is_optimizing:
            # Second click OR optimization - immediate stop
            _do_immediate_stop(panel, already_graceful_stopping)
        else:
            # First click - graceful stop
            _do_graceful_stop_request(panel)
    
    # Start background thread immediately and return
    try:
        submit_background("hrm-stop", _handle_stop, pool=getattr(panel, "_worker_pool", None))
    except RuntimeError as exc:
        logger.error("Failed to queue HRM stop handler: %s", exc)


def _do_immediate_stop(panel: Any, already_graceful_stopping: bool) -> None:
    """Handle immediate stop in background thread."""
    if already_graceful_stopping:
        logger.warning("EMERGENCY STOP: User requested immediate termination (graceful stop in progress)")
    else:
        logger.info("IMMEDIATE STOP: User requested forceful termination")
    
    _thread_safe_log(panel, "[hrm] ========================================")
    _thread_safe_log(panel, "[hrm] IMMEDIATE STOP requested - forcing termination" if already_graceful_stopping else "[hrm] Stopping optimization...")
    _thread_safe_log(panel, "[hrm] ========================================")
    panel._stop_requested = True
    panel._graceful_stop_requested = False  # Clear flag
        
    panel._stop_requested = True
    panel._graceful_stop_requested = False  # Clear flag
    
    # Update UI
    try:
        panel.after(0, lambda: panel.stop_btn.config(text="Stopping...", style="TButton", state="disabled"))
        panel.after(0, lambda: panel.progress_lbl.config(text="stopping…"))
        panel.after(0, lambda: panel.progress.configure(mode="indeterminate"))
        panel.after(0, lambda: panel.progress.start(15))
    except Exception:
        pass
    
    # Stop active optimizer immediately if running
    logger.debug("Stopping active optimizer processes")
    _stop_optimizer(panel)
    
    # Signal stop event
    logger.debug("Signaling stop event to training process")
    _write_stop_file(panel)
    
    # Start forceful termination
    logger.info("Initiating graceful shutdown with fallback to forced termination")
    _graceful_stop_then_terminate(panel)
    
    # Handle optimization background thread
    logger.debug("Stopping background optimization thread")
    _stop_background_thread(panel)


def _do_graceful_stop_request(panel: Any) -> None:
    """Handle graceful stop request in background thread."""
    logger.info("GRACEFUL STOP: User requested stop after current chunk completion")
    
    panel._graceful_stop_requested = True
    panel._stop_requested = True
    
    # Track when stop was requested
    try:
        panel._stop_request_time = time.time()
        logger.debug(f"Graceful stop requested at {panel._stop_request_time}")
    except Exception:
        pass
    
    # Update UI
    try:
        def _update_ui():
            try:
                from tkinter import ttk
                style = ttk.Style()
                style.configure("Red.TButton", foreground="red", font=("TkDefaultFont", 10, "bold"))
                try:
                    style.map("Red.TButton", foreground=[("active", "darkred"), ("!active", "red")], background=[("active", "#ffcccc")])
                except Exception:
                    pass
                panel.stop_btn.config(text="⚠ STOP ⚠", style="Red.TButton")
                panel.progress_lbl.config(text="finishing chunk...")
                panel.progress.configure(mode="indeterminate")
                panel.progress.start(15)
            except Exception as e:
                logger.error(f"UI update error during graceful stop: {e}")
        
        panel.after(0, _update_ui)
    except Exception as e:
        logger.error(f"UI schedule error during graceful stop: {e}")
    
    # Log and signal
    _thread_safe_log(panel, "[hrm] GRACEFUL STOP requested - will finish current chunk then exit")
    _thread_safe_log(panel, "[hrm] Click STOP again for immediate termination")
    
    logger.debug("Signaling graceful stop event to training process")
    _write_graceful_stop_file(panel)
    
    logger.debug("Starting graceful stop monitor")
    _monitor_graceful_stop(panel)
    
    # Save state
    try:
        if callable(getattr(panel, "_save_state_fn", None)):
            panel._save_state_fn()
            logger.debug("Saved panel state after graceful stop request")
    except Exception as e:
        logger.warning(f"Failed to save state after graceful stop: {e}")


def get_default_stop_file(panel: Any) -> str:
    """Get default STOP file path."""
    try:
        return os.path.join(panel._project_root, "training_data", "actv1", "STOP")
    except Exception:
        return "training_data/actv1/STOP"


def _stop_optimizer(panel: Any) -> None:
    """Stop active optimizer process if running."""
    try:
        active_optimizer = getattr(panel, "_active_optimizer", None)
        if active_optimizer is not None:
            _thread_safe_log(panel, "[hrm] Stop: terminating optimizer processes")
            try:
                # Call force_stop if available, otherwise regular stop
                if hasattr(active_optimizer, "force_stop"):
                    active_optimizer.force_stop()
                elif hasattr(active_optimizer, "stop"):
                    active_optimizer.stop()
            except Exception as e:
                _thread_safe_log(panel, f"[hrm] Optimizer stop error: {e}")
            panel._active_optimizer = None
    except Exception as e:
        _thread_safe_log(panel, f"[hrm] Optimizer access error: {e}")


def _write_stop_file(panel: Any) -> None:
    """Signal immediate stop using multiprocessing.Event."""
    try:
        stop_event = getattr(panel, "_stop_event", None)
        
        if stop_event is not None:
            stop_event.set()
            _thread_safe_log(panel, "[hrm] Immediate stop requested (Event signaled)")
        else:
            _thread_safe_log(panel, "[hrm] Warning: Stop event not available (training may not have started yet)")
    except Exception as e:
        _thread_safe_log(panel, f"[hrm] Failed to signal immediate stop: {e}")


def _write_graceful_stop_file(panel: Any) -> None:
    """Signal graceful stop using multiprocessing.Event."""
    try:
        graceful_stop_event = getattr(panel, "_graceful_stop_event", None)
        
        if graceful_stop_event is not None:
            graceful_stop_event.set()
            _thread_safe_log(panel, "[hrm] Graceful stop requested (Event signaled)")
        else:
            _thread_safe_log(panel, "[hrm] Warning: Graceful stop event not available (training may not have started yet)")
    except Exception as e:
        _thread_safe_log(panel, f"[hrm] Failed to signal graceful stop: {e}")


def _update_ui_stopping(panel: Any) -> None:
    """Update UI to show stopping state."""
    try:
        panel.progress_lbl.config(text="stopping…")
        panel.progress.configure(mode="indeterminate")
        panel.progress.start(15)
    except Exception:
        pass


def _graceful_stop_then_terminate(panel: Any) -> None:
    """
    Wait for graceful shutdown, then escalate to terminate if needed.
    
    For multiprocessing.Process:
    1. Wait 120 seconds for graceful exit
    2. proc.terminate() + wait 10 seconds
    3. proc.kill() + wait 2 seconds
    """
    try:
        proc = getattr(panel, "_proc", None)
        if proc is None:
            return
        
        # Check if process is already dead before waiting
        if not proc.is_alive():
            _thread_safe_log(panel, "[hrm] Training process already terminated")
            # Clean up
            panel._proc = None
            panel._graceful_stop_requested = False
            panel._stop_requested = False
            _cleanup_stop_files(panel)
            return
        
        _thread_safe_log(panel, "[hrm] Waiting for graceful shutdown (finishing chunk + merging checkpoints)…")
        
        # Wait up to 120 seconds for graceful shutdown
        for i in range(480):  # 480 * 0.25 = 120 seconds
            if not proc.is_alive():
                _thread_safe_log(panel, "[hrm] Training process exited gracefully")
                return
            
            # Progress indicator every 10 seconds
            if i > 0 and i % 40 == 0:
                elapsed = i * 0.25
                if elapsed < 30:
                    _thread_safe_log(panel, f"[hrm] Finishing current batch... ({elapsed:.0f}s)")
                elif elapsed < 60:
                    _thread_safe_log(panel, f"[hrm] Saving GPU checkpoints... ({elapsed:.0f}s)")
                elif elapsed < 90:
                    _thread_safe_log(panel, f"[hrm] Merging checkpoints... ({elapsed:.0f}s)")
                else:
                    _thread_safe_log(panel, f"[hrm] Still waiting for finalization... ({elapsed:.0f}s)")
            
            time.sleep(0.25)
        
        # If still running after 120 seconds, force terminate
        if proc.is_alive():
            _thread_safe_log(panel, "[hrm] Grace period expired (120s), terminating process…")
            proc.terminate()
            
            # Wait 10 more seconds for terminate to work
            for _ in range(40):  # 10 seconds
                if not proc.is_alive():
                    return
                time.sleep(0.25)
        
        # Last resort: kill
        if proc.is_alive():
            _thread_safe_log(panel, "[hrm] escalation: attempting kill()…")
            proc.kill()
            
            # Wait briefly for kill to take effect
            for _ in range(8):  # 2 seconds
                if not proc.is_alive():
                    return
                time.sleep(0.25)
        
    finally:
        # Always clean up at the end
        try:
            panel._proc = None
            panel._graceful_stop_requested = False
            panel._stop_requested = False
            _cleanup_stop_files(panel)
        except Exception:
            pass


def _monitor_graceful_stop(panel: Any) -> None:
    """Monitor graceful stop - wait indefinitely for chunk to complete."""
    try:
        proc = getattr(panel, "_proc", None)
        if proc is None:
            return
        
        # Wait indefinitely for graceful completion
        # No timeout - user can click stop again for immediate termination
        elapsed = 0
        last_log_time = 0
        
        while True:
            if not proc.is_alive():
                _thread_safe_log(panel, "[hrm] Training process exited gracefully after finishing chunk")
                # Reset button appearance and clear flags
                try:
                    panel.after(0, lambda: panel.stop_btn.config(text="Stop", style="TButton", state="normal"))
                except Exception:
                    pass
                
                # Clear process reference and flags
                panel._proc = None
                panel._graceful_stop_requested = False
                panel._stop_requested = False
                
                # Clean up stop files
                try:
                    _cleanup_stop_files(panel)
                except Exception:
                    pass
                
                return
            
            # Progress indicator every 30 seconds (less verbose)
            if elapsed > 0 and (elapsed - last_log_time) >= 30:
                _thread_safe_log(panel, f"[hrm] Finishing current chunk... ({int(elapsed)}s elapsed)")
                last_log_time = elapsed
            
            time.sleep(0.25)
            elapsed += 0.25
        
    except Exception:
        pass


def _cleanup_stop_files(panel: Any) -> None:
    """Clear stop events after training finishes."""
    try:
        stop_event = getattr(panel, "_stop_event", None)
        graceful_stop_event = getattr(panel, "_graceful_stop_event", None)
        
        if stop_event:
            stop_event.clear()
        if graceful_stop_event:
            graceful_stop_event.clear()
            
        _thread_safe_log(panel, "[hrm] Cleared stop events")
    except Exception:
        pass


def _stop_background_thread(panel: Any) -> None:
    """Stop optimization background thread if running."""
    try:
        bg_future = getattr(panel, "_bg_future", None)
        if bg_future is not None and not getattr(bg_future, "done", lambda: True)():
            _thread_safe_log(panel, "[hrm] Stop: waiting for background task to finish")

            def _await_future() -> None:
                try:
                    bg_future.result(timeout=2.0)
                except Exception:
                    pass

            try:
                submit_background(
                    "hrm-stop-wait",
                    _await_future,
                    pool=getattr(panel, "_worker_pool", None),
                )
            except RuntimeError:
                _await_future()
        else:
            bg_thread = getattr(panel, "_bg_thread", None)
            if bg_thread is not None and getattr(bg_thread, "is_alive", lambda: False)():
                _thread_safe_log(panel, "[hrm] Stop: signaling optimization thread to terminate")

                # Thread should check _stop_requested and exit; give it 2 seconds
                def _join_bg() -> None:
                    try:
                        bg_thread.join(timeout=2.0)
                        if bg_thread.is_alive():
                            _thread_safe_log(panel, "[hrm] Warning: optimization thread still running")
                    except Exception:
                        pass

                try:
                    submit_background(
                        "hrm-stop-wait-thread",
                        _join_bg,
                        pool=getattr(panel, "_worker_pool", None),
                    )
                except RuntimeError:
                    _join_bg()
    except Exception as e:
        _thread_safe_log(panel, f"[hrm] Background thread handling error: {e}")
