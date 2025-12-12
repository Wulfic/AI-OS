"""Training stop handler for HRM panel.

This module handles graceful and forceful termination of training processes.
"""

from __future__ import annotations
import logging
import os
import signal
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
    ack_timeout = 15.0
    acknowledged = _wait_for_stop_ack(panel, ack_timeout)
    if not acknowledged:
        _thread_safe_log(panel, f"[hrm] Stop acknowledgement not received within {int(ack_timeout)}s; will continue monitoring")
    _graceful_stop_then_terminate(panel, ack_waited=acknowledged)
    
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
    ack_timeout = 20.0
    if not _wait_for_graceful_ack(panel, ack_timeout):
        _thread_safe_log(panel, f"[hrm] Waiting for graceful stop acknowledgement (>{int(ack_timeout)}s)")
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
        return os.path.join(panel._project_root, "training_datasets", "actv1", "STOP")
    except Exception:
        return "training_datasets/actv1/STOP"


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


def _wait_for_ack_event(panel: Any, attr: str, timeout: float) -> bool:
    """Wait for a multiprocessing ack event to become set."""
    if timeout <= 0:
        timeout = 0
    event = getattr(panel, attr, None)
    if event is None:
        return False
    # Fast path: check without blocking first
    try:
        if event.is_set():
            return True
    except Exception:
        pass

    deadline = time.time() + timeout
    poll = 0.25
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        try:
            if event.wait(timeout=min(poll, remaining)):
                return True
        except Exception as exc:
            logger.debug("Ack wait failed for %s: %s", attr, exc)
            return False


def _wait_for_stop_ack(panel: Any, timeout: float) -> bool:
    """Wait for immediate-stop acknowledgement."""
    acknowledged = _wait_for_ack_event(panel, "_stop_ack_event", timeout)
    if acknowledged and not getattr(panel, "_stop_ack_notified", False):
        _thread_safe_log(panel, "[hrm] Training acknowledged immediate stop request")
        try:
            panel._stop_ack_notified = True
        except Exception:
            pass
    return acknowledged


def _wait_for_graceful_ack(panel: Any, timeout: float) -> bool:
    """Wait for graceful-stop acknowledgement."""
    acknowledged = _wait_for_ack_event(panel, "_graceful_stop_ack_event", timeout)
    if acknowledged and not getattr(panel, "_graceful_stop_ack_notified", False):
        _thread_safe_log(panel, "[hrm] Training acknowledged graceful stop request")
        try:
            panel._graceful_stop_ack_notified = True
        except Exception:
            pass
    return acknowledged


def _attempt_signal(panel: Any, proc: Any, sig: int | None, description: str) -> bool:
    """Try sending a signal to the training process."""
    if sig is None:
        return False
    try:
        pid = proc.pid
    except Exception:
        return False
    if pid is None:
        return False
    try:
        os.kill(pid, sig)
        if description:
            _thread_safe_log(panel, f"[hrm] {description}")
        return True
    except Exception as exc:
        logger.debug("Failed to send signal %s: %s", description or sig, exc)
        return False


def _update_ui_stopping(panel: Any) -> None:
    """Update UI to show stopping state."""
    try:
        panel.progress_lbl.config(text="stopping…")
        panel.progress.configure(mode="indeterminate")
        panel.progress.start(15)
    except Exception:
        pass


def _graceful_stop_then_terminate(panel: Any, ack_waited: bool = False) -> None:
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
        ack_observed = ack_waited
        if not ack_observed:
            ack_observed = _wait_for_stop_ack(panel, 5.0)
        if ack_observed:
            _thread_safe_log(panel, "[hrm] Stop acknowledged; allowing worker to finish cleanup")
        else:
            _thread_safe_log(panel, "[hrm] Stop acknowledgement still pending; grace period shortened to 45s")

        grace_seconds = 120 if ack_observed else 45
        start_time = time.time()
        deadline = start_time + grace_seconds
        last_log = 0
        _thread_safe_log(panel, f"[hrm] Waiting up to {int(grace_seconds)}s for graceful shutdown (finishing chunk + merging checkpoints)…")

        while True:
            if not proc.is_alive():
                _thread_safe_log(panel, "[hrm] Training process exited gracefully")
                return

            now = time.time()
            if now >= deadline:
                break

            elapsed = now - start_time
            log_bucket = int(elapsed // 10)
            if log_bucket > last_log and elapsed >= 10:
                last_log = log_bucket
                if elapsed < 30:
                    _thread_safe_log(panel, f"[hrm] Finishing current batch... ({elapsed:.0f}s)")
                elif elapsed < 60:
                    _thread_safe_log(panel, f"[hrm] Saving GPU checkpoints... ({elapsed:.0f}s)")
                elif elapsed < 90:
                    _thread_safe_log(panel, f"[hrm] Merging checkpoints... ({elapsed:.0f}s)")
                else:
                    _thread_safe_log(panel, f"[hrm] Still waiting for finalization... ({elapsed:.0f}s)")

            if not ack_observed:
                try:
                    ack_event = getattr(panel, "_stop_ack_event", None)
                    if ack_event is not None and ack_event.is_set():
                        ack_observed = True
                        _wait_for_stop_ack(panel, 0)
                        _thread_safe_log(panel, "[hrm] Stop acknowledgement received while waiting; extending grace window")
                        grace_seconds = 120
                        deadline = max(deadline, time.time() + 120)
                except Exception:
                    pass

            time.sleep(0.25)

        if not proc.is_alive():
            return

        sigint_sent = _attempt_signal(panel, proc, getattr(signal, "SIGINT", None), "Sent SIGINT to training process")
        if sigint_sent:
            for _ in range(40):  # 10 seconds
                if not proc.is_alive():
                    return
                time.sleep(0.25)

        if proc.is_alive():
            _thread_safe_log(panel, "[hrm] Grace period expired, invoking terminate()…")
            proc.terminate()
            for _ in range(40):  # 10 seconds
                if not proc.is_alive():
                    return
                time.sleep(0.25)

        if proc.is_alive():
            _attempt_signal(panel, proc, getattr(signal, "SIGUSR2", None), "Requesting stack trace before kill (SIGUSR2)")
            _thread_safe_log(panel, "[hrm] Escalation: attempting kill()…")
            proc.kill()
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
        ack_observed = _wait_for_graceful_ack(panel, 5.0)
        ack_missing_logged = False
        
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
            
            if not ack_observed:
                try:
                    ack_event = getattr(panel, "_graceful_stop_ack_event", None)
                    if ack_event is not None and ack_event.is_set():
                        ack_observed = True
                        _wait_for_graceful_ack(panel, 0)
                except Exception:
                    pass
                if not ack_observed and elapsed >= 10 and not ack_missing_logged:
                    _thread_safe_log(panel, "[hrm] Waiting for worker to acknowledge graceful stop request…")
                    ack_missing_logged = True
            
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
        stop_ack_event = getattr(panel, "_stop_ack_event", None)
        graceful_stop_ack_event = getattr(panel, "_graceful_stop_ack_event", None)
        
        if stop_event:
            stop_event.clear()
        if graceful_stop_event:
            graceful_stop_event.clear()
        if stop_ack_event:
            try:
                stop_ack_event.clear()
            except Exception:
                pass
        if graceful_stop_ack_event:
            try:
                graceful_stop_ack_event.clear()
            except Exception:
                pass
        try:
            panel._stop_ack_notified = False
            panel._graceful_stop_ack_notified = False
        except Exception:
            pass
            
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
