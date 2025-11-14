"""Emergency stop handler for HRM panel.

This module provides forceful termination of all training processes,
used when closing the application or in emergency scenarios.
"""

from __future__ import annotations
import time
from typing import Any

from ....utils.resource_management import submit_background


def stop_all(panel: Any) -> None:
    """
    Emergency stop: forcefully terminate all training/optimization processes.
    
    This function returns IMMEDIATELY and does all work in a background thread.
    """
    def _do_stop_all():
        # Disable metrics polling
        try:
            panel._metrics_polling_active = False
        except Exception:
            pass
        
        # Attempt graceful stop first
        try:
            from .stop_training import on_stop
            on_stop(panel)
        except Exception:
            pass
        
        # Wait for background thread to exit
        try:
            future = getattr(panel, "_bg_future", None)
            if future is not None and not getattr(future, "done", lambda: True)():
                future.result(timeout=2.0)
            else:
                t = getattr(panel, "_bg_thread", None)
                if t is not None and getattr(t, "is_alive", lambda: False)():
                    t.join(timeout=2.0)
        except Exception:
            pass
        
        # CRITICAL: Ensure subprocess is actually terminated
        # This runs synchronously to guarantee cleanup
        _force_terminate_subprocess(panel)
    
    # Run everything via dispatcher
    try:
        submit_background(
            "hrm-emergency-stop",
            _do_stop_all,
            pool=getattr(panel, "_worker_pool", None),
        )
    except RuntimeError:
        _do_stop_all()


def _force_terminate_subprocess(panel: Any) -> None:
    """Forcefully terminate multiprocessing.Process if still running."""
    try:
        proc = getattr(panel, "_proc", None)
        if proc is None or not proc.is_alive():
            return  # No process or already dead
        
        # Log for debugging
        try:
            panel._log(f"[hrm] Force-terminating training process (PID: {proc.pid})")
        except Exception:
            pass
        
        # Give graceful shutdown thread a moment
        time.sleep(0.5)
        
        # If still running, force terminate
        if proc.is_alive():
            _terminate_forcefully(panel, proc)
    except Exception as e:
        try:
            panel._log(f"[hrm] Force termination error: {e}")
        except Exception:
            pass


def _terminate_forcefully(panel: Any, proc: Any) -> None:
    """
    Terminate multiprocessing.Process forcefully.
    
    Uses terminate() first, then kill() if needed.
    """
    try:
        # First try terminate (SIGTERM on Unix, TerminateProcess on Windows)
        proc.terminate()
        
        # Wait for process to die
        for _ in range(8):  # 2 seconds
            if not proc.is_alive():
                panel._log("[hrm] Process terminated successfully")
                return
            time.sleep(0.25)
        
        # If still alive, escalate to kill (SIGKILL on Unix)
        if proc.is_alive():
            proc.kill()
            for _ in range(4):  # 1 second
                if not proc.is_alive():
                    return
                time.sleep(0.25)
        
        # If still alive, log warning
        if proc.is_alive():
            panel._log(f"[hrm] WARNING: Process {proc.pid} did not terminate!")
    except Exception as e:
        try:
            panel._log(f"[hrm] Termination error: {e}")
        except Exception:
            pass
