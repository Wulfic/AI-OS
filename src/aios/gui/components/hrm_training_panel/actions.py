"""Training action wrappers for HRM Training Panel.

Delegates to hrm_training package helpers for start, stop, select, optimize actions.
"""

from __future__ import annotations
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def on_start_wrapper(panel: HRMTrainingPanel) -> None:
    """Start training wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import on_start as _on_start_helper
    _on_start_helper(panel)


def on_stop_wrapper(panel: HRMTrainingPanel) -> None:
    """Stop training wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import on_stop as _on_stop_helper
    _on_stop_helper(panel)


def stop_all_wrapper(panel: HRMTrainingPanel) -> None:
    """Stop all training wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import stop_all as _stop_all_helper
    _stop_all_helper(panel)


def on_select_student(panel: HRMTrainingPanel) -> None:
    """Select student brain wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import select_student as _select_student_helper
    _select_student_helper(panel)


def open_rank_logs(panel: HRMTrainingPanel) -> None:
    """Open rank logs wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import open_rank_logs as _open_rank_logs_helper
    _open_rank_logs_helper(panel)


def on_optimize(panel: HRMTrainingPanel) -> None:
    """Run pre-flight optimization to find good training/gen settings.

    Executes in a background thread and updates UI fields on success.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from .helpers import log
    from ..hrm_training.optimizer_progressive import optimize_from_gui_progressive
    
    # Require a selected student/brain
    try:
        si = (panel.student_init_var.get() or "").strip()
        if not si:
            # Try to resolve from brain name bundle
            bname = (panel.brain_name_var.get() or "").strip()
            if bname:
                import os
                bdir = os.path.join(panel._project_root, "artifacts", "brains", "actv1", bname)
                cand = os.path.join(bdir, "actv1_student.safetensors")
                if os.path.exists(cand) or os.path.isdir(bdir):
                    try:
                        panel.student_init_var.set(cand)
                        si = cand
                    except Exception:
                        pass
        if not si:
            log(panel, "[opt] Please select a student/brain before optimizing → click 'Select Student'.")
            try:
                on_select_student(panel)
            except Exception:
                pass
            return
        else:
            try:
                log(panel, f"[opt] Using student for optimization: {si}")
            except Exception:
                pass
    except Exception:
        log(panel, "[opt] Failed to resolve selected student; select a brain first.")
        return
    
    # Guard against concurrent runs
    if getattr(panel, "_run_in_progress", False):
        try:
            log(panel, "[opt] Busy: wait for current run to finish.")
        except Exception:
            pass
        return
    
    try:
        panel._run_in_progress = True
        panel.start_btn.config(state="disabled")
        panel.progress_lbl.config(text="optimizing…")
        panel.progress.configure(mode="indeterminate", value=0)
        panel.progress.start(10)
    except Exception:
        pass
    
    # Set optimization state flags
    panel._run_in_progress = True
    panel._stop_requested = False
    
    # Start metrics polling during optimization
    if not panel._metrics_polling_active:
        panel._metrics_polling_active = True
        try:
            panel.after(1000, panel._poll_metrics)
        except Exception:
            pass
    
    def _bg():
        try:
            optimize_from_gui_progressive(panel)
        except Exception as e:
            try:
                log(panel, f"[opt] error: {e}")
            except Exception:
                pass
        finally:
            def _done():
                try:
                    panel.start_btn.config(state="normal")
                    panel.progress.stop()
                    panel.progress.configure(mode="determinate", value=0)
                    panel.progress_lbl.config(text="idle")
                    panel._stop_requested = False
                    # Save updated settings after optimization
                    if callable(getattr(panel, "_save_state_fn", None)):
                        try:
                            panel._save_state_fn()
                        except Exception:
                            pass
                except Exception:
                    pass
                panel._run_in_progress = False
            try:
                panel.after(0, _done)
            except Exception:
                _done()
    
    panel._bg_thread = threading.Thread(target=_bg, daemon=True)
    panel._bg_thread.start()
