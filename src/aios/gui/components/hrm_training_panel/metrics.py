"""Metrics polling wrappers for HRM Training Panel.

Delegates to hrm_training package helpers for metrics polling.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .panel_main import HRMTrainingPanel


def poll_metrics_wrapper(panel: HRMTrainingPanel) -> None:
    """Poll metrics wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import poll_metrics as _poll_metrics_helper
    _poll_metrics_helper(panel)


def show_stopped_dialog_wrapper(panel: HRMTrainingPanel) -> None:
    """Show stopped dialog wrapper - delegates to hrm_training helper.
    
    Args:
        panel: The HRMTrainingPanel instance
    """
    from ..hrm_training import show_stopped_dialog as _show_stopped_dialog_helper
    _show_stopped_dialog_helper(panel)
