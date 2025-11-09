"""Helpers for HRMTrainingPanel to keep the main panel concise.

Functions operate on the panel instance to avoid tight coupling.
"""

from .actions import on_start, on_stop, stop_all
from .metrics import poll_metrics, show_stopped_dialog
from .selection import select_student
from .logs import open_rank_logs
from .optimizer_progressive import optimize_from_gui_progressive as optimize_settings

__all__ = [
    "on_start", "on_stop", "stop_all", "poll_metrics", "show_stopped_dialog", "select_student", "open_rank_logs", "optimize_settings",
]
