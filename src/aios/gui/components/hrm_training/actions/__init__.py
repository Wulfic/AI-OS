"""Training action handlers for HRM panel.

This package provides functions for managing HRM training processes:
- on_start: Start training run
- on_stop: Stop training gracefully
- stop_all: Emergency stop (forceful termination)

Example usage:
    >>> from aios.gui.components.hrm_training.actions import on_start, on_stop
    >>> 
    >>> # In HRM training panel
    >>> on_start(self)  # Start training
    >>> on_stop(self)   # Stop training gracefully
"""

from .start_training import on_start
from .stop_training import on_stop
from .emergency_stop import stop_all

__all__ = ["on_start", "on_stop", "stop_all"]
