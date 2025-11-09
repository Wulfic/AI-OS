"""Unified configuration for HRM ActV1 training.

This module provides a single source of truth for all training parameters,
ensuring consistency across CLI, GUI, and any future interfaces.
"""

from .config_main import TrainingConfig

__all__ = ["TrainingConfig"]
