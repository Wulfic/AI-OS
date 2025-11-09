"""Training utilities for AI-OS core.

This module provides:
- TrainConfig: Configuration dataclass for training
- NumpyMLP: Simple numpy-based MLP implementation
- Trainer: Main training class with torch/numpy backends
- average_checkpoints_npz: Utility for averaging checkpoints
"""

from __future__ import annotations

from .config import TrainConfig
from .numpy_model import NumpyMLP
from .trainer import Trainer
from .checkpointing import average_checkpoints_npz

__all__ = [
    "TrainConfig",
    "NumpyMLP",
    "Trainer",
    "average_checkpoints_npz",
]
