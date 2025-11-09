"""Main Trainer class composing all training functionality."""

from __future__ import annotations

from typing import Optional

from .config import TrainConfig
from .trainer_base import TrainerBase
from .checkpointing import CheckpointMixin
from .training_methods import TrainingMixin
from .width_management import WidthManagementMixin
from .sleep_cycle import SleepCycleMixin


class Trainer(
    TrainerBase,
    CheckpointMixin,
    TrainingMixin,
    WidthManagementMixin,
    SleepCycleMixin,
):
    """Unified Trainer class with all training capabilities.
    
    Combines:
    - Base initialization and device setup (TrainerBase)
    - Checkpoint save/load (CheckpointMixin)
    - Training step methods (TrainingMixin)
    - Dynamic width adjustment (WidthManagementMixin)
    - Sleep/consolidation (SleepCycleMixin)
    """

    def __init__(self, cfg: Optional[TrainConfig] = None):
        """Initialize trainer with optional configuration."""
        super().__init__(cfg)
