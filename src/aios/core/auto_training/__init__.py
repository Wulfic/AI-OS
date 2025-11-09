"""Auto-Training module for Dynamic Subbrains.

This module provides automatic expert creation and training based on user intent.
"""

from .orchestrator import (
    LearningIntent,
    TrainingTaskStatus,
    TrainingTask,
    IntentDetector,
    AutoTrainingOrchestrator,
    NoDatasetFoundError,
)

__all__ = [
    "LearningIntent",
    "TrainingTaskStatus",
    "TrainingTask",
    "IntentDetector",
    "AutoTrainingOrchestrator",
    "NoDatasetFoundError",
]
