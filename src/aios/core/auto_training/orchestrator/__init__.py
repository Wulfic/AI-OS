"""Auto-Training Orchestrator for Dynamic Subbrains.

This module provides automatic expert creation and training based on user intent.
When a user expresses a learning intent (e.g., "learn Python programming"), the
orchestrator:
1. Detects the intent and extracts domain/categories
2. Searches for appropriate datasets
3. Creates a new expert with metadata
4. Starts background training
5. Tracks progress and provides updates

Example:
    >>> orchestrator = AutoTrainingOrchestrator(
    ...     dataset_registry=registry,
    ...     expert_registry=expert_reg
    ... )
    >>> task = orchestrator.create_learning_task("Learn Python programming")
    >>> print(f"Training {task.expert_id} on {task.dataset_id}")
"""

from .models import LearningIntent, TrainingTask, TrainingTaskStatus
from .intent_detector import IntentDetector
from .orchestrator import AutoTrainingOrchestrator
from .exceptions import NoDatasetFoundError

__all__ = [
    "LearningIntent",
    "TrainingTask",
    "TrainingTaskStatus",
    "IntentDetector",
    "AutoTrainingOrchestrator",
    "NoDatasetFoundError",
]
