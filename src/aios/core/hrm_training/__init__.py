"""Core HRM (Hierarchical Retraining Mechanism) training infrastructure.

This module contains the core training configuration and implementation shared by all interfaces
(CLI, GUI, etc.). The training configuration and implementation are interface-agnostic.

Note: This is separate from aios.core.hrm which contains the HRM execution engine.
"""

from .training_config import TrainingConfig

__all__ = ["TrainingConfig"]
