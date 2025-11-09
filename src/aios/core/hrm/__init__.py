"""Core HRM (Hierarchical Retraining Mechanism) training infrastructure.

This module contains the core training implementation shared by all interfaces
(CLI, GUI, etc.). The training configuration and implementation are interface-agnostic.
"""

from .training_config import TrainingConfig

__all__ = ["TrainingConfig"]
