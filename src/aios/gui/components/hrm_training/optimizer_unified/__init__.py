"""Unified optimizer package - batch size optimization system.

This optimizer focuses on finding optimal batch sizes for training workloads.
It is used by the 'aios optimize' CLI command.

Note: The GUI "Optimize Settings" button uses ProgressiveOptimizer instead,
which provides comprehensive optimization including batch sizing, optimization
levels, and intelligent LoRA parameter testing.

For CLI usage:
    aios optimize --model <model_path>

For GUI optimization:
    Use the "Optimize Settings" button in the HRM Training panel
"""

from .config import OptimizationConfig
from .optimizer import UnifiedOptimizer
from .api import (
    optimize_from_config,
    optimize_from_dict,
    optimize_from_gui,
    optimize_cli
)

__all__ = [
    'OptimizationConfig',
    'UnifiedOptimizer',
    'optimize_from_config',
    'optimize_from_dict',
    'optimize_from_gui',
    'optimize_cli',
]
