"""Progressive optimization system for finding optimal training settings.

This package implements an intelligent, progressive approach to finding optimal
training settings by testing different optimization levels with progressive batch sizing.
"""

from __future__ import annotations

from .models import OptimizationLevel, BatchTestResult, OptimizationConfig, LoRAConfig
from .optimizer import ProgressiveOptimizer
from .gui_adapter import optimize_from_gui_progressive

__all__ = [
    'OptimizationLevel',
    'BatchTestResult',
    'OptimizationConfig',
    'LoRAConfig',
    'ProgressiveOptimizer',
    'optimize_from_gui_progressive',
]
