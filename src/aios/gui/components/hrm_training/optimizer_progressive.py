"""
Progressive Optimization System

This system implements an intelligent, progressive approach to finding optimal training settings:
1. Start with minimal optimizations (gradient checkpointing only)
2. Start with batch size 1
3. Incrementally increase batch size until GPU memory cap is reached
4. If training fails, progressively add optimizations one at a time
5. Test all optimization combinations systematically
6. Provide clear feedback when all options are exhausted

The optimization order is designed to add features from least to most aggressive:
- Gradient Checkpointing (baseline - always enabled)
- AMP (Mixed Precision)
- LoRA/PEFT
- CPU Offload
- DeepSpeed ZeRO Stage 1
- DeepSpeed ZeRO Stage 2
- DeepSpeed ZeRO Stage 3

This module has been refactored into a package. All public APIs are re-exported here.
"""

from __future__ import annotations

# Re-export all public APIs from the refactored package
from .optimizer_progressive import (
    OptimizationLevel,
    BatchTestResult,
    OptimizationConfig,
    LoRAConfig,
    ProgressiveOptimizer,
    optimize_from_gui_progressive,
)

__all__ = [
    'OptimizationLevel',
    'BatchTestResult',
    'OptimizationConfig',
    'LoRAConfig',
    'ProgressiveOptimizer',
    'optimize_from_gui_progressive',
]
