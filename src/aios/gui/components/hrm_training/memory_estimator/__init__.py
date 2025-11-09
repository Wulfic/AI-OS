"""Memory estimation package for HRM training.

This package provides accurate memory estimation for HuggingFace transformer training,
including VRAM (GPU) and RAM (system) requirements.

Main exports:
    - MemoryEstimator: Main estimation class
    - quick_estimate: Quick estimation function for prototyping

Example usage:
    >>> from aios.gui.components.hrm_training.memory_estimator import MemoryEstimator
    >>> 
    >>> estimator = MemoryEstimator(
    ...     total_params=124_000_000,  # GPT-2 small
    ...     hidden_size=768,
    ...     num_layers=12,
    ...     num_heads=12,
    ...     seq_len=1024,
    ...     batch_size=4,
    ...     use_amp=True,
    ...     use_gradient_checkpointing=True,
    ... )
    >>> 
    >>> summary = estimator.get_summary()
    >>> print(f"VRAM needed: {summary['total_vram_gb']:.2f} GB")
    >>> print(f"RAM needed: {summary['total_ram_gb']:.2f} GB")
    >>> 
    >>> # Get recommendations
    >>> recs = estimator.get_recommendations(available_vram_gb=11.0, available_ram_gb=32.0)
    >>> if not recs['feasible']:
    ...     print("Training not feasible with current config!")
    ...     for suggestion in recs['suggestions']:
    ...         print(suggestion)
"""

from .estimator import MemoryEstimator, quick_estimate
from .constants import BYTES_FP32, BYTES_FP16, BYTES_INT32, GB

__all__ = [
    "MemoryEstimator",
    "quick_estimate",
    "BYTES_FP32",
    "BYTES_FP16",
    "BYTES_INT32",
    "GB",
]
