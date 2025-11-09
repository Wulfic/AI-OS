"""
Chunked training utilities for extreme context lengths in HRM models.

This module provides memory-efficient training for sequences that exceed available VRAM
by processing them in chunks while maintaining the recurrent carry state across chunks.

EXTREME-SCALE OPTIMIZATIONS:
- Supports contexts up to 1M+ tokens on consumer GPUs
- Ultra-aggressive memory management and garbage collection
- Mixed precision (FP16/BF16) aware
- CPU offloading capable for carry states
- Gradient accumulation support for large effective batch sizes
"""

from .core import chunked_segment_rollout
from .memory_estimation import estimate_memory_usage, recommend_chunk_size
from .reward_helpers import _compute_min_halt_steps, _exact_match_reward, _exact_match_reward_from_preds
from .carry_management import _compress_carry_state, _checkpoint_forward

__all__ = [
    "chunked_segment_rollout",
    "estimate_memory_usage",
    "recommend_chunk_size",
    "_compute_min_halt_steps",
    "_exact_match_reward",
    "_exact_match_reward_from_preds",
    "_compress_carry_state",
    "_checkpoint_forward",
]
