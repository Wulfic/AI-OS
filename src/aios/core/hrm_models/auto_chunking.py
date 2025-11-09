"""
Automatic chunking wrapper for HRM training.

This module provides a drop-in replacement for segment_rollout that automatically
uses chunking when sequences exceed a threshold.
"""

from typing import Dict, Tuple, Optional, Callable
import torch


def auto_chunked_segment_rollout(
    max_seq_len: int,
    chunk_threshold: int = 999999,
    chunk_size: int = 2048,
    gradient_checkpointing: bool = False,
    use_cpu_offload: bool = False,
) -> Callable:
    """
    Create a segment_rollout function that uses chunking based on threshold.
    
    Args:
        max_seq_len: Maximum sequence length configured for training
        chunk_threshold: Sequences longer than this will use chunking (default: 999999, effectively disabled)
                        Set to 0 to force chunking, or pass max_seq_len to use standard rollout.
                        Caller should respect user's explicit use_chunked_training setting.
        chunk_size: Size of chunks when chunking is used (default: 2048)
        gradient_checkpointing: Whether to use gradient checkpointing
        use_cpu_offload: Whether to offload carry states to CPU between chunks (for 100K+ contexts)
        
    Returns:
        A segment_rollout function with the same signature as the original
    """
    # Decide whether to use chunking
    use_chunking = max_seq_len > chunk_threshold
    
    if use_chunking:
        from aios.core.hrm_models.chunked_training import chunked_segment_rollout
        
        def wrapper(model, batch, max_segments, epsilon=0.0, ce_loss_fn_arg=None, ignore_index=-100):
            return chunked_segment_rollout(
                model=model,
                batch=batch,
                max_segments=max_segments,
                chunk_size=chunk_size,
                epsilon=epsilon,
                ce_loss_fn_arg=ce_loss_fn_arg,
                ignore_index=ignore_index,
                gradient_checkpointing=gradient_checkpointing,
                use_cpu_offload=use_cpu_offload,
            )
        
        return wrapper
    else:
        # Use standard segment_rollout for short sequences
        from aios.core.hrm_models.train_utils import segment_rollout
        return segment_rollout


def get_recommended_chunk_size(
    max_seq_len: int, 
    available_vram_gb: float = 20.0,
    model_params: int = 124_000_000,
    using_ddp: bool = False
) -> int:
    """
    Recommend a chunk size based on sequence length, VRAM, and model size.
    
    ULTRA-AGGRESSIVE for extreme contexts (100K+ tokens) and large models (500M+ params).
    Designed to handle massive scale on limited hardware.
    
    Args:
        max_seq_len: Target sequence length
        available_vram_gb: Available VRAM in GB
        model_params: Number of model parameters
        using_ddp: Whether DDP is being used (requires more conservative chunks)
        
    Returns:
        Recommended chunk size
    """
    # Adjust for model size - larger models need MUCH smaller chunks
    if model_params > 500_000_000:  # 500M+ parameters
        vram_multiplier = 0.3
    elif model_params > 200_000_000:  # 200M+ parameters
        vram_multiplier = 0.5
    elif model_params > 100_000_000:  # 100M+ parameters
        vram_multiplier = 0.7
    else:
        vram_multiplier = 0.85
    
    # DDP requires more conservative memory usage (model replicated per GPU)
    if using_ddp:
        vram_multiplier *= 0.7  # Further reduce effective VRAM for DDP overhead
    
    effective_vram = available_vram_gb * vram_multiplier
    
    # ULTRA-AGGRESSIVE defaults for extreme scale
    # Further reduced for low-VRAM scenarios (< 8GB effective)
    if effective_vram >= 30:
        base_chunk = 1024
    elif effective_vram >= 18:
        base_chunk = 768
    elif effective_vram >= 12:
        base_chunk = 512
    elif effective_vram >= 8:
        base_chunk = 384
    elif effective_vram >= 6:
        base_chunk = 256
    elif effective_vram >= 4:
        base_chunk = 128  # More aggressive for <6GB effective
    else:
        base_chunk = 64   # Ultra-aggressive for <4GB effective
    
    # EXTREME context handling - the longer the context, the smaller the chunks
    if max_seq_len >= 500_000:  # 500K+ tokens
        return min(base_chunk, 192)
    elif max_seq_len >= 200_000:  # 200K+ tokens
        return min(base_chunk, 256)
    elif max_seq_len >= 100_000:  # 100K+ tokens
        return min(base_chunk, 384)
    elif max_seq_len >= 50_000:  # 50K+ tokens
        return min(base_chunk, 512)
    elif max_seq_len >= 20_000:  # 20K+ tokens
        return min(base_chunk, 640)
    elif max_seq_len >= 10_000:  # 10K+ tokens
        return min(base_chunk, 384)  # More aggressive for 10K+ contexts
    else:
        return base_chunk
