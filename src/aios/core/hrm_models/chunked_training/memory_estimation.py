"""
Memory estimation utilities for chunked training.

Functions for estimating VRAM usage and recommending optimal chunk sizes.
"""

from __future__ import annotations

from typing import Dict


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    hidden_size: int,
    vocab_size: int,
    num_params: int,
) -> Dict[str, float]:
    """
    Estimate VRAM usage for chunked training.
    
    Args:
        batch_size: Batch size
        seq_len: Full sequence length
        chunk_size: Chunk size for processing
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_params: Total model parameters
        
    Returns:
        Dict with memory estimates in GB for different components
    """
    bytes_per_param = 4  # fp32
    bytes_per_activation = 2  # bf16
    
    # Model weights
    model_memory = num_params * bytes_per_param
    
    # Optimizer states (AdamW = 2x params for momentum + variance)
    optimizer_memory = num_params * bytes_per_param * 2
    
    # Gradients
    gradient_memory = num_params * bytes_per_param
    
    # Carry states (small, fixed size)
    carry_memory = batch_size * hidden_size * bytes_per_activation * 2  # H and L states
    
    # Chunk activations (only one chunk in memory at a time)
    chunk_activations = batch_size * chunk_size * hidden_size * bytes_per_activation
    
    # Output logits for chunk
    chunk_logits = batch_size * chunk_size * vocab_size * bytes_per_param
    
    # PyTorch overhead (empirical ~10%)
    pytorch_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.1
    
    # Total
    total_memory = (
        model_memory
        + optimizer_memory
        + gradient_memory
        + carry_memory
        + chunk_activations
        + chunk_logits
        + pytorch_overhead
    )
    
    # Convert to GB
    gb = 1024 ** 3
    return {
        "model_gb": model_memory / gb,
        "optimizer_gb": optimizer_memory / gb,
        "gradients_gb": gradient_memory / gb,
        "carry_gb": carry_memory / gb,
        "chunk_activations_gb": chunk_activations / gb,
        "chunk_logits_gb": chunk_logits / gb,
        "pytorch_overhead_gb": pytorch_overhead / gb,
        "total_gb": total_memory / gb,
        "effective_batch_size": batch_size,
        "effective_seq_len": seq_len,
        "chunk_size": chunk_size,
        "num_chunks": (seq_len + chunk_size - 1) // chunk_size,
    }


def recommend_chunk_size(
    available_vram_gb: float,
    batch_size: int,
    seq_len: int,
    hidden_size: int = 768,
    vocab_size: int = 50257,
    num_params: int = 124_000_000,
    safety_margin: float = 0.15,
) -> int:
    """
    Recommend optimal chunk size given VRAM constraints.
    
    Args:
        available_vram_gb: Available VRAM in GB
        batch_size: Desired batch size
        seq_len: Target sequence length
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        num_params: Total model parameters
        safety_margin: Safety margin fraction (default 15%)
        
    Returns:
        Recommended chunk size
    """
    # Start with a reasonable chunk size
    chunk_candidates = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    
    # Effective available memory after safety margin
    effective_vram = available_vram_gb * (1 - safety_margin)
    
    # Find largest chunk that fits
    best_chunk = 512  # Conservative default
    
    for chunk in chunk_candidates:
        if chunk > seq_len:
            break
        
        mem_est = estimate_memory_usage(
            batch_size=batch_size,
            seq_len=seq_len,
            chunk_size=chunk,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_params=num_params,
        )
        
        if mem_est["total_gb"] <= effective_vram:
            best_chunk = chunk
        else:
            break
    
    return best_chunk
