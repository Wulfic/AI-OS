"""
Memory optimization utilities for HRM training.

This module helps users optimize their training configuration for available VRAM,
providing recommendations and warnings before training starts.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any
import math


def estimate_training_memory(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    num_params: int,
    use_deepspeed: bool = False,
    zero_stage: int = 0,
    world_size: int = 1,
    gradient_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    Estimate VRAM usage for HRM training with detailed breakdown.
    
    Args:
        batch_size: Training batch size
        seq_len: Full sequence length
        chunk_size: Chunk size for processing
        hidden_size: Model hidden dimension
        num_layers: Total number of layers (H + L)
        vocab_size: Vocabulary size
        num_params: Total model parameters
        use_deepspeed: Whether DeepSpeed is enabled
        zero_stage: DeepSpeed ZeRO stage (0, 1, 2, or 3)
        world_size: Number of GPUs for DDP/DeepSpeed
        gradient_checkpointing: Whether gradient checkpointing is enabled
        
    Returns:
        Dict with memory estimates in GB for different components
    """
    bytes_per_param_fp32 = 4
    bytes_per_param_fp16 = 2
    bytes_per_activation = 2  # Mixed precision training (bf16/fp16)
    
    # Model weights
    model_memory = num_params * bytes_per_param_fp32
    
    # DeepSpeed ZeRO sharding
    if use_deepspeed and world_size > 1:
        if zero_stage == 3:
            # ZeRO-3: Shard parameters, gradients, and optimizer states
            model_memory = model_memory / world_size
            optimizer_sharding = world_size
            gradient_sharding = world_size
        elif zero_stage == 2:
            # ZeRO-2: Shard gradients and optimizer states
            model_memory = model_memory  # No sharding for model weights
            optimizer_sharding = world_size
            gradient_sharding = world_size
        elif zero_stage == 1:
            # ZeRO-1: Shard only optimizer states
            model_memory = model_memory
            optimizer_sharding = world_size
            gradient_sharding = 1
        else:
            optimizer_sharding = 1
            gradient_sharding = 1
    else:
        optimizer_sharding = 1
        gradient_sharding = 1
    
    # Optimizer states (AdamW = 2x params for momentum + variance)
    optimizer_memory = (num_params * bytes_per_param_fp32 * 2) / optimizer_sharding
    
    # Gradients
    gradient_memory = (num_params * bytes_per_param_fp32) / gradient_sharding
    
    # Carry states (recurrent state for HRM)
    # H-state and L-state, small and fixed size
    carry_memory = batch_size * hidden_size * bytes_per_activation * 2
    
    # Chunk activations (only one chunk in memory at a time with chunking)
    # Activations scale with number of layers and sequence length
    activation_multiplier = 2 if gradient_checkpointing else num_layers
    chunk_activations = (
        batch_size * chunk_size * hidden_size * bytes_per_activation * activation_multiplier
    )
    
    # Output logits for chunk (stored only as predictions now after optimization)
    # OLD: batch_size * chunk_size * vocab_size * bytes_per_param_fp32
    # NEW: batch_size * chunk_size * 4 bytes (int32 for argmax)
    chunk_predictions_memory = batch_size * chunk_size * 4
    
    # Number of chunks
    num_chunks = math.ceil(seq_len / chunk_size)
    
    # Stored predictions for all chunks (for reward computation)
    stored_predictions = batch_size * seq_len * 4  # int32 per token
    
    # DeepSpeed overhead (communication buffers, etc.)
    # WARNING: Zero-3 has MASSIVE overhead for small models!
    deepspeed_overhead = 0
    if use_deepspeed:
        if zero_stage == 3:
            # Zero-3: Parameter all-gather overhead per layer
            # Empirical: 10-20x overhead for models < 500M params on Windows/gloo
            # This is because all-gather happens per layer with temp allocations
            if num_params < 500_000_000:
                # Small model: Zero-3 is VERY inefficient
                deepspeed_overhead = (model_memory + optimizer_memory + gradient_memory) * 12.0
            else:
                # Large model: Zero-3 overhead is reasonable
                deepspeed_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.15
        elif zero_stage == 2:
            # Zero-2: Moderate overhead from gradient sharding
            deepspeed_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.10
        elif zero_stage == 1:
            # Zero-1: Minimal overhead, just optimizer sharding
            deepspeed_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.05
        else:
            # No zero stage, minimal DeepSpeed overhead
            deepspeed_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.03
    
    # PyTorch overhead (caching, fragmentation, etc.)
    pytorch_overhead = (model_memory + optimizer_memory + gradient_memory) * 0.1
    
    # Total per-GPU memory
    total_memory = (
        model_memory
        + optimizer_memory
        + gradient_memory
        + carry_memory
        + chunk_activations
        + chunk_predictions_memory
        + stored_predictions
        + deepspeed_overhead
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
        "chunk_predictions_gb": chunk_predictions_memory / gb,
        "stored_predictions_gb": stored_predictions / gb,
        "deepspeed_overhead_gb": deepspeed_overhead / gb,
        "pytorch_overhead_gb": pytorch_overhead / gb,
        "total_gb": total_memory / gb,
        "per_gpu_gb": total_memory / gb,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "world_size": world_size,
        "zero_stage": float(zero_stage) if use_deepspeed else 0.0,
    }


def recommend_optimal_config(
    available_vram_gb: float,
    target_seq_len: int,
    num_params: int,
    hidden_size: int = 512,
    num_layers: int = 16,
    vocab_size: int = 50257,
    world_size: int = 1,
    use_deepspeed: bool = False,
    zero_stage: int = 0,
    safety_margin: float = 0.15,
) -> Dict[str, Any]:
    """
    Recommend optimal training configuration for available VRAM.
    
    Args:
        available_vram_gb: Available VRAM per GPU
        target_seq_len: Desired sequence length
        num_params: Total model parameters
        hidden_size: Model hidden dimension
        num_layers: Total layers (H + L)
        vocab_size: Vocabulary size
        world_size: Number of GPUs
        use_deepspeed: Whether DeepSpeed is enabled
        zero_stage: DeepSpeed ZeRO stage
        safety_margin: Safety margin fraction (default 15%)
        
    Returns:
        Dict with recommended configuration and warnings
    """
    effective_vram = available_vram_gb * (1 - safety_margin)
    
    # Try different configurations
    chunk_sizes = [512, 1024, 2048, 4096, 8192]
    batch_sizes = [1, 2, 4, 8]
    
    best_config = None
    best_throughput = 0
    
    configs_tried = []
    
    for batch_size in batch_sizes:
        for chunk_size in chunk_sizes:
            if chunk_size > target_seq_len:
                continue
                
            mem = estimate_training_memory(
                batch_size=batch_size,
                seq_len=target_seq_len,
                chunk_size=chunk_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                vocab_size=vocab_size,
                num_params=num_params,
                use_deepspeed=use_deepspeed,
                zero_stage=zero_stage,
                world_size=world_size,
                gradient_checkpointing=True,
            )
            
            fits = mem["per_gpu_gb"] <= effective_vram
            
            # Throughput estimate: batch_size / num_chunks
            throughput = batch_size / mem["num_chunks"]
            
            configs_tried.append({
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "memory_gb": mem["per_gpu_gb"],
                "fits": fits,
                "throughput": throughput,
            })
            
            if fits and throughput > best_throughput:
                best_throughput = throughput
                best_config = {
                    "batch_size": batch_size,
                    "chunk_size": chunk_size,
                    "memory_estimate": mem,
                }
    
    # Generate warnings
    warnings = []
    recommendations = []
    
    # Check for Zero-3 with small models
    if use_deepspeed and zero_stage == 3 and num_params < 500_000_000:
        warnings.append(
            f"⚠️  DeepSpeed Zero-3 is VERY inefficient for models < 500M params! "
            f"Your model ({num_params:,} params) will have ~12x overhead. "
            f"Recommendation: Use plain DDP (no DeepSpeed) for best performance."
        )
        recommendations.append(
            "Remove --zero-stage flag entirely for ~10x memory savings"
        )
    elif use_deepspeed and zero_stage >= 2 and num_params < 200_000_000:
        warnings.append(
            f"Zero-{zero_stage} overhead may exceed benefits for models < 200M params. "
            f"Consider using plain DDP or Zero-1 instead."
        )
    
    if target_seq_len > 50000:
        warnings.append(
            f"Sequence length {target_seq_len:,} is very large. "
            "Consider starting with 10k-20k for testing."
        )
    
    if best_config is None:
        warnings.append(
            f"No configuration fits in {available_vram_gb:.1f}GB VRAM! "
            "Try reducing sequence length or model size."
        )
        # Find closest config
        configs_tried.sort(key=lambda x: x["memory_gb"])
        if configs_tried:
            closest = configs_tried[0]
            recommendations.append(
                f"Smallest config needs {closest['memory_gb']:.1f}GB "
                f"(batch={closest['batch_size']}, chunk={closest['chunk_size']})"
            )
    else:
        mem = best_config["memory_estimate"]
        utilization = (mem["per_gpu_gb"] / effective_vram) * 100
        
        recommendations.append(
            f"Optimal: batch_size={best_config['batch_size']}, "
            f"chunk_size={best_config['chunk_size']}"
        )
        recommendations.append(
            f"Expected VRAM usage: {mem['per_gpu_gb']:.2f}GB "
            f"({utilization:.1f}% of {effective_vram:.1f}GB safe limit)"
        )
        recommendations.append(
            f"Number of chunks: {mem['num_chunks']}"
        )
        
        if utilization > 90:
            warnings.append("Configuration uses >90% of safe VRAM - may be unstable")
    
    return {
        "best_config": best_config,
        "warnings": warnings,
        "recommendations": recommendations,
        "configs_tried": configs_tried,
        "available_vram_gb": available_vram_gb,
        "effective_vram_gb": effective_vram,
    }


def print_memory_analysis(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    hidden_size: int,
    num_layers: int,
    vocab_size: int,
    num_params: int,
    available_vram_gb: float,
    use_deepspeed: bool = False,
    zero_stage: int = 0,
    world_size: int = 1,
) -> bool:
    """
    Print detailed memory analysis and return whether config will fit.
    
    Returns:
        True if configuration fits in available VRAM, False otherwise
    """
    mem = estimate_training_memory(
        batch_size=batch_size,
        seq_len=seq_len,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        num_params=num_params,
        use_deepspeed=use_deepspeed,
        zero_stage=zero_stage,
        world_size=world_size,
        gradient_checkpointing=True,
    )
    
    print("=" * 70)
    print("MEMORY ESTIMATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Sequence length: {seq_len:,} tokens")
    print(f"  Chunk size: {chunk_size} tokens")
    print(f"  Batch size: {batch_size}")
    print(f"  Model parameters: {num_params:,}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")
    print(f"  GPUs: {world_size}")
    if use_deepspeed:
        print(f"  DeepSpeed ZeRO-{zero_stage}: Enabled")
    print()
    
    print("Memory breakdown (per GPU):")
    print(f"  Model weights:        {mem['model_gb']:>8.2f} GB")
    print(f"  Optimizer states:     {mem['optimizer_gb']:>8.2f} GB")
    print(f"  Gradients:            {mem['gradients_gb']:>8.2f} GB")
    print(f"  Carry states:         {mem['carry_gb']:>8.2f} GB")
    print(f"  Chunk activations:    {mem['chunk_activations_gb']:>8.2f} GB")
    print(f"  Predictions storage:  {mem['stored_predictions_gb']:>8.2f} GB")
    if use_deepspeed:
        print(f"  DeepSpeed overhead:   {mem['deepspeed_overhead_gb']:>8.2f} GB")
    print(f"  PyTorch overhead:     {mem['pytorch_overhead_gb']:>8.2f} GB")
    print(f"  " + "-" * 32)
    print(f"  TOTAL:                {mem['per_gpu_gb']:>8.2f} GB")
    print()
    print(f"Number of chunks: {mem['num_chunks']}")
    print(f"Available VRAM: {available_vram_gb:.2f} GB per GPU")
    
    safety_limit = available_vram_gb * 0.85  # 15% safety margin
    fits = mem['per_gpu_gb'] <= safety_limit
    
    if fits:
        utilization = (mem['per_gpu_gb'] / safety_limit) * 100
        print(f"✓ Configuration FITS in available VRAM ({utilization:.1f}% utilization)")
    else:
        overage = mem['per_gpu_gb'] - safety_limit
        print(f"✗ Configuration EXCEEDS available VRAM by {overage:.2f} GB!")
        print(f"  Recommendations:")
        print(f"    - Reduce batch_size (currently {batch_size})")
        print(f"    - Reduce max_seq_len (currently {seq_len:,})")
        print(f"    - Increase chunk_size (currently {chunk_size})")
        print(f"    - Use smaller model")
    
    print("=" * 70)
    
    return fits


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MEMORY OPTIMIZER FOR HRM TRAINING")
    print("=" * 70)
    
    # Example: User's configuration from the error log
    print("\nAnalyzing failed configuration:")
    print("-" * 70)
    
    fits = print_memory_analysis(
        batch_size=1,
        seq_len=100_000,
        chunk_size=512,
        hidden_size=512,
        num_layers=16,  # 8 H + 8 L
        vocab_size=50257,
        num_params=87_115_778,
        available_vram_gb=11.0,
        use_deepspeed=True,
        zero_stage=3,
        world_size=2,
    )
    
    print("\n\nFinding optimal configuration:")
    print("-" * 70)
    
    result = recommend_optimal_config(
        available_vram_gb=11.0,
        target_seq_len=100_000,
        num_params=87_115_778,
        hidden_size=512,
        num_layers=16,
        vocab_size=50257,
        world_size=2,
        use_deepspeed=True,
        zero_stage=3,
    )
    
    if result["best_config"]:
        print(f"\n✓ RECOMMENDED CONFIGURATION:")
        print(f"  Batch size: {result['best_config']['batch_size']}")
        print(f"  Chunk size: {result['best_config']['chunk_size']}")
        mem = result['best_config']['memory_estimate']
        print(f"  Expected memory: {mem['per_gpu_gb']:.2f} GB per GPU")
        print(f"  Number of chunks: {mem['num_chunks']}")
    
    if result["warnings"]:
        print(f"\n⚠ WARNINGS:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    
    if result["recommendations"]:
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")
    
    print("\n" + "=" * 70)
