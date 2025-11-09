"""
Extreme-scale optimizations for training massive models with huge contexts.

This module provides advanced memory management techniques for pushing the limits:
- Training 500M+ parameter models on consumer GPUs
- Handling 500K+ token contexts (0.5M to 1M tokens)
- CPU offloading for carry states
- Gradient accumulation for large effective batch sizes
- Ultra-aggressive memory management
"""

from __future__ import annotations
from typing import Optional, Any
import gc
import torch


def enable_extreme_memory_mode():
    """
    Enable ultra-aggressive memory management for extreme-scale training.
    Call this at the start of training for 100K+ contexts or 500M+ param models.
    """
    import os
    
    # PyTorch CUDA allocator settings for extreme scale
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'
    
    # Force aggressive garbage collection
    gc.set_threshold(500, 5, 5)  # More aggressive than default (700, 10, 10)
    
    print({
        "extreme_memory_mode": "enabled",
        "settings": {
            "cuda_alloc": "aggressive_fragmentation_control",
            "gc_threshold": "500,5,5",
            "note": "Optimized for 100K+ contexts and 500M+ params"
        }
    })


def offload_carry_to_cpu(carry, device: torch.device) -> tuple[Any, dict]:
    """
    Offload carry state to CPU to save GPU memory.
    
    For extreme contexts (500K+ tokens), the carry state can grow large.
    Moving it to CPU between chunks saves precious VRAM.
    
    Args:
        carry: The carry state to offload
        device: Current device (for metadata)
        
    Returns:
        (cpu_carry, metadata) - carry on CPU and metadata for restoration
    """
    if not hasattr(carry, 'inner_carry'):
        return carry, {}
    
    metadata = {'device': str(device)}
    
    # Move inner carry tensors to CPU
    inner = carry.inner_carry
    if hasattr(inner, 'z_H') and inner.z_H is not None:
        inner.z_H = inner.z_H.cpu()
    if hasattr(inner, 'z_L') and inner.z_L is not None:
        inner.z_L = inner.z_L.cpu()
    
    # CRITICAL: Synchronize to ensure async .cpu() transfers complete
    # before clearing cache or continuing. This prevents race conditions
    # especially in parallel multi-GPU training.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        gc.collect()
    
    return carry, metadata


def restore_carry_to_gpu(carry, metadata: dict, device: torch.device):
    """
    Restore carry state from CPU back to GPU.
    
    Args:
        carry: The carry state to restore
        metadata: Metadata from offload operation
        device: Target device
        
    Returns:
        carry on GPU
    """
    if not hasattr(carry, 'inner_carry'):
        return carry
    
    inner = carry.inner_carry
    if hasattr(inner, 'z_H') and inner.z_H is not None:
        inner.z_H = inner.z_H.to(device, non_blocking=True)
    if hasattr(inner, 'z_L') and inner.z_L is not None:
        inner.z_L = inner.z_L.to(device, non_blocking=True)
    
    # CRITICAL: Synchronize to ensure async transfers complete before
    # the restored tensors are used. Using non_blocking=True for performance
    # but synchronizing before return ensures data integrity.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    return carry


def estimate_extreme_context_memory(
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    hidden_size: int,
    h_layers: int,
    l_layers: int,
    num_heads: int,
    expansion: float,
    vocab_size: int,
    model_params: int,
    use_amp: bool = True,
    h_cycles: int = 2,
    l_cycles: int = 2,
) -> dict[str, Any]:
    """
    Estimate memory usage for extreme-scale HRM ACTv1 training.
    
    This estimator is specifically designed for the HRM ACTv1 architecture:
    - Two-level hierarchy (H and L)
    - Multiple cycles per level
    - Carry state (z_H and z_L) maintained across chunks
    - Self-attention + SwiGLU MLP per block
    - RoPE or learned positional encodings
    
    Args:
        batch_size: Batch size
        seq_len: Full sequence length (can be 100K+)
        chunk_size: Chunk size for processing
        hidden_size: Model hidden dimension
        h_layers: Number of H-level (high-level) layers
        l_layers: Number of L-level (low-level) layers
        num_heads: Number of attention heads
        expansion: MLP expansion factor (typically 2.0)
        vocab_size: Vocabulary size (typically 50257)
        model_params: Total model parameters (for verification)
        use_amp: Whether using mixed precision (FP16/BF16)
        h_cycles: H-level cycles per segment
        l_cycles: L-level cycles per segment
        
    Returns:
        Dict with detailed memory estimates in GB
    """
    bytes_per_param = 4  # FP32 for weights always
    bytes_per_activation = 2 if use_amp else 4  # FP16/BF16 vs FP32
    
    # === 1. MODEL PARAMETERS (weights) ===
    # Embeddings: vocab → hidden
    embedding_params = vocab_size * hidden_size
    
    # LM head: hidden → vocab (typically shared with embeddings, but count separately)
    lm_head_params = hidden_size * vocab_size
    
    # Q-head for halt prediction: hidden → 2
    q_head_params = hidden_size * 2 + 2  # +bias
    
    # Per-layer parameters (H and L have same structure):
    # - Self-attention: Q, K, V, O projections
    head_dim = hidden_size // num_heads
    attn_params_per_layer = (
        hidden_size * hidden_size * 4  # Q, K, V, O projections
    )
    
    # - SwiGLU MLP: gate + up + down projections
    mlp_hidden = int(hidden_size * expansion)
    mlp_params_per_layer = (
        hidden_size * mlp_hidden +  # gate projection
        hidden_size * mlp_hidden +  # up projection  
        mlp_hidden * hidden_size    # down projection
    )
    
    params_per_layer = attn_params_per_layer + mlp_params_per_layer
    
    # Total for all layers
    h_level_params = h_layers * params_per_layer
    l_level_params = l_layers * params_per_layer
    
    # RoPE or learned positional encodings (learned adds more params)
    pos_encoding_params = 0  # RoPE has no params; learned would add seq_len * hidden_size
    
    # Init states (H_init, L_init)
    init_params = hidden_size * 2
    
    total_params = (
        embedding_params + 
        lm_head_params + 
        q_head_params + 
        h_level_params + 
        l_level_params + 
        pos_encoding_params + 
        init_params
    )
    
    model_memory_gb = (total_params * bytes_per_param) / (1024**3)
    
    # Verification (warn if mismatch)
    param_mismatch = abs(total_params - model_params) / max(model_params, 1)
    if param_mismatch > 0.1:  # >10% mismatch
        print(f"⚠️  Warning: Estimated params ({total_params:,}) differs from actual ({model_params:,}) by {param_mismatch*100:.1f}%")
    
    # === 2. OPTIMIZER STATES (AdamW) ===
    # AdamW stores: momentum (1st moment) + variance (2nd moment)
    optimizer_memory_gb = (total_params * bytes_per_param * 2) / (1024**3)
    
    # === 3. ACTIVATIONS (for ONE chunk) ===
    # Key insight: We only keep activations for current chunk due to chunking!
    # Not the full sequence.
    
    # Input embeddings: batch × chunk × hidden
    embedding_activations = batch_size * chunk_size * hidden_size * bytes_per_activation
    
    # Per-layer activations (attention + MLP):
    # - Attention: Q, K, V, scores, output
    attn_activations_per_layer = (
        batch_size * chunk_size * hidden_size * 5 * bytes_per_activation  # Q, K, V, scores (approximated), output
    )
    
    # - MLP: gate, up, down activations
    mlp_activations_per_layer = (
        batch_size * chunk_size * mlp_hidden * 3 * bytes_per_activation
    )
    
    activations_per_layer = attn_activations_per_layer + mlp_activations_per_layer
    
    # Total activations for H and L levels (considering cycles)
    # Each cycle goes through all layers
    h_level_activations = h_layers * activations_per_layer * h_cycles
    l_level_activations = l_layers * activations_per_layer * l_cycles
    
    # With gradient checkpointing, activations are recomputed, saving ~50%
    checkpoint_factor = 0.5  # Assume gradient checkpointing is enabled
    
    chunk_activations_gb = (
        (embedding_activations + h_level_activations + l_level_activations) * 
        checkpoint_factor / (1024**3)
    )
    
    # === 4. CARRY STATE (z_H and z_L) ===
    # Critical: After compression, carry is only LAST position + puzzle embeddings
    # Not the full chunk size!
    # Assume puzzle_emb_len = 0 for simplicity, so carry = [batch, 1, hidden]
    carry_positions = 1  # Compressed to just last position
    carry_state_gb = (
        batch_size * carry_positions * hidden_size * 2 *  # 2 for z_H and z_L
        4 / (1024**3)  # FP32 for carry state
    )
    
    # === 5. GRADIENTS (same size as parameters) ===
    gradients_gb = model_memory_gb  # FP32
    
    # === 6. LOGITS OUTPUT ===
    # Final output: batch × chunk × vocab
    logits_gb = (batch_size * chunk_size * vocab_size * bytes_per_activation) / (1024**3)
    
    # === 7. CUDA OVERHEAD ===
    # PyTorch reserves extra memory for fragmentation, caching, etc.
    # Typically 10-20% of total allocated
    subtotal = model_memory_gb + optimizer_memory_gb + chunk_activations_gb + carry_state_gb + gradients_gb + logits_gb
    cuda_overhead_gb = subtotal * 0.15
    
    # === TOTAL ===
    total_gb = subtotal + cuda_overhead_gb
    
    return {
        "model_params_gb": round(model_memory_gb, 2),
        "optimizer_states_gb": round(optimizer_memory_gb, 2),
        "chunk_activations_gb": round(chunk_activations_gb, 2),
        "carry_state_gb": round(carry_state_gb, 3),
        "gradients_gb": round(gradients_gb, 2),
        "logits_output_gb": round(logits_gb, 2),
        "cuda_overhead_gb": round(cuda_overhead_gb, 2),
        "total_estimated_gb": round(total_gb, 2),
        "parameter_breakdown": {
            "embeddings": embedding_params,
            "lm_head": lm_head_params,
            "h_level": h_level_params,
            "l_level": l_level_params,
            "total": total_params,
            "model_params_input": model_params,
            "match": param_mismatch < 0.1
        },
        "notes": {
            "seq_len": seq_len,
            "chunk_size": chunk_size,
            "num_chunks": (seq_len + chunk_size - 1) // chunk_size,
            "use_amp": use_amp,
            "batch_size": batch_size,
            "gradient_checkpointing": True,
            "carry_compressed": True,
            "architecture": f"{h_layers}H + {l_layers}L, {hidden_size}d, {num_heads}heads"
        }
    }


def get_gradient_accumulation_steps(
    target_batch_size: int,
    available_vram_gb: float,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    num_heads: int,
    expansion: float,
    vocab_size: int,
    model_params: int,
    seq_len: int,
    chunk_size: int
) -> tuple[int, int]:
    """
    Calculate optimal gradient accumulation steps for large effective batch sizes.
    
    When you want batch_size=8 but only have VRAM for batch_size=1,
    use gradient accumulation to achieve the same effect.
    
    Args:
        target_batch_size: Desired effective batch size
        available_vram_gb: Available VRAM
        h_layers: Number of H-level layers
        l_layers: Number of L-level layers
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        expansion: MLP expansion factor
        vocab_size: Vocabulary size
        model_params: Model parameters
        seq_len: Sequence length
        chunk_size: Chunk size
        
    Returns:
        (micro_batch_size, accumulation_steps)
    """
    # Estimate memory for batch_size=1
    estimate = estimate_extreme_context_memory(
        batch_size=1,
        seq_len=seq_len,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
        h_layers=h_layers,
        l_layers=l_layers,
        num_heads=num_heads,
        expansion=expansion,
        vocab_size=vocab_size,
        model_params=model_params,
        use_amp=True
    )
    
    single_batch_memory = estimate['total_estimated_gb']
    
    # How many batches can we fit?
    max_micro_batch = max(1, int(available_vram_gb / single_batch_memory))
    
    # Gradient accumulation steps needed
    if max_micro_batch >= target_batch_size:
        return target_batch_size, 1  # No accumulation needed
    else:
        accumulation_steps = (target_batch_size + max_micro_batch - 1) // max_micro_batch
        return max_micro_batch, accumulation_steps


def print_extreme_scale_recommendations(
    model_params: int,
    seq_len: int,
    available_vram_gb: float,
    h_layers: int = 8,
    l_layers: int = 8,
    hidden_size: int = 512,
    num_heads: int = 8,
    expansion: float = 2.0,
    vocab_size: int = 50257
):
    """
    Print recommendations for extreme-scale training.
    
    Args:
        model_params: Total model parameters
        seq_len: Target sequence length
        available_vram_gb: Available VRAM
        h_layers: Number of H-level layers (default: 8)
        l_layers: Number of L-level layers (default: 8)
        hidden_size: Hidden dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        expansion: MLP expansion factor (default: 2.0)
        vocab_size: Vocabulary size (default: 50257)
    """
    from .auto_chunking import get_recommended_chunk_size
    
    chunk_size = get_recommended_chunk_size(seq_len, available_vram_gb, model_params)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    # Determine scale category
    if seq_len >= 500_000 or model_params >= 500_000_000:
        scale = "EXTREME"
        color = "🔴"
    elif seq_len >= 100_000 or model_params >= 200_000_000:
        scale = "VERY LARGE"
        color = "🟠"
    elif seq_len >= 50_000 or model_params >= 100_000_000:
        scale = "LARGE"
        color = "🟡"
    else:
        scale = "STANDARD"
        color = "🟢"
    
    estimate = estimate_extreme_context_memory(
        batch_size=1,
        seq_len=seq_len,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
        h_layers=h_layers,
        l_layers=l_layers,
        num_heads=num_heads,
        expansion=expansion,
        vocab_size=vocab_size,
        model_params=model_params,
        use_amp=True
    )
    
    print(f"\n{color} EXTREME-SCALE TRAINING: {scale}")
    print(f"{'='*60}")
    print(f"Model Parameters: {model_params:,}")
    print(f"Architecture: {h_layers}H + {l_layers}L layers, {hidden_size}d, {num_heads} heads")
    print(f"Context Length: {seq_len:,} tokens")
    print(f"Available VRAM: {available_vram_gb:.1f} GB")
    print(f"\nOptimal Configuration:")
    print(f"  Chunk Size: {chunk_size} tokens")
    print(f"  Num Chunks: {num_chunks}")
    print(f"  Estimated VRAM: {estimate['total_estimated_gb']:.2f} GB")
    print(f"\nMemory Breakdown:")
    print(f"  Model + Optimizer: {estimate['model_params_gb'] + estimate['optimizer_states_gb']:.2f} GB")
    print(f"  Activations (per chunk): {estimate['chunk_activations_gb']:.2f} GB")
    print(f"  Carry State (compressed): {estimate['carry_state_gb']:.3f} GB")
    print(f"  Gradients: {estimate['gradients_gb']:.2f} GB")
    print(f"  Logits Output: {estimate['logits_output_gb']:.2f} GB")
    print(f"  CUDA Overhead: {estimate['cuda_overhead_gb']:.2f} GB")
    
    # Parameter verification
    param_breakdown = estimate.get('parameter_breakdown', {})
    if not param_breakdown.get('match', True):
        print(f"\n⚠️  Parameter Count Mismatch:")
        print(f"   Estimated: {param_breakdown.get('total', 0):,}")
        print(f"   Actual: {param_breakdown.get('model_params_input', 0):,}")
    
    if estimate['total_estimated_gb'] > available_vram_gb:
        print(f"\n⚠️  WARNING: Estimated usage exceeds available VRAM!")
        overage = estimate['total_estimated_gb'] - available_vram_gb
        print(f"   Overage: {overage:.2f} GB ({overage/available_vram_gb*100:.0f}%)")
        print(f"   Consider:")
        print(f"   - Reducing context length to {seq_len // 2:,} tokens")
        print(f"   - Reducing model size (fewer layers or smaller hidden dim)")
        print(f"   - Reducing batch size to 1 (if not already)")
        print(f"   - Using CPU offloading (slower but works)")
        
        # Suggest specific architecture reductions
        if h_layers > 4 or l_layers > 4:
            print(f"   - Try {max(4, h_layers//2)}H + {max(4, l_layers//2)}L layers")
        if hidden_size > 384:
            print(f"   - Try hidden_size = {max(256, hidden_size//2)}")
    else:
        headroom = available_vram_gb - estimate['total_estimated_gb']
        print(f"\n✅ Configuration should work!")
        print(f"   Headroom: {headroom:.2f} GB ({headroom/available_vram_gb*100:.0f}%)")
        
        # Suggest optimizations if there's significant headroom
        if headroom > 3.0:
            print(f"\n💡 You have significant headroom! Consider:")
            if seq_len < 100_000:
                print(f"   - Increasing context to {min(500_000, seq_len * 2):,} tokens")
            print(f"   - Increasing batch_size to 2")
            if h_layers < 12 and l_layers < 12:
                print(f"   - Scaling up to {h_layers + 2}H + {l_layers + 2}L layers")
    
    print(f"{'='*60}\n")
