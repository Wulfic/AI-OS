"""VRAM (GPU memory) estimation logic for HRM training.

This module contains the complex VRAM estimation algorithm that accounts for:
- Model weights and precision
- Optimizer state (AdamW with momentum/variance)
- Gradients
- Activations (including O(n²) attention memory)
- Framework overhead
- PyTorch memory reservation
"""

from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .estimator import MemoryEstimator

from .constants import BYTES_FP32, BYTES_FP16, GB


def estimate_vram(estimator: "MemoryEstimator") -> Dict[str, Any]:
    """
    Estimate VRAM usage per GPU with detailed breakdown.
    
    Args:
        estimator: MemoryEstimator instance with configuration
    
    Returns dict with:
        - model_gb: Model weights
        - optimizer_gb: Optimizer state (AdamW)
        - gradients_gb: Gradient tensors
        - activations_gb: Forward activations
        - overhead_gb: Framework overhead
        - total_gb: Total per GPU
        - breakdown: Dict with detailed component breakdown
    """
    
    # Determine precision for different components
    model_bytes_per_param = BYTES_FP32  # Models usually keep FP32 master weights
    
    if estimator.use_amp:
        # With AMP: activations in FP16, but model/optimizer stay FP32
        activation_bytes = BYTES_FP16
        gradient_bytes = BYTES_FP16  # Gradients also in FP16
    else:
        activation_bytes = BYTES_FP32
        gradient_bytes = BYTES_FP32
    
    # ===== 1. MODEL WEIGHTS =====
    model_gb = (estimator.total_params * model_bytes_per_param) / GB
    
    # LoRA adds adapter parameters
    if estimator.use_lora:
        lora_adapter_gb = (estimator.lora_adapter_params * model_bytes_per_param) / GB
        model_gb += lora_adapter_gb
    
    # ZeRO-3 partitions model weights across GPUs
    # IMPORTANT: ZeRO-3 with single GPU provides NO memory savings!
    # It only adds overhead from all-gather operations during forward/backward
    # NOTE: In practice, many users have multiple GPUs but only train on 1
    if estimator.zero_stage == "zero3" and estimator.num_gpus > 1:
        model_gb_per_gpu = model_gb / estimator.num_gpus
    else:
        model_gb_per_gpu = model_gb
    
    # ===== 2. OPTIMIZER STATE =====
    # AdamW needs 2x parameters for momentum + variance
    if estimator.use_lora:
        # Only optimize trainable parameters
        optimizer_params = estimator.trainable_params + estimator.lora_adapter_params
    else:
        optimizer_params = estimator.total_params
    
    optimizer_gb = (optimizer_params * model_bytes_per_param * 2) / GB
    
    # 8-bit optimizer reduces to INT8 (75% reduction)
    if estimator.use_8bit_optimizer:
        optimizer_gb *= 0.25
    
    # ZeRO-1, 2, 3 partition optimizer across GPUs
    if estimator.zero_stage in ["zero1", "zero2", "zero3"] and estimator.num_gpus > 1:
        optimizer_gb_per_gpu = optimizer_gb / estimator.num_gpus
    else:
        optimizer_gb_per_gpu = optimizer_gb
    
    # CPU offload moves optimizer to RAM
    if estimator.use_cpu_offload:
        optimizer_gb_vram = 0
        optimizer_gb_ram = optimizer_gb_per_gpu
    else:
        optimizer_gb_vram = optimizer_gb_per_gpu
        optimizer_gb_ram = 0
    
    # ===== 3. GRADIENTS =====
    if estimator.use_lora:
        gradient_params = estimator.trainable_params + estimator.lora_adapter_params
    else:
        gradient_params = estimator.total_params
    
    gradients_gb = (gradient_params * gradient_bytes) / GB
    
    # ZeRO-2 and ZeRO-3 partition gradients
    if estimator.zero_stage in ["zero2", "zero3"] and estimator.num_gpus > 1:
        gradients_gb_per_gpu = gradients_gb / estimator.num_gpus
    else:
        gradients_gb_per_gpu = gradients_gb
    
    # ===== 4. ACTIVATIONS =====
    # Use effective sequence length (chunk size for chunked training)
    effective_seq = estimator.chunk_size if estimator.use_chunking else estimator.seq_len
    
    # Per-layer activations:
    # - Input: batch * seq * hidden
    # - Attention scores: batch * heads * seq * seq  (THIS IS THE KILLER FOR LONG SEQUENCES!)
    # - Attention output: batch * seq * hidden
    # - FFN intermediate: batch * seq * (hidden * expansion)
    num_heads = max(1, estimator.hidden_size // 64)  # Estimate
    expansion = 2.0  # HRM ACT-v1 uses 2.0 expansion typically
    
    # CRITICAL: Attention memory scales O(n²) with sequence length!
    # For seq_len=10000: attention_scores needs 10000² = 100M elements per head per batch
    # With 8 heads: 800M elements = 3.2 GB in FP32 (or 1.6 GB in FP16) PER LAYER!
    # 
    # IMPORTANT: With chunking, the attention matrix is ONLY computed over chunk_size!
    # The KV cache is reused across chunks, but the O(n²) attention scores are only chunk_size²
    # This is THE KEY to enabling long context training on consumer GPUs.
    if estimator.use_chunking and effective_seq < estimator.seq_len:
        # With chunking: attention matrix is chunk_size x chunk_size
        attention_seq = effective_seq  # Use chunk_size, not geometric mean!
    else:
        # Without chunking: full sequence attention
        attention_seq = estimator.seq_len
    
    attention_memory = (
        estimator.batch_size * num_heads * attention_seq * attention_seq  # Score matrix
    ) * activation_bytes
    
    per_layer_act = (
        estimator.batch_size * effective_seq * estimator.hidden_size +  # Input embedding
        attention_memory +  # Attention scores (O(n²) - dominates for long seq!)
        estimator.batch_size * effective_seq * estimator.hidden_size +  # Attention output
        estimator.batch_size * effective_seq * int(estimator.hidden_size * expansion)  # FFN intermediate
    ) * activation_bytes
    
    # Total activations for all layers
    total_activations = per_layer_act * estimator.num_layers
    
    # ===== OUTPUT LOGITS MEMORY (CRITICAL FOR LARGE VOCABULARIES!) =====
    # The output logits tensor: batch * seq * vocab_size
    # This is a MAJOR memory consumer with large vocabularies (150K+ tokens)
    # For Qwen 2.5: vocab=151,657 with seq=5000 = 758M float elements = ~2.8 GB in FP32 (or ~1.4 GB in FP16)!
    # 
    # IMPORTANT: This is stored during forward pass AND additional copies created during:
    # - Loss computation (cross_entropy creates temporary tensors)
    # - Validation checks (isnan/isinf materialized as boolean tensors)
    # - Backward pass (gradient w.r.t. logits)
    # 
    # Conservative estimate: 2x the base logits size to account for temporaries
    output_logits_memory = (
        estimator.batch_size * effective_seq * estimator.vocab_size * activation_bytes * 2
    )
    total_activations += output_logits_memory
    
    # HRM ACT-v1 SPECIFIC: Add overhead for dual H/L architecture
    # - Carry states (persistent across chunks, but small: batch * hidden)
    # - Halt prediction heads (minimal: batch * 2 floats per position)
    # - Cross-level communication buffers
    # Empirically measured from actual training: ~25% additional overhead
    # (Previous 45% estimate was too conservative)
    total_activations *= 1.25
    
    # Gradient checkpointing reduces stored activations by ~25%
    # We only store checkpoints at layer boundaries and recompute during backward
    # Real-world measurements show 20-30% reduction
    if estimator.use_gradient_checkpointing:
        total_activations *= 0.75
    
    # ===== ADD KV CACHE AND CARRY STATES (SPAN FULL SEQUENCE) =====
    # IMPORTANT: Even with chunking, KV cache and carry states span the FULL sequence!
    # This is because they accumulate across chunks for the entire context window.
    
    if estimator.use_chunking and effective_seq < estimator.seq_len:
        # KV cache: key + value for each layer, across full sequence
        # batch * seq_len * hidden * 2 (key + value) * bytes per element
        kv_cache_per_layer = (
            estimator.batch_size * estimator.seq_len * estimator.hidden_size * 2 * activation_bytes
        )
        kv_cache_total = kv_cache_per_layer * estimator.num_layers
        
        # HRM Carry states: persistent state across full sequence
        # batch * seq_len * hidden * bytes per element
        carry_states = (
            estimator.batch_size * estimator.seq_len * estimator.hidden_size * activation_bytes
        )
        
        # Add to activations
        total_activations += kv_cache_total + carry_states
        
        # ===== CRITICAL: CHUNK OVERLAP DURING BACKWARD PASS =====
        # During chunked training, there's a memory spike when:
        # 1. Chunk N is in backward pass (computing gradients)
        # 2. Chunk N+1 is loaded for forward pass (double buffering)
        # 3. ADDITIONAL temporary buffers for gradient accumulation
        # 4. Optimizer momentum buffers may be materialized
        # 
        # Users report spikes from ~4GB to ~11GB during these transitions.
        # This is a 2.75x multiplier on the base memory!
        #
        # The conservative approach: Account for FULL second chunk + extra buffers
        # This includes:
        # - Second chunk activations (full duplicate)
        # - Gradient accumulation workspace
        # - Temporary attention matrices during recompute
        # - Optimizer state materialization
        #
        # Calculate full chunk activations again
        per_layer_chunk_act = (
            estimator.batch_size * effective_seq * estimator.hidden_size +  # Input
            estimator.batch_size * num_heads * effective_seq * effective_seq +  # Attention
            estimator.batch_size * effective_seq * estimator.hidden_size +  # Output  
            estimator.batch_size * effective_seq * int(estimator.hidden_size * expansion)  # FFN
        ) * activation_bytes
        
        # Full second chunk (all layers)
        second_chunk = per_layer_chunk_act * estimator.num_layers
        
        # Apply HRM overhead to second chunk
        second_chunk *= 1.25
        
        # With gradient checkpointing, second chunk also gets some reduction
        # but not as much (we're recomputing during backward)
        if estimator.use_gradient_checkpointing:
            second_chunk *= 0.85  # Less reduction than first chunk
        
        # Add extra workspace for gradient accumulation and temporary buffers
        # This accounts for the gap between calculated and observed memory
        # CRITICAL: User observations show 15-20 GB actual vs ~10 GB calculated
        # This suggests MASSIVE temporary buffers during chunk transitions that aren't
        # captured by the simple second_chunk calculation. These include:
        # - DeepSpeed internal buffers for gradient accumulation
        # - CUDA graph workspace allocations
        # - Optimizer state materializations (momentum/variance temporarily loaded)
        # - Communication buffers for all-reduce operations
        # - Flash attention workspace (if enabled)
        # - Temporary tensors during autograd graph construction
        # Real-world observations: 2x-3x multiplier needed for large chunks with long sequences
        workspace_multiplier = 2.5  # 150% extra - aggressive but realistic
        second_chunk *= workspace_multiplier
        
        # Add overlap memory (this is the spike!)
        total_activations += second_chunk
    
    # Chunked training reduces peak memory proportionally
    # NOTE: Chunking benefits are ALREADY accounted for by using effective_seq/chunk_size
    # in the attention calculation above. DO NOT apply additional chunking_factor here
    # or we'll double-count the benefit!
    # The hybrid attention_seq calculation already handles the trade-off between
    # chunk size and full sequence memory requirements.
    
    activations_gb = total_activations / GB
    
    # Activations are NOT divided by num_gpus!
    # The batch_size parameter is already the PER-GPU batch size in DDP.
    # Each GPU computes activations for its own batch slice.
    # DO NOT divide by num_gpus or we massively underestimate!
    activations_gb_per_gpu = activations_gb
    
    # ===== 5. FRAMEWORK OVERHEAD =====
    # CUDA kernels, PyTorch overhead, cuDNN workspace, etc.
    base_total = model_gb_per_gpu + optimizer_gb_vram + gradients_gb_per_gpu + activations_gb_per_gpu
    
    # Base overhead: 8-15% for PyTorch internals
    # Higher percentage for smaller models (more proportional overhead)
    if base_total < 1.0:
        base_overhead_pct = 0.15
    elif base_total < 3.0:
        base_overhead_pct = 0.12
    else:
        base_overhead_pct = 0.10
    overhead_gb = base_total * base_overhead_pct
    
    # DDP overhead: Additional buffers for gradient synchronization
    if estimator.num_gpus > 1:
        # DDP needs communication buffers (~3% per GPU)
        overhead_gb += base_total * 0.03
    
    # ZeRO-3 has additional communication overhead
    if estimator.zero_stage == "zero3":
        if estimator.num_gpus == 1:
            # Single GPU ZeRO-3: MAJOR overhead from all-gather reconstruction!
            # Model weights are partitioned in optimizer but must be reconstructed
            # for forward/backward, causing temporary spikes to near-full memory.
            # Add 35% overhead to account for these reconstruction spikes
            overhead_gb += base_total * 0.35
        else:
            # Multi-GPU: normal 3% communication overhead
            overhead_gb += base_total * 0.03
    
    # Long sequence overhead: cuDNN workspace and temporary buffers
    # These grow with sequence length, especially for attention operations
    if estimator.seq_len > 2048:
        # For chunked training: workspace still needs space for chunk processing
        # Scale based on effective sequence (chunk size if chunked, else full seq)
        effective_calc_seq = estimator.chunk_size if estimator.use_chunking else estimator.seq_len
        # Empirically: ~0.2 GB base + scale with sqrt of sequence (diminishing returns)
        workspace_gb = 0.2 + (effective_calc_seq / 1000) * 0.15
        # Cap at reasonable maximum
        workspace_gb = min(workspace_gb, 1.0)
        overhead_gb += workspace_gb
        
        # Additional safety margin for very long sequences (>5K) with chunking
        # These have temporary spikes during chunk transitions and gradient accumulation
        if estimator.seq_len > 5000 and estimator.use_chunking:
            # Scale with sequence length: up to +20% for extreme contexts
            chunking_safety = base_total * min(0.20, (estimator.seq_len - 5000) / 20000)
            overhead_gb += chunking_safety
            
            # Extra overhead for large chunk sizes (>2048)
            # Large chunks increase activation memory and temporary buffers significantly
            if estimator.chunk_size > 2048:
                # Scale with chunk size: up to +15% for very large chunks (4096+)
                large_chunk_overhead = base_total * min(0.15, (estimator.chunk_size - 2048) / 4096 * 0.15)
                overhead_gb += large_chunk_overhead
    
    # ===== TOTAL VRAM PER GPU (BEFORE RESERVATION) =====
    total_vram_gb = model_gb_per_gpu + optimizer_gb_vram + gradients_gb_per_gpu + activations_gb_per_gpu + overhead_gb
    
    # ===== PYTORCH MEMORY RESERVATION =====
    # PyTorch allocates more than peak usage for caching and fragmentation.
    # The "reserved_gb" metric shows this reservation, which is typically 20-40% higher than peak.
    # With DeepSpeed and chunked training, the overhead is even higher due to:
    # - Internal buffer pools
    # - Gradient accumulation staging areas
    # - Communication buffer pre-allocation
    # - CUDA graph workspace reservation
    # 
    # Based on actual measurements from users:
    # - Tiny models (<0.5GB): Higher proportional overhead ~1.40x
    # - Small models (0.5-2GB): ~1.30x reservation
    # - Medium models (2-8GB): ~1.25x reservation  
    # - Large models (8-12GB): ~1.20x reservation
    # - Very large (>12GB): ~1.15x (less proportional overhead)
    if total_vram_gb < 0.5:
        reservation_factor = 1.40  # High proportional overhead for tiny models
    elif total_vram_gb < 2.0:
        reservation_factor = 1.30
    elif total_vram_gb < 8.0:
        reservation_factor = 1.25
    elif total_vram_gb < 12.0:
        reservation_factor = 1.20
    else:
        reservation_factor = 1.15
    
    total_vram_gb *= reservation_factor
    
    return {
        "model_gb": model_gb_per_gpu,
        "optimizer_gb": optimizer_gb_vram,
        "gradients_gb": gradients_gb_per_gpu,
        "activations_gb": activations_gb_per_gpu,
        "overhead_gb": overhead_gb,
        "total_gb": total_vram_gb,
        "breakdown": {
            "model_full": model_gb,
            "optimizer_full": optimizer_gb,
            "optimizer_ram": optimizer_gb_ram,
            "gradients_full": gradients_gb,
            "trainable_params": estimator.trainable_params,
            "lora_adapter_params": estimator.lora_adapter_params,
            "effective_seq": effective_seq,
            "num_gpus": estimator.num_gpus,
        }
    }
