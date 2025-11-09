"""Training optimization configuration fields.

DeepSpeed, quantization, and memory optimization settings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizationFields:
    """Training optimization and memory efficiency parameters."""
    
    # ============================================================================
    # DeepSpeed ZeRO Optimization
    # ============================================================================
    zero_stage: str = "none"
    """DeepSpeed ZeRO optimization stage.
    
    Options:
    - "none": No DeepSpeed (standard PyTorch training)
    - "zero1": Optimizer state partitioning (~25% VRAM reduction)
    - "zero2": Optimizer + gradient partitioning (~50% VRAM reduction, recommended)
    - "zero3": Optimizer + gradient + parameter partitioning (~75% VRAM reduction)
    
    Note: Requires DeepSpeed library installed. Auto-selected if --optimize is used.
    """
    
    # ============================================================================
    # Model Precision and Quantization Options
    # ============================================================================
    
    model_dtype: str = "fp32"
    """Model weight precision/data type.
    
    Controls the precision of model weights (parameters). Lower precision reduces
    memory usage and can speed up training, but may impact numerical stability.
    
    Options:
    - "fp32" (float32): Full precision, maximum stability, highest memory (default)
    - "fp16" (float16): Half precision, 50% memory reduction, faster on modern GPUs
    - "bf16" (bfloat16): Brain float 16, 50% memory reduction, better stability than fp16
    
    Recommendations:
    - fp32: Use for small models or when maximum precision is required
    - fp16: Best for NVIDIA GPUs with Tensor Cores (V100, A100, RTX 20xx+)
    - bf16: Best for Ampere+ GPUs (A100, RTX 30xx+) or TPUs, more stable than fp16
    
    Memory savings example (87M params model):
    - fp32: ~348 MB
    - fp16/bf16: ~174 MB (50% reduction)
    
    Note: This is separate from AMP (use_amp), which only affects activations.
    Model dtype affects the stored weights, while AMP affects computation.
    
    Compatibility:
    - Works with all optimizations (gradient checkpointing, PEFT, etc.)
    - bf16 requires PyTorch 1.10+ and compatible hardware
    - Combine with use_amp=True for maximum memory efficiency
    """
    
    load_in_8bit: bool = False
    """Load model weights in 8-bit precision using bitsandbytes.
    
    Quantizes model weights to INT8, reducing memory by ~75% with minimal quality loss.
    Particularly effective when combined with PEFT/LoRA (QLoRA pattern).
    
    Benefits:
    - 75% reduction in model memory (87M params: 348 MB → 87 MB)
    - Enables training larger models on consumer GPUs
    - Minimal accuracy degradation (<1% typically)
    - Works well with PEFT adapters
    
    Requirements:
    - bitsandbytes library (pip install bitsandbytes>=0.43.0)
    - CUDA-capable GPU (not supported on CPU/DML)
    - PyTorch 2.0+
    
    Memory example (87M params):
    - fp32: 348 MB
    - 8-bit: ~87 MB (75% reduction)
    - 8-bit + LoRA (r=16): ~90 MB total (only adapters trainable)
    
    Note: Quantization happens at model loading time. Training gradients remain
    in higher precision for numerical stability.
    
    Trade-offs:
    - Slower training (~10-20% overhead from quantization/dequantization)
    - May not work with custom model architectures (HRM may have limitations)
    - Cannot be combined with load_in_4bit
    
    Recommended for:
    - Memory-constrained scenarios (8-12 GB VRAM)
    - Fine-tuning with PEFT on large models
    - When model memory dominates total memory usage
    """
    
    load_in_4bit: bool = False
    """Load model weights in 4-bit precision using bitsandbytes (QLoRA).
    
    Quantizes model weights to 4-bit NormalFloat, reducing memory by ~87.5%.
    The most aggressive quantization option, enabling very large models on consumer hardware.
    
    Benefits:
    - 87.5% reduction in model memory (87M params: 348 MB → 43 MB)
    - Enables training models 4x larger than 8-bit
    - Surprisingly good quality with proper configuration
    - Perfect for QLoRA (4-bit base + LoRA adapters)
    
    Requirements:
    - bitsandbytes library (pip install bitsandbytes>=0.43.0)  
    - CUDA-capable GPU with compute capability 7.0+ (V100, T4, RTX 20xx+)
    - PyTorch 2.0+
    
    Memory example (87M params):
    - fp32: 348 MB
    - 4-bit: ~43 MB (87.5% reduction)
    - 4-bit + LoRA (r=16): ~45 MB total
    
    4-bit Quantization Types:
    - NF4 (NormalFloat4): Optimized for normally distributed weights (default)
    - FP4: Standard 4-bit float
    
    Note: 4-bit uses double quantization by default (quantizing the quantization
    constants) for additional memory savings with minimal quality impact.
    
    Trade-offs:
    - Slower than 8-bit (~20-30% training overhead)
    - More aggressive quality degradation (monitor val loss carefully)
    - Requires compatible GPU architecture
    - Cannot be combined with load_in_8bit
    - May not work with custom architectures
    
    Recommended for:
    - Extremely memory-constrained scenarios (4-8 GB VRAM)
    - When you need to fit a model 4x larger
    - QLoRA fine-tuning pattern
    - Experimentation before committing to full precision training
    """

    # ============================================================================
    # Gradient Accumulation
    # ============================================================================
    gradient_accumulation_steps: int = 1
    """Number of batches to accumulate gradients before updating weights.
    
    Enables training with larger effective batch sizes without increasing VRAM.
    The effective batch size is: physical_batch_size × gradient_accumulation_steps
    
    Benefits:
    - Fixes loss instability from small batch sizes
    - No VRAM increase (memory usage stays at physical batch size)
    - Smoother training dynamics
    - Better gradient estimates
    
    Example:
    - batch_size=8, gradient_accumulation_steps=4 → effective_batch_size=32
    - VRAM usage: ~10GB (for batch=8)
    - Training stability: equivalent to batch=32
    
    Recommended values:
    - 1: No accumulation (default, update every batch)
    - 2-4: Mild accumulation for slightly smoother training
    - 4-8: Moderate accumulation (recommended for most cases)
    - 8-16: High accumulation for very small batch sizes
    - 16+: Extreme accumulation (use when batch=1-2 required)
    
    How to choose:
    1. Start with current batch_size and desired effective_batch_size
    2. Calculate: gradient_accumulation_steps = effective_batch_size / batch_size
    3. Test and adjust based on loss stability
    
    Memory impact:
    - Gradients: +1× model size (same as normal training)
    - Activations: Only for physical batch size
    - Total overhead: Negligible (<5% of total memory)
    
    Performance impact:
    - Slightly slower due to more forward passes
    - ~5-15% overhead depending on accumulation_steps
    - Worth it for stability improvement
    
    Compatibility:
    - Works with: AMP, gradient checkpointing, DeepSpeed ZeRO, PEFT/LoRA
    - Works across: DDP, parallel independent, single-GPU modes
    - Scheduler: Automatically adjusted to step with weight updates
    """
