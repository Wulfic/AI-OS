"""RAM (system memory) estimation logic for HRM training.

This module estimates system RAM requirements including:
- Optimizer state (if CPU offloaded)
- Model CPU copy
- Tokenizer and vocabulary
- Dataset buffers
- PyTorch/CUDA/Python overhead
- Training-specific buffers
- GPU overflow to RAM
"""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .estimator import MemoryEstimator

from .constants import BYTES_FP32, BYTES_INT32, GB


def estimate_ram(estimator: "MemoryEstimator") -> Dict[str, float]:
    """
    Estimate system RAM usage.
    
    Args:
        estimator: MemoryEstimator instance with configuration
    
    Returns dict with:
        - optimizer_gb: Offloaded optimizer state (if CPU offload)
        - model_cpu_gb: Model copy in CPU RAM (before GPU transfer)
        - tokenizer_gb: Tokenizer vocabulary and buffers
        - dataset_gb: Dataset loading/streaming overhead
        - pytorch_gb: PyTorch/Python/CUDA overhead
        - training_gb: Training-specific buffers and staging
        - gpu_overflow_gb: Shared GPU memory (RAM used as VRAM overflow)
        - total_gb: Total system RAM needed
    """
    
    # Import vram estimation to get breakdown
    from .vram_estimation import estimate_vram
    
    # Optimizer offload (if enabled)
    vram_estimate = estimate_vram(estimator)
    breakdown = vram_estimate.get("breakdown", {})
    optimizer_ram_gb = breakdown.get("optimizer_ram", 0.0) if isinstance(breakdown, dict) else 0.0
    
    # ===== 1. MODEL CPU COPY =====
    # PyTorch loads model on CPU first, then transfers to GPU
    # Even after transfer, keeps CPU copy for checkpointing and optimizer
    model_bytes_per_param = BYTES_FP32
    model_cpu_gb = (estimator.total_params * model_bytes_per_param) / GB
    
    # Add buffer overhead (PyTorch allocates extra for safety)
    model_cpu_gb *= 1.5  # 50% buffer overhead
    
    # ===== 2. TOKENIZER + VOCABULARY =====
    # HuggingFace tokenizers load full vocabulary into RAM
    # Size varies by tokenizer:
    # - GPT-2: ~0.5 GB
    # - LLaMA/Mistral: ~1-2 GB  
    # - Very large vocabs: ~2-3 GB
    if estimator.vocab_size <= 50000:
        tokenizer_gb = 0.5  # Small vocab (GPT-2)
    elif estimator.vocab_size <= 100000:
        tokenizer_gb = 1.0  # Medium vocab
    else:
        tokenizer_gb = 2.0  # Large vocab
    
    # Add tokenizer overhead (regex patterns, special tokens, etc.)
    tokenizer_gb += 0.3
    
    # ===== 3. DATASET MEMORY =====
    # Even with streaming, need RAM for tokenized batches and prefetch buffers
    # For long sequences, this is significant!
    
    if estimator.seq_len > 8192:
        # Streaming mode with long context
        # Need buffer for current batch + prefetch buffer (typically 2-4 batches)
        prefetch_batches = 4
        batch_memory = (estimator.batch_size * estimator.seq_len * BYTES_INT32) / GB
        dataset_gb = batch_memory * prefetch_batches
        dataset_gb += 0.5  # Streaming dataset object overhead
    else:
        # Eager loading or short context streaming
        # Full dataset in RAM (or reasonable buffer)
        dataset_gb = (estimator.batch_size * estimator.seq_len * BYTES_INT32 * 10) / GB  # ~10 batches
        dataset_gb = max(1.0, min(dataset_gb, 4.0))  # 1-4 GB range
    
    # Long sequence overhead: tokenization and string processing uses extra RAM
    if estimator.seq_len > 4096:
        dataset_gb *= 1.5  # 50% extra for long sequences
    
    # ===== 4. PYTORCH/CUDA/PYTHON OVERHEAD =====
    # This is much larger than naive estimates!
    pytorch_base_gb = 0.0
    
    # Python interpreter
    pytorch_base_gb += 0.5
    
    # PyTorch library (includes all operators, autograd engine, etc.)
    pytorch_base_gb += 1.5
    
    # CUDA runtime and libraries (if using CUDA)
    # cuDNN, cuBLAS, NCCL, etc. all loaded in RAM
    if estimator.num_gpus > 0:
        pytorch_base_gb += 2.0  # CUDA libs
    
    # DeepSpeed overhead (if using ZeRO)
    if estimator.zero_stage != "none":
        pytorch_base_gb += 1.0  # DeepSpeed engine and communication buffers
    
    # Additional overhead for DDP processes (each process duplicates base overhead)
    if estimator.num_gpus > 1:
        pytorch_ddp_gb = pytorch_base_gb * (estimator.num_gpus - 1) * 0.7  # Each extra process ~70% of base
    else:
        pytorch_ddp_gb = 0
    
    pytorch_total_gb = pytorch_base_gb + pytorch_ddp_gb
    
    # ===== 5. TRAINING-SPECIFIC OVERHEAD =====
    # Gradient accumulation, mixed precision shadows, checkpoint staging
    training_gb = 0.0
    
    # Gradient accumulation staging area
    if estimator.use_gradient_checkpointing:
        training_gb += 0.5  # Recomputation buffers
    
    # Mixed precision CPU shadows (FP32 copies of FP16 gradients)
    if estimator.use_amp:
        gradients_fp32_gb = (estimator.total_params * BYTES_FP32) / GB
        training_gb += gradients_fp32_gb * 0.5  # Partial shadowing
    
    # Chunked training overhead (staging and buffer management)
    if estimator.use_chunking:
        # Chunk buffers, carry state staging
        chunk_buffer_gb = (estimator.batch_size * estimator.chunk_size * estimator.hidden_size * 2) / GB
        training_gb += chunk_buffer_gb * 2  # Double buffering
    
    # Checkpoint saving staging
    training_gb += 0.5  # Temporary checkpoint buffers
    
    # ===== 6. GPU OVERFLOW TO RAM (CRITICAL!) =====
    # When GPU VRAM is full, PyTorch and OS use system RAM as overflow
    # This shows as "Shared GPU Memory" in Task Manager
    # User's screenshot showed 4.8 GB shared GPU memory!
    
    gpu_overflow_gb = 0.0
    
    # If estimated VRAM exceeds typical GPU capacity, assume overflow
    # This is aggressive but realistic for memory-constrained training
    estimated_vram_gb = vram_estimate.get("total_gb", 0)
    
    # For consumer GPUs (8-24 GB), if we're near limit, expect significant overflow
    typical_gpu_vram = 11.0  # Assume RTX 2080 Ti or similar
    
    if estimated_vram_gb > typical_gpu_vram * 0.85:
        # Heavy memory pressure → RAM overflow
        # Estimate 20-40% of VRAM usage spills to RAM
        overflow_factor = 0.3
        gpu_overflow_gb = estimated_vram_gb * overflow_factor
    elif estimated_vram_gb > typical_gpu_vram * 0.70:
        # Moderate pressure
        overflow_factor = 0.15
        gpu_overflow_gb = estimated_vram_gb * overflow_factor
    
    # Add extra overflow for chunked training with long sequences
    # (frequent data movement between GPU and CPU)
    if estimator.use_chunking and estimator.seq_len > 8192:
        gpu_overflow_gb *= 1.5
    
    # ===== TOTAL RAM =====
    total_ram_gb = (
        optimizer_ram_gb +      # Offloaded optimizer (if enabled)
        model_cpu_gb +           # Model copy in CPU
        tokenizer_gb +           # Tokenizer vocabulary
        dataset_gb +             # Dataset buffers
        pytorch_total_gb +       # PyTorch/CUDA/Python
        training_gb +            # Training-specific overhead
        gpu_overflow_gb          # GPU overflow to RAM
    )
    
    return {
        "optimizer_gb": optimizer_ram_gb,
        "model_cpu_gb": model_cpu_gb,
        "tokenizer_gb": tokenizer_gb,
        "dataset_gb": dataset_gb,
        "pytorch_gb": pytorch_total_gb,
        "training_gb": training_gb,
        "gpu_overflow_gb": gpu_overflow_gb,
        "total_gb": total_ram_gb,
    }
