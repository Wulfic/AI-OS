"""Core training configuration fields.

Model, dataset, and basic optimization parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseFields:
    """Core training parameters for model, data, and basic optimization."""
    
    # ============================================================================
    # Model and Data
    # ============================================================================
    model: str = "artifacts/hf_implant/base_model"
    """HF model name or local path for tokenizer."""
    
    dataset_file: Optional[str] = None
    """Text/CSV/archive/dir to sample training lines from (REQUIRED)."""
    
    dataset_chunk_size: int = 4000
    """Number of samples to load per training cycle in iterate mode.
    
    Controls memory usage when streaming large datasets:
    - Smaller chunks (2000): Lower VRAM usage, slower processing
    - Default (4000): Balanced for most scenarios (~4GB VRAM per chunk)
    - Larger chunks (8000+): Faster but requires more VRAM
    
    This is the sub-chunk size within a dataset block. For example:
    - Block: 100k samples downloaded from HuggingFace
    - Chunk: 4000 samples loaded per training cycle
    - Steps: Training iterations per chunk
    
    Adjust based on available VRAM:
    - 8GB VRAM: 2000-3000 samples
    - 12GB VRAM: 4000 samples (default)
    - 24GB+ VRAM: 8000+ samples
    
    Default: 4000 samples
    """
    
    samples_per_block: int = 100000
    """Number of samples in each dataset block for streaming.
    
    When downloading from HuggingFace or processing large datasets,
    data is loaded in blocks of this size. Blocks are the fundamental
    unit for parallel GPU distribution and epoch tracking.
    
    Block hierarchy:
    - Dataset: Entire dataset (e.g., 10M samples)
    - Block: Downloaded chunk (samples_per_block, e.g., 100k samples)
    - Chunk: Training subdivision (dataset_chunk_size, e.g., 4k samples)
    - Batch: Training batch (batch_size, e.g., 8 samples)
    
    For parallel training:
    - Each GPU receives unique chunks from the current block
    - No duplicate training across GPUs within same block
    - Next block loaded when current block exhausted
    - Epoch completes when all blocks processed
    
    Larger blocks:
    - Fewer download operations (faster for HF datasets)
    - More RAM usage for cached blocks
    - Better for high-bandwidth connections
    
    Smaller blocks:
    - More frequent downloads (more overhead)
    - Less RAM usage
    - Better for memory-constrained systems
    
    Typical values:
    - 100000 (default): Balanced for most scenarios
    - 50000: Memory-constrained systems (<32GB RAM)
    - 200000: High-RAM systems (64GB+ RAM)
    
    Default: 100000 samples per block
    """
    
    max_seq_len: int = 128
    """Token sequence length (context length)."""
    
    batch_size: int = 8
    """Training batch size (may be auto-adjusted by OOM backoff)."""
    
    steps: int = 200
    """Number of training steps."""
    
    # ============================================================================
    # Optimization
    # ============================================================================
    lr: float = 2e-4
    """Learning rate."""
    
    auto_adjust_lr: bool = True
    """Automatically adjust learning rate based on model configuration.
    
    When enabled:
    - MoE models: Reduced to safer values (1e-5 to 2e-5) for router stability
    - Large models (>1B params): Slightly reduced for stability
    - Standard models: Uses configured lr value as-is
    
    Disable this to use your exact lr value without any adjustments.
    
    Default: True (recommended for stability across model types)
    """
    
    halt_max_steps: int = 2
    """Maximum ACT (Adaptive Computation Time) segments during training."""
    
    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce VRAM usage.
    
    Trades ~20% speed for 30-50% less memory. Recommended for large contexts.
    Default: enabled for better VRAM efficiency.
    """
    
    use_amp: bool = True
    """Use automatic mixed precision (FP16/BF16) for activations.
    
    Saves ~40-50% memory with minimal quality loss. Default: enabled.
    """
    
    use_cpu_offload: bool = False
    """Offload carry states to CPU between chunks.
    
    For extreme contexts (>500K tokens). Slower but uses less VRAM.
    Default: disabled (only beneficial for very large contexts).
    """
    
    use_8bit_optimizer: bool = False
    """Use 8-bit optimizer (bitsandbytes) for 75% optimizer memory savings.
    
    Reduces optimizer state memory from FP32 to INT8, saving ~75% memory
    with minimal impact on training quality. Particularly effective for
    large models (>100M parameters).
    
    Memory savings example:
    - 60M params with AdamW: ~360 MB saved
    - 500M params with AdamW: ~3 GB saved
    - 1B params with AdamW: ~6 GB saved
    
    Requires: bitsandbytes library (pip install bitsandbytes)
    Default: disabled (enable for memory-constrained scenarios)
    """
    
    use_chunked_training: bool = False
    """Enable chunked training for extreme context lengths.
    
    Splits long sequences into smaller chunks to reduce memory usage.
    User must explicitly enable this - no automatic enforcement.
    
    Benefits:
    - Allows training on sequences that would otherwise OOM
    - Reduces peak memory usage
    - Maintains gradient flow across chunks
    
    Trade-offs:
    - Slightly slower training (~10-20% overhead)
    - More complex training loop
    
    Recommendation: Use the 'Optimize Settings' button to determine if chunking is needed.
    Default: disabled (user-controlled)
    """
    
    chunk_size: int = 2048
    """Size of chunks when using chunked training.
    
    Smaller chunks use less memory but are slower.
    Typical values: 1024-4096 tokens.
    
    Recommendations:
    - 2048: Good balance for most scenarios
    - 1024: Very constrained memory (10GB VRAM)
    - 4096: More memory available (24GB+ VRAM)
    
    Default: 2048 tokens
    """
    
    sys_mem_cap_pct: Optional[int] = None
    """Soft cap for system memory usage percent.
    
    If system RAM usage exceeds this percentage, CPU batch size is auto-reduced.
    Typical values: 70-90. Default: None (no cap).
    """
