"""Advanced training configuration fields.

PEFT/LoRA, deprecated fields, and experimental features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AdvancedFields:
    """Advanced training features and deprecated parameters."""
    
    # ============================================================================
    # Advanced Options
    # ============================================================================
    kl: float = 0.0
    """KL divergence scaling factor for KL-regularized objectives.

    Default: 0.0 (disable KL regularization)."""

    kl_temp: float = 1.0
    """Temperature applied to KL term annealing schedules.

    Default: 1.0 (no annealing)."""

    resume: bool = False
    """Resume training from last checkpoint.
    
    When enabled and a checkpoint exists:
    - Loads model weights from checkpoint
    - Restores training configuration (dataset, batch size, learning rate, etc.)
    - Continues step counter from where training left off
    - Validates dataset matches to prevent breaking training continuity
    
    If checkpoint doesn't exist or dataset doesn't match, starts fresh training.
    
    Use cases:
    - Continue training after interruption
    - Add more training steps to an existing brain
    - Resume long-running training sessions
    
    Safety:
    - Dataset-aware: Warns if dataset changed
    - Config validation: Alerts if critical settings differ
    - Graceful degradation: Falls back to fresh start if resume fails
    
    Default: False (start fresh training)
    """
    
    iterate: bool = False
    """Repeat generation + training cycles indefinitely until stopped.
    
    WARNING: This is an experimental feature for continuous training loops.
    """
    
    stop_after_block: bool = False
    """Stop training after completing the current block.
    
    When enabled, training will complete the current block (downloaded dataset chunk)
    and then stop gracefully. A "block" is a portion of the dataset loaded into memory
    for training (typically 4000-100000 samples depending on dataset size).
    
    Use cases:
    - Stop after processing current downloaded chunk
    - Checkpoint after manageable data portions
    - Test training on limited data before full epoch
    
    Default: False
    """
    
    stop_after_epoch: bool = False
    """Stop training after completing the current epoch.
    
    When enabled, training will complete the current epoch (full dataset pass) and
    then stop gracefully, even if in iterate mode. This allows for clean checkpointing
    after a complete pass through the ENTIRE dataset.
    
    An "epoch" is defined as one complete pass through the ENTIRE dataset across ALL blocks.
    For large datasets (e.g., millions of samples), this means cycling through all available
    blocks until every sample has been seen once.
    
    Hierarchy:
    - Dataset: Entire dataset (e.g., 10M samples)
    - Block: Downloaded chunk (e.g., 100k samples) 
    - Chunk: Training subdivision (e.g., 4k samples)
    - Epoch: One complete pass through ALL blocks in dataset
    
    Use cases:
    - Stop after complete dataset pass (all blocks processed)
    - Ensure checkpoint is saved after full epoch
    - Multi-epoch training with clean boundaries
    
    Note: This is different from stop_after_block and stop_file mechanisms.
    
    Default: False (train for all specified steps/cycles)
    """
    
    # ============================================================================
    # Epoch Tracking (Internal - Set by Training Loop)
    # ============================================================================
    dataset_total_samples: Optional[int] = None
    """Total number of samples in the COMPLETE dataset.
    
    For HuggingFace datasets, this is the full dataset size (e.g., 10 million samples).
    For local files, this is the total line count.
    
    Automatically detected at training start.
    Used to calculate epoch completion across all blocks.
    Internal field - not set by user.
    """
    
    samples_per_block: Optional[int] = None
    """Number of samples in each downloaded block.
    
    A "block" is a portion of the dataset loaded from source (e.g., 100k samples from HF dataset).
    For streaming datasets, this is the download chunk size.
    For small local files, equals dataset_total_samples.
    
    Used to track which blocks have been processed.
    Internal field - not set by user.
    """
    
    total_blocks: Optional[int] = None
    """Total number of blocks in the complete dataset.
    
    Calculated as: ceil(dataset_total_samples / samples_per_block)
    For small datasets, equals 1.
    
    Used to determine when all blocks have been visited (epoch complete).
    Internal field - not set by user.
    """
    
    samples_processed_this_epoch: int = 0
    """Running count of samples processed in the current epoch.
    
    Incremented after each block is processed. Reset to 0 when epoch completes.
    Tracks progress through the ENTIRE dataset (all blocks).
    
    Saved/restored in checkpoints.
    Internal field - managed by training loop.
    """
    
    blocks_processed_this_epoch: str = ""
    """Comma-separated list of block indices processed in the current epoch.
    
    Tracks which blocks have been visited to detect when all blocks are covered.
    Example: "0,3,7,1,5,2,4" indicates blocks 0-7 have been processed.
    
    Reset to empty string when epoch completes (all blocks visited).
    Saved/restored in checkpoints.
    Internal field - managed by training loop.
    """
    
    current_epoch: int = 0
    """Current epoch number (0-indexed).
    
    Increments each time the FULL dataset has been covered (all blocks visited).
    Used for logging and progress tracking.
    
    Saved/restored in checkpoints.
    Internal field - managed by training loop.
    """
    
    current_block_samples: int = 0
    """Samples processed in the current block.
    
    Used to track when current block is complete (for stop_after_block).
    Reset to 0 when moving to next block.
    Internal field - managed by training loop.
    """
    
    optimize: bool = False
    """Automatically find optimal settings for max context and batch size.
    
    When enabled, searches for the largest context length (up to 100K) and
    batch size that fit in available VRAM. Overrides max-seq-len and batch-size.
    
    WARNING: This runs a search process that may take several minutes.
    """
    
    ascii_only: bool = False
    """Filter dataset to ASCII-only lines for English focus.
    
    Useful when training on mixed-language datasets but wanting English-only output.
    """
    
    linear_dataset: bool = False
    """Process dataset linearly without shuffling.
    
    When enabled (linear_dataset=True):
    - Dataset samples are processed in sequential order
    - Training position is tracked and saved in checkpoints
    - Can pause and resume without retraining the same data
    - Ensures 100% dataset coverage before wrapping to the beginning
    
    When disabled (linear_dataset=False, default):
    - Dataset is shuffled each epoch using deterministic seeds
    - Better for model generalization (recommended for most cases)
    - Each epoch presents data in a different order
    
    Use cases for linear mode:
    - When dataset order matters (sequential story, curriculum learning)
    - When you need precise tracking of training progress through dataset
    - When resuming training and want to continue exactly where left off
    - When debugging or validating training on specific data samples
    
    Note: Position tracking is saved to checkpoints automatically when using
    streaming datasets (>4GB). For smaller datasets, the entire dataset is
    loaded into memory and sampling is random (position tracking not applicable).
    
    Default: False (shuffled mode for better generalization)
    """

    force_train: bool = False
    """Force training even if the chunk tracker thinks the chunk is already trained.

    This is primarily a diagnostics/escape hatch for cases where:
    - you want to rerun a short smoke test without changing --save-dir, or
    - the chunk tracker state file got out of sync with your intent.

    When True, training will proceed even if the current chunk is marked
    complete in chunk_tracker_state.json.

    Default: False (respect chunk tracker and skip already-trained chunks).
    """

    adaptive_lr_config: Optional[str] = None
    """Optional path to an adaptive LR config file (JSON/TOML/YAML).

    When provided and auto_adjust_lr is enabled, the adaptive LR scheduler will
    load overrides from this file on top of safe defaults derived from --lr.

    Supported formats:
    - .json (built-in)
    - .toml (built-in via tomllib)
    - .yaml/.yml (requires PyYAML)
    """

    adaptive_lr_debug_level: Optional[int] = None
    """Override adaptive LR debug_level without editing the scheduler config file.

    Values:
    - 0: off
    - 1: adjustments only
    - 2: periodic window summaries
    - 3: very verbose

    If None (default), uses whatever the scheduler config resolves to.
    """

    adaptive_lr_emit_window_summary: Optional[bool] = None
    """Override emit_window_summary for adaptive LR.

    If None (default), uses whatever the scheduler config resolves to.
    """

    adaptive_lr_window_summary_every: Optional[int] = None
    """Override window_summary_every for adaptive LR (emit every N windows).

    If None (default), uses whatever the scheduler config resolves to.
    """

    adaptive_lr_state_path: Optional[str] = None
    """Optional path to persist AdaptiveLRScheduler.state_dict() as JSON.

    If provided, the scheduler will write a JSON state file as it runs.
    When resuming, this file can be loaded to continue LR behavior smoothly.

    If None (default), uses <save_dir>/adaptive_lr_state.json.
    """

    adaptive_lr_reset_state: bool = False
    """If True, ignore any persisted adaptive LR state when resuming.

    Default: False (attempt to restore scheduler state when available).
    """
    
    dataset_start_offset: int = 0
    """Starting sample index for resuming linear dataset training.
    
    Only used when linear_dataset=True. Allows manually specifying where
    to start in the dataset, useful for:
    - Resuming interrupted training runs
    - Training on specific portions of the dataset
    - Continuing from a known checkpoint position
    
    Value should be between 0 and (num_samples - 1).
    Default: 0 (start from beginning)
    """
    
    # ============================================================================
    # PEFT (Parameter-Efficient Fine-Tuning) Options
    # ============================================================================
    use_peft: bool = False
    """Enable PEFT (e.g., LoRA) for parameter-efficient training.
    
    When enabled, only adds small adapter layers (~1-2M params) instead of training
    all 87M+ parameters. Reduces memory by 5-10GB and speeds up training significantly.
    
    Benefits:
    - 95-99% reduction in trainable parameters
    - 40-60% reduction in VRAM usage
    - Faster training and convergence
    - Easy to merge adapters back into base model
    
    Compatible with all other optimizations (gradient checkpointing, AMP, etc.).
    """
    
    peft_method: str = "lora"
    """PEFT method to use.
    
    Supported methods:
    - 'lora': Low-Rank Adaptation (recommended, most stable)
    - 'adalora': Adaptive LoRA (adjusts rank dynamically)
    - 'ia3': Infused Adapter by Inhibiting and Amplifying Inner Activations
    - 'loha': Low-Rank Hadamard Product
    - 'lokr': Low-Rank Kronecker Product
    
    Default: 'lora' (best balance of quality and efficiency)
    """
    
    lora_r: int = 16
    """LoRA rank (adapter matrix dimension).
    
    Higher rank = more model capacity but more parameters and memory.
    
    Typical values:
    - r=8: Minimal (0.5M params, ~1-2GB VRAM) - good for fine-tuning
    - r=16: Balanced (2M params, ~2-3GB VRAM) - recommended default
    - r=32: High capacity (8M params, ~4-5GB VRAM) - for complex tasks
    - r=64: Very high (32M params, ~8-10GB VRAM) - rarely needed
    
    Rule of thumb: Use 2× rank for lora_alpha (e.g., r=16, alpha=32).
    """
    
    lora_alpha: int = 32
    """LoRA scaling parameter.
    
    Controls the scaling of adapter outputs. The effective learning rate of
    adapters is: (lora_alpha / lora_r) * base_lr
    
    Typical values:
    - lora_alpha = 2 × lora_r (standard scaling)
    - Higher alpha = stronger adapter influence
    - Lower alpha = more conservative adaptation
    
    Default: 32 (works well with r=16).
    """
    
    lora_dropout: float = 0.05
    """Dropout probability for LoRA adapter layers.
    
    Standard dropout applied to LoRA adapters for regularization.
    Typical values: 0.0-0.1. Default: 0.05.
    """
    
    lora_target_modules: str = "q_proj,v_proj"
    """Comma-separated list of module names to apply LoRA adapters to.
    
    HRM ACTv1 architecture modules:
    - Attention projections: q_proj, k_proj, v_proj, o_proj (recommended)
    - MLP projections: gate_proj, up_proj, down_proj (optional, more params)
    
    Recommendations:
    - Minimal: 'q_proj,v_proj' (default) - attention queries and values only
    - Balanced: 'q_proj,k_proj,v_proj,o_proj' - full attention
    - Full: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj' - attention + MLP
    
    Note: lm_head and q_head (HRM-specific halting head) always remain trainable
    to preserve architecture integrity.
    
    
    Default: 'q_proj,v_proj' (good balance of efficiency and quality).
    """

