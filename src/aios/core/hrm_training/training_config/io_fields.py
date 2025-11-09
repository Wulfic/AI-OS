"""I/O and persistence configuration fields.

Checkpointing, logging, and file paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class IOFields:
    """I/O and persistence parameters."""
    
    # ============================================================================
    # I/O and Persistence
    # ============================================================================
    save_dir: str = "training_data/actv1"
    """Directory where checkpoints and training state are saved."""
    
    stop_file: Optional[str] = None
    """If file exists, training stops gracefully.
    
    Useful for remote monitoring and controlled shutdown.
    """
    
    log_file: Optional[str] = None
    """Optional JSONL file to append training/eval metrics.
    
    Each line is a JSON object with step, loss, etc.
    """
    
    student_init: Optional[str] = None
    """Optional path to existing ACTV1 student state_dict (.pt) to continue training.
    
    If provided, training resumes from this checkpoint.
    """
    
    brain_name: Optional[str] = None
    """Optional brain bundle name.
    
    When provided, saves to --bundle-dir/<brain-name>.
    """
    
    default_goal: Optional[str] = None
    """Default training goal/directive for this brain.
    
    This goal guides what the brain learns during training.
    Automatically added to the brain's goals when training starts.
    """
    
    expert_id: Optional[str] = None
    """Expert ID for expert-specific training mode.
    
    When provided, trains only a FeedForward expert module instead of full HRM model.
    The expert is saved to artifacts/experts/{expert_id}/ and registered in the
    expert registry. This enables training specialized experts for specific goals
    without affecting the base model.
    
    Expert training features:
    - Trains lightweight FeedForward module (hidden_size → intermediate_size → hidden_size)
    - Saves to artifacts/experts/{expert_id}/expert.pt
    - Auto-updates expert_registry.json with metadata
    - Can be loaded at runtime via LazyExpertLoader
    - Supports all optimization features (AMP, gradient checkpointing, etc.)
    
    If None (default), trains full HRM ACT-v1 model normally.
    """
    
    bundle_dir: str = "artifacts/brains/actv1"
    """Base directory for ACTV1 brain bundles.
    
    Brain-specific subdirectories are created here.
    
    Note: All checkpoints are saved in safetensors format for security and performance.
    Safetensors provides:
    - Security: No arbitrary code execution (unlike Pickle-based .pt)
    - Performance: Faster load/save with memory-mapped I/O
    - Interoperability: Standard across HuggingFace ecosystem
    """
    
    # ============================================================================
    # Chunk/Block Tracking for Parallel Training
    # ============================================================================
    samples_per_block: int = 100000
    """Samples per streaming block downloaded from dataset source.
    
    When streaming large datasets (e.g., from HuggingFace), data is downloaded
    in blocks. Each block is then subdivided into chunks for training.
    
    Default: 100,000 samples per block (standard for HF streaming)
    """
    
    dataset_total_samples: Optional[int] = None
    """Total number of samples in the complete dataset.
    
    Used for epoch tracking and progress calculation. If None, will be
    determined automatically during training.
    """
    
    total_blocks: Optional[int] = None
    """Total number of blocks in the dataset.
    
    Calculated as: ceil(dataset_total_samples / samples_per_block)
    If None, will be determined automatically.
    """
    
    current_epoch: int = 0
    """Current epoch number (0-based).
    
    Increments after all blocks in dataset have been trained once.
    Used for tracking training progress across multiple passes.
    """
    
    samples_processed_this_epoch: int = 0
    """Number of samples processed in current epoch.
    
    Resets to 0 at start of each epoch. Used for epoch progress tracking.
    """
    
    blocks_processed_this_epoch: str = ""
    """Comma-separated list of block IDs processed in current epoch.
    
    Format: "0,1,5,8" indicates blocks 0, 1, 5, and 8 have been trained.
    Used to prevent duplicate block training within an epoch.
    Resets to empty string at start of each epoch.
    """
    
    current_block_samples: int = 0
    """Number of samples processed in the current block.
    
    When this reaches samples_per_block, the block is considered complete.
    Resets to 0 when starting a new block.
    """
