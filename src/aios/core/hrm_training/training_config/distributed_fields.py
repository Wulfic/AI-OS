"""Distributed training configuration fields.

Multi-GPU, device placement, and inference settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedFields:
    """Distributed training and device placement parameters."""
    
    # ============================================================================
    # Distributed Training
    # ============================================================================
    device: str = "auto"
    """Device for training: auto|cpu|cuda|xpu|mps|dml.
    
    "auto" will use CUDA if available, otherwise CPU.
    """
    
    cuda_ids: Optional[str] = None
    """Comma-separated CUDA device indices to use (e.g., '0,1' for 2 GPUs).
    
    If multiple IDs provided, DDP is automatically enabled.
    """
    
    ddp: bool = False
    """Enable multi-GPU training via torch.distributed (CUDA only).
    
    Automatically enabled when multiple cuda_ids are specified.
    """
    
    world_size: Optional[int] = None
    """Number of processes/GPUs to use for DDP.
    
    Defaults to number of cuda_ids or all visible GPUs.
    """
    
    strict: bool = False
    """Disallow device fallbacks.
    
    If True, error instead of falling back (e.g., no CPU fallback if CUDA requested).
    """
    
    parallel_independent: bool = False
    """Use parallel independent training (Windows-compatible multi-GPU).
    
    Trains different data blocks on different GPUs sequentially, then merges
    checkpoints via weight averaging. Bypasses torch.distributed (DDP) entirely,
    making it compatible with Windows where DDP is broken.
    
    Each GPU trains on its own data subset independently with no synchronization.
    After training, model checkpoints are merged by averaging weights.
    
    Expected performance: ~90% of DDP throughput, 95-98% final accuracy.
    Requires: multiple cuda_ids specified.
    """
    
    inference_device: Optional[str] = None
    """Specific GPU device for inference while training on another GPU.
    
    Example: "cuda:1" to run inference on GPU 1 while training on GPU 0.
    Enables simultaneous training and inference on multi-GPU systems.
    Requires at least 2 GPUs available.
    
    Note: Leave None to disable separate inference GPU (inference and training on same device).
    """
    
    hot_reload_steps: int = 0
    """Frequency (in training steps) to reload inference model from training checkpoint.
    
    If > 0 and inference_device is set, the inference model will be reloaded
    every N steps with the latest training weights. This enables real-time testing
    of the model during training on a separate GPU.
    
    Example: hot_reload_steps=100 reloads every 100 training steps.
    Default: 0 (disabled)
    """
    
    # ============================================================================
    # Evaluation
    # ============================================================================
    eval_file: Optional[str] = None
    """Held-out file/dir for final evaluation after training.
    
    If provided, enables evaluation metrics on validation data.
    """
    
    eval_batches: int = 10
    """Maximum eval batches for final evaluation.
    
    Set to 0 to disable evaluation. Default: 10 batches.
    """
