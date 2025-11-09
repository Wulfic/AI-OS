"""Unified configuration for HRM ActV1 training.

This module provides a single source of truth for all training parameters,
ensuring consistency across CLI, GUI, and any future interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Unified configuration for HRM ActV1 training.
    
    Single source of truth for all training parameters.
    Used by CLI, GUI, and any future interfaces.
    
    This ensures feature parity across interfaces and makes it easy to add
    new optimizations (Flash Attention, DeepSpeed ZeRO, etc.) that automatically
    work everywhere.
    """
    
    # ============================================================================
    # Model and Data
    # ============================================================================
    model: str = "artifacts/hf_implant/base_model"
    """HF model name or local path for tokenizer."""
    
    dataset_file: Optional[str] = None
    """Text/CSV/archive/dir to sample training lines from (REQUIRED)."""
    
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
    
    bundle_dir: str = "artifacts/brains/actv1"
    """Base directory for ACTV1 brain bundles.
    
    Brain-specific subdirectories are created here.
    """
    
    # ============================================================================
    # Model Architecture
    # ============================================================================
    h_layers: int = 2
    """Number of High-level layers in the HRM architecture."""
    
    l_layers: int = 2
    """Number of Low-level layers in the HRM architecture."""
    
    hidden_size: int = 512
    """Model hidden dimension.
    
    Examples:
    - 512 → ~50M params
    - 768 → ~87M params
    - 1024 → ~150M params
    - 1536 → ~230M params
    - 2048 → ~378M params
    """
    
    expansion: float = 2.0
    """FFN (Feed-Forward Network) expansion factor.
    
    Intermediate dimension = hidden_size * expansion.
    Typical values: 2.0 to 4.0.
    """
    
    num_heads: int = 8
    """Number of attention heads.
    
    Must divide hidden_size evenly (e.g., 768/8 = 96 per head).
    """
    
    h_cycles: int = 2
    """High-level recurrent cycles per segment."""
    
    l_cycles: int = 2
    """Low-level recurrent cycles per segment."""
    
    pos_encodings: str = "rope"
    """Position encodings type.
    
    Options:
    - "rope": Rotary Position Embeddings (recommended, handles long contexts)
    - "alibi": ALiBi position bias
    - "none": No position encodings (not recommended)
    """
    
    # ============================================================================
    # Advanced Options
    # ============================================================================
    iterate: bool = False
    """Repeat generation + training cycles indefinitely until stopped.
    
    WARNING: This is an experimental feature for continuous training loops.
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
    
    sys_mem_cap_pct: Optional[int] = None
    """Soft cap for system memory usage percent.
    
    If system RAM usage exceeds this percentage, CPU batch size is auto-reduced.
    Typical values: 70-90. Default: None (no cap).
    """
    
    # ============================================================================
    # Deprecated (Kept for Backward Compatibility)
    # ============================================================================
    kl: float = 0.0
    """[DEPRECATED] KL weight toward teacher distribution.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    kl_temp: float = 1.0
    """[DEPRECATED] Temperature for teacher/student softmax in KL.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    teacher: Optional[str] = None
    """[DEPRECATED] Optional teacher LM name/path.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    teacher_device: str = "cuda"
    """[DEPRECATED] Device for teacher model.
    
    Teacher-student distillation has been removed. This parameter is ignored.
    """
    
    # ============================================================================
    # Methods
    # ============================================================================
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.dataset_file is None:
            raise ValueError("dataset_file is required")
        
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        
        if self.steps < 1:
            raise ValueError("steps must be positive")
        
        if self.zero_stage not in {"none", "zero1", "zero2", "zero3"}:
            raise ValueError(
                f"Invalid zero_stage: {self.zero_stage}. "
                f"Must be one of: none, zero1, zero2, zero3"
            )
        
        if self.pos_encodings not in {"rope", "alibi", "none"}:
            raise ValueError(
                f"Invalid pos_encodings: {self.pos_encodings}. "
                f"Must be one of: rope, alibi, none"
            )
        
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        
        if self.lr <= 0:
            raise ValueError("lr (learning rate) must be positive")
        
        if self.eval_batches < 0:
            raise ValueError("eval_batches must be non-negative (0 to disable)")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create from dictionary.
        
        Args:
            d: Dictionary with configuration parameters.
            
        Returns:
            New TrainingConfig instance.
        """
        return cls(**d)
    
    def to_cli_args(self) -> list[str]:
        """Convert to CLI arguments for subprocess launching.
        
        This is primarily used by the GUI to launch training as a subprocess.
        
        Returns:
            List of CLI arguments like: ["hrm-hf", "train-actv1", "--model", "...", ...]
        """
        args = ["hrm-hf", "train-actv1"]
        
        # Required/primary parameters
        args.extend(["--model", self.model])
        args.extend(["--max-seq-len", str(self.max_seq_len)])
        args.extend(["--batch-size", str(self.batch_size)])
        args.extend(["--steps", str(self.steps)])
        
        # Optional string/path parameters
        if self.dataset_file:
            args.extend(["--dataset-file", self.dataset_file])
        if self.cuda_ids:
            args.extend(["--cuda-ids", self.cuda_ids])
        if self.eval_file:
            args.extend(["--eval-file", self.eval_file])
        if self.stop_file:
            args.extend(["--stop-file", self.stop_file])
        if self.log_file:
            args.extend(["--log-file", self.log_file])
        if self.student_init:
            args.extend(["--student-init", self.student_init])
        if self.brain_name:
            args.extend(["--brain-name", self.brain_name])
        
        # Numeric parameters
        args.extend(["--lr", str(self.lr)])
        args.extend(["--halt-max-steps", str(self.halt_max_steps)])
        args.extend(["--eval-batches", str(self.eval_batches)])
        if self.sys_mem_cap_pct is not None:
            args.extend(["--sys-mem-cap-pct", str(self.sys_mem_cap_pct)])
        
        # Device/distributed
        args.extend(["--device", self.device])
        if self.ddp:
            args.append("--ddp")
        if self.world_size is not None:
            args.extend(["--world-size", str(self.world_size)])
        
        # Boolean flags
        if self.gradient_checkpointing:
            args.append("--gradient-checkpointing")
        else:
            args.append("--no-gradient-checkpointing")
        
        if self.use_amp:
            args.append("--amp")
        else:
            args.append("--no-amp")
        
        if self.use_cpu_offload:
            args.append("--cpu-offload")
        else:
            args.append("--no-cpu-offload")
        
        if self.iterate:
            args.append("--iterate")
        if self.optimize:
            args.append("--optimize")
        if self.strict:
            args.append("--strict")
        if self.ascii_only:
            args.append("--ascii-only")
        
        # DeepSpeed ZeRO
        if self.zero_stage != "none":
            args.extend(["--zero-stage", self.zero_stage])
        
        # Architecture parameters
        args.extend(["--h-layers", str(self.h_layers)])
        args.extend(["--l-layers", str(self.l_layers)])
        args.extend(["--hidden-size", str(self.hidden_size)])
        args.extend(["--expansion", str(self.expansion)])
        args.extend(["--num-heads", str(self.num_heads)])
        args.extend(["--h-cycles", str(self.h_cycles)])
        args.extend(["--l-cycles", str(self.l_cycles)])
        args.extend(["--pos-encodings", self.pos_encodings])
        
        # Directories
        args.extend(["--save-dir", self.save_dir])
        args.extend(["--bundle-dir", self.bundle_dir])
        
        return args
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"TrainingConfig(\n"
            f"  model={self.model!r},\n"
            f"  dataset={self.dataset_file!r},\n"
            f"  seq_len={self.max_seq_len}, batch={self.batch_size}, steps={self.steps},\n"
            f"  arch={self.h_layers}h/{self.l_layers}l, hidden={self.hidden_size}, heads={self.num_heads},\n"
            f"  device={self.device!r}, ddp={self.ddp}, zero={self.zero_stage!r}\n"
            f")"
        )
