"""Main training configuration class.

Combines all field groups and provides validation/serialization methods.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .base_fields import BaseFields
from .architecture_fields import ArchitectureFields
from .optimization_fields import OptimizationFields
from .distributed_fields import DistributedFields
from .io_fields import IOFields
from .advanced_fields import AdvancedFields


@dataclass
class TrainingConfig(
    BaseFields,
    ArchitectureFields,
    OptimizationFields,
    DistributedFields,
    IOFields,
    AdvancedFields,
):
    """Unified configuration for HRM ActV1 training.
    
    Single source of truth for all training parameters.
    Used by CLI, GUI, and any future interfaces.
    
    This ensures feature parity across interfaces and makes it easy to add
    new optimizations (Flash Attention, DeepSpeed ZeRO, etc.) that automatically
    work everywhere.
    
    Field groups:
    - BaseFields: Model, data, basic optimization
    - ArchitectureFields: Model architecture and MoE settings
    - OptimizationFields: DeepSpeed, quantization, memory
    - DistributedFields: Multi-GPU and device placement
    - IOFields: Checkpointing, logging, paths
    - AdvancedFields: PEFT, experimental features, deprecated
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
        
        # Gradient accumulation
        if self.gradient_accumulation_steps > 1:
            args.extend(["--gradient-accumulation-steps", str(self.gradient_accumulation_steps)])
        
        args.extend(["--steps", str(self.steps)])
        
        # Optional string/path parameters
        if self.dataset_file:
            args.extend(["--dataset-file", self.dataset_file])
        args.extend(["--dataset-chunk-size", str(self.dataset_chunk_size)])
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
        if self.default_goal:
            args.extend(["--default-goal", self.default_goal])
        if self.expert_id:
            args.extend(["--expert-id", self.expert_id])
        
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
        
        if self.use_8bit_optimizer:
            args.append("--use-8bit-optimizer")
        
        if self.use_chunked_training:
            args.append("--use-chunked-training")
            args.extend(["--chunk-size", str(self.chunk_size)])
        
        if self.resume:
            args.append("--resume")
        if self.iterate:
            args.append("--iterate")
        if self.stop_after_epoch:
            args.append("--stop-after-epoch")
        if self.optimize:
            args.append("--optimize")
        if self.strict:
            args.append("--strict")
        if self.ascii_only:
            args.append("--ascii-only")
        
        # Window size for Flash Attention / sliding window
        if self.window_size is not None:
            args.extend(["--window-size", str(self.window_size)])
        
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
        
        # MoE parameters
        if self.use_moe:
            args.append("--use-moe")
        else:
            args.append("--no-moe")
        args.extend(["--num-experts", str(self.num_experts)])
        args.extend(["--num-experts-per-tok", str(self.num_experts_per_tok)])
        args.extend(["--moe-capacity-factor", str(self.moe_capacity_factor)])
        
        # MoE learning rate adjustment
        if self.auto_adjust_lr:
            args.append("--auto-adjust-lr")
        else:
            args.append("--no-auto-adjust-lr")
        
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
