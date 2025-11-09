"""Training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Configuration for the Trainer."""
    use_torch: bool = False  # default to numpy fallback for portability
    batch_size: int = 32
    lr: float = 1e-2
    hidden: int = 16
    input_dim: int = 8
    output_dim: int = 1
    max_steps: int = 100
    # Minimal CMDP knobs
    cost_coef: float = 0.1  # scales computed cost
    cost_budget: float = float("inf")  # total allowed cost; inf disables budget
    # Torch/GPU controls
    device: str = "auto"  # 'auto' | 'cpu' | 'cuda' | 'mps'
    amp: bool = True  # use autocast+GradScaler when CUDA available
    num_threads: int = 0  # torch CPU threads (0=auto)
    data_parallel: bool = True  # enable DataParallel when multiple CUDA devices
    cuda_devices: Optional[list[int]] = None  # specific CUDA device IDs, else auto
    # Distributed training
    ddp: bool = False  # wrap model in DistributedDataParallel (expects process group initialized)
    ddp_backend: Optional[str] = None  # backend override (auto if None)
    # Dynamic width controls (neuron growth/shrink)
    dynamic_width: bool = False
    width_min: int = 8
    width_max: int = 1024
    grow_patience: int = 200  # steps in window before considering growth
    shrink_patience: int = 400  # steps in window before considering shrink
    grow_factor: float = 2.0
    shrink_factor: float = 1.5
    grow_threshold: float = 1e-4  # minimum improvement to avoid growth
    # Storage guard for growth (per-model). If set, avoid growing beyond this memory budget.
    # Units: megabytes (MB). Uses a simple FP32 param-size estimate for the numpy MLP (safe upper-bound).
    width_storage_limit_mb: Optional[float] = None
    # Sleep/consolidation controls
    sleep_downscale: float = 0.01  # multiplicative downscale applied to weights during sleep
    sleep_consolidation_steps: int = 50  # optional replay/consolidation steps during sleep
