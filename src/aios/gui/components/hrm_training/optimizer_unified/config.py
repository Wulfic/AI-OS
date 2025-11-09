"""Configuration dataclass for unified optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, List, Any


@dataclass
class OptimizationConfig:
    """Unified configuration for optimization across all interfaces."""
    
    # Model configuration
    model: str = "base_model"
    teacher_model: str = ""
    max_seq_len: int = 512
    dataset_file: str = "training_data/curated_datasets/test_sample.txt"  # Dataset for optimization
    
    # Optimization parameters
    test_duration: int = 45  # seconds per test - increased for real work
    max_timeout: int = 240   # max subprocess timeout - increased for DDP init (160s) + training (30s) + buffer
    batch_sizes: Optional[List[int]] = None  # Will default to [1, 2, 4, 8]
    min_batch_size: int = 1
    max_batch_size: Optional[int] = None
    batch_growth_factor: float = 2.0
    
    # Training test parameters
    train_steps: int = 10  # Steps to test training throughput
    
    # GPU configuration
    use_multi_gpu: bool = True
    cuda_devices: str = ""  # e.g., "0,1"
    device: str = "auto"
    strict: bool = False
    target_util: Optional[int] = None  # Target GPU utilization (default: 90%)
    util_tolerance: int = 5
    monitor_interval: float = 1.0
    
    # Output configuration
    log_callback: Optional[Callable[[str], None]] = None
    stop_callback: Optional[Callable[[], bool]] = None  # Returns True if stop requested
    output_dir: str = "artifacts/optimization"
    
    def __post_init__(self):
        if not self.batch_sizes:
            self.batch_sizes = [1, 2, 4, 8]
        else:
            normalized: List[int] = []
            for value in self.batch_sizes:
                try:
                    ivalue = int(value)
                    if ivalue > 0:
                        normalized.append(ivalue)
                except Exception:
                    continue
            self.batch_sizes = sorted(set(normalized)) or [1, 2, 4, 8]

        self.min_batch_size = max(1, int(self.min_batch_size or 1))
        self.min_batch_size = max(self.min_batch_size, self.batch_sizes[0])

        if self.max_batch_size is None:
            self.max_batch_size = max(self.batch_sizes)
        else:
            self.max_batch_size = max(int(self.max_batch_size), self.min_batch_size)

        try:
            self.batch_growth_factor = float(self.batch_growth_factor)
        except Exception:
            self.batch_growth_factor = 2.0
        self.batch_growth_factor = max(1.2, self.batch_growth_factor)

        if not self.teacher_model:
            self.teacher_model = self.model

        if self.target_util is not None and self.target_util <= 0:
            self.target_util = None

        try:
            self.util_tolerance = max(0, int(self.util_tolerance))
        except Exception:
            self.util_tolerance = 5

        try:
            self.monitor_interval = float(self.monitor_interval)
        except Exception:
            self.monitor_interval = 1.0
        self.monitor_interval = max(0.5, self.monitor_interval)
