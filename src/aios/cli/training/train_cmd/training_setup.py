"""
Training setup and configuration.

Creates training configuration and handles GPU/checkpoint setup.
"""

from pathlib import Path
from typing import Optional

from aios.core.train import Trainer, TrainConfig


def create_train_config(
    use_torch: bool,
    steps: int,
    batch_size: int,
    cost_coef: float,
    device: str,
    amp: bool,
    num_threads: int,
    data_parallel: bool,
    ddp: bool,
    dynamic_width: bool,
    width_min: int,
    width_max: int,
    grow_patience: int,
    shrink_patience: int,
    grow_factor: float,
    shrink_factor: float,
    grow_threshold: float,
    sleep_downscale: float,
    sleep_steps: int,
    feature_dim: Optional[int] = None,
) -> TrainConfig:
    """Create training configuration."""
    tcfg = TrainConfig(
        use_torch=use_torch,
        max_steps=steps,
        batch_size=batch_size,
        cost_coef=cost_coef,
        device=device,
        amp=amp,
        num_threads=num_threads,
        data_parallel=data_parallel,
        ddp=ddp,
        dynamic_width=dynamic_width,
        width_min=width_min,
        width_max=width_max,
        grow_patience=grow_patience,
        shrink_patience=shrink_patience,
        grow_factor=grow_factor,
        shrink_factor=shrink_factor,
        grow_threshold=grow_threshold,
        sleep_downscale=sleep_downscale,
        sleep_consolidation_steps=sleep_steps,
    )
    
    if feature_dim and feature_dim > 0:
        tcfg.input_dim = int(feature_dim)
    
    return tcfg


def setup_gpu_memory(use_torch: bool, gpu_mem_frac: float):
    """Setup GPU memory fraction for CUDA."""
    if not use_torch:
        return
    
    try:
        import torch  # type: ignore
        frac = float(max(0.1, min(0.99, float(gpu_mem_frac))))
        if torch.cuda.is_available() and frac < 0.995:
            try:
                ndev = int(torch.cuda.device_count())
                for d in range(ndev):
                    try:
                        torch.cuda.set_per_process_memory_fraction(frac, device=d)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass


def setup_cuda_devices(tcfg: TrainConfig, cuda_devices: Optional[str]):
    """Configure CUDA device IDs."""
    if cuda_devices:
        try:
            tcfg.cuda_devices = [int(x.strip()) for x in str(cuda_devices).split(",") if x.strip() != ""]
        except Exception:
            tcfg.cuda_devices = None


def setup_cost_budget(tcfg: TrainConfig, cost_budget: Optional[float]):
    """Configure cost budget."""
    if cost_budget is not None:
        tcfg.cost_budget = float(cost_budget)


def load_checkpoint_if_provided(trainer: Trainer, load_ckpt: Optional[str]):
    """Load checkpoint if path provided."""
    if not load_ckpt:
        return
    
    try:
        ok = trainer.load_checkpoint(load_ckpt)
        if not ok:
            import logging
            logging.getLogger(__name__).warning("failed to load checkpoint: %s", load_ckpt)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("error loading checkpoint: %s", load_ckpt)
