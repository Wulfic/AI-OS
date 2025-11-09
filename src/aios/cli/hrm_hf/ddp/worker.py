from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aios.core.hrm_training import TrainingConfig


def ddp_worker(rank: int, world_size: int, init_file: str, config_dict: dict) -> None:
    """Spawn target for multi-process distributed training.

    Uses minimal imports to avoid Windows spawn overhead.
    """
    # Import the minimal worker entry point
    from .worker_main import main as _worker_main
    _worker_main(rank, world_size, init_file, config_dict)


def serialize_config_for_spawn(config: "TrainingConfig") -> dict:
    """Serialize TrainingConfig for DDP spawn workers.
    
    Args:
        config: The training configuration to serialize
        
    Returns:
        Dictionary representation suitable for pickling and spawning
    """
    return config.to_dict()

