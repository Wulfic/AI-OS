"""Command and environment building utilities for optimizer."""

from __future__ import annotations

import os
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OptimizationConfig


def build_command_env(
    config: "OptimizationConfig",
    gpu_config: Dict[str, Any],
    session_id: str
) -> Dict[str, str]:
    """Create environment variables for optimizer subprocess.
    
    Args:
        config: Optimization configuration
        gpu_config: GPU configuration dict with ids, target_util, multi_gpu
        session_id: Session identifier
        
    Returns:
        Environment variables dictionary
    """
    env = os.environ.copy()

    ids = gpu_config.get("ids", [])
    if ids:
        joined = ",".join(ids)
        env["CUDA_VISIBLE_DEVICES"] = joined
        env["AIOS_CUDA_IDS"] = joined
        
    if gpu_config.get("multi_gpu"):
        env.setdefault("AIOS_WORLD_SIZE", str(len(ids)))
        backend = "gloo" if os.name == "nt" else "nccl"
        env.setdefault("AIOS_DDP_BACKEND", backend)
        env.setdefault("AIOS_DDP_OPTIMIZE_MODE", "1")
        # Enable internal DDP spawn for GUI/optimizer compatibility
        env.setdefault("AIOS_DDP_SPAWN", "1")

    target = gpu_config.get("target_util")
    if target:
        env.setdefault("AIOS_GPU_UTIL_TARGET", str(target))

    env["AIOS_OPT_SESSION"] = session_id
    env["AIOS_OPT_STRICT"] = "1" if config.strict else "0"
    
    # Speed up transformers import in DDP workers (Windows optimization)
    # Disable slow import structure scanning which can take 60-120s per worker
    env.setdefault("TRANSFORMERS_OFFLINE", "1")  # Skip online checks
    env.setdefault("HF_DATASETS_OFFLINE", "1")   # Skip dataset checks
    
    return env


def extend_with_device_args(
    cmd: List[str],
    config: "OptimizationConfig",
    gpu_config: Dict[str, Any]
) -> None:
    """Add device- and GPU-related CLI arguments in-place.
    
    Args:
        cmd: Command list to modify in-place
        config: Optimization configuration
        gpu_config: GPU configuration dict with ids, multi_gpu
    """
    ids = gpu_config.get("ids", [])

    if ids and "--cuda-ids" not in cmd:
        cmd.extend(["--cuda-ids", ",".join(ids)])
        if gpu_config.get("multi_gpu"):
            if "--ddp" not in cmd:
                cmd.append("--ddp")
            if "--world-size" not in cmd:
                cmd.extend(["--world-size", str(len(ids))])

    device = (config.device or "auto").strip()
    if device and device != "auto" and "--device" not in cmd:
        cmd.extend(["--device", device])

    if config.strict and "--strict" not in cmd:
        cmd.append("--strict")


def parse_cuda_devices(raw_devices: Any) -> List[str]:
    """Return sanitized list of requested CUDA device identifiers.
    
    Args:
        raw_devices: Raw device specification (str, list, tuple, set, or None)
        
    Returns:
        List of device identifier strings
    """
    devices: List[str] = []

    if isinstance(raw_devices, (list, tuple, set)):
        iterable = raw_devices
    elif isinstance(raw_devices, str):
        iterable = raw_devices.split(",")
    elif raw_devices is None:
        iterable = []
    else:
        iterable = [raw_devices]

    for item in iterable:
        if item is None:
            continue
        token = str(item).strip()
        if token:
            devices.append(token)

    return devices
