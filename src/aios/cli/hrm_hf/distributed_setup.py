"""Distributed training setup (DDP and DeepSpeed)."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def initialize_deepspeed(
    model: Any,
    config: "TrainingConfig",
    device_obj: torch.device,
    log_fn
) -> Tuple[Optional[Any], bool]:
    """Initialize DeepSpeed ZeRO optimizer.
    
    Args:
        model: The model to wrap
        config: Training configuration
        device_obj: PyTorch device object
        log_fn: Logging function
        
    Returns:
        Tuple of (deepspeed_engine, use_deepspeed_optimizer)
    """
    zero_stage = config.zero_stage
    
    if not zero_stage or zero_stage == "none":
        return None, False
    
    dev = str(device_obj).split(':')[0]
    if dev != "cuda":
        log_fn({
            "deepspeed": "skipped",
            "reason": "Only CUDA devices supported",
            "device": dev
        })
        return None, False
    
    try:
        import deepspeed
        import importlib.util
        
        # Verify deepspeed is actually loadable
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError("deepspeed module not found in path")
        
        log_fn({
            "deepspeed": "import_success",
            "version": getattr(deepspeed, "__version__", "unknown"),
            "note": "Using DeepSpeed without distributed launcher for single-GPU ZeRO"
        })
        
        # Map zero_stage string to DeepSpeed config
        ds_config_map = {
            "zero1": "config/deepspeed_zero1.json",
            "zero2": "config/deepspeed_zero2.json",
            "zero3": "config/deepspeed_zero3.json",
        }
        
        ds_config_path = ds_config_map.get(zero_stage.lower())
        if not ds_config_path:
            log_fn({
                "deepspeed": "error",
                "message": f"Unknown ZeRO stage: {zero_stage}",
                "valid_options": list(ds_config_map.keys())
            })
            return None, False
        
        # Load and modify DeepSpeed config
        if Path(ds_config_path).exists():
            with open(ds_config_path, 'r') as f:
                ds_config = json.load(f)
        else:
            # Create minimal config if file doesn't exist
            stage_num = int(zero_stage.lower().replace("zero", ""))
            # Get gradient accumulation from config, default to 1
            gradient_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
            ds_config = {
                "train_batch_size": config.batch_size * gradient_accum_steps,  # DeepSpeed expects TOTAL batch size
                "gradient_accumulation_steps": gradient_accum_steps,
                "fp16": {"enabled": config.use_amp and config.model_dtype.lower() != "bf16"},
                "bf16": {"enabled": config.model_dtype.lower() == "bf16"},
                "zero_optimization": {"stage": stage_num},
            }
        
        # Update with current training config
        gradient_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
        ds_config["train_batch_size"] = config.batch_size * gradient_accum_steps  # Total effective batch size
        ds_config["gradient_accumulation_steps"] = gradient_accum_steps
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {"lr": config.lr}
        }
        
        # Initialize DeepSpeed
        # For single-GPU ZeRO-3, we need torch.distributed even with 1 process
        stage_num = int(zero_stage.lower().replace("zero", ""))
        needs_distributed = stage_num == 3
        
        if needs_distributed and not torch.distributed.is_initialized():
            # Initialize single-process distributed for ZeRO-3
            log_fn({
                "deepspeed": "initializing_distributed",
                "mode": "single_gpu_zero3",
                "reason": "ZeRO-3 requires torch.distributed even for single GPU"
            })
            
            # Set environment variables for single-process distributed
            if "RANK" not in os.environ:
                os.environ["RANK"] = "0"
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = "1"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
            
            try:
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=1,
                    rank=0
                )
                log_fn({
                    "torch_distributed": "initialized",
                    "backend": "nccl",
                    "world_size": 1,
                    "rank": 0
                })
            except Exception as e:
                log_fn({
                    "torch_distributed": "init_failed",
                    "error": str(e),
                    "fallback": "DeepSpeed will try to initialize itself"
                })
        
        log_fn({"deepspeed": "initializing", "mode": "single_gpu_zero", "stage": zero_stage})
        
        # Let DeepSpeed handle distributed init if needed (default behavior)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        
        log_fn({
            "deepspeed": "initialized",
            "zero_stage": zero_stage,
            "config": ds_config_path if Path(ds_config_path).exists() else "generated",
            "train_batch_size": ds_config["train_batch_size"],
            "fp16": ds_config.get("fp16", {}).get("enabled", False),
            "bf16": ds_config.get("bf16", {}).get("enabled", False),
        })
        
        return model_engine, True
        
    except ImportError as e:
        import traceback
        log_fn({
            "deepspeed": "import_error",
            "message": f"DeepSpeed not available: {e}",
            "traceback": traceback.format_exc()[:500],
            "solution": "Install with: pip install deepspeed",
            "fallback": "Training without ZeRO optimization"
        })
        return None, False
    except Exception as e:
        import traceback
        log_fn({
            "deepspeed": "initialization_error",
            "message": f"Failed to initialize DeepSpeed: {e}",
            "traceback": traceback.format_exc()[:500],
            "fallback": "Training without ZeRO optimization"
        })
        return None, False


def wrap_with_ddp(
    model: Any,
    device_obj: torch.device,
    is_distributed: bool,
    rank_id: int,
    log_fn
) -> Any:
    """Wrap model with DistributedDataParallel.
    
    Args:
        model: The model to wrap
        device_obj: PyTorch device object
        is_distributed: Whether distributed training is active
        rank_id: Process rank ID
        log_fn: Logging function
        
    Returns:
        Wrapped model (or original if not using DDP)
    """
    if not is_distributed:
        return model
    
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Get device ID for DDP
        device_ids = None
        output_device = None
        
        if str(device_obj).startswith("cuda"):
            device_id = int(str(device_obj).split(":")[-1]) if ":" in str(device_obj) else rank_id
            device_ids = [device_id]
            output_device = device_id
        
        ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=True,  # Needed for HRM's dynamic architecture
        )
        
        log_fn({
            "ddp": "initialized",
            "rank": rank_id,
            "device": str(device_obj),
            "device_ids": device_ids,
        })
        
        return ddp_model
        
    except Exception as e:
        log_fn({
            "ddp": "error",
            "message": f"Failed to wrap with DDP: {e}",
            "fallback": "Training without DDP"
        })
        return model


def handle_windows_zero_multi_gpu(
    config: "TrainingConfig",
    log_fn
) -> Tuple[bool, Optional[int]]:
    """Handle Windows-specific ZeRO + multi-GPU incompatibility.
    
    Windows doesn't support ZeRO with multi-GPU due to gloo backend limitations.
    Auto-convert to single GPU when detected.
    
    Args:
        config: Training configuration
        log_fn: Logging function
        
    Returns:
        Tuple of (ddp_enabled, world_size)
    """
    if os.name != "nt":
        return config.ddp, config.world_size
    
    if not config.zero_stage or config.zero_stage == "none":
        return config.ddp, config.world_size
    
    # Check if multi-GPU is configured
    num_gpus = 0
    if config.cuda_ids:
        try:
            num_gpus = len([x for x in str(config.cuda_ids).split(",") if x.strip()])
        except Exception:
            num_gpus = 0
    
    if num_gpus > 1 or (config.ddp and config.world_size and config.world_size > 1):
        # Auto-convert to single GPU
        log_fn({
            "windows_zero_multi_gpu_detected": True,
            "action": "auto_convert_to_single_gpu",
            "reason": "Windows does not support ZeRO with multi-GPU (gloo backend limitations)",
            "previous_config": {
                "cuda_ids": config.cuda_ids,
                "world_size": config.world_size,
                "ddp": config.ddp
            },
            "new_config": "single GPU (first device)"
        })
        
        # Keep only first GPU
        if config.cuda_ids:
            first_gpu = str(config.cuda_ids).split(",")[0].strip()
            os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
        
        log_fn({
            "note": "To use multi-GPU on Windows, disable ZeRO and use standard DDP"
        })
        
        return False, 1
    
    return config.ddp, config.world_size


def ensure_model_on_device(
    model: Any,
    device_obj: torch.device,
    zero_stage: Optional[str],
    log_fn
) -> Any:
    """Ensure model is on the correct device.
    
    ZeRO-3 needs model on CPU initially; others need GPU.
    
    Args:
        model: The model
        device_obj: Target device
        zero_stage: DeepSpeed ZeRO stage
        log_fn: Logging function
        
    Returns:
        Model (possibly moved to device)
    """
    will_use_zero3 = (
        zero_stage and 
        zero_stage.lower() == "zero3" and
        str(device_obj).startswith("cuda")
    )
    
    if will_use_zero3:
        log_fn({
            "model_placement": "cpu",
            "reason": "ZeRO-3 will handle GPU placement during initialization"
        })
        return model
    
    # Check if model is already on GPU
    try:
        first_param = next(model.parameters())
        if first_param.device.type == str(device_obj).split(':')[0]:
            return model
    except StopIteration:
        pass
    
    # Move to device
    model.to(device_obj)
    log_fn({
        "model_placement": str(device_obj),
        "reason": "not using ZeRO-3"
    })
    
    return model
