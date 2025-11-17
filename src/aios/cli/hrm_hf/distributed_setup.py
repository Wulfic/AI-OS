"""Distributed training setup (DDP and DeepSpeed)."""
from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

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
    
    logger.info(f"Initializing DeepSpeed with ZeRO stage: {zero_stage}")
    
    if not zero_stage or zero_stage == "none":
        logger.debug("DeepSpeed disabled (zero_stage=none)")
        return None, False
    
    dev = str(device_obj).split(':')[0]
    if dev != "cuda":
        logger.warning(f"DeepSpeed skipped - only CUDA supported, got device: {dev}")
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
        
        ds_version = getattr(deepspeed, "__version__", "unknown")
        world_size = _infer_world_size(config)
        logger.info(f"DeepSpeed version: {ds_version} (world_size={world_size})")
        log_mode = "ddp_zero" if world_size > 1 else "single_process_zero"
        log_fn({
            "deepspeed": "import_success",
            "version": ds_version,
            "note": f"Using DeepSpeed in {log_mode} mode"
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
            with open(ds_config_path, 'r', encoding="utf-8") as f:
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
        
        ds_config = _apply_runtime_overrides(
            ds_config=ds_config,
            config=config,
            stage=zero_stage,
            world_size=world_size,
        )
        
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
        
        init_mode = "ddp_zero" if world_size > 1 else "single_gpu_zero"
        log_fn({"deepspeed": "initializing", "mode": init_mode, "stage": zero_stage, "world_size": world_size})
        
        init_kwargs = {
            "model": model,
            "config": ds_config,
        }

        try:
            params_fn = getattr(model, "parameters", None)
            if callable(params_fn):
                init_kwargs["model_parameters"] = params_fn()
        except Exception:
            # Fall back to DeepSpeed's internal parameter handling
            pass

        # Let DeepSpeed handle distributed init if needed (default behavior)
        model_engine, optimizer, _, _ = deepspeed.initialize(**init_kwargs)
        
        log_fn({
            "deepspeed": "initialized",
            "zero_stage": zero_stage,
            "config": ds_config_path if Path(ds_config_path).exists() else "generated",
            "train_batch_size": ds_config["train_batch_size"],
            "fp16": ds_config.get("fp16", {}).get("enabled", False),
            "bf16": ds_config.get("bf16", {}).get("enabled", False),
            "world_size": world_size,
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


def _infer_world_size(config: "TrainingConfig") -> int:
    """Best-effort inference of intended world size."""
    try:
        world_size = getattr(config, "world_size", None)
        if world_size is not None and int(world_size) > 0:
            return int(world_size)
    except Exception:
        pass

    cuda_ids = getattr(config, "cuda_ids", None)
    if cuda_ids:
        try:
            if isinstance(cuda_ids, str):
                entries = [item.strip() for item in cuda_ids.split(",") if item.strip()]
            else:
                entries = [str(item).strip() for item in cuda_ids if str(item).strip()]
            if entries:
                return max(len(entries), 1)
        except Exception:
            pass

    try:
        if torch.distributed.is_initialized():
            return max(torch.distributed.get_world_size(), 1)
    except Exception:
        pass

    return 1


def _apply_runtime_overrides(
    *,
    ds_config: dict,
    config: "TrainingConfig",
    stage: str,
    world_size: int,
) -> dict:
    """Normalize DeepSpeed config values at runtime."""

    def _safe_int(value: Any, default: int) -> int:
        try:
            return max(int(value), 1)
        except Exception:
            return default

    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    gradient_accum_steps = _safe_int(
        getattr(config, "gradient_accumulation_steps", 1),
        default=1,
    )
    micro_batch = _safe_int(getattr(config, "batch_size", 1), default=1)

    steps_candidate = (
        getattr(config, "num_train_steps", None)
        or getattr(config, "steps", None)
        or getattr(config, "max_steps", None)
        or getattr(config, "total_steps", None)
    )
    total_steps = _safe_int(steps_candidate, default=gradient_accum_steps)
    total_steps = max(total_steps, gradient_accum_steps)

    lr = _safe_float(getattr(config, "lr", 1e-4), default=1e-4)
    zero_stage_num = _safe_int(stage.lower().replace("zero", ""), default=1)

    train_batch = micro_batch * gradient_accum_steps * max(world_size, 1)

    ds_config["train_batch_size"] = train_batch
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch
    ds_config["gradient_accumulation_steps"] = gradient_accum_steps

    optimizer_cfg = ds_config.setdefault("optimizer", {})
    optimizer_cfg.setdefault("type", "AdamW")
    optimizer_params = optimizer_cfg.setdefault("params", {})
    optimizer_params["lr"] = lr

    scheduler_cfg = ds_config.get("scheduler")
    if isinstance(scheduler_cfg, dict):
        scheduler_params = scheduler_cfg.setdefault("params", {})
        default_warmup = max(1, min(total_steps, int(max(total_steps * 0.1, 1))))
        scheduler_params["warmup_max_lr"] = _safe_float(
            scheduler_params.get("warmup_max_lr", lr),
            default=lr,
        )
        scheduler_params["warmup_min_lr"] = _safe_float(
            scheduler_params.get("warmup_min_lr", 0.0),
            default=0.0,
        )
        scheduler_params["total_num_steps"] = _safe_int(
            scheduler_params.get("total_num_steps", total_steps),
            default=total_steps,
        )
        scheduler_params["warmup_num_steps"] = _safe_int(
            scheduler_params.get("warmup_num_steps", default_warmup),
            default=default_warmup,
        )

    model_dtype = str(getattr(config, "model_dtype", "fp32")).lower()
    use_bf16 = model_dtype == "bf16"
    use_amp = bool(getattr(config, "use_amp", False))
    use_fp16 = use_amp and not use_bf16

    if use_fp16:
        ds_config.setdefault("fp16", {})["enabled"] = True
    elif "fp16" in ds_config:
        ds_config["fp16"]["enabled"] = False

    if use_bf16:
        ds_config.setdefault("bf16", {})["enabled"] = True
    elif "bf16" in ds_config:
        ds_config["bf16"]["enabled"] = False

    zero_cfg = ds_config.setdefault("zero_optimization", {})
    zero_cfg["stage"] = zero_stage_num

    if zero_stage_num == 3:
        zero_cfg.setdefault("stage3_prefetch_bucket_size", 5e8)
        zero_cfg.setdefault("stage3_param_persistence_threshold", 1e6)
        zero_cfg.setdefault("stage3_max_live_parameters", 1e9)
        zero_cfg.setdefault("stage3_max_reuse_distance", 1e9)
        zero_cfg.setdefault("stage3_gather_16bit_weights_on_model_save", True)

    return ds_config


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
    logger.debug(f"Checking Windows ZeRO multi-GPU compatibility (os.name={os.name})")
    
    if os.name != "nt":
        logger.debug("Not on Windows, no special handling needed")
        return config.ddp, config.world_size
    
    if not config.zero_stage or config.zero_stage == "none":
        logger.debug("ZeRO not enabled, no special handling needed")
        return config.ddp, config.world_size
    
    # Check if multi-GPU is configured
    num_gpus = 0
    if config.cuda_ids:
        try:
            num_gpus = len([x for x in str(config.cuda_ids).split(",") if x.strip()])
        except Exception:
            num_gpus = 0
    
    logger.debug(f"Detected {num_gpus} GPUs configured")
    
    if num_gpus > 1 or (config.ddp and config.world_size and config.world_size > 1):
        # Auto-convert to single GPU
        logger.warning(f"Windows ZeRO multi-GPU incompatibility detected: Converting to single GPU (first device)")
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
            logger.info(f"Set CUDA_VISIBLE_DEVICES={first_gpu} (first GPU only)")
        
        log_fn({
            "note": "To use multi-GPU on Windows, disable ZeRO and use standard DDP"
        })
        
        return False, 1
    
    logger.debug("Windows ZeRO configuration is compatible")
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
        logger.info("Keeping model on CPU for ZeRO-3 initialization")
        log_fn({
            "model_placement": "cpu",
            "reason": "ZeRO-3 will handle GPU placement during initialization"
        })
        return model
    
    # Check if model is already on GPU
    try:
        first_param = next(model.parameters())
        if first_param.device.type == str(device_obj).split(':')[0]:
            logger.debug(f"Model already on {device_obj}")
            return model
    except StopIteration:
        logger.warning("Model has no parameters to check device placement")
        pass
    
    # Move to device
    logger.info(f"Moving model to {device_obj}")
    model.to(device_obj)
    log_fn({
        "model_placement": str(device_obj),
        "reason": "not using ZeRO-3"
    })
    
    return model
