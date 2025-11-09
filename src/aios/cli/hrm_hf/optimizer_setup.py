"""Optimizer and AMP scaler setup utilities."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def create_optimizer(
    model: Any,
    config: "TrainingConfig",
    use_deepspeed_optimizer: bool,
    log_fn
) -> Any:
    """Create optimizer for training.
    
    Args:
        model: The model to optimize
        config: Training configuration
        use_deepspeed_optimizer: Whether DeepSpeed is managing the optimizer
        log_fn: Logging function
        
    Returns:
        Optimizer instance (or None if DeepSpeed manages it)
    """
    if use_deepspeed_optimizer:
        log_fn({
            "optimizer": "DeepSpeed managed",
            "note": "Optimizer created by DeepSpeed with ZeRO optimizations"
        })
        return None
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config.use_8bit_optimizer:
        try:
            from aios.core.hrm_models.memory_optimizations import create_8bit_optimizer
            opt = create_8bit_optimizer(params, lr=config.lr, optimizer_type='adamw')
            log_fn({
                "optimizer": "AdamW8bit",
                "lr": config.lr,
                "note": "Using 8-bit optimizer for memory efficiency"
            })
            return opt
        except ImportError:
            log_fn({
                "optimizer": "AdamW8bit unavailable",
                "fallback": "Using standard AdamW",
                "hint": "Install bitsandbytes for 8-bit optimizer support"
            })
    
    # Standard optimizer
    OptClass = getattr(torch.optim, "AdamW", None) or getattr(torch.optim, "Adam")
    opt = OptClass(params, lr=config.lr)
    log_fn({
        "optimizer": "AdamW",
        "lr": config.lr,
        "parameters": len(params)
    })
    
    return opt


def setup_amp_scaler(
    use_amp: bool,
    dev: str,
    log_fn
) -> Optional[Any]:
    """Setup AMP GradScaler for mixed precision training.
    
    Args:
        use_amp: Whether to use automatic mixed precision
        dev: Device type string
        log_fn: Logging function
        
    Returns:
        GradScaler instance or None
    """
    scaler = None
    
    if use_amp and dev == "cuda" and torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
            log_fn({"amp_enabled": True, "device": "cuda"})
        except Exception as e:
            log_fn({
                "amp_enabled": False,
                "error": str(e),
                "fallback": "FP32 training"
            })
    elif not use_amp:
        log_fn({
            "amp_enabled": False,
            "note": "FP32 training (user choice or AMP not requested)"
        })
    
    return scaler


def setup_inference_manager(
    config: "TrainingConfig",
    model: Any,
    tokenizer: Any,
    log_fn
) -> Optional[Any]:
    """Setup multi-GPU inference manager if configured.
    
    Args:
        config: Training configuration
        model: The training model
        tokenizer: Model tokenizer
        log_fn: Logging function
        
    Returns:
        InferenceManager instance or None
    """
    if not config.inference_device or config.hot_reload_steps <= 0:
        return None
    
    try:
        from .inference_manager import InferenceManager
        
        inference_manager = InferenceManager(
            inference_device=config.inference_device,
            training_device=config.device,
            hot_reload_steps=config.hot_reload_steps,
            model_architecture=model,
            tokenizer=tokenizer,
            log_fn=log_fn,
        )
        
        log_fn({
            "inference_manager": "initialized",
            "inference_device": config.inference_device,
            "training_device": config.device,
            "hot_reload_steps": config.hot_reload_steps,
            "note": "Separate GPU will handle inference during training"
        })
        
        return inference_manager
        
    except Exception as e:
        log_fn({
            "inference_manager": "failed",
            "error": str(e),
            "fallback": "Single GPU mode (training only)"
        })
        return None


def log_optimizer_memory(
    memory_tracker: Any,
    use_8bit_optimizer: bool,
    use_deepspeed_optimizer: bool,
    log_fn
) -> None:
    """Log memory snapshot after optimizer creation.
    
    Args:
        memory_tracker: MemoryTracker instance
        use_8bit_optimizer: Whether using 8-bit optimizer
        use_deepspeed_optimizer: Whether using DeepSpeed optimizer
        log_fn: Logging function
    """
    try:
        memory_tracker.snapshot('optimizer_created', metadata={
            '8bit_optimizer': use_8bit_optimizer,
            'deepspeed_managed': use_deepspeed_optimizer,
        })
    except Exception as e:
        log_fn({
            "memory_snapshot": "failed",
            "stage": "optimizer_created",
            "error": str(e)
        })
