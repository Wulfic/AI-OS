"""Configuration validation and processing."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def extract_and_log_config(config: "TrainingConfig", log_fn) -> dict:
    """Extract configuration values and log MoE/auto-adjust state.
    
    Args:
        config: Training configuration
        log_fn: Logging function
        
    Returns:
        Dictionary of extracted config values
    """
    log_fn({
        "event": "config_check",
        "use_moe": config.use_moe,
        "auto_adjust_lr": config.auto_adjust_lr,
        "lr_before_adjustment": config.lr,
        "num_experts": config.num_experts,
    })
    
    return {
        "model": config.model,
        "dataset_file": config.dataset_file,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "steps": config.steps,
        "lr": config.lr,
        "device": config.device,
        "halt_max_steps": config.halt_max_steps,
        "save_dir": config.save_dir,
        "kl": config.kl,
        "kl_temp": config.kl_temp,
        "ascii_only": config.ascii_only,
        "eval_file": config.eval_file,
        "eval_batches": config.eval_batches,
        "sys_mem_cap_pct": config.sys_mem_cap_pct,
        "stop_file": config.stop_file,
        "log_file": config.log_file,
        "student_init": config.student_init,
        "brain_name": config.brain_name,
        "bundle_dir": config.bundle_dir,
    }


def auto_adjust_moe_learning_rate(config: "TrainingConfig", log_fn) -> float:
    """Auto-adjust learning rate based on model configuration.
    
    Applies intelligent learning rate adjustments for different model types:
    - MoE models: Conservative rates for router network stability
    - Large models: Slightly reduced for training stability
    - Standard models: Uses configured rate as-is
    
    Args:
        config: Training configuration
        log_fn: Logging function
        
    Returns:
        Adjusted learning rate
    """
    lr = config.lr
    
    # Check if auto-adjust is enabled (applies to all models)
    auto_adjust_enabled = getattr(config, 'auto_adjust_lr', True)
    
    if not auto_adjust_enabled:
        # Respect user's exact LR choice
        log_fn({
            "learning_rate": lr,
            "auto_adjust": "disabled",
            "note": "Using user-specified learning rate without adjustments"
        })
        return lr
    
    original_lr = lr
    adjustment_made = False
    adjustment_reason = ""
    
    # MoE models need adjusted learning rates for router stability
    if config.use_moe:
        if lr >= 0.002:
            lr = 0.001  # Start at 0.001 for high initial LR
            adjustment_reason = f"MoE with {config.num_experts} experts - set to 0.001 (standard starting rate)"
            adjustment_made = True
        elif lr < 0.0001:
            lr = 0.0001  # Minimum threshold
            adjustment_reason = f"MoE with {config.num_experts} experts - increased to minimum 0.0001"
            adjustment_made = True
        # If LR is between 0.0001 and 0.002, use as-is (already in good range)
    
    if adjustment_made:
        log_fn({
            "event": "lr_auto_adjustment",
            "message": "Learning rate adjusted for model stability",
            "original_lr": original_lr,
            "adjusted_lr": lr,
            "reason": adjustment_reason,
            "model_type": "MoE" if config.use_moe else "standard",
        })
    else:
        log_fn({
            "learning_rate": lr,
            "auto_adjust": "enabled",
            "note": "No adjustment needed - learning rate is appropriate for model configuration"
        })
    
    return lr


def setup_cuda_devices(cuda_ids: Optional[str], log_fn) -> None:
    """Setup CUDA visible devices if specified.
    
    Args:
        cuda_ids: Comma-separated CUDA device IDs
        log_fn: Logging function
    """
    if not cuda_ids:
        return
    
    try:
        device_list = [
            str(int(x)) for x in str(cuda_ids).split(",") if str(x).strip() != ""
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)
        log_fn({
            "cuda_devices": device_list,
            "note": "CUDA_VISIBLE_DEVICES set"
        })
    except Exception as e:
        log_fn({
            "cuda_devices": "error",
            "cuda_ids": cuda_ids,
            "error": str(e)
        })


def validate_dependencies(log_fn) -> bool:
    """Validate required dependencies are available.
    
    Args:
        log_fn: Logging function
        
    Returns:
        True if all dependencies available, False otherwise
    """
    try:
        import warnings as _warnings
        from pathlib import Path
        import torch
        
        # Suppress deprecation warnings BEFORE importing transformers
        _warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)
        _warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Suppress noisy deprecation warnings from Transformers
        try:
            from transformers.utils import logging as _hf_logging
            _hf_logging.set_verbosity_error()
        except Exception:
            pass
        
        from aios.core.hrm_models import build_act_v1
        from aios.core.hrm_models.auto_chunking import auto_chunked_segment_rollout
        
        return True
        
    except Exception as e:
        log_fn({
            "started": False,
            "error": f"Missing deps: {e}",
            "hint": "pip install -e .[hf]"
        })
        return False


def setup_output_directory(
    brain_name: Optional[str],
    bundle_dir: str,
    log_file: Optional[str],
    save_dir: str,
    log_fn
) -> tuple[Optional[Any], str, Optional[str]]:
    """Setup output/bundle directory based on brain_name.
    
    Args:
        brain_name: Brain name
        bundle_dir: Bundle directory path
        log_file: Log file path
        save_dir: Save directory path
        log_fn: Logging function
        
    Returns:
        Tuple of (out_dir_path, save_dir, log_file)
    """
    from pathlib import Path
    
    out_dir_path = None
    
    log_fn({
        "setup_output_directory": "called",
        "brain_name": brain_name,
        "bundle_dir": bundle_dir,
        "save_dir_input": save_dir,
        "log_file_input": log_file
    })
    
    if brain_name:
        try:
            out_dir_path = Path(bundle_dir) / str(brain_name)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Default log file to bundle if not explicitly provided
            if not log_file:
                log_file = str(out_dir_path / "metrics.jsonl")
            
            # Always prefer saving artifacts into the bundle dir
            save_dir = str(out_dir_path)
            
            log_fn({
                "output_directory": str(out_dir_path),
                "log_file": log_file,
                "save_dir_output": save_dir,
                "note": "Brain artifacts will be saved to bundle directory"
            })
        except Exception as e:
            log_fn({
                "output_directory": "error",
                "error": str(e),
                "fallback": save_dir
            })
            out_dir_path = None
    
    return out_dir_path, save_dir, log_file
