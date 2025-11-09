"""Configuration processing for train_actv1.

Handles config extraction and MoE learning rate adjustment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Dict, Any

from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def extract_and_process_config(config: "TrainingConfig") -> Tuple[Dict[str, Any], float]:
    """Extract and process configuration values.
    
    Args:
        config: Training configuration object
        
    Returns:
        Tuple of (config_dict, adjusted_lr)
    """
    # Extract all configuration values into a dict for easy access
    config_values = {
        "model": config.model,
        "dataset_file": config.dataset_file,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "steps": config.steps,
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
        # Architecture
        "h_layers": config.h_layers,
        "l_layers": config.l_layers,
        "hidden_size": config.hidden_size,
        "expansion": config.expansion,
        "num_heads": config.num_heads,
        "h_cycles": config.h_cycles,
        "l_cycles": config.l_cycles,
        "pos_encodings": config.pos_encodings,
        "window_size": config.window_size,
        # MoE
        "use_moe": config.use_moe,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_capacity_factor": config.moe_capacity_factor,
        # Other flags
        "iterate": config.iterate,
        "linear_dataset": config.linear_dataset,
        "dataset_start_offset": config.dataset_start_offset,
        "stop_after_epoch": config.stop_after_epoch,
    }
    
    # Debug: Log MoE and auto-adjust state
    print({
        "event": "config_check",
        "use_moe": config.use_moe,
        "auto_adjust_moe_lr": config.auto_adjust_moe_lr,
        "lr_before_adjustment": config.lr,
        "num_experts": config.num_experts,
    })
    
    # Apply MoE LR adjustment if enabled
    adjusted_lr = apply_moe_lr_adjustment(
        lr=config.lr,
        use_moe=config.use_moe,
        auto_adjust_moe_lr=config.auto_adjust_moe_lr,
        num_experts=config.num_experts
    )
    
    return config_values, adjusted_lr


def apply_moe_lr_adjustment(
    lr: float,
    use_moe: bool,
    auto_adjust_moe_lr: bool,
    num_experts: int
) -> float:
    """Apply learning rate adjustment for MoE models.
    
    MoE models with router networks are sensitive to high learning rates.
    This function automatically reduces LR when enabled by user.
    
    Args:
        lr: Original learning rate
        use_moe: Whether MoE is enabled
        auto_adjust_moe_lr: Whether to auto-adjust LR for MoE
        num_experts: Number of experts in MoE layer
        
    Returns:
        Adjusted learning rate
    """
    if use_moe and auto_adjust_moe_lr:
        # Tiered approach based on original LR
        if lr >= 1e-4:
            original_lr = lr
            lr = 1e-6  # Ultra-conservative for very high LR
            print({
                "event": "moe_lr_adjustment",
                "message": "Aggressive LR reduction for MoE stability (user enabled auto-adjust)",
                "original_lr": original_lr,
                "adjusted_lr": lr,
                "reason": f"MoE with {num_experts} experts requires ultra-conservative LR (<=1e-6) to prevent loss spikes",
            })
        elif lr >= 5e-5:
            original_lr = lr
            lr = 2e-6  # Very conservative for high LR
            print({
                "event": "moe_lr_adjustment",
                "message": "Strong LR reduction for MoE stability (user enabled auto-adjust)",
                "original_lr": original_lr,
                "adjusted_lr": lr,
                "reason": f"MoE with {num_experts} experts requires very conservative LR (<=2e-6) to prevent gradient explosions",
            })
    else:
        # Respect user's exact LR choice
        try:
            print({
                "learning_rate": lr,
                "auto_adjust": "disabled" if not auto_adjust_moe_lr else "not_applicable",
                "note": "Respecting user's explicit LR choice"
            })
        except Exception:
            pass
    
    return lr
