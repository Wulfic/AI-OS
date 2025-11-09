"""Learning rate auto-adjustment for MoE models."""
from __future__ import annotations


def adjust_moe_learning_rate(config, log_fn) -> float:
    """Auto-adjust learning rate for MoE models if enabled.
    
    Returns:
        Adjusted learning rate
    """
    lr = config.lr
    
    # Debug logging
    log_fn({
        "event": "config_check",
        "use_moe": config.use_moe,
        "auto_adjust_lr": config.auto_adjust_lr,
        "lr_before_adjustment": config.lr,
        "num_experts": config.num_experts,
    })
    
    # Auto-adjust learning rate for MoE models ONLY if user enables the checkbox
    if config.use_moe and config.auto_adjust_lr:
        # MoE models: target 0.001 starting rate with 0.0001 minimum
        if lr >= 0.002:
            original_lr = lr
            lr = 0.001
            log_fn({
                "event": "moe_lr_adjustment",
                "message": "LR adjusted to standard MoE starting rate (user enabled auto-adjust)",
                "original_lr": original_lr,
                "adjusted_lr": lr,
                "reason": f"MoE with {config.num_experts} experts - set to 0.001 (standard starting rate)",
            })
        elif lr < 0.0001:
            original_lr = lr
            lr = 0.0001
            log_fn({
                "event": "moe_lr_adjustment",
                "message": "LR increased to minimum threshold (user enabled auto-adjust)",
                "original_lr": original_lr,
                "adjusted_lr": lr,
                "reason": f"MoE with {config.num_experts} experts - increased to minimum 0.0001",
            })
        # If LR is between 0.0001 and 0.002, use as-is (already in good range)
    else:
        # Respect user's exact LR choice
        try:
            log_fn({
                "learning_rate": lr,
                "auto_adjust": "disabled" if not config.auto_adjust_lr else "not_applicable",
                "note": "Respecting user's explicit LR choice"
            })
        except Exception:
            pass
    
    return lr