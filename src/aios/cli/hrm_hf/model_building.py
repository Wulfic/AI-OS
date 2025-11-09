"""Model configuration and building utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

from .model import build_student as _build_student, build_actv1_config as _build_cfg


def calculate_vocab_size(tokenizer: Any, log_fn) -> int:
    """Calculate actual vocab size including special tokens.
    
    Some tokenizers have special tokens with IDs >= vocab_size, so we need
    to extend the vocab_size to accommodate them.
    
    Args:
        tokenizer: The tokenizer
        log_fn: Logging function
        
    Returns:
        Adjusted vocabulary size
    """
    vocab_size = int(getattr(tokenizer, "vocab_size", 50257) or 50257)
    
    # Check if tokenizer has special tokens beyond vocab_size
    if hasattr(tokenizer, 'all_special_ids') and tokenizer.all_special_ids:
        max_special_id = max(tokenizer.all_special_ids)
        if max_special_id >= vocab_size:
            # Extend vocab_size to accommodate special tokens
            actual_vocab_size = max_special_id + 1
            log_fn({
                "vocab_size_adjustment": {
                    "original_vocab_size": vocab_size,
                    "max_special_token_id": max_special_id,
                    "adjusted_vocab_size": actual_vocab_size,
                    "reason": "Special tokens extend beyond base vocabulary"
                }
            })
            return actual_vocab_size
    
    return vocab_size


def build_model_config(
    config: "TrainingConfig",
    vocab_size: int,
    log_fn
) -> dict:
    """Build ACT V1 model configuration.
    
    Args:
        config: Training configuration
        vocab_size: Vocabulary size
        log_fn: Logging function
        
    Returns:
        Model configuration dictionary
    """
    cfg = _build_cfg(
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        vocab_size=vocab_size,
        h_cycles=config.h_cycles,
        l_cycles=config.l_cycles,
        h_layers=config.h_layers,
        l_layers=config.l_layers,
        hidden_size=config.hidden_size,
        expansion=config.expansion,
        num_heads=config.num_heads,
        pos_encodings=config.pos_encodings,
        halt_max_steps=config.halt_max_steps,
        use_flash_attn=config.use_flash_attn,
        use_gradient_checkpointing=config.gradient_checkpointing,
        window_size=config.window_size,
        use_moe=config.use_moe,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_capacity_factor=config.moe_capacity_factor,
    )
    
    try:
        log_fn({
            "arch": {
                "H_layers": int(cfg["H_layers"]),
                "L_layers": int(cfg["L_layers"]),
                "hidden_size": int(cfg["hidden_size"]),
                "num_heads": int(cfg["num_heads"]),
                "expansion": float(cfg["expansion"]),
            }
        })
    except Exception:
        pass
    
    return cfg


def build_model(
    config: dict,
    student_init: str | None,
    log_fn
) -> Any:
    """Build the student model.
    
    Args:
        config: Model configuration dictionary
        student_init: Optional checkpoint path for initialization
        log_fn: Logging function
        
    Returns:
        Built model
    """
    return _build_student(config, student_init=student_init, print_fn=log_fn)


def count_model_parameters(model: Any) -> int:
    """Count total model parameters.
    
    Args:
        model: The model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def calculate_actv1_params(
    vocab_size: int,
    hidden_size: int,
    h_layers: int,
    l_layers: int,
    expansion: float = 2.0,
    use_moe: bool = False,
    num_experts: int = 1
) -> int:
    """Calculate ACTV1 model parameters from architecture.
    
    This is the single source of truth for parameter calculations.
    All parameter estimates (GUI, brain.json, training logs) should use this function.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        h_layers: Number of H (heavy) layers
        l_layers: Number of L (light) layers
        expansion: FFN expansion factor (default: 2.0)
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts (if MoE enabled)
        
    Returns:
        Total parameter count
        
    Formula:
        Embeddings: vocab_size * hidden_size
        Per-layer:
            - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
            - FFN: 
                - Dense: hidden_size * ffn_hidden + ffn_hidden * hidden_size
                - MoE: num_experts * (hidden_size * ffn_hidden + ffn_hidden * hidden_size) + router
            - LayerNorm: 2 * hidden_size * 2
        Output: hidden_size * vocab_size
    """
    # Embedding layer
    embed_params = vocab_size * hidden_size
    
    # Per-layer calculations
    attn_params_per_layer = 4 * hidden_size * hidden_size
    ffn_hidden = int(hidden_size * expansion)
    
    if use_moe:
        # MoE: each expert is a full FFN, plus router (hidden_size * num_experts)
        ffn_params_per_layer = num_experts * (
            hidden_size * ffn_hidden + ffn_hidden * hidden_size
        ) + hidden_size * num_experts
    else:
        # Dense FFN
        ffn_params_per_layer = hidden_size * ffn_hidden + ffn_hidden * hidden_size
    
    # LayerNorm: 2 params (scale, shift) per hidden dim, 2 LNs per layer
    ln_params_per_layer = 2 * hidden_size * 2
    
    # Total layers
    total_layers = h_layers + l_layers
    layer_params = total_layers * (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer)
    
    # Output projection
    output_params = hidden_size * vocab_size
    
    # Total
    total_params = embed_params + layer_params + output_params
    
    return int(total_params)


def print_extreme_scale_recommendations(
    max_seq_len: int,
    total_params: int,
    log_fn
) -> None:
    """Print recommendations for extreme-scale training.
    
    Args:
        max_seq_len: Maximum sequence length
        total_params: Total model parameters
        log_fn: Logging function
    """
    if max_seq_len < 50_000 and total_params < 200_000_000:
        return
    
    try:
        from aios.core.hrm_models.extreme_scale_optimizations import (
            print_extreme_scale_recommendations as _print_recs
        )
        # Note: print_extreme_scale_recommendations doesn't have print_fn parameter
        # It prints directly, so we just call it
        _print_recs(
            model_params=total_params,
            seq_len=max_seq_len,
            available_vram_gb=16.0  # Default assumption, could be improved
        )
    except Exception as e:
        log_fn({
            "extreme_scale_recommendations": "unavailable",
            "error": str(e)
        })
