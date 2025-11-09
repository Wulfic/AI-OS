"""Parameter estimation for HRM student models.

Calculates total and active parameter counts from architecture configuration,
including sparse MoE overhead.
"""

from __future__ import annotations
from typing import Tuple, Dict

# Import centralized parameter calculation (single source of truth)
from aios.cli.hrm_hf.model_building import calculate_actv1_params


def estimate_parameters(
    hidden_size: int,
    h_layers: int,
    l_layers: int,
    num_heads: int,
    expansion: float,
    vocab_size: int = 50257,
    use_moe: bool = False,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
) -> Tuple[int, int, str, str]:
    """
    Estimate model parameters from architecture configuration.
    
    Args:
        hidden_size: Hidden dimension size
        h_layers: Number of H (hierarchical) layers
        l_layers: Number of L (local) layers
        num_heads: Number of attention heads
        expansion: FFN expansion factor
        vocab_size: Tokenizer vocabulary size
        use_moe: Whether using sparse MoE
        num_experts: Number of MoE experts (if MoE enabled)
        num_experts_per_tok: Active experts per token (if MoE enabled)
    
    Returns:
        Tuple of (total_params, active_params, param_text, breakdown_text)
        - total_params: Total parameter count (including all MoE experts)
        - active_params: Active parameters per forward pass (MoE sparsity)
        - param_text: Human-readable summary (e.g., "Total: 10.5M params...")
        - breakdown_text: Detailed breakdown by component
    """
    
    # ===== TOTAL PARAMETERS (use centralized function) =====
    # This is the single source of truth for total parameter calculations
    total_params = calculate_actv1_params(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        h_layers=h_layers,
        l_layers=l_layers,
        expansion=expansion,
        use_moe=use_moe,
        num_experts=num_experts
    )
    
    # ===== COMPONENT SIZES (for breakdown display) =====
    embed_params = vocab_size * hidden_size
    attn_per_layer = 4 * hidden_size * hidden_size
    ffn_hidden_size = int(hidden_size * expansion)
    ffn_params = 2 * hidden_size * ffn_hidden_size
    total_layers = h_layers + l_layers
    
    # ===== ACTIVE PARAMETERS (MoE SPARSITY) =====
    if use_moe:
        # Only num_experts_per_tok experts are active per token
        # Active FFN params per layer: ffn_params × num_experts_per_tok
        active_ffn_per_layer = ffn_params * num_experts_per_tok
        
        # Router params
        router_params_per_layer = hidden_size * num_experts
        router_total_params = router_params_per_layer * total_layers
        
        # Active total = embeddings + attention + active FFN + router
        active_params = (
            embed_params +
            total_layers * attn_per_layer +
            total_layers * active_ffn_per_layer +
            router_total_params
        )
        
        # Compute reduction percentage
        compute_reduction = (1 - active_params / total_params) * 100
        
        # ===== TEXT SUMMARIES =====
        param_text = (
            f"Total: {total_params/1e6:.1f}M params "
            f"({active_params/1e6:.1f}M active, ~{compute_reduction:.0f}% reduction)"
        )
        
        breakdown_text = (
            f"Embeddings: {embed_params/1e6:.1f}M  •  "
            f"Attention: {total_layers * attn_per_layer/1e6:.1f}M  •  "
            f"MoE Experts: {(ffn_params * num_experts * total_layers)/1e6:.1f}M ({num_experts} experts)\n"
            f"Active per forward: {active_params/1e6:.1f}M ({num_experts_per_tok} experts/token)"
        )
        
    else:
        # Dense model (no MoE) - all params active
        active_params = total_params
        
        param_text = f"Total: {total_params/1e6:.1f}M params (dense model, no sparsity)"
        
        breakdown_text = (
            f"Embeddings: {embed_params/1e6:.1f}M  •  "
            f"Attention: {total_layers * attn_per_layer/1e6:.1f}M  •  "
            f"FFN: {total_layers * ffn_params/1e6:.1f}M"
        )
    
    return total_params, active_params, param_text, breakdown_text


def estimate_from_preset(preset: str) -> Tuple[int, int, str, str]:
    """
    Estimate parameters from a named preset (1M, 5M, 10M, 20M, 50M).
    
    Args:
        preset: Preset name (e.g., "5M", "10M")
    
    Returns:
        Same as estimate_parameters()
    """
    # Preset configurations (matching panel preset logic)
    presets = {
        "1M": {
            "hidden_size": 128,
            "h_layers": 2,
            "l_layers": 2,
            "num_heads": 8,
            "expansion": 2.0,
            "use_moe": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "5M": {
            "hidden_size": 256,
            "h_layers": 2,
            "l_layers": 2,
            "num_heads": 8,
            "expansion": 2.0,
            "use_moe": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "10M": {
            "hidden_size": 384,
            "h_layers": 2,
            "l_layers": 2,
            "num_heads": 8,
            "expansion": 2.0,
            "use_moe": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "20M": {
            "hidden_size": 512,
            "h_layers": 2,
            "l_layers": 2,
            "num_heads": 8,
            "expansion": 2.0,
            "use_moe": True,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "50M": {
            "hidden_size": 768,
            "h_layers": 2,
            "l_layers": 2,
            "num_heads": 12,
            "expansion": 2.0,
            "use_moe": True,
            "num_experts": 12,
            "num_experts_per_tok": 2,
        },
    }
    
    config = presets.get(preset)
    if not config:
        raise ValueError(f"Unknown preset: {preset}")
    
    return estimate_parameters(**config)
