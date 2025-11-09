"""Sparse MoE utilities for replacing FFN layers with MoE layers."""

from __future__ import annotations

from typing import Any


def replace_ffn_with_moe(
    model: Any,
    is_peft: bool,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_capacity_factor: float,
    device: Any,
) -> int:
    """Replace feed-forward layers with sparse MoE layers.
    
    This function identifies FFN/MLP blocks in transformer models and replaces them
    with DynamicMoELayer instances for sparse expert routing.
    
    Args:
        model: The transformer model (may be PEFT-wrapped)
        is_peft: Whether the model is wrapped with PEFT (LoRA)
        hidden_size: Model hidden dimension
        num_experts: Total number of experts per MoE layer
        num_experts_per_tok: Top-k experts to route each token to
        moe_capacity_factor: Capacity factor for load balancing
        device: Target device for MoE layers
        
    Returns:
        Number of FFN layers replaced
    """
    try:
        from aios.core.hrm_models.dynamic_moe import DynamicMoELayer
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError(f"MoE support requires dynamic_moe module: {e}") from e
    
    # Define wrapper class to make MoE compatible with original MLP interface
    class MoEWrapper(nn.Module):
        """Wrapper to make MoE layer compatible with original MLP interface."""
        def __init__(self, moe: DynamicMoELayer):
            super().__init__()
            self.moe = moe
        
        def forward(self, x):
            output, _ = self.moe(x)
            return output
    
    # Get the underlying base model (unwrap PEFT if present)
    base_model = model.base_model if is_peft else model
    
    replaced_count = 0
    
    # Try GPT-2 style (model.transformer.h[i].mlp)
    if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
        for layer_idx, layer in enumerate(base_model.transformer.h):  # type: ignore
            if hasattr(layer, 'mlp'):
                # Get intermediate size from existing MLP
                try:
                    # GPT-2 uses c_fc (input projection) to determine intermediate size
                    if hasattr(layer.mlp, 'c_fc'):
                        intermediate_size = layer.mlp.c_fc.out_features
                    else:
                        # Fallback: 4x hidden size (standard transformer FFN)
                        intermediate_size = hidden_size * 4
                except Exception:
                    intermediate_size = hidden_size * 4
                
                # Create MoE layer
                moe_layer = DynamicMoELayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts_per_tok=num_experts_per_tok,
                    capacity_factor=moe_capacity_factor,
                    lazy_loading=False,  # Keep experts in memory for now
                    device=str(device),
                )
                
                # Add default experts
                for expert_idx in range(num_experts):
                    from aios.core.hrm_models.expert_metadata import create_expert_metadata
                    metadata = create_expert_metadata(
                        expert_id=f"layer{layer_idx}_expert{expert_idx}",
                        name=f"Layer {layer_idx} Expert {expert_idx}",
                        description=f"Auto-created MoE expert for layer {layer_idx}",
                        category="general",
                    )
                    moe_layer.add_expert(f"layer{layer_idx}_expert{expert_idx}", metadata=metadata)
                
                # Convert all experts to match the model's dtype (critical for bfloat16/fp16 models)
                model_dtype = next(base_model.parameters()).dtype
                for expert_id in list(moe_layer.experts.keys()):
                    moe_layer.experts[expert_id] = moe_layer.experts[expert_id].to(dtype=model_dtype)
                
                # Also convert the router to match model dtype
                moe_layer.router = moe_layer.router.to(dtype=model_dtype)
                
                # Replace the MLP with wrapped MoE
                layer.mlp = MoEWrapper(moe_layer)
                replaced_count += 1
    
    # Try LLaMA style (model.model.layers[i].mlp)
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        for layer_idx, layer in enumerate(base_model.model.layers):  # type: ignore
            if hasattr(layer, 'mlp'):
                # Get intermediate size
                try:
                    if hasattr(layer.mlp, 'gate_proj'):
                        intermediate_size = layer.mlp.gate_proj.out_features
                    elif hasattr(layer.mlp, 'up_proj'):
                        intermediate_size = layer.mlp.up_proj.out_features
                    else:
                        intermediate_size = hidden_size * 4
                except Exception:
                    intermediate_size = hidden_size * 4
                
                moe_layer = DynamicMoELayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts_per_tok=num_experts_per_tok,
                    capacity_factor=moe_capacity_factor,
                    lazy_loading=False,
                    device=str(device),
                )
                
                for expert_idx in range(num_experts):
                    from aios.core.hrm_models.expert_metadata import create_expert_metadata
                    metadata = create_expert_metadata(
                        expert_id=f"layer{layer_idx}_expert{expert_idx}",
                        name=f"Layer {layer_idx} Expert {expert_idx}",
                        description=f"Auto-created MoE expert for layer {layer_idx}",
                        category="general",
                    )
                    moe_layer.add_expert(f"layer{layer_idx}_expert{expert_idx}", metadata=metadata)
                
                # Convert all experts to match the model's dtype (critical for bfloat16/fp16 models)
                model_dtype = next(base_model.parameters()).dtype
                for expert_id in list(moe_layer.experts.keys()):
                    moe_layer.experts[expert_id] = moe_layer.experts[expert_id].to(dtype=model_dtype)
                
                # Also convert the router to match model dtype
                moe_layer.router = moe_layer.router.to(dtype=model_dtype)
                
                layer.mlp = MoEWrapper(moe_layer)
                replaced_count += 1
    
    if replaced_count > 0:
        print(f"[HFAdapter] Replaced {replaced_count} FFN layers with sparse MoE ({num_experts} experts, top-{num_experts_per_tok} routing)")
    else:
        print(f"[HFAdapter] Warning: MoE enabled but no FFN layers found to replace in {type(base_model).__name__}")
    
    return replaced_count
