"""Mixture of Experts (MoE) layer implementation for AI-OS.

This module provides the core MoE architecture that will be extended
for dynamic subbrains functionality.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Standard feed-forward network used as an expert.
    
    This is the basic building block for experts in the MoE layer.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        """Initialize feed-forward network.
        
        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension (typically 4x hidden_size)
            dropout: Dropout probability
        """
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq, hidden]
            
        Returns:
            Output tensor [batch, seq, hidden]
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TopKRouter(nn.Module):
    """Top-k routing network for selecting experts.
    
    This learns to route tokens to the most appropriate experts.
    """
    
    def __init__(self, hidden_size: int, num_experts: int):
        """Initialize router.
        
        Args:
            hidden_size: Input dimension
            num_experts: Number of experts to route between
        """
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to top-k experts.
        
        Args:
            hidden_states: Input tensor [batch, seq, hidden]
            top_k: Number of experts to activate per token
            
        Returns:
            Tuple of:
                - top_k_weights: Routing weights [batch, seq, top_k]
                - top_k_indices: Expert indices [batch, seq, top_k]
                - logits: Full routing logits [batch, seq, num_experts]
        """
        # Compute routing logits
        logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # Softmax over selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        return top_k_weights, top_k_indices, logits


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing.
    
    This is the base MoE layer that replaces dense FFN layers.
    It will be extended to DynamicMoELayer for dynamic expert management.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        intermediate_size: Optional[int] = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        """Initialize MoE layer.
        
        Args:
            hidden_size: Input/output dimension
            num_experts: Number of expert networks
            num_experts_per_tok: How many experts to activate per token (top-k)
            intermediate_size: Expert hidden dimension (default: 4x hidden_size)
            capacity_factor: Factor for expert capacity (load balancing)
            dropout: Dropout probability
        """
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.capacity_factor = capacity_factor
        
        # Router network
        self.router = TopKRouter(hidden_size, num_experts)
        
        # Expert networks (parallel FFNs)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size, dropout)
            for _ in range(num_experts)
        ])
        
        # Store last router logits for load balancing loss
        self.last_router_logits: Optional[torch.Tensor] = None
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse expert routing.
        
        Args:
            hidden_states: Input tensor [batch, seq, hidden]
        
        Returns:
            Tuple of:
                - output: Output tensor [batch, seq, hidden]
                - router_logits: Routing logits [batch, seq, num_experts]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for easier routing
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]
        
        # Route to top-k experts
        top_k_weights, top_k_indices, router_logits = self.router(
            hidden_states, self.num_experts_per_tok
        )
        
        # Store for load balancing loss
        self.last_router_logits = router_logits
        
        # Flatten routing info
        flat_top_k_weights = top_k_weights.view(-1, self.num_experts_per_tok)
        flat_top_k_indices = top_k_indices.view(-1, self.num_experts_per_tok)
        
        # Initialize output
        output = torch.zeros_like(flat_hidden)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (flat_top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = flat_hidden[expert_mask]
                
                # Ensure expert is on the same device and dtype as input
                # to prevent device/dtype mismatch errors during matrix multiplication
                input_device = expert_input.device
                input_dtype = expert_input.dtype
                expert = self.experts[expert_idx]
                
                try:
                    expert_device = next(expert.parameters()).device
                    expert_dtype = next(expert.parameters()).dtype
                    
                    # Synchronize device
                    if expert_device != input_device:
                        expert = expert.to(input_device)
                        self.experts[expert_idx] = expert
                    
                    # Synchronize dtype to prevent "expected mat1 and mat2 to have the same dtype" error
                    if expert_dtype != input_dtype:
                        expert = expert.to(input_dtype)
                        self.experts[expert_idx] = expert
                        
                except StopIteration:
                    # Expert has no parameters, try moving it anyway
                    expert = expert.to(device=input_device, dtype=input_dtype)
                    self.experts[expert_idx] = expert
                
                # Process with expert
                expert_output = expert(expert_input)
                
                # Get routing weights for this expert
                # Find which position in top_k this expert is
                expert_positions = (flat_top_k_indices[expert_mask] == expert_idx)
                expert_weights = flat_top_k_weights[expert_mask][expert_positions].unsqueeze(-1)
                
                # Apply routing weights and accumulate
                output[expert_mask] += expert_output * expert_weights
        
        # Reshape to original dimensions
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, router_logits


def load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Compute load balancing loss to encourage uniform expert usage.
    
    This auxiliary loss prevents some experts from being underutilized.
    
    Args:
        router_logits: Routing logits [batch, seq, num_experts]
        num_experts: Number of experts
    
    Returns:
        Scalar load balancing loss
    """
    # Compute routing probabilities
    routing_weights = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]
    
    # Average probability of routing to each expert across all tokens
    expert_usage = routing_weights.mean(dim=[0, 1])  # [num_experts]
    
    # Ideal usage: uniform distribution (1/num_experts for each)
    ideal_usage = 1.0 / num_experts
    
    # L2 penalty for deviation from uniform
    loss = torch.sum((expert_usage - ideal_usage) ** 2)
    
    # Scale by num_experts to make loss magnitude consistent
    return loss * num_experts


def get_expert_usage_stats(router_logits: torch.Tensor) -> dict:
    """Compute statistics about expert usage from router logits.
    
    Args:
        router_logits: Routing logits [batch, seq, num_experts]
        
    Returns:
        Dictionary with usage statistics per expert
    """
    # Get routing probabilities
    routing_weights = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]
    
    # Average usage per expert
    avg_usage = routing_weights.mean(dim=[0, 1])  # [num_experts]
    
    # Max usage per expert
    max_usage = routing_weights.max(dim=0)[0].max(dim=0)[0]  # [num_experts]
    
    # Number of tokens routed to each expert (top-1 assignment)
    top1_assignments = routing_weights.argmax(dim=-1)  # [batch, seq]
    token_counts = torch.bincount(
        top1_assignments.flatten(),
        minlength=routing_weights.shape[-1]
    )
    
    return {
        "avg_routing_prob": avg_usage.cpu().tolist(),
        "max_routing_prob": max_usage.cpu().tolist(),
        "token_counts": token_counts.cpu().tolist(),
        "total_tokens": top1_assignments.numel(),
    }


# Testing
if __name__ == "__main__":
    print("Testing MoE Layer...")
    
    # Create MoE layer
    batch, seq, hidden = 2, 10, 256
    moe = MoELayer(
        hidden_size=hidden,
        num_experts=8,
        num_experts_per_tok=2,
    )
    
    # Forward pass
    x = torch.randn(batch, seq, hidden)
    output, router_logits = moe(x)
    
    # Check shapes
    assert output.shape == (batch, seq, hidden), f"Expected {(batch, seq, hidden)}, got {output.shape}"
    assert router_logits.shape == (batch, seq, 8), f"Expected {(batch, seq, 8)}, got {router_logits.shape}"
    
    # Compute load balancing loss
    lb_loss = load_balancing_loss(router_logits, num_experts=8)
    assert lb_loss.shape == (), f"Expected scalar, got {lb_loss.shape}"
    assert lb_loss >= 0, f"Load balancing loss should be non-negative, got {lb_loss}"
    
    # Get usage stats
    stats = get_expert_usage_stats(router_logits)
    assert len(stats["avg_routing_prob"]) == 8
    assert sum(stats["token_counts"]) == batch * seq
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Router logits shape: {router_logits.shape}")
    print(f"✅ Load balancing loss: {lb_loss.item():.4f}")
    print(f"✅ Expert usage (avg): {[f'{p:.3f}' for p in stats['avg_routing_prob']]}")
    print(f"✅ Token counts: {stats['token_counts']}")
    print("✅ All tests passed!")
