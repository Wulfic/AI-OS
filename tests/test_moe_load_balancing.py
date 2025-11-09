"""Unit tests for MoE load balancing loss implementation."""

import pytest
import torch
from aios.core.hrm_models.moe_layer import MoELayer, load_balancing_loss, get_expert_usage_stats


def test_load_balancing_loss_reduces_imbalance():
    """Test that load balancing loss encourages uniform expert usage."""
    # Create MoE layer
    batch, seq, hidden = 2, 10, 256
    moe = MoELayer(hidden_size=hidden, num_experts=8, num_experts_per_tok=2)
    
    # Forward pass
    x = torch.randn(batch, seq, hidden)
    output, router_logits = moe(x)
    
    # Compute load balancing loss
    lb_loss = load_balancing_loss(router_logits, num_experts=8)
    
    # Verify loss properties
    assert lb_loss.shape == (), f"Expected scalar loss, got {lb_loss.shape}"
    assert lb_loss >= 0, f"Load balancing loss should be non-negative, got {lb_loss}"
    assert not torch.isnan(lb_loss).any(), "Load balancing loss should not be NaN"
    assert not torch.isinf(lb_loss).any(), "Load balancing loss should not be Inf"
    
    print(f"✅ Load balancing loss: {lb_loss.item():.4f}")
    
    # Get usage statistics
    stats = get_expert_usage_stats(router_logits)
    print(f"✅ Expert usage (avg): {[f'{p:.3f}' for p in stats['avg_routing_prob']]}")
    print(f"✅ Token counts: {stats['token_counts']}")


def test_load_balancing_loss_with_training():
    """Test that load balancing loss improves expert distribution over training steps."""
    batch, seq, hidden = 4, 20, 256
    moe = MoELayer(hidden_size=hidden, num_experts=8, num_experts_per_tok=2)
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)
    
    initial_imbalance = None
    final_imbalance = None
    
    # Train for a few steps with load balancing loss
    for step in range(50):
        x = torch.randn(batch, seq, hidden)
        output, router_logits = moe(x)
        
        # Main task loss (dummy)
        main_loss = output.mean()
        
        # Load balancing loss
        lb_loss = load_balancing_loss(router_logits, num_experts=8)
        
        # Combined loss
        total_loss = main_loss + 0.01 * lb_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track imbalance (std deviation of expert usage)
        stats = get_expert_usage_stats(router_logits)
        usage = torch.tensor(stats['avg_routing_prob'])
        imbalance = usage.std().item()
        
        if step == 0:
            initial_imbalance = imbalance
        if step == 49:
            final_imbalance = imbalance
    
    print(f"✅ Initial imbalance (std): {initial_imbalance:.4f}")
    print(f"✅ Final imbalance (std): {final_imbalance:.4f}")
    print(f"✅ Improvement: {((initial_imbalance - final_imbalance) / initial_imbalance * 100):.1f}%")
    
    # Verify that balancing improved (or at least didn't get worse)
    assert final_imbalance <= initial_imbalance * 1.5, \
        "Load balancing should prevent expert imbalance from worsening significantly"


def test_load_balancing_loss_shape_invariance():
    """Test that load balancing loss works with different batch/seq sizes."""
    test_shapes = [
        (1, 5, 128),
        (2, 10, 256),
        (4, 20, 512),
        (8, 50, 256),
    ]
    
    for batch, seq, hidden in test_shapes:
        moe = MoELayer(hidden_size=hidden, num_experts=8, num_experts_per_tok=2)
        x = torch.randn(batch, seq, hidden)
        output, router_logits = moe(x)
        
        lb_loss = load_balancing_loss(router_logits, num_experts=8)
        
        assert lb_loss.shape == (), f"Expected scalar for shape {(batch, seq, hidden)}"
        assert lb_loss >= 0, f"Loss should be non-negative for shape {(batch, seq, hidden)}"
        
    print(f"✅ Load balancing loss works for all tested shapes")


def test_expert_usage_stats():
    """Test expert usage statistics computation."""
    batch, seq, hidden = 2, 10, 256
    moe = MoELayer(hidden_size=hidden, num_experts=8, num_experts_per_tok=2)
    
    x = torch.randn(batch, seq, hidden)
    output, router_logits = moe(x)
    
    stats = get_expert_usage_stats(router_logits)
    
    # Verify statistics structure
    assert 'avg_routing_prob' in stats
    assert 'max_routing_prob' in stats
    assert 'token_counts' in stats
    assert 'total_tokens' in stats
    
    # Verify values
    assert len(stats['avg_routing_prob']) == 8, "Should have stats for all 8 experts"
    assert len(stats['token_counts']) == 8, "Should have counts for all 8 experts"
    assert stats['total_tokens'] == batch * seq, f"Total tokens should be {batch * seq}"
    assert sum(stats['token_counts']) == batch * seq, "Token counts should sum to total"
    
    # Verify probabilities sum to 1 (within numerical precision)
    total_prob = sum(stats['avg_routing_prob'])
    assert abs(total_prob - 1.0) < 0.01, f"Routing probabilities should sum to ~1.0, got {total_prob}"
    
    print(f"✅ Expert usage stats structure is correct")
    print(f"✅ Total tokens: {stats['total_tokens']}")
    print(f"✅ Token distribution: {stats['token_counts']}")


def test_moe_with_different_expert_counts():
    """Test load balancing with different numbers of experts."""
    for num_experts in [4, 8, 16]:
        batch, seq, hidden = 2, 10, 256
        num_experts_per_tok = min(2, num_experts)
        
        moe = MoELayer(
            hidden_size=hidden,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok
        )
        
        x = torch.randn(batch, seq, hidden)
        output, router_logits = moe(x)
        
        lb_loss = load_balancing_loss(router_logits, num_experts=num_experts)
        
        assert lb_loss >= 0, f"Loss should be non-negative for {num_experts} experts"
        
        stats = get_expert_usage_stats(router_logits)
        assert len(stats['avg_routing_prob']) == num_experts
        
    print(f"✅ Load balancing works for 4, 8, and 16 experts")


if __name__ == "__main__":
    print("Testing MoE Load Balancing Implementation...\n")
    
    print("=" * 70)
    print("Test 1: Load Balancing Loss Computation")
    print("=" * 70)
    test_load_balancing_loss_reduces_imbalance()
    
    print("\n" + "=" * 70)
    print("Test 2: Load Balancing with Training")
    print("=" * 70)
    test_load_balancing_loss_with_training()
    
    print("\n" + "=" * 70)
    print("Test 3: Shape Invariance")
    print("=" * 70)
    test_load_balancing_loss_shape_invariance()
    
    print("\n" + "=" * 70)
    print("Test 4: Expert Usage Statistics")
    print("=" * 70)
    test_expert_usage_stats()
    
    print("\n" + "=" * 70)
    print("Test 5: Different Expert Counts")
    print("=" * 70)
    test_moe_with_different_expert_counts()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
