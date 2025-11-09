"""
Tests for DynamicMoELayer.

Run with: python -m aios.core.hrm_models.dynamic_moe.tests
"""

import torch
import tempfile

from .dynamic_layer import DynamicMoELayer
from aios.core.hrm_models.expert_metadata import create_expert_metadata


def run_tests():
    """Run comprehensive tests for DynamicMoELayer."""
    print("Testing DynamicMoELayer...")
    
    # Force CPU for testing to avoid device issues
    device = "cpu"
    
    # Test 1: Basic creation
    print("\n[Test 1] Creating DynamicMoELayer...")
    moe = DynamicMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts_per_tok=2,
        lazy_loading=False,
        device=device,
    )
    print(f"[OK] Created DynamicMoELayer with 0 experts initially")
    
    # Test 2: Add experts
    print("\n[Test 2] Adding experts...")
    
    for i in range(4):
        expert_id = f"expert_{i}"
        metadata = create_expert_metadata(
            expert_id=expert_id,
            name=f"Expert {i}",
            description=f"Test expert {i}",
            category="general" if i < 2 else "specialized",
        )
        moe.add_expert(expert_id, metadata=metadata)
    
    active_ids = moe.get_active_expert_ids()
    print(f"[OK] Added 4 experts: {active_ids}")
    assert len(active_ids) == 4
    
    # Test 3: Forward pass
    print("\n[Test 3] Testing forward pass...")
    x = torch.randn(2, 10, 256)
    output, router_logits = moe(x)
    print(f"[OK] Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Router logits shape: {router_logits.shape}")
    assert output.shape == (2, 10, 256)
    assert router_logits.shape == (2, 10, 4)
    
    # Test 4: Freeze/unfreeze expert
    print("\n[Test 4] Testing freeze/unfreeze...")
    moe.freeze_expert("expert_0")
    assert "expert_0" in moe.frozen_experts
    print(f"[OK] Froze expert_0")
    
    moe.unfreeze_expert("expert_0")
    assert "expert_0" not in moe.frozen_experts
    print(f"[OK] Unfroze expert_0")
    
    # Test 5: Remove expert
    print("\n[Test 5] Testing expert removal...")
    moe.remove_expert("expert_3")
    active_ids = moe.get_active_expert_ids()
    print(f"[OK] Removed expert_3, remaining: {active_ids}")
    assert len(active_ids) == 3
    assert "expert_3" not in active_ids
    
    # Test 6: Save/load checkpoint
    print("\n[Test 6] Testing save/load checkpoint...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        moe.save_checkpoint(tmpdir)
        print(f"[OK] Saved checkpoint to {tmpdir}")
        
        # Create new layer and load
        moe2 = DynamicMoELayer(
            hidden_size=256,
            intermediate_size=1024,
            num_experts_per_tok=2,
            device=device,
        )
        moe2.load_checkpoint(tmpdir)
        
        loaded_ids = moe2.get_active_expert_ids()
        print(f"[OK] Loaded checkpoint, experts: {loaded_ids}")
        assert len(loaded_ids) == 3
        assert set(loaded_ids) == set(active_ids)
    
    # Test 7: Lazy loading
    print("\n[Test 7] Testing lazy loading...")
    moe_lazy = DynamicMoELayer(
        hidden_size=256,
        intermediate_size=1024,
        num_experts_per_tok=2,
        lazy_loading=True,
        max_gpu_experts=2,
        device=device,
    )
    
    # Add experts with lazy loading
    for i in range(4):
        expert_id = f"lazy_expert_{i}"
        metadata = create_expert_metadata(
            expert_id=expert_id,
            name=f"Lazy Expert {i}",
            description=f"Lazy test expert {i}",
            category="general"
        )
        moe_lazy.add_expert(expert_id, metadata=metadata)
    
    print(f"[OK] Created lazy MoE with 4 experts")
    
    # Get lazy loader stats
    if moe_lazy.lazy_loader:
        stats = moe_lazy.lazy_loader.get_stats()
        print(f"   Lazy loader stats: {stats}")
    
    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    run_tests()
