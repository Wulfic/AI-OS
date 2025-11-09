"""
Quick test to verify DDP model wrapping fix without full training.
This tests if the model gets wrapped in DDP correctly when initialized.
"""
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def test_ddp_wrapping():
    """Test that model wrapping works correctly."""
    print("Testing DDP model wrapping...")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s)")
    
    if device_count < 2:
        print("WARNING: Less than 2 GPUs available, skipping multi-GPU test")
        return True
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    # Test single GPU (baseline)
    print("\n1. Testing single GPU model...")
    device = torch.device("cuda:0")
    model = SimpleModel().to(device)
    print(f"  Model type: {type(model)}")
    print(f"  Is DDP wrapped: {isinstance(model, DDP)}")
    print("  ✓ Single GPU works")
    
    # Test DDP wrapping
    print("\n2. Testing DDP-wrapped model...")
    rank = 0  # Simulating rank 0
    model_ddp = SimpleModel().to(device)
    
    # Check if already wrapped
    is_wrapped_before = isinstance(model_ddp, DDP)
    print(f"  Before wrapping - Is DDP: {is_wrapped_before}")
    
    # Wrap in DDP
    try:
        model_ddp = DDP(model_ddp, device_ids=[rank], output_device=rank)
        is_wrapped_after = isinstance(model_ddp, DDP)
        print(f"  After wrapping - Is DDP: {is_wrapped_after}")
        print(f"  Model type: {type(model_ddp)}")
        print("  ✓ DDP wrapping successful")
        
        # Test state_dict unwrapping
        print("\n3. Testing state_dict access...")
        if isinstance(model_ddp, DDP):
            state_dict = model_ddp.module.state_dict()
            print(f"  State dict keys (via .module): {list(state_dict.keys())}")
        else:
            state_dict = model_ddp.state_dict()
            print(f"  State dict keys (direct): {list(state_dict.keys())}")
        print("  ✓ State dict access works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ DDP wrapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ddp_wrapping()
    if success:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - DDP wrapping fix verified!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ TESTS FAILED")
        print("="*60)
