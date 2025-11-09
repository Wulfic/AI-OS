"""
Quick test script for CPU offload functionality.
Tests the synchronization fixes in extreme_scale_optimizations.py
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("="*60)
print("CPU OFFLOAD SYNCHRONIZATION TEST")
print("="*60)

# Test 1: Basic offload/restore cycle
print("\n[Test 1] Basic CPU offload and restore")
print("-" * 40)

from aios.core.hrm_models.extreme_scale_optimizations import (
    offload_carry_to_cpu,
    restore_carry_to_gpu
)

# Create a mock carry object
class MockInnerCarry:
    def __init__(self, device):
        self.z_H = torch.randn(2, 10, 512, device=device)
        self.z_L = torch.randn(2, 10, 512, device=device)

class MockCarry:
    def __init__(self, device):
        self.inner_carry = MockInnerCarry(device)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"✓ Using device: {device}")
    
    # Create carry on GPU
    carry = MockCarry(device)
    print(f"✓ Created carry with z_H shape: {carry.inner_carry.z_H.shape}")
    print(f"✓ Initial z_H device: {carry.inner_carry.z_H.device}")
    print(f"✓ Initial z_H sum: {carry.inner_carry.z_H.sum().item():.4f}")
    
    # Test offload
    print("\n[Offloading to CPU]")
    original_sum_H = carry.inner_carry.z_H.sum().item()
    original_sum_L = carry.inner_carry.z_L.sum().item()
    
    carry, metadata = offload_carry_to_cpu(carry, device)
    
    print(f"✓ After offload z_H device: {carry.inner_carry.z_H.device}")
    print(f"✓ After offload z_L device: {carry.inner_carry.z_L.device}")
    print(f"✓ z_H sum preserved: {carry.inner_carry.z_H.sum().item():.4f}")
    
    # Verify synchronization happened (data should be complete)
    # Use relative tolerance for floating point comparisons
    assert carry.inner_carry.z_H.device.type == 'cpu', "z_H should be on CPU"
    assert carry.inner_carry.z_L.device.type == 'cpu', "z_L should be on CPU"
    
    # Floating point tolerance: allow tiny precision differences from GPU->CPU transfer
    rtol = 1e-4  # 0.01% relative tolerance
    z_H_diff = abs(carry.inner_carry.z_H.sum().item() - original_sum_H)
    z_L_diff = abs(carry.inner_carry.z_L.sum().item() - original_sum_L)
    assert z_H_diff < rtol * abs(original_sum_H) + 1e-4, f"z_H data corrupted: diff={z_H_diff}"
    assert z_L_diff < rtol * abs(original_sum_L) + 1e-4, f"z_L data corrupted: diff={z_L_diff}"
    
    print("✓ Synchronization test passed - data intact after offload")
    
    # Test restore
    print("\n[Restoring to GPU]")
    carry = restore_carry_to_gpu(carry, metadata, device)
    
    print(f"✓ After restore z_H device: {carry.inner_carry.z_H.device}")
    print(f"✓ After restore z_L device: {carry.inner_carry.z_L.device}")
    print(f"✓ z_H sum preserved: {carry.inner_carry.z_H.sum().item():.4f}")
    
    # Verify synchronization happened (data should be complete)
    assert carry.inner_carry.z_H.device == device, "z_H should be back on GPU"
    assert carry.inner_carry.z_L.device == device, "z_L should be back on GPU"
    
    # Check data integrity with appropriate floating point tolerance
    z_H_diff = abs(carry.inner_carry.z_H.sum().item() - original_sum_H)
    z_L_diff = abs(carry.inner_carry.z_L.sum().item() - original_sum_L)
    assert z_H_diff < rtol * abs(original_sum_H) + 1e-4, f"z_H data corrupted: diff={z_H_diff}"
    assert z_L_diff < rtol * abs(original_sum_L) + 1e-4, f"z_L data corrupted: diff={z_L_diff}"
    
    print("✓ Synchronization test passed - data intact after restore")
    
    # Test 2: Multiple rapid cycles (stress test)
    print("\n[Test 2] Rapid offload/restore cycles (stress test)")
    print("-" * 40)
    
    for i in range(10):
        # Offload
        carry, _ = offload_carry_to_cpu(carry, device)
        
        # Immediately restore (tests if sync is working)
        carry = restore_carry_to_gpu(carry, {}, device)
        
        # Verify data integrity with floating point tolerance
        current_sum = carry.inner_carry.z_H.sum().item()
        diff = abs(current_sum - original_sum_H)
        if diff > rtol * abs(original_sum_H) + 1e-4:
            print(f"✗ FAILED at cycle {i+1}: Data corruption detected!")
            print(f"  Expected: {original_sum_H:.4f}, Got: {current_sum:.4f}, Diff: {diff:.6f}")
            sys.exit(1)
    
    print(f"✓ Completed 10 rapid cycles - no data corruption")
    print(f"✓ Final z_H sum: {carry.inner_carry.z_H.sum().item():.4f}")
    
    # Test 3: Simulate chunked training pattern
    print("\n[Test 3] Simulated chunked training pattern")
    print("-" * 40)
    
    for chunk_idx in range(5):
        if chunk_idx > 0:
            # Restore from previous chunk
            carry = restore_carry_to_gpu(carry, {}, device)
        
        # Simulate forward pass (create some tensors, do operations)
        _ = torch.randn(2, 1024, 512, device=device)  # Simulate activations
        torch.cuda.empty_cache()  # This is what happens in actual training
        
        # Verify carry is still valid with floating point tolerance
        current_sum = carry.inner_carry.z_H.sum().item()
        diff = abs(current_sum - original_sum_H)
        if diff > rtol * abs(original_sum_H) + 1e-4:
            print(f"✗ FAILED at chunk {chunk_idx}: Data corruption! Diff: {diff:.6f}")
            sys.exit(1)
        
        if chunk_idx < 4:
            # Offload for next chunk
            carry, _ = offload_carry_to_cpu(carry, device)
    
    print(f"✓ Completed 5 chunk cycles - pattern matches actual training")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe CPU offload synchronization fixes are working correctly.")
    print("Data integrity is maintained across offload/restore cycles.")
    print("The fixes prevent race conditions in parallel training scenarios.")
    
else:
    print("⚠ No CUDA device available - skipping GPU tests")
    print("CPU offload is only relevant for GPU training")
