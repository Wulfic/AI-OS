"""
Debug script to investigate the precision issue in CPU offload
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.core.hrm_models.extreme_scale_optimizations import (
    offload_carry_to_cpu,
    restore_carry_to_gpu
)

class MockInnerCarry:
    def __init__(self, device):
        self.z_H = torch.randn(2, 10, 512, device=device)
        self.z_L = torch.randn(2, 10, 512, device=device)

class MockCarry:
    def __init__(self, device):
        self.inner_carry = MockInnerCarry(device)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
    carry = MockCarry(device)
    
    print("Before offload:")
    print(f"  z_H device: {carry.inner_carry.z_H.device}")
    print(f"  z_H dtype: {carry.inner_carry.z_H.dtype}")
    print(f"  z_H sum: {carry.inner_carry.z_H.sum().item()}")
    print(f"  z_L device: {carry.inner_carry.z_L.device}")
    print(f"  z_L dtype: {carry.inner_carry.z_L.dtype}")
    print(f"  z_L sum: {carry.inner_carry.z_L.sum().item()}")
    
    original_sum_H = carry.inner_carry.z_H.sum().item()
    original_sum_L = carry.inner_carry.z_L.sum().item()
    
    # Test direct .cpu() without wrapper
    print("\nDirect .cpu() test:")
    z_L_cpu = carry.inner_carry.z_L.cpu()
    torch.cuda.synchronize(device)
    print(f"  After .cpu() sum: {z_L_cpu.sum().item()}")
    print(f"  Difference: {abs(z_L_cpu.sum().item() - original_sum_L)}")
    
    # Reset
    carry = MockCarry(device)
    original_sum_H = carry.inner_carry.z_H.sum().item()
    original_sum_L = carry.inner_carry.z_L.sum().item()
    
    # Test through wrapper
    print("\nThrough offload_carry_to_cpu:")
    carry, metadata = offload_carry_to_cpu(carry, device)
    
    print(f"  After offload:")
    print(f"  z_H sum: {carry.inner_carry.z_H.sum().item()}")
    print(f"  z_H diff: {abs(carry.inner_carry.z_H.sum().item() - original_sum_H)}")
    print(f"  z_L sum: {carry.inner_carry.z_L.sum().item()}")
    print(f"  z_L diff: {abs(carry.inner_carry.z_L.sum().item() - original_sum_L)}")
    
else:
    print("No CUDA device")
