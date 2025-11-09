#!/usr/bin/env python3
"""Test if gradient_accumulation_steps is properly stored in TrainingConfig"""
import sys
sys.path.insert(0, 'src')

from aios.core.hrm_training import TrainingConfig

# Create config with gradient_accumulation_steps=16
c = TrainingConfig(
    model='gpt2',
    dataset_file='test.txt',
    batch_size=1,
    gradient_accumulation_steps=16
)

print("=" * 60)
print("TESTING TrainingConfig.gradient_accumulation_steps")
print("=" * 60)
print(f"\n✓ Config created successfully")
print(f"Has attribute 'gradient_accumulation_steps': {hasattr(c, 'gradient_accumulation_steps')}")
print(f"Value: {c.gradient_accumulation_steps}")
print(f"Type: {type(c.gradient_accumulation_steps)}")
print(f"getattr result: {getattr(c, 'gradient_accumulation_steps', 1)}")

print(f"\n{'='*60}")
print("All gradient/accumulation related fields:")
print("=" * 60)
for name in sorted(dir(c)):
    if not name.startswith('_') and ('grad' in name.lower() or 'accum' in name.lower()):
        val = getattr(c, name)
        if not callable(val):
            print(f"  {name:40} = {val}")

print(f"\n{'='*60}")
print("RESULT: ", end="")
if c.gradient_accumulation_steps == 16:
    print("✓ SUCCESS - Value is correctly 16")
else:
    print(f"✗ FAIL - Expected 16, got {c.gradient_accumulation_steps}")
print("=" * 60)
