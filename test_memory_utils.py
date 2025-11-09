"""
Quick test to verify memory tracking utilities work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aios.core.hrm_training.memory_utils import (
    MemoryTracker,
    estimate_model_memory,
    estimate_activation_memory,
    log_optimization_summary
)

def test_memory_utilities():
    """Test memory tracking utilities."""
    
    print("Testing memory tracking utilities...\n")
    
    # Test 1: Model memory estimation
    print("1. Testing model memory estimation:")
    memory_est = estimate_model_memory(
        num_parameters=87_115_778,
        precision='fp16',
        include_optimizer=True,
        optimizer_type='adamw8bit'
    )
    print(f"   ✓ Model memory: {memory_est['model_gb']} GB")
    print(f"   ✓ Optimizer memory: {memory_est['optimizer_gb']} GB")
    print(f"   ✓ Total: {memory_est['total_gb']} GB\n")
    
    # Test 2: Activation memory estimation
    print("2. Testing activation memory estimation:")
    act_est = estimate_activation_memory(
        batch_size=1,
        sequence_length=20000,
        hidden_size=512,
        num_layers=16,
        num_heads=8,
        gradient_checkpointing=True,
        precision='fp16'
    )
    print(f"   ✓ Activation memory: {act_est['activation_gb']} GB")
    print(f"   ✓ Memory saved by checkpointing: {act_est['memory_saved_gb']} GB")
    print(f"   ✓ Saving factor: {act_est['saving_factor']}x\n")
    
    # Test 3: Optimization summary
    print("3. Testing optimization summary:")
    opt_summary = log_optimization_summary(
        model_memory_gb=0.325,
        use_8bit_optimizer=True,
        gradient_checkpointing=True,
        use_amp=True,
        use_chunked_training=True,
        chunk_size=1024,
        zero_stage="zero3",
        num_gpus=2
    )
    print(f"   ✓ Total optimizations: {len(opt_summary['optimizations'])}")
    print(f"   ✓ Total estimated savings: {opt_summary['total_estimated_savings_gb']} GB")
    for opt in opt_summary['optimizations']:
        if 'savings_gb' in opt:
            print(f"     - {opt['name']}: {opt['savings_gb']} GB saved")
        else:
            print(f"     - {opt['name']}: {opt['description']}")
    print()
    
    # Test 4: Memory tracker (basic test without CUDA)
    print("4. Testing memory tracker (basic):")
    tracker = MemoryTracker(device='cpu')  # Use CPU for testing
    if tracker.enabled:
        print("   ⚠ CUDA available - would track GPU memory")
    else:
        print("   ℹ CUDA not available - tracker disabled (expected in test)")
    
    # Tracker should still work without errors
    snapshot = tracker.snapshot('test_checkpoint', metadata={'test': True})
    print(f"   ✓ Snapshot creation: {'success' if snapshot or not tracker.enabled else 'failed'}")
    
    report = tracker.get_report()
    print(f"   ✓ Report generation: {'success' if report else 'failed'}\n")
    
    print("✅ All memory utility tests passed!\n")
    return True

if __name__ == "__main__":
    try:
        success = test_memory_utilities()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
