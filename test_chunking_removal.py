#!/usr/bin/env python3
"""Test script to verify auto-chunking removal."""

from aios.core.hrm_models.auto_chunking import auto_chunked_segment_rollout

def test_auto_chunking_disabled():
    """Verify that auto-chunking is disabled by default."""
    
    print("Testing auto-chunking behavior...")
    print("=" * 70)
    
    # Test 1: Very long context without explicit chunking should NOT chunk
    print("\n✅ Test 1: Long context (50000 tokens) without explicit chunking")
    print("   Expected: Should use standard rollout (no chunking)")
    
    rollout_fn = auto_chunked_segment_rollout(
        max_seq_len=50000,
        # chunk_threshold defaults to 999999 (effectively disabled)
        chunk_size=2048
    )
    
    # Check what we got
    module = rollout_fn.__module__ if hasattr(rollout_fn, '__module__') else str(rollout_fn)
    if 'chunked_training' in module or 'wrapper' in rollout_fn.__name__:
        print("   ❌ FAILED: Chunking was auto-enabled!")
        return False
    else:
        print(f"   ✅ PASSED: Using standard rollout: {rollout_fn.__name__}")
    
    # Test 2: Explicit chunking enabled (threshold=0) should chunk
    print("\n✅ Test 2: Explicit chunking enabled (threshold=0)")
    print("   Expected: Should use chunked rollout")
    
    rollout_fn_chunked = auto_chunked_segment_rollout(
        max_seq_len=50000,
        chunk_threshold=0,  # Force chunking
        chunk_size=2048
    )
    
    if 'wrapper' in rollout_fn_chunked.__name__:
        print(f"   ✅ PASSED: Using chunked rollout: {rollout_fn_chunked.__name__}")
    else:
        print("   ❌ FAILED: Chunking not enabled when explicitly requested!")
        return False
    
    # Test 3: Old behavior (8192) would have auto-chunked
    print("\n✅ Test 3: 10K context with old threshold would have auto-chunked")
    print("   Expected: Now requires explicit enable")
    
    rollout_fn_10k = auto_chunked_segment_rollout(
        max_seq_len=10000,
        # Using default threshold (999999)
        chunk_size=2048
    )
    
    if 'chunked_training' not in str(rollout_fn_10k.__module__) and 'wrapper' not in rollout_fn_10k.__name__:
        print(f"   ✅ PASSED: No auto-chunking for 10K context")
    else:
        print("   ❌ FAILED: Still auto-chunking at 10K!")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 All tests passed! Auto-chunking removal successful!")
    print("=" * 70)
    print("\nSummary:")
    print("  • Default chunk_threshold: 999999 (effectively disabled)")
    print("  • User must explicitly enable chunking")
    print("  • No hardcoded 8192 token limit")
    print("  • User settings are respected")
    
    return True

if __name__ == "__main__":
    try:
        success = test_auto_chunking_disabled()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
