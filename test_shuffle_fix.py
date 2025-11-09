"""
Quick test to verify streaming dataset shuffle fix.
This test verifies that calling __iter__ multiple times on the same dataset produces different shuffles.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from aios.cli.hrm_hf.streaming_dataset import StreamingTextDataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
    
    def __call__(self, texts, **kwargs):
        """Mock tokenization."""
        batch_size = len(texts)
        max_length = kwargs.get('max_length', 128)
        
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def test_multiple_iterations_different_shuffles():
    """Test that multiple iterations on same dataset produce different shuffles."""
    lines = [f"Sample {i}" for i in range(50)]
    tokenizer = MockTokenizer()
    
    # Create ONE dataset
    dataset = StreamingTextDataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=True,
        epoch=0,
    )
    
    print(f"✓ Created dataset with {dataset.num_samples} samples")
    print(f"  Initial epoch: {dataset.epoch}, internal_epoch: {dataset._internal_epoch}")
    
    # First iteration
    print("\n1st iteration:")
    for _ in dataset:
        pass
    stats1 = dataset.get_sample_stats()
    print(f"  First 5 indices: {stats1['first_5_indices']}")
    print(f"  Internal epoch after: {stats1['internal_epoch']}")
    
    # Second iteration (simulates StopIteration restart in training loop)
    print("\n2nd iteration:")
    for _ in dataset:
        pass
    stats2 = dataset.get_sample_stats()
    print(f"  First 5 indices: {stats2['first_5_indices']}")
    print(f"  Internal epoch after: {stats2['internal_epoch']}")
    
    # Third iteration
    print("\n3rd iteration:")
    for _ in dataset:
        pass
    stats3 = dataset.get_sample_stats()
    print(f"  First 5 indices: {stats3['first_5_indices']}")
    print(f"  Internal epoch after: {stats3['internal_epoch']}")
    
    # Verify different shuffles
    assert stats1['first_5_indices'] != stats2['first_5_indices'], \
        "ERROR: First and second iteration have SAME shuffle!"
    assert stats2['first_5_indices'] != stats3['first_5_indices'], \
        "ERROR: Second and third iteration have SAME shuffle!"
    assert stats1['first_5_indices'] != stats3['first_5_indices'], \
        "ERROR: First and third iteration have SAME shuffle!"
    
    print("\n✓ SUCCESS: All three iterations produced DIFFERENT shuffles!")
    print(f"  Iteration 1: {stats1['first_5_indices']}")
    print(f"  Iteration 2: {stats2['first_5_indices']}")
    print(f"  Iteration 3: {stats3['first_5_indices']}")
    
    # Verify internal epoch incremented
    assert stats1['internal_epoch'] == 1, "Internal epoch should be 1 after first iteration"
    assert stats2['internal_epoch'] == 2, "Internal epoch should be 2 after second iteration"
    assert stats3['internal_epoch'] == 3, "Internal epoch should be 3 after third iteration"
    
    print("\n✓ Internal epoch counter working correctly!")
    return True


def test_no_shuffle_consistent():
    """Test that shuffle=False still produces consistent order."""
    lines = [f"Sample {i}" for i in range(20)]
    tokenizer = MockTokenizer()
    
    dataset = StreamingTextDataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=5,
        shuffle=False,
        epoch=0,
    )
    
    print("\n\nTesting shuffle=False (should be consistent):")
    
    # First iteration
    for _ in dataset:
        pass
    stats1 = dataset.get_sample_stats()
    
    # Second iteration
    for _ in dataset:
        pass
    stats2 = dataset.get_sample_stats()
    
    # Should be identical (no shuffle)
    assert stats1['first_5_indices'] == stats2['first_5_indices'] == [0, 1, 2, 3, 4], \
        "No-shuffle mode should produce consistent sequential order"
    
    print(f"  ✓ Both iterations: {stats1['first_5_indices']} (consistent)")
    return True


if __name__ == "__main__":
    print("="*70)
    print("Testing Streaming Dataset Shuffle Fix")
    print("="*70)
    
    try:
        test_multiple_iterations_different_shuffles()
        test_no_shuffle_consistent()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe streaming dataset now properly shuffles on each iteration.")
        print("This ensures varied training data across epochs within a single run.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
