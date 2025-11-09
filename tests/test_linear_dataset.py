"""
Test linear dataset progression feature.

Verifies that:
1. Linear mode processes data sequentially
2. Position tracking works correctly
3. Resume from offset works as expected
4. Shuffled mode behaves differently from linear mode
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.cli.hrm_hf.streaming_dataset import create_streaming_dataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
    
    def __call__(self, texts, **kwargs):
        """Mock tokenization."""
        import torch
        batch_size = len(texts)
        max_length = kwargs.get('max_length', 128)
        
        # Create simple token IDs (use hash of text for variety)
        input_ids = []
        attention_mask = []
        
        for text in texts:
            tokens = [hash(text) % 1000 for _ in range(max_length)]
            input_ids.append(tokens)
            attention_mask.append([1] * max_length)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }


def test_linear_progression():
    """Test that linear mode processes data sequentially."""
    print("\n" + "="*60)
    print("TEST 1: Linear Progression (Sequential Order)")
    print("="*60)
    
    # Create test dataset
    lines = [f"Sample {i}" for i in range(100)]
    tokenizer = MockTokenizer()
    
    # Create streaming dataset in linear mode
    dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=8,
        shuffle=False,  # Linear mode
        start_offset=0,
    )
    
    print(f"Created dataset with {len(lines)} samples")
    print(f"Batch size: 8")
    print(f"Expected batches: {len(dataset)}")
    
    # Iterate through first 3 batches
    iterator = iter(dataset)
    sample_indices = []
    
    for i in range(3):
        input_ids, labels, puzzle_ids = next(iterator)
        stats = dataset.get_sample_stats()
        
        print(f"\nBatch {i+1}:")
        print(f"  Position: {dataset.get_position()}")
        print(f"  Samples processed: {stats['total_samples_yielded']}")
        print(f"  First 5 indices: {stats['first_5_indices']}")
        
        sample_indices.extend(stats['first_5_indices'])
    
    # Verify sequential processing
    expected_indices = list(range(24))  # 3 batches * 8 samples
    actual_indices = dataset._samples_yielded[:24]
    
    if actual_indices == expected_indices:
        print("\n✅ SUCCESS: Linear mode processes data sequentially!")
        print(f"   Expected: {expected_indices[:10]}...")
        print(f"   Actual:   {actual_indices[:10]}...")
        return True
    else:
        print("\n❌ FAILED: Data was not processed sequentially")
        print(f"   Expected: {expected_indices[:10]}...")
        print(f"   Actual:   {actual_indices[:10]}...")
        return False


def test_position_tracking():
    """Test that position tracking works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Position Tracking")
    print("="*60)
    
    lines = [f"Sample {i}" for i in range(100)]
    tokenizer = MockTokenizer()
    
    dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=False,
        start_offset=0,
    )
    
    print(f"Initial position: {dataset.get_position()}")
    
    # Process 3 batches
    iterator = iter(dataset)
    positions = [dataset.get_position()]
    
    for i in range(3):
        next(iterator)
        positions.append(dataset.get_position())
        print(f"After batch {i+1}: position = {dataset.get_position()}")
    
    # Verify position increases by batch_size
    expected_positions = [0, 10, 20, 30]
    
    if positions == expected_positions:
        print("\n✅ SUCCESS: Position tracking works correctly!")
        print(f"   Positions: {positions}")
        return True
    else:
        print("\n❌ FAILED: Position tracking incorrect")
        print(f"   Expected: {expected_positions}")
        print(f"   Actual:   {positions}")
        return False


def test_resume_from_offset():
    """Test resuming from a specific offset."""
    print("\n" + "="*60)
    print("TEST 3: Resume from Offset")
    print("="*60)
    
    lines = [f"Sample {i}" for i in range(100)]
    tokenizer = MockTokenizer()
    
    # Start from offset 50
    start_offset = 50
    dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=False,
        start_offset=start_offset,
    )
    
    print(f"Starting from offset: {start_offset}")
    print(f"Initial position: {dataset.get_position()}")
    
    # Process first batch
    iterator = iter(dataset)
    next(iterator)
    
    stats = dataset.get_sample_stats()
    first_indices = stats['first_5_indices']
    
    print(f"First batch indices: {first_indices}")
    
    # Should start from index 50
    expected_start = list(range(50, 55))
    
    if first_indices == expected_start:
        print("\n✅ SUCCESS: Resume from offset works correctly!")
        print(f"   Started from index {start_offset}")
        return True
    else:
        print("\n❌ FAILED: Did not resume from correct offset")
        print(f"   Expected: {expected_start}")
        print(f"   Actual:   {first_indices}")
        return False


def test_wrap_around():
    """Test that dataset wraps around at the end."""
    print("\n" + "="*60)
    print("TEST 4: Wrap Around at End of Dataset")
    print("="*60)
    
    lines = [f"Sample {i}" for i in range(20)]  # Small dataset
    tokenizer = MockTokenizer()
    
    # Start near the end
    dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=8,
        shuffle=False,
        start_offset=16,  # 16 samples remain
    )
    
    print(f"Dataset size: {len(lines)} samples")
    print(f"Starting at: 16 (4 samples to end)")
    print(f"Batch size: 8")
    
    # Process first batch (should get samples 16-19, then wrap to 0-3)
    iterator = iter(dataset)
    next(iterator)
    
    position_after_first = dataset.get_position()
    print(f"Position after first batch: {position_after_first}")
    
    # Position should wrap around: (16 + 8) % 20 = 4
    expected_position = 4
    
    if position_after_first == expected_position:
        print("\n✅ SUCCESS: Dataset wraps around correctly!")
        print(f"   Wrapped to position {position_after_first}")
        return True
    else:
        print("\n❌ FAILED: Wrap around didn't work correctly")
        print(f"   Expected: {expected_position}")
        print(f"   Actual:   {position_after_first}")
        return False


def test_shuffled_vs_linear():
    """Test that shuffled mode behaves differently from linear mode."""
    print("\n" + "="*60)
    print("TEST 5: Shuffled vs Linear Mode Comparison")
    print("="*60)
    
    lines = [f"Sample {i}" for i in range(100)]
    tokenizer = MockTokenizer()
    
    # Linear mode
    linear_dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=False,
        start_offset=0,
    )
    
    # Shuffled mode
    shuffled_dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=True,
        start_offset=0,
    )
    
    # Get first batch indices from each
    linear_iter = iter(linear_dataset)
    shuffled_iter = iter(shuffled_dataset)
    
    next(linear_iter)
    next(shuffled_iter)
    
    linear_stats = linear_dataset.get_sample_stats()
    shuffled_stats = shuffled_dataset.get_sample_stats()
    
    linear_indices = linear_stats['first_5_indices']
    shuffled_indices = shuffled_stats['first_5_indices']
    
    print(f"Linear mode first 5:   {linear_indices}")
    print(f"Shuffled mode first 5: {shuffled_indices}")
    print(f"\nLinear shuffle_mode:   {linear_stats['shuffle_mode']}")
    print(f"Shuffled shuffle_mode: {shuffled_stats['shuffle_mode']}")
    
    # Linear should be [0,1,2,3,4], shuffled should be different
    is_sequential = linear_indices == [0, 1, 2, 3, 4]
    is_different = linear_indices != shuffled_indices
    
    if is_sequential and is_different:
        print("\n✅ SUCCESS: Linear and shuffled modes behave differently!")
        print(f"   Linear mode is sequential: {is_sequential}")
        print(f"   Modes produce different order: {is_different}")
        return True
    else:
        print("\n❌ FAILED: Modes didn't behave as expected")
        print(f"   Linear sequential: {is_sequential}")
        print(f"   Different orders: {is_different}")
        return False


def test_set_position():
    """Test manually setting position."""
    print("\n" + "="*60)
    print("TEST 6: Manually Set Position")
    print("="*60)
    
    lines = [f"Sample {i}" for i in range(100)]
    tokenizer = MockTokenizer()
    
    dataset = create_streaming_dataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=128,
        batch_size=10,
        shuffle=False,
        start_offset=0,
    )
    
    print(f"Initial position: {dataset.get_position()}")
    
    # Set position to 75
    new_position = 75
    dataset.set_position(new_position)
    
    print(f"Set position to: {new_position}")
    print(f"Current position: {dataset.get_position()}")
    
    # Process a batch
    iterator = iter(dataset)
    next(iterator)
    
    stats = dataset.get_sample_stats()
    first_indices = stats['first_5_indices']
    
    print(f"First batch indices: {first_indices}")
    
    # Should start from index 75
    expected_start = list(range(75, 80))
    
    if first_indices == expected_start and dataset.get_position() == 85:
        print("\n✅ SUCCESS: set_position() works correctly!")
        return True
    else:
        print("\n❌ FAILED: set_position() didn't work as expected")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" LINEAR DATASET PROGRESSION TESTS")
    print("="*80)
    
    tests = [
        ("Linear Progression", test_linear_progression),
        ("Position Tracking", test_position_tracking),
        ("Resume from Offset", test_resume_from_offset),
        ("Wrap Around", test_wrap_around),
        ("Shuffled vs Linear", test_shuffled_vs_linear),
        ("Set Position", test_set_position),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
