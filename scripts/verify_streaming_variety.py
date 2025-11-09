"""
Verification script to test that streaming datasets provide different data each iteration.

This script simulates what happens during training with --iterate mode and
verifies that different data is being used across cycles.
"""

from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.cli.hrm_hf.streaming_dataset import create_streaming_dataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token_id = 0
    
    def __call__(self, texts: List[str], **kwargs):
        """Mock tokenization - just return dummy tensors."""
        import torch
        batch_size = len(texts)
        max_length = kwargs.get('max_length', 128)
        
        # Create dummy tensors
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }


def test_streaming_variety():
    """Test that streaming provides different data across epochs."""
    print("=" * 80)
    print("STREAMING DATASET VARIETY VERIFICATION")
    print("=" * 80)
    
    # Create test data
    num_samples = 100
    lines = [f"Sample text line {i} with some content" for i in range(num_samples)]
    
    tokenizer = MockTokenizer()
    max_seq_len = 128
    batch_size = 10
    num_epochs = 5
    
    print(f"\nTest Configuration:")
    print(f"  Total samples: {num_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs to test: {num_epochs}")
    print(f"  Expected batches per epoch: {(num_samples + batch_size - 1) // batch_size}")
    
    # Track which samples are used in each epoch
    epoch_samples = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}")
        print(f"{'='*80}")
        
        # Create dataset for this epoch (simulates what happens in iterate mode)
        dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=True,
            epoch=epoch,
        )
        
        # Iterate through all batches
        samples_this_epoch = []
        for batch_idx, (input_ids, labels, puzzle_ids) in enumerate(dataset):
            # We can't directly see which samples, but we can track order
            samples_this_epoch.append(batch_idx)
        
        # Get statistics
        stats = dataset.get_sample_stats()
        print(f"\nEpoch {epoch} Statistics:")
        print(f"  Coverage: {stats['coverage_percent']}% of dataset")
        print(f"  Unique samples: {stats['unique_samples_used']} / {stats['total_samples']}")
        print(f"  First 5 indices: {stats['first_5_indices']}")
        print(f"  Last 5 indices: {stats['last_5_indices']}")
        
        epoch_samples.append(stats['first_5_indices'])
    
    # Compare epochs to verify variety
    print(f"\n{'='*80}")
    print("CROSS-EPOCH COMPARISON")
    print(f"{'='*80}")
    
    all_different = True
    for i in range(len(epoch_samples) - 1):
        for j in range(i + 1, len(epoch_samples)):
            if epoch_samples[i] == epoch_samples[j]:
                print(f"⚠️  WARNING: Epoch {i} and {j} have IDENTICAL sample order!")
                print(f"   Epoch {i} first 5: {epoch_samples[i]}")
                print(f"   Epoch {j} first 5: {epoch_samples[j]}")
                all_different = False
    
    if all_different:
        print("✅ SUCCESS: All epochs use DIFFERENT sample orders!")
        print("\nFirst 5 samples per epoch:")
        for i, samples in enumerate(epoch_samples):
            print(f"  Epoch {i}: {samples}")
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")
    
    return all_different


def test_iterate_mode_simulation():
    """Simulate what happens in --iterate mode to verify data variety."""
    print("\n\n" + "=" * 80)
    print("ITERATE MODE SIMULATION")
    print("=" * 80)
    
    # Simulate loading different lines each cycle (like _load_or_generate_lines)
    all_lines = [f"Line {i} from dataset" for i in range(1000)]
    
    tokenizer = MockTokenizer()
    max_seq_len = 128
    batch_size = 4
    num_cycles = 3
    
    print(f"\nSimulation Configuration:")
    print(f"  Total lines in dataset: {len(all_lines)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Cycles to simulate: {num_cycles}")
    
    for cycle in range(num_cycles):
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle}")
        print(f"{'='*80}")
        
        # Simulate loading a subset of lines (like in real iterate mode)
        # In practice, all lines might be loaded, or a subset - doesn't matter
        # What matters is that we create a NEW dataset each cycle
        lines = all_lines  # or could be: all_lines[cycle*100:(cycle+1)*100]
        
        print(f"Creating NEW streaming dataset with {len(lines)} lines, epoch={cycle}")
        
        # Create fresh dataset (THIS is what the fix does)
        dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=True,
            epoch=cycle,  # Different epoch each cycle!
        )
        
        # Train on a few batches
        batch_count = 0
        for input_ids, labels, puzzle_ids in dataset:
            batch_count += 1
            if batch_count >= 3:  # Just sample a few batches
                break
        
        # Check what we got
        stats = dataset.get_sample_stats()
        print(f"\nCycle {cycle} Results:")
        print(f"  First 5 sample indices: {stats['first_5_indices']}")
        print(f"  Last 5 sample indices: {stats['last_5_indices']}")
        print(f"  → These should be DIFFERENT each cycle due to epoch-based shuffling")
    
    print(f"\n{'='*80}")
    print("If you see different 'first 5 indices' above, the fix is working! ✅")
    print(f"{'='*80}")


if __name__ == "__main__":
    success = test_streaming_variety()
    test_iterate_mode_simulation()
    
    if success:
        print("\n✅ All tests passed! Dataset streaming provides variety across iterations.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Dataset might be using same data repeatedly.")
        sys.exit(1)
