"""
Unit tests for streaming dataset functionality and data variety verification.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aios.cli.hrm_hf.streaming_dataset import StreamingTextDataset, create_streaming_dataset


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


class TestStreamingDataset:
    """Tests for StreamingTextDataset class."""
    
    def test_basic_creation(self):
        """Test basic dataset creation."""
        lines = [f"Sample {i}" for i in range(10)]
        tokenizer = MockTokenizer()
        
        dataset = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=2,
            shuffle=False,
            epoch=0,
        )
        
        assert dataset.num_samples == 10
        assert dataset.num_batches == 5
        assert len(dataset) == 5
    
    def test_empty_lines_filtered(self):
        """Test that empty lines are filtered out."""
        lines = [f"Sample {i}" if i % 2 == 0 else "" for i in range(10)]
        tokenizer = MockTokenizer()
        
        dataset = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=2,
            epoch=0,
        )
        
        # Should have 5 non-empty lines
        assert dataset.num_samples == 5
    
    def test_max_samples_limit(self):
        """Test max_samples parameter."""
        lines = [f"Sample {i}" for i in range(100)]
        tokenizer = MockTokenizer()
        
        dataset = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            max_samples=50,
            epoch=0,
        )
        
        assert dataset.num_samples == 50
    
    def test_iteration_yields_batches(self):
        """Test that iteration yields correct number of batches."""
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
        
        batch_count = 0
        for input_ids, labels, puzzle_ids in dataset:
            batch_count += 1
            assert input_ids.shape[0] <= 5  # Batch size
            assert labels.shape[0] == input_ids.shape[0]
            assert puzzle_ids.shape[0] == input_ids.shape[0]
        
        assert batch_count == 4  # 20 samples / 5 per batch
    
    def test_sample_stats_tracking(self):
        """Test that sample statistics are tracked correctly."""
        lines = [f"Sample {i}" for i in range(10)]
        tokenizer = MockTokenizer()
        
        dataset = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=3,
            shuffle=False,
            epoch=0,
        )
        
        # Iterate through dataset
        for _ in dataset:
            pass
        
        # Check statistics
        stats = dataset.get_sample_stats()
        assert stats['epoch'] == 0
        assert stats['total_samples'] == 10
        assert stats['unique_samples_used'] == 10
        assert stats['coverage_percent'] == 100.0
        assert len(stats['first_5_indices']) == 5
        assert len(stats['last_5_indices']) == 5


class TestEpochBasedShuffling:
    """Tests for epoch-based shuffling functionality."""
    
    def test_different_epochs_different_order(self):
        """Test that different epochs produce different sample orders."""
        lines = [f"Sample {i}" for i in range(100)]
        tokenizer = MockTokenizer()
        
        # Create datasets with different epochs
        dataset0 = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            shuffle=True,
            epoch=0,
        )
        
        dataset1 = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            shuffle=True,
            epoch=1,
        )
        
        # Iterate through both
        for _ in dataset0:
            pass
        for _ in dataset1:
            pass
        
        # Get stats
        stats0 = dataset0.get_sample_stats()
        stats1 = dataset1.get_sample_stats()
        
        # Should have different first 5 indices
        assert stats0['first_5_indices'] != stats1['first_5_indices']
    
    def test_same_epoch_same_order(self):
        """Test that same epoch produces same order (reproducibility)."""
        lines = [f"Sample {i}" for i in range(100)]
        tokenizer = MockTokenizer()
        
        # Create two datasets with same epoch
        dataset1 = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            shuffle=True,
            epoch=5,
        )
        
        dataset2 = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            shuffle=True,
            epoch=5,
        )
        
        # Iterate through both
        for _ in dataset1:
            pass
        for _ in dataset2:
            pass
        
        # Get stats
        stats1 = dataset1.get_sample_stats()
        stats2 = dataset2.get_sample_stats()
        
        # Should have identical first 5 indices (reproducible)
        assert stats1['first_5_indices'] == stats2['first_5_indices']
    
    def test_no_shuffle_consistent_order(self):
        """Test that shuffle=False gives consistent order."""
        lines = [f"Sample {i}" for i in range(50)]
        tokenizer = MockTokenizer()
        
        dataset = StreamingTextDataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=10,
            shuffle=False,
            epoch=0,
        )
        
        # Iterate through dataset
        for _ in dataset:
            pass
        
        stats = dataset.get_sample_stats()
        
        # With no shuffle, first 5 should be [0, 1, 2, 3, 4]
        assert stats['first_5_indices'] == [0, 1, 2, 3, 4]


class TestIterateMode:
    """Tests simulating iterate mode behavior."""
    
    def test_dataset_recreation_with_new_lines(self):
        """Test that recreating dataset with new epoch uses different order."""
        tokenizer = MockTokenizer()
        
        # Simulate iterate mode: different cycles with different epochs
        all_stats = []
        
        for cycle in range(3):
            # Load lines (in reality these might be different each cycle)
            lines = [f"Cycle {cycle} Sample {i}" for i in range(100)]
            
            # Create dataset with cycle as epoch
            dataset = create_streaming_dataset(
                lines=lines,
                tokenizer=tokenizer,
                max_seq_len=128,
                batch_size=10,
                shuffle=True,
                epoch=cycle,
            )
            
            # Train on dataset
            for _ in dataset:
                pass
            
            # Collect stats
            stats = dataset.get_sample_stats()
            all_stats.append(stats['first_5_indices'])
        
        # All cycles should have different sample orders
        assert all_stats[0] != all_stats[1]
        assert all_stats[1] != all_stats[2]
        assert all_stats[0] != all_stats[2]
    
    def test_coverage_across_cycles(self):
        """Test that full dataset coverage is maintained across cycles."""
        lines = [f"Sample {i}" for i in range(100)]
        tokenizer = MockTokenizer()
        
        for cycle in range(3):
            dataset = create_streaming_dataset(
                lines=lines,
                tokenizer=tokenizer,
                max_seq_len=128,
                batch_size=10,
                shuffle=True,
                epoch=cycle,
            )
            
            # Iterate through entire dataset
            for _ in dataset:
                pass
            
            stats = dataset.get_sample_stats()
            
            # Should have 100% coverage every cycle
            assert stats['coverage_percent'] == 100.0
            assert stats['unique_samples_used'] == 100


class TestFactoryFunction:
    """Tests for create_streaming_dataset factory."""
    
    def test_factory_creates_valid_dataset(self):
        """Test that factory creates a valid dataset."""
        lines = [f"Sample {i}" for i in range(10)]
        tokenizer = MockTokenizer()
        
        dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=128,
            batch_size=2,
            shuffle=True,
            epoch=0,
        )
        
        assert isinstance(dataset, StreamingTextDataset)
        assert dataset.num_samples == 10
    
    def test_factory_passes_all_parameters(self):
        """Test that factory correctly passes all parameters."""
        lines = [f"Sample {i}" for i in range(20)]
        tokenizer = MockTokenizer()
        
        dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=256,
            batch_size=5,
            shuffle=False,
            max_samples=10,
            epoch=3,
        )
        
        assert dataset.max_seq_len == 256
        assert dataset.batch_size == 5
        assert dataset.shuffle == False
        assert dataset.num_samples == 10  # max_samples applied
        assert dataset.epoch == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
