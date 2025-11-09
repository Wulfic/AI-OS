"""
Tests for streaming dataset chunk cache functionality.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_cache_instance_singleton(temp_cache_dir):
    """Test that cache uses singleton pattern."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache1 = StreamingChunkCache.get_instance(cache_dir=temp_cache_dir)
    cache2 = StreamingChunkCache.get_instance()
    
    assert cache1 is cache2


def test_save_and_retrieve_chunk(temp_cache_dir):
    """Test basic save and retrieve operations."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir, max_chunks_per_dataset=3)
    
    # Test data
    dataset_path = "test/dataset"
    config = None
    split = "train"
    chunk_index = 0
    test_lines = [f"Line {i}" for i in range(100)]
    
    # Save chunk
    success = cache.save_chunk(dataset_path, config, split, chunk_index, test_lines)
    assert success
    
    # Retrieve chunk
    retrieved = cache.get_cached_chunk(dataset_path, config, split, chunk_index)
    assert retrieved is not None
    assert len(retrieved) == len(test_lines)
    assert retrieved == test_lines


def test_lru_eviction(temp_cache_dir):
    """Test that LRU eviction works correctly."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir, max_chunks_per_dataset=3, max_size_mb=1000)
    
    dataset_path = "test/dataset"
    config = None
    split = "train"
    
    # Save 4 chunks (should evict oldest)
    for i in range(4):
        lines = [f"Chunk {i} Line {j}" for j in range(50)]
        cache.save_chunk(dataset_path, config, split, i, lines)
        time.sleep(0.01)  # Ensure different timestamps
    
    # First chunk (index 0) should be evicted
    chunk_0 = cache.get_cached_chunk(dataset_path, config, split, 0)
    assert chunk_0 is None
    
    # Chunks 1, 2, 3 should still exist
    chunk_1 = cache.get_cached_chunk(dataset_path, config, split, 1)
    chunk_2 = cache.get_cached_chunk(dataset_path, config, split, 2)
    chunk_3 = cache.get_cached_chunk(dataset_path, config, split, 3)
    
    assert chunk_1 is not None
    assert chunk_2 is not None
    assert chunk_3 is not None


def test_age_expiration(temp_cache_dir):
    """Test that old caches are rejected."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir)
    
    dataset_path = "test/dataset"
    config = None
    split = "train"
    chunk_index = 0
    test_lines = ["Line 1", "Line 2"]
    
    # Save chunk
    cache.save_chunk(dataset_path, config, split, chunk_index, test_lines)
    
    # Should retrieve with max_age_hours=1000
    retrieved = cache.get_cached_chunk(dataset_path, config, split, chunk_index, max_age_hours=1000.0)
    assert retrieved is not None
    
    # Should NOT retrieve with max_age_hours=0 (too old)
    retrieved = cache.get_cached_chunk(dataset_path, config, split, chunk_index, max_age_hours=0.0)
    assert retrieved is None


def test_clear_dataset_cache(temp_cache_dir):
    """Test clearing cache for specific dataset."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir)
    
    # Save chunks for two datasets
    for dataset in ["dataset1", "dataset2"]:
        for i in range(2):
            lines = [f"{dataset} Chunk {i}"]
            cache.save_chunk(dataset, None, "train", i, lines)
    
    # Clear dataset1
    removed = cache.clear_dataset_cache("dataset1")
    assert removed == 2
    
    # Dataset1 should be gone
    assert cache.get_cached_chunk("dataset1", None, "train", 0) is None
    assert cache.get_cached_chunk("dataset1", None, "train", 1) is None
    
    # Dataset2 should remain
    assert cache.get_cached_chunk("dataset2", None, "train", 0) is not None
    assert cache.get_cached_chunk("dataset2", None, "train", 1) is not None


def test_cache_stats(temp_cache_dir):
    """Test cache statistics generation."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir, max_size_mb=1000)
    
    # Save some chunks
    for i in range(3):
        lines = [f"Line {j}" for j in range(50)]
        cache.save_chunk("test/dataset", None, "train", i, lines)
    
    # Get stats
    stats = cache.get_cache_stats()
    
    assert stats['total_chunks'] == 3
    assert stats['datasets_cached'] == 1
    assert 'test/dataset' in stats['chunks_per_dataset']
    assert stats['chunks_per_dataset']['test/dataset'] == 3
    assert stats['cache_size_mb'] > 0
    assert 'size_limit_status' in stats
    assert 'max_size_mb' in stats


def test_cleanup_old_caches(temp_cache_dir):
    """Test cleanup of old caches."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir, max_size_mb=1000)
    
    # Save chunks
    for i in range(3):
        lines = [f"Line {j}" for j in range(20)]
        cache.save_chunk("test/dataset", None, "train", i, lines)
    
    # Cleanup with max_age_hours=0 (all should be removed)
    removed = cache.cleanup_old_caches(max_age_hours=0.0)
    assert removed == 3
    
    # All chunks should be gone
    for i in range(3):
        assert cache.get_cached_chunk("test/dataset", None, "train", i) is None


def test_size_limit_enforcement(temp_cache_dir):
    """Test that cache size limit is enforced."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    # Set a very small cache size limit (0.01 MB = 10KB)
    cache = StreamingChunkCache(cache_dir=temp_cache_dir, max_chunks_per_dataset=10, max_size_mb=0.01)
    
    # Save multiple chunks that would exceed the limit
    for i in range(5):
        # Each chunk has ~1KB of text
        lines = [f"Chunk {i} " + "x" * 100 for j in range(10)]
        cache.save_chunk("test/dataset", None, "train", i, lines)
    
    # Check that size is kept under limit
    stats = cache.get_cache_stats()
    assert stats['cache_size_mb'] <= cache.max_size_mb * 1.1  # Allow 10% buffer


def test_cache_key_uniqueness(temp_cache_dir):
    """Test that different dataset parameters create unique cache keys."""
    from aios.data.streaming_cache import StreamingChunkCache
    
    cache = StreamingChunkCache(cache_dir=temp_cache_dir)
    
    test_lines = ["Test line"]
    
    # Save with different parameters
    cache.save_chunk("dataset1", None, "train", 0, test_lines)
    cache.save_chunk("dataset1", "config1", "train", 0, test_lines)
    cache.save_chunk("dataset1", None, "test", 0, test_lines)
    cache.save_chunk("dataset1", None, "train", 1, test_lines)
    
    # All should be retrievable independently
    assert cache.get_cached_chunk("dataset1", None, "train", 0) is not None
    assert cache.get_cached_chunk("dataset1", "config1", "train", 0) is not None
    assert cache.get_cached_chunk("dataset1", None, "test", 0) is not None
    assert cache.get_cached_chunk("dataset1", None, "train", 1) is not None
    
    # Wrong parameters should return None
    assert cache.get_cached_chunk("dataset2", None, "train", 0) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
