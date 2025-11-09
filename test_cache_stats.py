"""Test cache statistics retrieval."""
from aios.data.streaming_cache import get_cache

cache = get_cache()
stats = cache.get_cache_stats()

print(f'Cache directory: {stats["cache_dir"]}')
print(f'Total chunks: {stats["total_chunks"]}')
print(f'Max size: {stats["max_size_mb"]} MB')
print(f'Datasets cached: {stats["datasets_cached"]}')
print(f'Size status: {stats["size_limit_status"]}')
print(f'Chunks per dataset: {stats["chunks_per_dataset"]}')
print(f'Actual cache size: {stats["cache_size_mb"]:.4f} MB')
