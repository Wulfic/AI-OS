"""
Streaming Dataset Chunk Cache

Caches downloaded chunks from streaming datasets to reduce wait time between
training runs. Maintains a rotating cache of a few chunks per dataset to
balance speed and disk usage.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import yaml


class StreamingChunkCache:
    """
    Manages cached chunks of streaming datasets.
    
    Features:
    - LRU eviction when max chunks per dataset is reached
    - Thread-safe operations
    - Automatic cleanup of old/invalid caches
    - Configurable chunk count per dataset
    """
    
    _instance: Optional['StreamingChunkCache'] = None
    _lock = threading.Lock()
    
    def __init__(self, cache_dir: Optional[Path] = None, max_chunks_per_dataset: Optional[int] = None, max_size_mb: Optional[float] = None):
        """
        Initialize the chunk cache.
        
        Args:
            cache_dir: Directory for cached chunks (default: training_data/hf_cache/streaming_chunks)
            max_chunks_per_dataset: Maximum number of chunks to keep per dataset (default: from config or 5)
            max_size_mb: Maximum total cache size in MB (default: from config or 100)
        """
        if cache_dir is None:
            # Use HF_HOME if set, otherwise use default location
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                cache_dir = Path(hf_home) / "streaming_chunks"
            else:
                cache_dir = Path.cwd() / "training_data" / "hf_cache" / "streaming_chunks"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration from config file if not specified
        config = self._load_config()
        self.max_chunks_per_dataset = max_chunks_per_dataset or config.get('max_chunks_per_dataset', 5)
        self.max_size_mb = max_size_mb or config.get('max_size_mb', 100.0)
        self._cache_lock = threading.Lock()
        
        # Metadata file for tracking cache state
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    @classmethod
    def get_instance(cls, cache_dir: Optional[Path] = None, max_chunks_per_dataset: Optional[int] = None, max_size_mb: Optional[float] = None) -> 'StreamingChunkCache':
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir, max_chunks_per_dataset, max_size_mb)
        return cls._instance
    
    def _load_config(self) -> Dict[str, Any]:
        """Load streaming cache configuration from user config file."""
        try:
            # Import centralized config loader
            try:
                from ..gui.config_loader import load_config
                config = load_config()
                return config.get('streaming_cache', {})
            except ImportError:
                # Fallback for CLI-only usage
                import os
                env_path = os.environ.get("AIOS_CONFIG")
                if env_path:
                    config_path = Path(env_path)
                else:
                    user_config = Path.home() / ".config" / "aios" / "config.yaml"
                    if user_config.exists():
                        config_path = user_config
                    else:
                        config_path = Path.cwd() / "config" / "default.yaml"
                
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        return config.get('streaming_cache', {})
        except Exception:
            pass
        return {}
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception:
                # Corrupted metadata, start fresh
                self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass  # Non-critical failure
    
    def _get_cache_key(self, dataset_path: str, config: Optional[str], split: str, chunk_index: int, max_lines: int = 4000) -> str:
        """
        Generate a cache key for a dataset chunk.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Dataset configuration name
            split: Dataset split (train/test/validation)
            chunk_index: Index of the chunk (based on max_lines)
            max_lines: Chunk size (number of lines per chunk)
            
        Returns:
            Hex string cache key
        """
        # Create deterministic key from dataset parameters INCLUDING chunk size
        # This ensures different chunk sizes don't reuse incompatible cached data
        key_str = f"{dataset_path}:{config or 'default'}:{split}:{chunk_index}:{max_lines}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
    
    def _get_dataset_key(self, dataset_path: str, config: Optional[str], split: str) -> str:
        """Get a key for the dataset (without chunk index)."""
        key_str = f"{dataset_path}:{config or 'default'}:{split}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
    
    def get_cached_chunk(
        self,
        dataset_path: str,
        config: Optional[str],
        split: str,
        chunk_index: int,
        max_age_hours: float = 72.0,
        max_lines: int = 4000
    ) -> Optional[List[str]]:
        """
        Retrieve a cached chunk if available and not too old.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Dataset configuration name
            split: Dataset split
            chunk_index: Chunk index to retrieve
            max_age_hours: Maximum age of cache in hours (default: 72h = 3 days)
            max_lines: Chunk size (number of lines per chunk)
            
        Returns:
            List of text lines if cached and valid, None otherwise
        """
        with self._cache_lock:
            cache_key = self._get_cache_key(dataset_path, config, split, chunk_index, max_lines)
            
            # Check if cache exists in metadata
            if cache_key not in self.metadata:
                return None
            
            cache_info = self.metadata[cache_key]
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Check if file exists
            if not cache_file.exists():
                # Orphaned metadata, clean it up
                del self.metadata[cache_key]
                self._save_metadata()
                return None
            
            # Check age
            cached_time = cache_info.get('timestamp', 0)
            age_hours = (time.time() - cached_time) / 3600
            
            if age_hours > max_age_hours:
                # Cache too old, remove it
                try:
                    cache_file.unlink()
                    del self.metadata[cache_key]
                    self._save_metadata()
                except Exception:
                    pass
                return None
            
            # Load cached data
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lines = data.get('lines', [])
                
                # Update access time for LRU
                cache_info['last_access'] = time.time()
                self._save_metadata()
                
                return lines
            except Exception:
                # Corrupted cache, remove it
                try:
                    cache_file.unlink()
                    del self.metadata[cache_key]
                    self._save_metadata()
                except Exception:
                    pass
                return None
    
    def _get_total_cache_size_mb(self) -> float:
        """Calculate total cache size in MB."""
        total_size = 0.0
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name != "cache_metadata.json":
                try:
                    total_size += cache_file.stat().st_size
                except Exception:
                    pass
        return total_size / (1024 * 1024)
    
    def _enforce_size_limit(self) -> None:
        """Remove oldest caches if total size exceeds limit."""
        current_size = self._get_total_cache_size_mb()
        
        if current_size <= self.max_size_mb:
            return
        
        # Sort all chunks by last access time (oldest first)
        all_chunks = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        # Remove oldest chunks until we're under the limit
        for key, _ in all_chunks:
            if current_size <= self.max_size_mb * 0.9:  # Keep 10% buffer
                break
            
            cache_file = self.cache_dir / f"{key}.json"
            try:
                if cache_file.exists():
                    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                    cache_file.unlink()
                    current_size -= file_size_mb
                del self.metadata[key]
            except Exception:
                pass
        
        self._save_metadata()
    
    def save_chunk(
        self,
        dataset_path: str,
        config: Optional[str],
        split: str,
        chunk_index: int,
        lines: List[str],
        max_lines: int = 4000
    ) -> bool:
        """
        Save a chunk to cache, enforcing per-dataset limits and total size limit.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Dataset configuration name
            split: Dataset split
            chunk_index: Chunk index
            lines: Text lines to cache
            
        Returns:
            True if saved successfully
        """
        with self._cache_lock:
            cache_key = self._get_cache_key(dataset_path, config, split, chunk_index, max_lines)
            dataset_key = self._get_dataset_key(dataset_path, config, split)
            
            # Find all chunks for this dataset
            dataset_chunks = [
                (key, info) for key, info in self.metadata.items()
                if info.get('dataset_key') == dataset_key
            ]
            
            # If we're at the limit, evict oldest (LRU)
            if len(dataset_chunks) >= self.max_chunks_per_dataset:
                # Sort by last access time (oldest first)
                dataset_chunks.sort(key=lambda x: x[1].get('last_access', 0))
                
                # Remove oldest chunk(s) to make room
                chunks_to_remove = len(dataset_chunks) - self.max_chunks_per_dataset + 1
                for old_key, _ in dataset_chunks[:chunks_to_remove]:
                    old_file = self.cache_dir / f"{old_key}.json"
                    try:
                        if old_file.exists():
                            old_file.unlink()
                        del self.metadata[old_key]
                    except Exception:
                        pass
            
            # Save the new chunk
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            try:
                cache_data = {
                    'lines': lines,
                    'dataset_path': dataset_path,
                    'config': config,
                    'split': split,
                    'chunk_index': chunk_index,
                    'num_lines': len(lines),
                }
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                
                # Update metadata
                self.metadata[cache_key] = {
                    'dataset_path': dataset_path,
                    'config': config,
                    'split': split,
                    'chunk_index': chunk_index,
                    'dataset_key': dataset_key,
                    'num_lines': len(lines),
                    'timestamp': time.time(),
                    'last_access': time.time(),
                    'created': datetime.now().isoformat(),
                }
                
                self._save_metadata()
                
                # Enforce total cache size limit
                self._enforce_size_limit()
                
                return True
                
            except Exception as e:
                # Failed to save
                return False
    
    def clear_dataset_cache(self, dataset_path: str, config: Optional[str] = None, split: Optional[str] = None) -> int:
        """
        Clear all cached chunks for a specific dataset.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Optional config filter
            split: Optional split filter
            
        Returns:
            Number of chunks removed
        """
        with self._cache_lock:
            removed = 0
            keys_to_remove = []
            
            for key, info in self.metadata.items():
                if info.get('dataset_path') != dataset_path:
                    continue
                if config is not None and info.get('config') != config:
                    continue
                if split is not None and info.get('split') != split:
                    continue
                
                # Remove file
                cache_file = self.cache_dir / f"{key}.json"
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(key)
                    removed += 1
                except Exception:
                    pass
            
            # Update metadata
            for key in keys_to_remove:
                del self.metadata[key]
            
            if keys_to_remove:
                self._save_metadata()
            
            return removed
    
    def has_chunks_with_different_size(
        self,
        dataset_path: str,
        config: Optional[str],
        split: str,
        current_max_lines: int
    ) -> bool:
        """
        Check if there are cached chunks for this dataset with a different chunk size.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Dataset configuration name
            split: Dataset split
            current_max_lines: Current chunk size to compare against
            
        Returns:
            True if chunks with different max_lines exist
        """
        with self._cache_lock:
            dataset_key = self._get_dataset_key(dataset_path, config, split)
            
            for key, info in self.metadata.items():
                if info.get('dataset_key') != dataset_key:
                    continue
                
                # Extract max_lines from the cache key
                # Cache key format: dataset:config:split:chunk_index:max_lines
                try:
                    # Reconstruct the cache key to extract max_lines
                    stored_path = info.get('dataset_path', '')
                    stored_config = info.get('config')
                    stored_split = info.get('split', '')
                    stored_chunk_idx = info.get('chunk_index', 0)
                    
                    # Try all possible max_lines values to see which one matches this key
                    for test_lines in [100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]:
                        test_key = self._get_cache_key(stored_path, stored_config, stored_split, stored_chunk_idx, test_lines)
                        if test_key == key and test_lines != current_max_lines:
                            return True
                except Exception:
                    continue
            
            return False
    
    def clear_dataset_cache_except_size(
        self,
        dataset_path: str,
        config: Optional[str],
        split: str,
        keep_max_lines: int
    ) -> int:
        """
        Clear all cached chunks for a dataset except those with specified chunk size.
        
        Args:
            dataset_path: HuggingFace dataset path
            config: Dataset configuration name
            split: Dataset split
            keep_max_lines: Chunk size to keep (clear all others)
            
        Returns:
            Number of chunks removed
        """
        with self._cache_lock:
            removed = 0
            keys_to_remove = []
            dataset_key = self._get_dataset_key(dataset_path, config, split)
            
            for key, info in self.metadata.items():
                if info.get('dataset_key') != dataset_key:
                    continue
                
                # Check if this key matches the keep_max_lines size
                try:
                    stored_path = info.get('dataset_path', '')
                    stored_config = info.get('config')
                    stored_split = info.get('split', '')
                    stored_chunk_idx = info.get('chunk_index', 0)
                    
                    # Generate key for the size we want to keep
                    keep_key = self._get_cache_key(stored_path, stored_config, stored_split, stored_chunk_idx, keep_max_lines)
                    
                    # If this isn't the key we want to keep, remove it
                    if key != keep_key:
                        cache_file = self.cache_dir / f"{key}.json"
                        try:
                            if cache_file.exists():
                                cache_file.unlink()
                            keys_to_remove.append(key)
                            removed += 1
                        except Exception:
                            pass
                except Exception:
                    continue
            
            # Update metadata
            for key in keys_to_remove:
                del self.metadata[key]
            
            if keys_to_remove:
                self._save_metadata()
            
            return removed
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._cache_lock:
            total_chunks = len(self.metadata)
            
            # Group by dataset
            datasets: Dict[str, int] = {}
            for info in self.metadata.values():
                dataset_path = info.get('dataset_path', 'unknown')
                datasets[dataset_path] = datasets.get(dataset_path, 0) + 1
            
            # Calculate total cache size
            total_size_mb = 0.0
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_metadata.json":
                    try:
                        total_size_mb += cache_file.stat().st_size / (1024 * 1024)
                    except Exception:
                        pass
            
            return {
                'total_chunks': total_chunks,
                'datasets_cached': len(datasets),
                'chunks_per_dataset': datasets,
                # Do not round here so very small caches still register as > 0 in tests
                'cache_size_mb': total_size_mb,
                'cache_dir': str(self.cache_dir),
                'max_chunks_per_dataset': self.max_chunks_per_dataset,
                'max_size_mb': self.max_size_mb,
                'size_limit_status': f"{round(total_size_mb, 2)}/{self.max_size_mb} MB ({round(100 * total_size_mb / self.max_size_mb, 1)}%)",
            }
    
    def cleanup_old_caches(self, max_age_hours: float = 168.0) -> int:
        """
        Remove caches older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours (default: 168h = 7 days)
            
        Returns:
            Number of chunks removed
        """
        with self._cache_lock:
            removed = 0
            keys_to_remove = []
            current_time = time.time()
            
            for key, info in self.metadata.items():
                cached_time = info.get('timestamp', 0)
                age_hours = (current_time - cached_time) / 3600
                
                if age_hours > max_age_hours:
                    cache_file = self.cache_dir / f"{key}.json"
                    try:
                        if cache_file.exists():
                            cache_file.unlink()
                        keys_to_remove.append(key)
                        removed += 1
                    except Exception:
                        pass
            
            # Update metadata
            for key in keys_to_remove:
                del self.metadata[key]
            
            if keys_to_remove:
                self._save_metadata()
            
            return removed


def get_cache() -> StreamingChunkCache:
    """Get the global streaming chunk cache instance."""
    return StreamingChunkCache.get_instance()


# Example usage
if __name__ == "__main__":
    cache = get_cache()
    
    # Example: Cache a chunk
    test_lines = [f"This is line {i}" for i in range(1000)]
    cache.save_chunk("test/dataset", None, "train", 0, test_lines)
    
    # Retrieve it
    retrieved = cache.get_cached_chunk("test/dataset", None, "train", 0)
    print(f"Retrieved {len(retrieved) if retrieved else 0} lines")
    
    # Stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
