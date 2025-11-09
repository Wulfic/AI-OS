"""Block Manager for Dataset Streaming and Distribution.

Handles downloading HuggingFace datasets in blocks (e.g., 100k samples),
metadata prefetching to detect last block, and distributing chunks to GPUs.

Key optimizations:
- Only loads chunks (e.g., 100 samples) on demand, not full blocks (100k samples)
- Prefetches metadata only (sample count, is_last) for next block, not data
- Aggressive memory cleanup to prevent OOM
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class DataBlock:
    """Represents metadata about a block of dataset samples.
    
    Samples are NOT stored here - they're loaded per-chunk on demand
    to minimize memory usage (100 samples per chunk vs 100k per block).
    """
    
    block_id: int
    """Sequential block identifier."""
    
    total_samples: int
    """Total number of samples in this block."""
    
    is_last_block: bool = False
    """Whether this is the last block in the dataset."""
    
    def chunk_count(self, chunk_size: int) -> int:
        """Calculate number of chunks in this block."""
        return (self.total_samples + chunk_size - 1) // chunk_size


class BlockManager:
    """Manages dataset blocks for parallel GPU training.
    
    Features:
    - Downloads HuggingFace datasets in blocks (default 100k samples)
    - Prefetches metadata for next block to detect dataset end (no data loading)
    - Thread-safe block access for parallel GPUs
    - Loads only requested chunks (e.g., 100 samples) on demand
    - Supports both HF streaming and local file datasets
    """
    
    def __init__(
        self,
        dataset_path: str,
        samples_per_block: int = 100000,
        dataset_chunk_size: int = 4000,
        ascii_only: bool = False,
        read_text_lines_sample_any: Optional[Callable] = None,
        enable_prefetch: bool = True,
    ):
        """Initialize BlockManager.
        
        Args:
            dataset_path: Path to dataset (local file, directory, or hf://... URL)
            samples_per_block: Number of samples per block (for HF datasets)
            dataset_chunk_size: Chunk size for training (within a block)
            ascii_only: Filter to ASCII-only text
            read_text_lines_sample_any: Function to read dataset lines
            enable_prefetch: Enable metadata prefetching for next block (lightweight, no data loaded)
        """
        self.dataset_path = dataset_path
        self.samples_per_block = samples_per_block
        self.dataset_chunk_size = dataset_chunk_size
        self.ascii_only = ascii_only
        self.enable_prefetch = enable_prefetch
        
        # Import here to avoid circular dependency
        if read_text_lines_sample_any is None:
            from aios.data.datasets import read_text_lines_sample_any as read_fn
            self.read_fn = read_fn
        else:
            self.read_fn = read_text_lines_sample_any
        
        # Block metadata storage (lightweight - no sample data)
        self.blocks: dict[int, DataBlock] = {}
        self.lock = threading.Lock()
        
        # Chunk data cache: {(block_id, chunk_id): List[str]}
        # Only stores actually-used chunks (100 samples each) instead of full blocks (100k)
        self.chunk_cache: dict[Tuple[int, int], List[str]] = {}
        self._chunk_cache_lock = threading.Lock()
        
        # Per-block locks for preventing duplicate loads
        self._block_locks: dict[int, threading.Lock] = {}
        self._block_locks_lock = threading.Lock()
        
        # HuggingFace streaming state
        self._hf_iterator = None
        self._hf_current_position = 0
        self._hf_text_column = None
        
        # Initialize StreamingCache for persistent disk caching
        from aios.data.streaming_cache import StreamingChunkCache
        self._cache = StreamingChunkCache.get_instance()
        
        # Dataset state
        self.current_block_id = 0
        self.is_dataset_exhausted = False
        self.total_blocks_detected: Optional[int] = None
        
        # Pre-fetch next block to detect end
        self._prefetch_block_id: Optional[int] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        
        # Detect total blocks at initialization for local datasets
        self._detect_total_blocks_at_init()
    
    def get_chunk(self, block_id: int, chunk_id: int, chunk_size: int) -> Optional[List[str]]:
        """Get a specific chunk from a block, loading only that chunk into memory.
        
        This is the key memory optimization: instead of loading 100k samples (full block),
        we only load X amount of samples (one chunk) at a time.
        
        Args:
            block_id: Block ID
            chunk_id: Chunk ID within the block
            chunk_size: Number of samples per chunk
            
        Returns:
            List of text samples for this chunk, or None if beyond dataset end
        """
        chunk_key = (block_id, chunk_id)
        
        # Check chunk cache first
        with self._chunk_cache_lock:
            if chunk_key in self.chunk_cache:
                try:
                    cached_chunk = self.chunk_cache[chunk_key]
                    print(f"[BlockManager] Using cached chunk for Block {block_id}, Chunk {chunk_id} ({len(cached_chunk)} samples)")
                    return cached_chunk
                except Exception:
                    return self.chunk_cache[chunk_key]
        
        # Get block metadata (lightweight)
        block = self.get_block(block_id)
        if block is None:
            return None
        
        # Check if chunk_id is valid for this block
        if chunk_id >= block.chunk_count(chunk_size):
            return None
        
        # Load this specific chunk from persistent cache or HF
        chunk_samples = self._load_chunk(block_id, chunk_id, chunk_size)
        
        if chunk_samples is not None:
            # Cache this chunk in memory for potential reuse
            with self._chunk_cache_lock:
                self.chunk_cache[chunk_key] = chunk_samples
                
                # Limit chunk cache size (keep last 10 chunks = ~5 per GPU for 2 GPUs)
                if len(self.chunk_cache) > 10:
                    # Remove oldest chunk (simple FIFO)
                    oldest_key = next(iter(self.chunk_cache))
                    del self.chunk_cache[oldest_key]
        
        return chunk_samples
    
    def get_block(self, block_id: int, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[DataBlock]:
        """Get a specific block, loading it if necessary.
        
        Args:
            block_id: Block ID to retrieve
            progress_callback: Optional callback for progress updates (non-blocking)
            
        Returns:
            DataBlock if available, None if beyond dataset end
        """
        # Check cache first (quick check without full lock)
        with self.lock:
            if block_id in self.blocks:
                if progress_callback:
                    progress_callback(f"[BlockManager] Block {block_id} already loaded")
                return self.blocks[block_id]
            
            # Check if we know this is beyond dataset end
            if self.is_dataset_exhausted and (
                self.total_blocks_detected is not None and block_id >= self.total_blocks_detected
            ):
                return None
        
        # Get or create a lock for this specific block
        with self._block_locks_lock:
            if block_id not in self._block_locks:
                self._block_locks[block_id] = threading.Lock()
            block_lock = self._block_locks[block_id]
        
        # Acquire block-specific lock to prevent duplicate loads
        with block_lock:
            # Double-check cache (another thread might have loaded it)
            with self.lock:
                if block_id in self.blocks:
                    if progress_callback:
                        progress_callback(f"[BlockManager] Block {block_id} loaded by another thread")
                    return self.blocks[block_id]
            
            if progress_callback:
                progress_callback(f"[BlockManager] Loading Block {block_id}...")
            
            # Load the block (outside main lock to allow other operations)
            block = self._load_block(block_id, progress_callback)
            
            if block is None:
                return None
            
            # Cache the block metadata
            with self.lock:
                self.blocks[block_id] = block
                
                # Aggressive cleanup: Only keep current block and optionally next block metadata
                blocks_to_remove = []
                for cached_id in list(self.blocks.keys()):
                    if cached_id == block_id:
                        continue
                    if self.enable_prefetch and cached_id == block_id + 1:
                        continue
                    blocks_to_remove.append(cached_id)
                
                for old_id in blocks_to_remove:
                    del self.blocks[old_id]
                
                # Prefetch metadata for next block (lightweight - no data loading)
                if self.enable_prefetch and self._prefetch_block_id != block_id + 1:
                    self._start_prefetch_metadata(block_id + 1)
            
            return block
    
    def _load_block(self, block_id: int, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[DataBlock]:
        """Load a specific block from the dataset.
        
        Args:
            block_id: Block ID to load
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataBlock if successful, None if no more data
        """
        # Determine if this is an HF streaming dataset
        is_hf_streaming = isinstance(self.dataset_path, str) and self.dataset_path.startswith("hf://")
        
        if is_hf_streaming:
            if progress_callback:
                progress_callback(f"[BlockManager] Streaming Block {block_id} from HuggingFace...")
            return self._load_hf_block(block_id)
        else:
            if progress_callback:
                progress_callback(f"[BlockManager] Loading Block {block_id} from local files...")
            return self._load_local_block(block_id)
    
    def _load_chunk(self, block_id: int, chunk_id: int, chunk_size: int) -> Optional[List[str]]:
        """Load a specific chunk from a block.
        
        This loads ONLY the requested chunk (e.g., 100 samples) instead of
        the full block (100k samples), providing 1000x memory reduction.
        
        Args:
            block_id: Block ID
            chunk_id: Chunk ID within block
            chunk_size: Number of samples per chunk
            
        Returns:
            List of text samples for this chunk
        """
        # Determine if this is an HF streaming dataset
        is_hf_streaming = isinstance(self.dataset_path, str) and self.dataset_path.startswith("hf://")
        
        if is_hf_streaming:
            # For HF datasets, load from persistent cache
            hf_path = self.dataset_path[5:]
            parts = hf_path.split(":")
            dataset_path = parts[0]
            config = parts[1] if len(parts) > 1 else None
            split = parts[2] if len(parts) > 2 else "train"
            
            # Get cached block data
            cached_block = self._cache.get_cached_chunk(
                dataset_path=dataset_path,
                config=config,
                split=split,
                chunk_index=block_id,
                max_age_hours=168.0,
                max_lines=self.samples_per_block
            )
            
            if cached_block:
                # Extract just our chunk from the cached block
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(cached_block))
                chunk_samples = cached_block[start_idx:end_idx]
                print(f"[BlockManager] Loaded Block {block_id} Chunk {chunk_id}: {len(chunk_samples)} samples")
                return chunk_samples
            else:
                print(f"[BlockManager] Warning: Block {block_id} not in cache, cannot load chunk {chunk_id}")
                return None
        else:
            # For local datasets with block structure, check block-specific cache first
            if hasattr(self, '_block_samples_cache') and block_id in self._block_samples_cache:
                all_samples = self._block_samples_cache[block_id]
                # Calculate chunk within this block's samples
                start_idx = chunk_id * chunk_size
                end_idx = min(start_idx + chunk_size, len(all_samples))
                
                if start_idx >= len(all_samples):
                    return None
                
                chunk_samples = all_samples[start_idx:end_idx]
                print(f"[BlockManager] Loaded Block {block_id} Chunk {chunk_id}: {len(chunk_samples)} samples (from block cache)")
                return chunk_samples
            
            # Fall back to loading from all samples cache (non-block-structured datasets)
            all_samples = getattr(self, '_all_local_samples', [])
            start_idx = (block_id * self.samples_per_block) + (chunk_id * chunk_size)
            end_idx = min(start_idx + chunk_size, len(all_samples))
            
            if start_idx >= len(all_samples):
                return None
            
            chunk_samples = all_samples[start_idx:end_idx]
            return chunk_samples
    
    def _load_hf_block(self, block_id: int) -> Optional[DataBlock]:
        """Load a block from HuggingFace streaming dataset.
        
        Args:
            block_id: Block ID to load
            
        Returns:
            DataBlock if successful, None if no more data
        """
        import time
        start_time = time.time()
        
        # Parse dataset info for cache lookup
        hf_path = self.dataset_path[5:]  # Remove 'hf://' prefix
        parts = hf_path.split(":")
        dataset_path = parts[0]
        config = parts[1] if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 else "train"
        
        # Try to load from cache first
        cached_samples = self._cache.get_cached_chunk(
            dataset_path=dataset_path,
            config=config,
            split=split,
            chunk_index=block_id,
            max_age_hours=168.0,  # 7 days
            max_lines=self.samples_per_block
        )
        
        if cached_samples is not None:
            elapsed = time.time() - start_time
            # Check if this was a partial block (last block)
            is_last = len(cached_samples) < self.samples_per_block
            
            print(f"[BlockManager] Block {block_id} metadata loaded from cache: {len(cached_samples):,} samples in {elapsed:.1f}s")
            
            # Store metadata only - actual samples loaded per-chunk on demand
            block = DataBlock(
                block_id=block_id,
                total_samples=len(cached_samples),
                is_last_block=is_last
            )
            
            if is_last:
                with self.lock:
                    self.is_dataset_exhausted = True
                    self.total_blocks_detected = block_id + 1
            
            return block
        
        # Cache miss - download from HuggingFace
        print(f"[BlockManager] Downloading Block {block_id} from HuggingFace...")
        
        try:
            from datasets import load_dataset
            
            # Initialize streaming iterator on first use
            if self._hf_iterator is None:
                hf_path = self.dataset_path[5:]
                parts = hf_path.split(":")
                
                dataset_path = parts[0]
                config = parts[1] if len(parts) > 1 else None
                split = parts[2] if len(parts) > 2 else "train"
                
                # Load with streaming
                dataset = load_dataset(
                    dataset_path,
                    name=config,
                    split=split,
                    streaming=True
                )
                
                self._hf_iterator = iter(dataset)
                self._hf_current_position = 0
            
            # Skip to the start of this block if needed
            target_position = block_id * self.samples_per_block
            
            if target_position < self._hf_current_position:
                # Need to restart iterator
                hf_path = self.dataset_path[5:]
                parts = hf_path.split(":")
                dataset_path = parts[0]
                config = parts[1] if len(parts) > 1 else None
                split = parts[2] if len(parts) > 2 else "train"
                
                dataset = load_dataset(dataset_path, name=config, split=split, streaming=True)
                self._hf_iterator = iter(dataset)
                self._hf_current_position = 0
            
            # Skip to target position
            skip_count = target_position - self._hf_current_position
            if skip_count > 0:
                for _ in range(skip_count):
                    try:
                        next(self._hf_iterator)
                        self._hf_current_position += 1
                    except StopIteration:
                        with self.lock:
                            self.is_dataset_exhausted = True
                            self.total_blocks_detected = block_id
                        return None
            
            # Try common text column names
            text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
            
            # Collect items for this block
            samples = []
            for _ in range(self.samples_per_block):
                try:
                    item = next(self._hf_iterator)
                    self._hf_current_position += 1
                    
                    # Find text column on first item
                    if self._hf_text_column is None and isinstance(item, dict):
                        for col in text_columns:
                            if col in item and item[col]:
                                self._hf_text_column = col
                                print(f"[BlockManager] Using text column: '{col}'")
                                break
                        if self._hf_text_column is None:
                            for key, value in item.items():
                                if isinstance(value, str) and value.strip():
                                    self._hf_text_column = key
                                    print(f"[BlockManager] Using text column: '{key}'")
                                    break
                    
                    # Extract text
                    if self._hf_text_column:
                        text = item.get(self._hf_text_column, "") if isinstance(item, dict) else item[self._hf_text_column]
                        if text and str(text).strip():
                            if not self.ascii_only or self._is_ascii(str(text)):
                                samples.append(str(text).strip())
                    
                except StopIteration:
                    # End of dataset
                    break
                except Exception as e:
                    print(f"[BlockManager] Warning: Error processing item: {e}")
                    continue
            
            # Check if this is the last block
            is_last = len(samples) < self.samples_per_block
            
            # Save to cache for future use
            if len(samples) > 0:
                self._cache.save_chunk(
                    dataset_path=dataset_path,
                    config=config,
                    split=split,
                    chunk_index=block_id,
                    lines=samples,
                    max_lines=self.samples_per_block
                )
            
            elapsed = time.time() - start_time
            print(f"[BlockManager] Block {block_id} downloaded: {len(samples):,} samples in {elapsed:.1f}s{' (last block)' if is_last else ''}")
            
            if len(samples) == 0:
                # No data in this block - beyond dataset end
                with self.lock:
                    self.is_dataset_exhausted = True
                    self.total_blocks_detected = block_id
                return None
            
            # Store metadata only - actual samples loaded per-chunk on demand
            block = DataBlock(
                block_id=block_id,
                total_samples=len(samples),
                is_last_block=is_last
            )
            
            if is_last:
                with self.lock:
                    self.is_dataset_exhausted = True
                    self.total_blocks_detected = block_id + 1
            
            return block
            
        except Exception as e:
            print(f"[BlockManager] Error loading HF block {block_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_local_block(self, block_id: int) -> Optional[DataBlock]:
        """Load a block from local file/directory dataset.
        
        For local files, we treat the entire file as blocks of samples_per_block size.
        For directories with block structure (block_0, block_1, etc.), load from those.
        
        Args:
            block_id: Block ID to load
            
        Returns:
            DataBlock if successful, None if no more data
        """
        try:
            from pathlib import Path
            dataset_path = Path(self.dataset_path)
            
            # Check if this is a directory with pre-processed block structure
            if dataset_path.is_dir():
                block_dir = dataset_path / f"block_{block_id}"
                samples_file = block_dir / "samples.txt"
                
                if block_dir.is_dir() and samples_file.exists():
                    # Load from pre-processed block directory
                    print(f"[BlockManager] Loading Block {block_id} from pre-processed directory: {block_dir}")
                    
                    samples = []
                    with open(samples_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            
                            # Apply ASCII filter if needed
                            if self.ascii_only and not self._is_ascii(stripped):
                                continue
                            
                            samples.append(stripped)
                    
                    # Store samples for chunk loading
                    # For block-structured datasets, store in a block-specific cache
                    if not hasattr(self, '_block_samples_cache'):
                        self._block_samples_cache = {}
                    self._block_samples_cache[block_id] = samples
                    
                    # Check if this is the last block (next block doesn't exist)
                    next_block_dir = dataset_path / f"block_{block_id + 1}"
                    is_last = not (next_block_dir.is_dir() and (next_block_dir / "samples.txt").exists())
                    
                    # Create metadata block
                    block = DataBlock(
                        block_id=block_id,
                        total_samples=len(samples),
                        is_last_block=is_last
                    )
                    
                    if is_last:
                        with self.lock:
                            self.is_dataset_exhausted = True
                            if self.total_blocks_detected is None:
                                self.total_blocks_detected = block_id + 1
                    
                    print(f"[BlockManager] Loaded Block {block_id}: {len(samples):,} samples{' (last block)' if is_last else ''}")
                    return block
            
            # Fall back to traditional loading for single files or non-block-structured directories
            max_lines = self.samples_per_block
            
            # Load all lines on first block, then slice appropriately
            # This is simpler for local files since they're typically smaller
            if block_id == 0 or len(self.blocks) == 0:
                # Load initial samples
                all_samples = self.read_fn(
                    self.dataset_path,
                    max_lines=max_lines * 10,  # Load more to have multiple blocks
                    cycle=0
                )
                
                # Filter ASCII if needed
                if self.ascii_only:
                    all_samples = [s for s in all_samples if self._is_ascii(s)]
                
                # Cache total size for future blocks
                self._all_local_samples = all_samples
                
                # Calculate total blocks now that we know sample count
                if len(all_samples) > 0:
                    total_blocks = (len(all_samples) + self.samples_per_block - 1) // self.samples_per_block
                    with self.lock:
                        if self.total_blocks_detected is None:
                            self.total_blocks_detected = total_blocks
                    print(f"[BlockManager] Detected {total_blocks} blocks from local file ({len(all_samples):,} samples)")
            else:
                # Use cached samples
                all_samples = getattr(self, '_all_local_samples', [])
            
            # Calculate block boundaries
            start_idx = block_id * self.samples_per_block
            end_idx = start_idx + self.samples_per_block
            
            if start_idx >= len(all_samples):
                # Beyond dataset end
                with self.lock:
                    self.is_dataset_exhausted = True
                    if self.total_blocks_detected is None:
                        self.total_blocks_detected = block_id
                return None
            
            block_samples = all_samples[start_idx:end_idx]
            is_last = end_idx >= len(all_samples)
            
            if len(block_samples) == 0:
                with self.lock:
                    self.is_dataset_exhausted = True
                    if self.total_blocks_detected is None:
                        self.total_blocks_detected = block_id
                return None
            
            # Store metadata only - actual samples loaded per-chunk on demand
            block = DataBlock(
                block_id=block_id,
                total_samples=len(block_samples),
                is_last_block=is_last
            )
            
            if is_last:
                with self.lock:
                    self.is_dataset_exhausted = True
                    if self.total_blocks_detected is None:
                        self.total_blocks_detected = block_id + 1
            
            return block
            
        except Exception as e:
            print(f"[BlockManager] Error loading local block {block_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _start_prefetch_metadata(self, block_id: int) -> None:
        """Start prefetching metadata for next block in the background.
        
        This is lightweight - only loads block metadata (sample count, is_last),
        NOT the actual sample data. Data is loaded per-chunk on demand.
        
        Args:
            block_id: Block ID to prefetch metadata for
        """
        # Only prefetch if not already in cache and not already prefetching this block
        if block_id in self.blocks or self._prefetch_block_id == block_id:
            return
        
        # Stop any existing prefetch thread
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            # Let it finish naturally
            pass
        
        self._prefetch_block_id = block_id
        
        def prefetch_worker():
            block = self._load_block(block_id)
            if block:
                with self.lock:
                    self.blocks[block_id] = block
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def free_block(self, block_id: int) -> None:
        """Explicitly free a block metadata from memory.
        
        Args:
            block_id: Block ID to free
        """
        with self.lock:
            if block_id in self.blocks:
                del self.blocks[block_id]
    
    def get_next_block(self) -> Optional[DataBlock]:
        """Get the next sequential block.
        
        Returns:
            Next DataBlock, or None if dataset exhausted
        """
        with self.lock:
            block_id = self.current_block_id
            self.current_block_id += 1
        
        return self.get_block(block_id)
    
    def reset(self) -> None:
        """Reset to beginning of dataset (for new epoch in iterate mode)."""
        with self.lock:
            self.current_block_id = 0
            # Keep blocks cached for efficiency
    
    def is_last_block(self, block_id: int) -> bool:
        """Check if a block is the last in the dataset.
        
        Args:
            block_id: Block ID to check
            
        Returns:
            True if this is the last block
        """
        block = self.get_block(block_id)
        return block.is_last_block if block else False
    
    def get_total_blocks(self) -> Optional[int]:
        """Get total number of blocks in dataset.
        
        Returns:
            Total blocks if known, None if not yet detected
        """
        with self.lock:
            return self.total_blocks_detected
    
    def _detect_total_blocks_at_init(self) -> None:
        """Detect total blocks for local datasets at initialization.
        
        This allows progress tracking to show total blocks from the start.
        For HF streaming datasets, blocks are detected lazily as we download.
        """
        # Skip HF streaming datasets (blocks detected during download)
        if isinstance(self.dataset_path, str) and self.dataset_path.startswith("hf://"):
            return
        
        try:
            from pathlib import Path
            dataset_path = Path(self.dataset_path)
            
            # Check if this is a directory with pre-processed blocks
            if dataset_path.is_dir():
                total_blocks = self._count_blocks_in_directory(dataset_path)
                if total_blocks is not None and total_blocks > 0:
                    with self.lock:
                        self.total_blocks_detected = total_blocks
                    print(f"[BlockManager] Detected {total_blocks} pre-processed blocks in {dataset_path}")
                    return
                
                # Check if directory has dataset_info.json with block information
                info_file = dataset_path / "dataset_info.json"
                if info_file.exists():
                    import json
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    
                    total_blocks = info.get("total_blocks")
                    if total_blocks is not None:
                        with self.lock:
                            self.total_blocks_detected = total_blocks
                        print(f"[BlockManager] Detected {total_blocks} blocks from dataset_info.json")
                        return
                
                # For directories without blocks, load all samples and calculate blocks
                # We'll do this lazily when first block is requested to avoid startup delay
                pass
            
            # For single files, calculate blocks based on file size estimate
            elif dataset_path.is_file():
                # We'll detect blocks lazily when loading
                pass
                
        except Exception as e:
            print(f"[BlockManager] Warning: Could not detect total blocks at init: {e}")
    
    def _count_blocks_in_directory(self, directory: Path) -> Optional[int]:
        """Count pre-processed block directories (block_0, block_1, etc.).
        
        Args:
            directory: Path to dataset directory
            
        Returns:
            Number of blocks found, or None if no block structure detected
        """
        try:
            # Look for block_0, block_1, block_2, ... pattern
            block_count = 0
            while True:
                block_dir = directory / f"block_{block_count}"
                # Check for block directory with samples file
                if not block_dir.is_dir():
                    break
                
                # Verify block has data
                samples_file = block_dir / "samples.txt"
                if not samples_file.exists():
                    break
                
                block_count += 1
            
            # Return count only if we found at least one block
            return block_count if block_count > 0 else None
            
        except Exception as e:
            print(f"[BlockManager] Warning: Error counting blocks in {directory}: {e}")
            return None
    
    @staticmethod
    def _is_ascii(s: str) -> bool:
        """Check if string is ASCII-only."""
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False
