"""
Streaming dataset utilities for memory-efficient training with large datasets.

This module provides lazy loading and tokenization to avoid loading entire datasets into RAM/VRAM.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple, Any, Optional
import torch


class StreamingTextDataset:
    """
    Memory-efficient streaming dataset that tokenizes on-the-fly.
    
    Instead of loading all data into memory, this yields batches as needed,
    tokenizing just-in-time. This enables training on datasets much larger
    than available RAM.
    """
    
    def __init__(
        self,
        lines: List[str],
        tokenizer: Any,
        max_seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        epoch: int = 0,
        start_offset: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            lines: List of text lines (kept in RAM as strings - lightweight)
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length for tokenization
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle lines between epochs (False for linear progression)
            max_samples: Optional limit on number of samples (for testing)
            epoch: Epoch/cycle number for deterministic shuffling (default: 0)
            start_offset: Starting sample index for resuming training (default: 0)
        """
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.epoch = epoch
        self._internal_epoch = epoch  # Internal counter that auto-increments on each __iter__ call
        self._samples_yielded: List[int] = []  # Track which samples are used
        self.start_offset = start_offset  # Starting position for resume capability
        self._current_position = start_offset  # Track current position in dataset
        
        # Filter empty lines
        all_lines = [ln for ln in lines if ln and str(ln).strip()]
        
        # DDP: Shard data across ranks to avoid duplicate work
        if self.world_size > 1:
            # Each rank gets a subset of the data
            samples_per_rank = len(all_lines) // self.world_size
            rank_start = self.rank * samples_per_rank
            rank_end = rank_start + samples_per_rank if self.rank < self.world_size - 1 else len(all_lines)
            self.lines = all_lines[rank_start:rank_end]
        else:
            self.lines = all_lines
        
        if self.max_samples:
            self.lines = self.lines[:self.max_samples]
        
        self.num_samples = len(self.lines)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
    
    def __len__(self) -> int:
        """Return number of batches (not samples)."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches, tokenizing on-the-fly.
        
        For linear progression (shuffle=False):
        - Starts from start_offset/current_position
        - Processes samples sequentially
        - Wraps around to beginning after reaching end
        
        For shuffled mode (shuffle=True):
        - Applies shuffling with epoch-based seed
        - Still respects start_offset for initial position
        
        Yields:
            (input_ids, labels, puzzle_ids) for each batch
        """
        # Get indices
        indices = list(range(self.num_samples))
        
        # Shuffle if requested (with epoch-based seed for reproducible variety)
        if self.shuffle:
            import random
            # Use internal epoch counter + content hash as seed to ensure:
            # 1. Different shuffle each time __iter__ is called (auto-increments)
            # 2. Reproducible for same epoch + same content (debugging)
            # Use first few lines as content fingerprint instead of object id
            content_hash = hash(tuple(self.lines[:min(10, len(self.lines))]))
            seed = hash((content_hash, self._internal_epoch, len(self.lines))) % (2**32)
            rng = random.Random(seed)
            rng.shuffle(indices)
            
            # Auto-increment internal epoch for next iteration
            self._internal_epoch += 1
        
        # For linear progression, adjust indices to start from current position
        if not self.shuffle and self._current_position > 0:
            # Rotate indices to start from current position
            # This ensures linear progression: [pos, pos+1, ..., end, 0, 1, ..., pos-1]
            indices = indices[self._current_position:] + indices[:self._current_position]
            
        # Clear previous tracking
        self._samples_yielded.clear()
        
        # Yield batches
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            # Get batch lines
            batch_indices = indices[start_idx:end_idx]
            batch_lines = [self.lines[i] for i in batch_indices]
            
            # Track which samples we're using
            self._samples_yielded.extend(batch_indices)
            
            # Update current position (for linear progression tracking)
            if not self.shuffle:
                # In linear mode, advance position by batch size
                samples_in_batch = len(batch_indices)
                self._current_position = (self._current_position + samples_in_batch) % self.num_samples
            
            # Tokenize this batch only
            try:
                enc = self.tokenizer(
                    batch_lines,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_len,
                    return_tensors="pt",
                )
                
                input_ids = enc["input_ids"]
                labels = input_ids.clone()
                
                # Apply ignore_index for padding
                if self.tokenizer.pad_token_id is not None:
                    labels[enc["attention_mask"] == 0] = -100
                
                # Create puzzle identifiers (all zeros for now)
                batch_size_actual = input_ids.shape[0]
                puzzle_ids = torch.zeros(batch_size_actual, dtype=torch.long)
                
                yield input_ids, labels, puzzle_ids
                
            except Exception as e:
                # Skip problematic batches
                print(f"Warning: Skipping batch {batch_idx} due to error: {e}")
                continue
    
    def get_sample_stats(self) -> dict:
        """Get statistics about which samples were yielded.
        
        Returns:
            Dict with sample usage information for verification
        """
        unique_samples = len(set(self._samples_yielded))
        total_yielded = len(self._samples_yielded)
        
        return {
            "epoch": self.epoch,
            "internal_epoch": self._internal_epoch,  # Show actual iteration count
            "total_samples": self.num_samples,
            "unique_samples_used": unique_samples,
            "total_samples_yielded": total_yielded,
            "coverage_percent": round(100 * unique_samples / max(1, self.num_samples), 2),
            "current_position": self._current_position,
            "shuffle_mode": self.shuffle,
            "first_5_indices": self._samples_yielded[:5] if self._samples_yielded else [],
            "last_5_indices": self._samples_yielded[-5:] if len(self._samples_yielded) > 5 else [],
        }
    
    def get_position(self) -> int:
        """Get current position in the dataset.
        
        Returns:
            Current sample index (0 to num_samples-1)
        """
        return self._current_position
    
    def set_position(self, position: int) -> None:
        """Set current position in the dataset for resuming.
        
        Args:
            position: Sample index to resume from (0 to num_samples-1)
        """
        if not 0 <= position < self.num_samples:
            raise ValueError(f"Position {position} out of range [0, {self.num_samples})")
        self._current_position = position
        self.start_offset = position


def create_streaming_dataset(
    lines: List[str],
    tokenizer: Any,
    max_seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    epoch: int = 0,
    start_offset: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> StreamingTextDataset:
    """
    Factory function to create a streaming dataset.
    
    Args:
        lines: List of text lines
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle (False for linear progression)
        max_samples: Optional limit on samples
        epoch: Epoch/cycle number for shuffling variance
        start_offset: Starting sample index for resuming training
        rank: DDP rank (default: 0)
        world_size: DDP world size (default: 1)
        
    Returns:
        StreamingTextDataset instance
    """
    return StreamingTextDataset(
        lines=lines,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=shuffle,
        max_samples=max_samples,
        epoch=epoch,
        start_offset=start_offset,
        rank=rank,
        world_size=world_size,
    )


def estimate_dataset_memory(
    num_samples: int,
    max_seq_len: int,
    batch_size: int,
    eager_loading: bool = True,
) -> dict:
    """
    Estimate memory usage for dataset loading.
    
    Args:
        num_samples: Number of samples in dataset
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        eager_loading: If True, entire dataset loaded; if False, streaming
        
    Returns:
        Dict with memory estimates in GB
    """
    bytes_per_token = 4  # int32/int64
    
    if eager_loading:
        # All samples tokenized and in memory
        total_tokens = num_samples * max_seq_len
        input_ids_gb = (total_tokens * bytes_per_token) / (1024 ** 3)
        labels_gb = input_ids_gb  # Labels are same size
        total_gb = input_ids_gb + labels_gb
        
        return {
            "mode": "eager (all in memory)",
            "input_ids_gb": round(input_ids_gb, 2),
            "labels_gb": round(labels_gb, 2),
            "total_gb": round(total_gb, 2),
            "num_samples": num_samples,
            "max_seq_len": max_seq_len,
        }
    else:
        # Only one batch tokenized at a time
        batch_tokens = batch_size * max_seq_len
        input_ids_gb = (batch_tokens * bytes_per_token) / (1024 ** 3)
        labels_gb = input_ids_gb
        total_gb = input_ids_gb + labels_gb
        
        return {
            "mode": "streaming (one batch at a time)",
            "input_ids_gb": round(input_ids_gb, 4),
            "labels_gb": round(labels_gb, 4),
            "total_gb": round(total_gb, 4),
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "savings_vs_eager": f"{round((1 - (batch_size / num_samples)) * 100, 1)}%",
        }


if __name__ == "__main__":
    # Example: Compare memory usage
    print("Dataset Memory Estimation")
    print("=" * 60)
    
    scenarios = [
        ("Small dataset, 10K context", 1000, 10000, 2),
        ("Medium dataset, 10K context", 10000, 10000, 2),
        ("Large dataset (TinyStories-like), 10K context", 100000, 10000, 2),
    ]
    
    for name, num_samples, max_seq_len, batch_size in scenarios:
        print(f"\n{name}:")
        print(f"  Samples: {num_samples:,}, Seq len: {max_seq_len:,}, Batch: {batch_size}")
        
        eager = estimate_dataset_memory(num_samples, max_seq_len, batch_size, eager_loading=True)
        streaming = estimate_dataset_memory(num_samples, max_seq_len, batch_size, eager_loading=False)
        
        print(f"\n  Eager loading (current):")
        print(f"    Input IDs: {eager['input_ids_gb']} GB")
        print(f"    Labels: {eager['labels_gb']} GB")
        print(f"    TOTAL: {eager['total_gb']} GB")
        
        print(f"\n  Streaming loading (optimized):")
        print(f"    Input IDs: {streaming['input_ids_gb']} GB")
        print(f"    Labels: {streaming['labels_gb']} GB")
        print(f"    TOTAL: {streaming['total_gb']} GB")
        print(f"    Memory savings: {streaming['savings_vs_eager']}")
