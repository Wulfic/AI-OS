"""Chunk and Block Tracking System for Parallel Training.

This module provides a persistent tracking system for dataset chunks and blocks
to enable true parallel training speedup by ensuring:
1. Each GPU trains on unique chunks
2. No chunk is trained twice within an epoch
3. Progress persists across training sessions
4. Epoch completion properly resets tracking

Architecture:
- Block: 100k samples streamed from dataset (e.g., from HuggingFace)
- Chunk: dataset_chunk_size samples (default 4000) - subdivides blocks
- Each block contains ~25 chunks (100k / 4k)
- Chunks are identified as "block_id:chunk_id" strings
- Progress tracked in brain.json under last_session.chunk_progress

Parallel Training Strategy:
- Dataset divided into blocks of 100k samples
- Each block divided into chunks (configured via dataset_chunk_size)
- Each GPU assigned unique untrained chunks from current block
- After block completion, move to next block
- After all blocks trained, epoch completes and tracking resets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ChunkProgress:
    """Progress tracking for a single chunk."""
    
    block_id: int
    """Block index (0-based)."""
    
    chunk_id: int
    """Chunk index within block (0-based)."""
    
    start_sample: int
    """Starting sample index in dataset."""
    
    end_sample: int
    """Ending sample index in dataset (exclusive)."""
    
    trained: bool = False
    """Whether this chunk has been trained."""
    
    gpu_id: Optional[int] = None
    """Which GPU trained this chunk (if any)."""
    
    step_count: int = 0
    """Number of training steps completed on this chunk."""
    
    @property
    def chunk_key(self) -> str:
        """Unique identifier for this chunk: 'block_id:chunk_id'."""
        return f"{self.block_id}:{self.chunk_id}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "ChunkProgress":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class EpochProgress:
    """Progress tracking for current epoch."""
    
    current_epoch: int = 0
    """Current epoch number (0-based)."""
    
    total_samples: int = 0
    """Total samples in complete dataset."""
    
    samples_per_block: int = 100000
    """Samples per block (streaming chunk from HuggingFace)."""
    
    samples_per_chunk: int = 4000
    """Samples per training chunk (dataset_chunk_size)."""
    
    samples_processed: int = 0
    """Total samples processed in current epoch."""
    
    blocks_processed: Set[int] = None  # Will be initialized in __post_init__
    """Set of block IDs that have been completed in current epoch."""
    
    current_block_id: int = 0
    """Current block being processed."""
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.blocks_processed is None:
            self.blocks_processed = set()
    
    @property
    def total_blocks(self) -> int:
        """Calculate total number of blocks in dataset."""
        if self.total_samples <= 0:
            return 0
        return (self.total_samples + self.samples_per_block - 1) // self.samples_per_block
    
    @property
    def chunks_per_block(self) -> int:
        """Calculate chunks per block based on actual block size.
        
        For small datasets where total_samples < samples_per_block,
        we calculate based on actual dataset size.
        """
        # For small datasets, use actual size instead of block size
        if self.total_samples > 0 and self.total_samples < self.samples_per_block:
            effective_block_size = self.total_samples
        else:
            effective_block_size = self.samples_per_block
        
        return max(1, (effective_block_size + self.samples_per_chunk - 1) // self.samples_per_chunk)
    
    @property
    def total_chunks(self) -> int:
        """Calculate total chunks in dataset."""
        return self.total_blocks * self.chunks_per_block
    
    @property
    def epoch_complete(self) -> bool:
        """Check if current epoch is complete."""
        if self.total_blocks == 0:
            return False
        return len(self.blocks_processed) >= self.total_blocks
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert set to list for JSON
        data['blocks_processed'] = list(self.blocks_processed)
        return data
    
    @classmethod
    def from_dict(cls, d: dict) -> "EpochProgress":
        """Create from dictionary."""
        # Convert list back to set
        d = d.copy()
        if 'blocks_processed' in d:
            d['blocks_processed'] = set(d['blocks_processed'])
        return cls(**d)


class ChunkTracker:
    """Manages chunk/block tracking for parallel training.
    
    Responsibilities:
    - Track which chunks have been trained
    - Assign unique chunks to parallel GPUs
    - Persist progress to brain.json
    - Handle epoch completion and reset
    - Resume from saved state
    
    Example usage:
        tracker = ChunkTracker(
            brain_path=Path("artifacts/brains/actv1/my_brain"),
            samples_per_block=100000,
            samples_per_chunk=4000,
            total_samples=1000000
        )
        
        # Load previous progress (if any)
        tracker.load_progress()
        
        # Get chunks for parallel GPUs
        chunks = tracker.get_chunks_for_gpus(num_gpus=4)
        
        # Mark chunks as trained
        for chunk in chunks:
            tracker.mark_chunk_trained(chunk, gpu_id=0, steps=100)
        
        # Save progress
        tracker.save_progress()
    """
    
    def __init__(
        self,
        brain_path: Path,
        samples_per_block: int = 100000,
        samples_per_chunk: int = 4000,
        total_samples: Optional[int] = None,
    ):
        """Initialize chunk tracker.
        
        Args:
            brain_path: Path to brain directory (contains brain.json)
            samples_per_block: Samples per streaming block (default 100k)
            samples_per_chunk: Samples per training chunk (default 4k)
            total_samples: Total samples in dataset (None if unknown)
        """
        self.brain_path = Path(brain_path)
        self.brain_json_path = self.brain_path / "brain.json"
        
        # Initialize epoch progress
        self.epoch = EpochProgress(
            samples_per_block=samples_per_block,
            samples_per_chunk=samples_per_chunk,
            total_samples=total_samples or 0,
        )
        
        # Chunk tracking
        self.chunks: Dict[str, ChunkProgress] = {}  # chunk_key -> ChunkProgress
    
    def load_progress(self) -> bool:
        """Load progress from brain.json.
        
        Returns:
            True if progress was loaded, False if starting fresh.
        """
        if not self.brain_json_path.exists():
            return False
        
        try:
            with self.brain_json_path.open('r', encoding='utf-8') as f:
                brain_data = json.load(f)
            
            last_session = brain_data.get('last_session')
            if not last_session or not isinstance(last_session, dict):
                return False
            
            # Load epoch progress
            epoch_tracking = last_session.get('epoch_tracking', {})
            if epoch_tracking:
                self.epoch = EpochProgress.from_dict(epoch_tracking)
            
            # Load chunk progress
            chunk_progress = last_session.get('chunk_progress', {})
            if chunk_progress:
                for chunk_key, chunk_data in chunk_progress.items():
                    self.chunks[chunk_key] = ChunkProgress.from_dict(chunk_data)
            
            return True
            
        except Exception as e:
            print(f"[ChunkTracker] Warning: Failed to load progress: {e}")
            return False
    
    def save_progress(self) -> bool:
        """Save progress to brain.json.
        
        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.brain_json_path.exists():
            # Create brain.json if it doesn't exist
            self.brain_path.mkdir(parents=True, exist_ok=True)
            brain_data = {
                "name": self.brain_path.name,
                "type": "actv1",
                "last_session": {}
            }
        else:
            try:
                with self.brain_json_path.open('r', encoding='utf-8') as f:
                    brain_data = json.load(f)
            except Exception:
                brain_data = {"last_session": {}}
        
        # Ensure last_session exists
        if 'last_session' not in brain_data:
            brain_data['last_session'] = {}
        
        # Save epoch progress
        brain_data['last_session']['epoch_tracking'] = self.epoch.to_dict()
        
        # Save chunk progress
        chunk_progress = {
            chunk_key: chunk.to_dict()
            for chunk_key, chunk in self.chunks.items()
        }
        brain_data['last_session']['chunk_progress'] = chunk_progress
        
        # Write atomically
        try:
            tmp_path = self.brain_json_path.with_suffix('.json.tmp')
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(brain_data, f, indent=2)
            tmp_path.replace(self.brain_json_path)
            return True
        except Exception as e:
            print(f"[ChunkTracker] Error: Failed to save progress: {e}")
            return False
    
    def get_chunks_for_gpus(
        self,
        num_gpus: int,
        block_id: Optional[int] = None
    ) -> List[ChunkProgress]:
        """Get unique untrained chunks for parallel GPUs.
        
        Args:
            num_gpus: Number of GPUs to assign chunks to
            block_id: Specific block to get chunks from (None = current block)
        
        Returns:
            List of ChunkProgress objects (one per GPU).
            Returns empty list if no untrained chunks available.
        """
        if block_id is None:
            block_id = self.epoch.current_block_id
        
        # Get all chunks for this block
        block_chunks = self._get_or_create_block_chunks(block_id)
        
        # Filter untrained chunks and ensure they're not empty
        untrained = [
            c for c in block_chunks 
            if not c.trained and (c.end_sample - c.start_sample) > 0
        ]
        
        # Return up to num_gpus chunks
        return untrained[:num_gpus]
    
    def _get_or_create_block_chunks(self, block_id: int) -> List[ChunkProgress]:
        """Get or create chunks for a specific block.
        
        Args:
            block_id: Block index
        
        Returns:
            List of ChunkProgress objects for this block.
        """
        chunks = []
        
        # Calculate block boundaries
        block_start = block_id * self.epoch.samples_per_block
        block_end = min((block_id + 1) * self.epoch.samples_per_block, self.epoch.total_samples)
        block_size = block_end - block_start
        
        # Calculate actual chunks needed for this block
        chunks_needed = max(1, (block_size + self.epoch.samples_per_chunk - 1) // self.epoch.samples_per_chunk)
        
        for chunk_id in range(chunks_needed):
            chunk_key = f"{block_id}:{chunk_id}"
            
            if chunk_key in self.chunks:
                chunks.append(self.chunks[chunk_key])
            else:
                # Create new chunk with proper bounds
                start_sample = block_start + (chunk_id * self.epoch.samples_per_chunk)
                end_sample = min(
                    start_sample + self.epoch.samples_per_chunk,
                    block_end,
                    self.epoch.total_samples
                )
                
                # Skip if this would be an empty chunk
                if start_sample >= end_sample:
                    continue
                
                chunk = ChunkProgress(
                    block_id=block_id,
                    chunk_id=chunk_id,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    trained=False
                )
                self.chunks[chunk_key] = chunk
                chunks.append(chunk)
        
        return chunks
    
    def mark_chunk_trained(
        self,
        chunk: ChunkProgress,
        gpu_id: int,
        steps: int
    ) -> None:
        """Mark a chunk as trained.
        
        Args:
            chunk: Chunk that was trained
            gpu_id: GPU that trained this chunk
            steps: Number of training steps completed
        """
        chunk.trained = True
        chunk.gpu_id = gpu_id
        chunk.step_count = steps
        
        # Update samples processed
        chunk_samples = chunk.end_sample - chunk.start_sample
        self.epoch.samples_processed += chunk_samples
    
    def mark_block_complete(self, block_id: int) -> None:
        """Mark a block as complete.
        
        Args:
            block_id: Block that was completed
        """
        self.epoch.blocks_processed.add(block_id)
        
        # Move to next block if current block is complete
        if block_id == self.epoch.current_block_id:
            self.epoch.current_block_id += 1
    
    def check_epoch_complete(self) -> bool:
        """Check if current epoch is complete.
        
        Returns:
            True if all blocks in dataset have been trained.
        """
        return self.epoch.epoch_complete
    
    def reset_epoch(self) -> None:
        """Reset tracking for new epoch.
        
        Clears all chunk training status but preserves epoch metadata.
        """
        self.epoch.current_epoch += 1
        self.epoch.samples_processed = 0
        self.epoch.blocks_processed = set()
        self.epoch.current_block_id = 0
        
        # Reset all chunk training status
        for chunk in self.chunks.values():
            chunk.trained = False
            chunk.gpu_id = None
            chunk.step_count = 0
    
    def get_progress_summary(self) -> dict:
        """Get summary of training progress.
        
        Returns:
            Dictionary with progress statistics.
        """
        total_chunks = len(self.chunks)
        trained_chunks = sum(1 for c in self.chunks.values() if c.trained)
        
        return {
            'current_epoch': self.epoch.current_epoch,
            'current_block': self.epoch.current_block_id,
            'total_blocks': self.epoch.total_blocks,
            'blocks_complete': len(self.epoch.blocks_processed),
            'samples_processed': self.epoch.samples_processed,
            'total_samples': self.epoch.total_samples,
            'chunks_trained': trained_chunks,
            'chunks_total': total_chunks,
            'epoch_progress_pct': (
                self.epoch.samples_processed / self.epoch.total_samples * 100
                if self.epoch.total_samples > 0 else 0
            ),
            'epoch_complete': self.epoch.epoch_complete,
        }
    
    def get_next_block_id(self) -> Optional[int]:
        """Get next block ID that has untrained chunks.
        
        Returns:
            Next block ID, or None if no untrained blocks remain.
        """
        # Check current block first
        if self.epoch.current_block_id < self.epoch.total_blocks:
            block_chunks = self._get_or_create_block_chunks(self.epoch.current_block_id)
            if any(not c.trained for c in block_chunks):
                return self.epoch.current_block_id
        
        # Search remaining blocks
        for block_id in range(self.epoch.current_block_id + 1, self.epoch.total_blocks):
            if block_id not in self.epoch.blocks_processed:
                return block_id
        
        return None  # All blocks trained
    
    def estimate_remaining_time(
        self,
        steps_per_second: float
    ) -> Tuple[float, str]:
        """Estimate remaining training time for current epoch.
        
        Args:
            steps_per_second: Average training throughput
        
        Returns:
            Tuple of (seconds_remaining, human_readable_string)
        """
        if self.epoch.total_samples <= 0 or steps_per_second <= 0:
            return (0.0, "Unknown")
        
        samples_remaining = self.epoch.total_samples - self.epoch.samples_processed
        
        # Rough estimate: assume 1 step per sample (actual may vary)
        steps_remaining = samples_remaining
        seconds_remaining = steps_remaining / steps_per_second
        
        # Format human-readable
        if seconds_remaining < 60:
            time_str = f"{seconds_remaining:.0f}s"
        elif seconds_remaining < 3600:
            minutes = seconds_remaining / 60
            time_str = f"{minutes:.1f}m"
        else:
            hours = seconds_remaining / 3600
            time_str = f"{hours:.1f}h"
        
        return (seconds_remaining, time_str)
