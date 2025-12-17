"""Chunk Tracker for Training Progress Management.

Tracks which chunks from which blocks have been trained at which steps.
Ensures no duplicate training and enables resume capability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


CLAIM_TIMEOUT_SECONDS = 600


def sanitize_dataset_name(dataset_path: str) -> str:
    """Create a safe filename from a dataset path.
    
    Args:
        dataset_path: Full path or URL to dataset
        
    Returns:
        Sanitized name suitable for use in filenames
    """
    if not dataset_path:
        return "unknown"
    
    # Extract just the name from the path
    from pathlib import Path as _Path
    
    if dataset_path.startswith("hf://"):
        # HuggingFace dataset: hf://owner/name:config:split -> owner_name
        name = dataset_path[5:].split(":")[0].replace("/", "_")
    else:
        # Local path: extract directory/file name
        name = _Path(dataset_path).stem
    
    # Sanitize: keep only alphanumeric, underscore, hyphen
    name = re.sub(r'[^\w\-]', '_', name)
    
    # Limit length and remove consecutive underscores
    name = re.sub(r'_+', '_', name).strip('_')[:50]
    
    return name or "unknown"


def get_dataset_state_file(brain_dir: Path, dataset_name: str, chunk_size: Optional[int] = None) -> Path:
    """Get the dataset-specific chunk tracker state file path.
    
    Args:
        brain_dir: Brain directory path
        dataset_name: Sanitized dataset name
        chunk_size: Optional chunk size for variation tracking
        
    Returns:
        Path to dataset-specific state file
    """
    # Create datasets subdirectory
    datasets_dir = brain_dir / "datasets"
    
    if chunk_size:
        state_file = datasets_dir / f"{dataset_name}_cs{chunk_size}" / "chunk_tracker_state.json"
    else:
        state_file = datasets_dir / dataset_name / "chunk_tracker_state.json"
    
    return state_file


def find_dataset_state_files(brain_dir: Path, dataset_name: str) -> List[Tuple[Path, Optional[int]]]:
    """Find all state files for a dataset (possibly with different chunk sizes).
    
    Args:
        brain_dir: Brain directory path
        dataset_name: Sanitized dataset name
        
    Returns:
        List of (state_file_path, chunk_size) tuples
    """
    datasets_dir = brain_dir / "datasets"
    if not datasets_dir.exists():
        return []
    
    results = []
    
    # Look for dataset directories with optional chunk size suffix
    for entry in datasets_dir.iterdir():
        if not entry.is_dir():
            continue
        
        dir_name = entry.name
        state_file = entry / "chunk_tracker_state.json"
        
        if not state_file.exists():
            continue
        
        # Check if this matches our dataset
        if dir_name == dataset_name:
            # Exact match, no chunk size suffix
            results.append((state_file, None))
        elif dir_name.startswith(f"{dataset_name}_cs"):
            # Has chunk size suffix
            try:
                chunk_size = int(dir_name.split("_cs")[1])
                results.append((state_file, chunk_size))
            except (ValueError, IndexError):
                pass
    
    return results


class _InterProcessLock:
    """Cross-platform file-based lock for coordinating multiple processes."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._fh: Optional[object] = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append binary to ensure file exists and pointer at end
        self._fh = open(self._path, "a+b")
        if os.name == "nt":
            import msvcrt  # type: ignore

            # Windows locking requires seek to the start
            self._fh.seek(0)
            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except PermissionError:
                    time.sleep(0.01)
        else:
            import fcntl  # type: ignore

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        if not self._fh:
            return

        if os.name == "nt":
            import msvcrt  # type: ignore

            self._fh.seek(0)
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self._fh.close()
        else:
            import fcntl  # type: ignore

            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()

        self._fh = None

    def __enter__(self) -> "_InterProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass
class ChunkProgress:
    """Tracks progress for a specific chunk."""
    
    block_id: int
    """Block ID this chunk belongs to."""
    
    chunk_id: int
    """Chunk ID within the block."""
    
    gpu_id: int
    """GPU that trained this chunk."""
    
    optimizer_step: int
    """Optimizer step when this chunk was completed."""
    
    steps: int = 0
    """Training steps (samples/rows trained) for this chunk."""


class ChunkTracker:
    """Thread-safe tracker for training progress across blocks and chunks.
    
    Features:
    - Tracks (block_id, chunk_id, step) for each trained chunk
    - Per-brain and per-dataset progress tracking
    - Saves/loads state from disk for resume capability
    - Thread-safe for parallel GPU access
    - Detects epoch completion when all blocks visited
    """
    
    def __init__(
        self, 
        state_file: Optional[Path] = None, 
        brain_id: Optional[str] = None, 
        dataset_name: Optional[str] = None,
        start_block_id: int = 0,
        start_chunk_id: int = 0
    ):
        """Initialize ChunkTracker.
        
        Args:
            state_file: Path to save/load training state
            brain_id: Unique brain identifier (YYYYMMDDHHMMSS format)
            dataset_name: Dataset identifier for this training session
            start_block_id: Starting block ID lower bound (Phase 6.4)
            start_chunk_id: Starting chunk ID lower bound (Phase 6.4)
        """
        self.state_file = state_file or Path("training_state/chunk_tracker_state.json")
        self.brain_id = brain_id
        self.dataset_name = dataset_name
        self.start_block_id = start_block_id
        self.start_chunk_id = start_chunk_id
        logger.info(f"Initializing ChunkTracker with state file: {self.state_file}")
        if brain_id:
            logger.info(f"  Brain ID: {brain_id}")
        if dataset_name:
            logger.info(f"  Dataset: {dataset_name}")
        if start_block_id > 0 or start_chunk_id > 0:
            logger.info(f"  Start Position: Block {start_block_id}, Chunk {start_chunk_id} (lower bound)")
        self.lock = threading.RLock()
        self._ipc_lock = _InterProcessLock(self.state_file.with_suffix(".lock"))
        self._state_mtime: Optional[float] = None
        
        # Track completed chunks: {(block_id, chunk_id): ChunkProgress}
        self.completed_chunks: Dict[Tuple[int, int], ChunkProgress] = {}
        
        # Track chunks currently in progress with metadata (gpu_id, timestamp)
        self.in_progress_chunks: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        # Track blocks that have been started: {block_id}
        self.started_blocks: Set[int] = set()
        
        # Track blocks that have been fully completed: {block_id}
        self.completed_blocks: Set[int] = set()
        
        # Epoch tracking
        self.current_epoch = 0
        self.blocks_this_epoch: Set[int] = set()
        self.total_blocks_in_dataset: Optional[int] = None
        
        # Training statistics
        self.total_steps = 0  # Aggregate steps (samples trained) across all GPUs
        
        # Session tracking (not persisted - resets each training run)
        self.session_start_chunk_count = 0
        self.session_chunks_completed = 0
        self.session_start_steps = 0  # Steps at session start
        
        # Load existing state if available (synchronized across processes)
        logger.debug("Loading existing chunk tracker state if available")
        with self._locked(refresh=True):
            pass
        
        # After loading, mark session start point
        self.session_start_chunk_count = len(self.completed_chunks)
        self.session_start_steps = self.total_steps
        logger.info(f"ChunkTracker initialized: {len(self.completed_chunks)} chunks completed, "
                   f"{self.total_steps} steps, epoch {self.current_epoch}")
    
    @contextmanager
    def _locked(self, refresh: bool = True) -> None:
        """Acquire both inter-process and intra-process locks."""
        try:
            self._ipc_lock.acquire()
            self.lock.acquire()
            if refresh:
                self._sync_from_disk_locked(force=False)
            yield
        finally:
            self.lock.release()
            self._ipc_lock.release()

    def _sync_from_disk_locked(self, force: bool = False) -> None:
        """Refresh in-memory state from disk if changed."""
        try:
            stat = self.state_file.stat()
            mtime = stat.st_mtime
        except FileNotFoundError:
            if force or self._state_mtime is not None:
                self._reset_state_locked()
                self._state_mtime = None
            return

        if force or self._state_mtime is None or mtime > (self._state_mtime or 0):
            with self.state_file.open("r", encoding="utf-8") as f:
                state = json.load(f)
            self._load_state_from_dict_locked(state)
            self._state_mtime = mtime
            self._cleanup_stale_in_progress_locked()

    def _reset_state_locked(self) -> None:
        self.completed_chunks = {}
        self.in_progress_chunks = {}
        self.started_blocks = set()
        self.completed_blocks = set()
        self.current_epoch = 0
        self.blocks_this_epoch = set()
        self.total_blocks_in_dataset = None
        self.total_steps = 0

    def _load_state_from_dict_locked(self, state: Dict) -> None:
        # Load brain_id and dataset_name from state (may be None for old states)
        if self.brain_id is None:
            self.brain_id = state.get("brain_id")
        if self.dataset_name is None:
            self.dataset_name = state.get("dataset_name")
        
        self.completed_chunks = {}
        for chunk_data in state.get("completed_chunks", []):
            chunk_key = (chunk_data["block_id"], chunk_data["chunk_id"])
            # Backward compat: old files have "step" and "samples_trained"/"true_steps"
            optimizer_step = chunk_data.get("optimizer_step", chunk_data.get("step", 0))
            steps = chunk_data.get("steps", chunk_data.get("true_steps", chunk_data.get("samples_trained", 0)))
            self.completed_chunks[chunk_key] = ChunkProgress(
                block_id=chunk_data["block_id"],
                chunk_id=chunk_data["chunk_id"],
                gpu_id=chunk_data["gpu_id"],
                optimizer_step=optimizer_step,
                steps=steps,
            )

        self.started_blocks = set(state.get("started_blocks", []))
        self.completed_blocks = set(state.get("completed_blocks", []))
        self.blocks_this_epoch = set(state.get("blocks_this_epoch", []))
        self.current_epoch = state.get("current_epoch", 0)
        self.total_blocks_in_dataset = state.get("total_blocks_in_dataset")
        # Backward compat: try total_steps, fall back to total_true_steps, then total_samples_trained
        self.total_steps = state.get("total_steps", state.get("total_true_steps", state.get("total_samples_trained", 0)))
        self.in_progress_chunks = {
            (entry["block_id"], entry["chunk_id"]): {
                "gpu_id": entry.get("gpu_id", -1),
                "timestamp": entry.get("timestamp", 0.0),
            }
            for entry in state.get("in_progress_chunks", [])
        }

    def _save_state_locked(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "brain_id": self.brain_id,
            "dataset_name": self.dataset_name,
            "completed_chunks": [
                {
                    "block_id": progress.block_id,
                    "chunk_id": progress.chunk_id,
                    "gpu_id": progress.gpu_id,
                    "optimizer_step": progress.optimizer_step,
                    "steps": progress.steps,
                }
                for progress in self.completed_chunks.values()
            ],
            "started_blocks": list(self.started_blocks),
            "completed_blocks": list(self.completed_blocks),
            "current_epoch": self.current_epoch,
            "blocks_this_epoch": list(self.blocks_this_epoch),
            "total_blocks_in_dataset": self.total_blocks_in_dataset,
            "total_steps": self.total_steps,
            "in_progress_chunks": [
                {
                    "block_id": block_id,
                    "chunk_id": chunk_id,
                    "gpu_id": meta.get("gpu_id", -1),
                    "timestamp": meta.get("timestamp", 0.0),
                }
                for (block_id, chunk_id), meta in self.in_progress_chunks.items()
            ],
        }

        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        try:
            self._state_mtime = self.state_file.stat().st_mtime
        except FileNotFoundError:
            self._state_mtime = None

    def _cleanup_stale_in_progress_locked(self) -> None:
        now = time.time()
        stale_keys = [
            key
            for key, meta in self.in_progress_chunks.items()
            if now - float(meta.get("timestamp", 0.0)) > CLAIM_TIMEOUT_SECONDS
        ]
        for key in stale_keys:
            logger.debug(f"Pruning stale in-progress chunk claim: {key}")
            del self.in_progress_chunks[key]

    def claim_chunk(
        self,
        block_id: int,
        chunk_id: int,
        gpu_id: int
    ) -> bool:
        """Claim a chunk for training on a specific GPU.
        
        Phase 6.4: Enforces start position lower bound - chunks before
        (start_block_id, start_chunk_id) cannot be claimed.
        
        Args:
            block_id: Block ID
            chunk_id: Chunk ID within block
            gpu_id: GPU claiming the chunk
            
        Returns:
            True if chunk was successfully claimed, False if already trained
            or before start position lower bound
        """
        logger.debug(f"GPU {gpu_id} attempting to claim chunk (block={block_id}, chunk={chunk_id})")
        with self._locked(refresh=True):
            chunk_key = (block_id, chunk_id)
            
            # Phase 6.4: Check start position lower bound
            if block_id < self.start_block_id:
                logger.debug(f"Chunk {chunk_key} rejected: block {block_id} < start_block_id {self.start_block_id}")
                return False
            elif block_id == self.start_block_id and chunk_id < self.start_chunk_id:
                logger.debug(f"Chunk {chunk_key} rejected: same block but chunk {chunk_id} < start_chunk_id {self.start_chunk_id}")
                return False
            
            # Check if already completed
            if chunk_key in self.completed_chunks:
                logger.debug(f"Chunk {chunk_key} already completed, cannot claim")
                return False
            
            # Mark block as started
            self.started_blocks.add(block_id)
            logger.info(f"GPU {gpu_id} successfully claimed chunk (block={block_id}, chunk={chunk_id})")
            
            return True
    
    def mark_chunk_complete(
        self,
        block_id: int,
        chunk_id: int,
        gpu_id: int,
        optimizer_step: int,
        steps: int = 0
    ) -> None:
        """Mark a chunk as completed.
        
        Args:
            block_id: Block ID
            chunk_id: Chunk ID within block
            gpu_id: GPU that trained this chunk
            optimizer_step: Optimizer step when completed
            steps: Training steps (samples/rows trained) for this chunk
        """
        logger.debug(f"Marking chunk complete: block={block_id}, chunk={chunk_id}, gpu={gpu_id}, optimizer_step={optimizer_step}, steps={steps}")
        with self._locked(refresh=True):
            chunk_key = (block_id, chunk_id)
            
            # Remove from in-progress (if it's there)
            if chunk_key in self.in_progress_chunks:
                del self.in_progress_chunks[chunk_key]
            
            progress = ChunkProgress(
                block_id=block_id,
                chunk_id=chunk_id,
                gpu_id=gpu_id,
                optimizer_step=optimizer_step,
                steps=steps
            )
            
            self.completed_chunks[chunk_key] = progress
            self.total_steps += steps  # Aggregate steps across all GPUs
            self.session_chunks_completed += 1  # Track chunks completed this session
            
            # Track blocks visited this epoch
            self.blocks_this_epoch.add(block_id)
            
            print(f"[ChunkTracker] GPU {gpu_id} completed Block {block_id} Chunk {chunk_id} (Optimizer Step {optimizer_step}, Steps {steps})")
            logger.info(f"Chunk completed: block={block_id}, chunk={chunk_id}, gpu={gpu_id}, total_steps={self.total_steps}")
            
            # Auto-save state after every chunk for better resume granularity
            # This matches checkpoint save frequency in single-GPU and parallel modes
            self._save_state_locked()
    
    def is_chunk_trained(self, block_id: int, chunk_id: int) -> bool:
        """Check if a chunk has already been trained.
        
        Args:
            block_id: Block ID
            chunk_id: Chunk ID within block
            
        Returns:
            True if chunk has been trained
        """
        with self._locked(refresh=True):
            return (block_id, chunk_id) in self.completed_chunks
    
    def mark_block_complete(self, block_id: int) -> None:
        """Mark a block as fully completed.
        
        Args:
            block_id: Block ID to mark complete
        """
        with self._locked(refresh=True):
            self.completed_blocks.add(block_id)
            self._save_state_locked()
    
    def is_block_complete(self, block_id: int) -> bool:
        """Check if a block has been fully completed.
        
        Args:
            block_id: Block ID to check
            
        Returns:
            True if block is complete
        """
        with self._locked(refresh=True):
            return block_id in self.completed_blocks
    
    def get_next_untrained_chunk(
        self,
        block_id: int,
        total_chunks_in_block: int,
        gpu_id: int
    ) -> Optional[int]:
        """Get the next untrained chunk ID in a block for a specific GPU.
        
        This method immediately claims the chunk as in-progress to prevent
        multiple GPUs from training on the same chunk (race condition).
        
        Respects start position bounds - will not return chunks before
        (start_block_id, start_chunk_id).
        
        Args:
            block_id: Block ID to search
            total_chunks_in_block: Total number of chunks in the block
            gpu_id: GPU requesting a chunk
            
        Returns:
            Chunk ID if found, None if all chunks trained or all chunks before start position
        """
        with self._locked(refresh=True):
            self._cleanup_stale_in_progress_locked()
            
            # Determine the starting chunk ID based on start position bounds
            start_chunk = 0
            if block_id == self.start_block_id:
                # If we're in the start block, start from start_chunk_id
                start_chunk = self.start_chunk_id
            elif block_id < self.start_block_id:
                # If block is before start_block_id, skip all chunks
                logger.debug(f"Block {block_id} is before start_block_id {self.start_block_id}, skipping all chunks")
                return None
            
            for chunk_id in range(start_chunk, total_chunks_in_block):
                chunk_key = (block_id, chunk_id)
                # Check if chunk is neither completed nor in progress
                if chunk_key not in self.completed_chunks and chunk_key not in self.in_progress_chunks:
                    # CRITICAL: Immediately claim it as in-progress to prevent other GPUs from taking it
                    self.in_progress_chunks[chunk_key] = {"gpu_id": int(gpu_id), "timestamp": time.time()}
                    self.started_blocks.add(block_id)
                    print(f"[ChunkTracker] GPU {gpu_id} claimed Block {block_id} Chunk {chunk_id}")
                    self._save_state_locked()
                    return chunk_id
            
            return None
    
    def check_epoch_complete(self, total_blocks: int) -> bool:
        """Check if the current epoch is complete.
        
        An epoch is complete when all blocks in the dataset have been visited.
        
        Args:
            total_blocks: Total number of blocks in the dataset
            
        Returns:
            True if epoch is complete
        """
        with self._locked(refresh=True):
            self.total_blocks_in_dataset = total_blocks
            is_complete = len(self.blocks_this_epoch) >= total_blocks
            if is_complete:
                logger.info(f"Epoch {self.current_epoch} complete: {len(self.blocks_this_epoch)}/{total_blocks} blocks visited")
            else:
                logger.debug(f"Epoch {self.current_epoch} progress: {len(self.blocks_this_epoch)}/{total_blocks} blocks visited")
            return is_complete
    
    def start_new_epoch(self) -> None:
        """Start a new epoch (reset block tracking)."""
        with self._locked(refresh=True):
            self.current_epoch += 1
            self.blocks_this_epoch.clear()
            logger.info(f"Started new epoch: {self.current_epoch}")
            # Don't clear completed_chunks - we still want to track all training
            self._save_state_locked()
    
    def get_progress_stats(self) -> Dict:
        """Get current training progress statistics.
        
        Returns:
            Dictionary with progress statistics including:
            - session_steps: Steps trained in current session (samples/rows)
            - total_steps: All-time cumulative steps (samples/rows)
            - total_optimizer_steps: Max optimizer step across all chunks
        """
        with self._locked(refresh=True):
            # Get the maximum optimizer step value (optimizer steps are cumulative, not per-chunk)
            total_optimizer_steps = max((p.optimizer_step for p in self.completed_chunks.values()), default=0)
            
            # Calculate session optimizer steps: max optimizer step from chunks completed this session
            session_chunks = list(self.completed_chunks.values())[-self.session_chunks_completed:] if self.session_chunks_completed > 0 else []
            session_optimizer_steps = max((p.optimizer_step for p in session_chunks), default=0)
            
            # Calculate session steps: total steps completed this session
            session_steps = self.total_steps - self.session_start_steps
            
            return {
                "total_optimizer_steps": total_optimizer_steps,  # Historical max optimizer steps (all sessions)
                "session_optimizer_steps": session_optimizer_steps,  # Current session max optimizer steps
                "session_steps": session_steps,  # Current session steps (samples trained, all GPUs)
                "total_steps": self.total_steps,  # All-time cumulative steps (samples trained, all GPUs, all sessions)
                "total_chunks_trained": len(self.completed_chunks),
                "session_chunks_trained": self.session_chunks_completed,
                "blocks_started": len(self.started_blocks),
                "blocks_completed": len(self.completed_blocks),
                "current_epoch": self.current_epoch,
                "blocks_this_epoch": len(self.blocks_this_epoch),
                "total_blocks_in_dataset": self.total_blocks_in_dataset,
            }
    
    def get_chunks_for_block(self, block_id: int) -> List[ChunkProgress]:
        """Get all completed chunks for a specific block.
        
        Args:
            block_id: Block ID
            
        Returns:
            List of ChunkProgress for the block
        """
        with self._locked(refresh=True):
            return [
                progress
                for (bid, _), progress in self.completed_chunks.items()
                if bid == block_id
            ]
    
    def reset_for_iterate(self) -> None:
        """Reset state for continuing in iterate mode.
        
        Clears block completion tracking but keeps chunk history for debugging.
        """
        with self._locked(refresh=True):
            # Keep chunk history but allow revisiting blocks
            self.completed_blocks.clear()
            self.blocks_this_epoch.clear()
            # Optionally clear completed_chunks if you want to allow retraining
            # For now, keep it to avoid duplicate training even across epochs
            self._save_state_locked()
    
    def save(self) -> None:
        """Explicitly save current state to disk."""
        with self._locked(refresh=False):
            self._save_state_locked()

    def refresh(self) -> None:
        """Reload latest state from disk into memory."""
        with self._locked(refresh=True):
            # No mutation required; context manager handles refresh
            pass
