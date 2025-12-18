"""
Block Processor for Dataset Downloads

Handles processing downloaded datasets into standardized 100k-sample blocks
for efficient training and memory management.

Block Structure:
- Each block contains up to 100,000 examples/rows
- Blocks are saved as individual files or Arrow datasets
- Enables streaming during training without loading full dataset

This module provides:
1. Streaming download with automatic block creation (save-as-you-go)
2. Post-processing of raw downloads into 100k blocks
3. Block integrity verification
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Default block size for downloaded datasets
DEFAULT_BLOCK_SIZE = 100_000  # 100k samples per block


@dataclass
class BlockMetadata:
    """Metadata for a single block of data."""
    
    block_id: int
    """Sequential block identifier (0-indexed)."""
    
    sample_count: int
    """Number of samples in this block."""
    
    start_index: int
    """Global start index in the original dataset."""
    
    end_index: int
    """Global end index (exclusive) in the original dataset."""
    
    file_path: Path
    """Path to the block file on disk."""
    
    size_bytes: int = 0
    """Size of the block file in bytes."""
    
    is_complete: bool = True
    """Whether the block was fully written (for partial downloads)."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "block_id": self.block_id,
            "sample_count": self.sample_count,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "file_path": str(self.file_path),
            "size_bytes": self.size_bytes,
            "is_complete": self.is_complete,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlockMetadata":
        """Create from dictionary."""
        return cls(
            block_id=data["block_id"],
            sample_count=data["sample_count"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            file_path=Path(data["file_path"]),
            size_bytes=data.get("size_bytes", 0),
            is_complete=data.get("is_complete", True),
        )


@dataclass
class DatasetBlockInfo:
    """Information about a processed dataset with block structure."""
    
    dataset_name: str
    """Name of the dataset."""
    
    total_samples: int
    """Total number of samples across all blocks."""
    
    total_blocks: int
    """Number of blocks in the dataset."""
    
    block_size: int
    """Target samples per block."""
    
    blocks: List[BlockMetadata]
    """List of block metadata."""
    
    text_column: Optional[str] = None
    """Name of the text column (for text datasets)."""
    
    output_dir: Optional[Path] = None
    """Directory containing the blocks."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "total_blocks": self.total_blocks,
            "block_size": self.block_size,
            "blocks": [b.to_dict() for b in self.blocks],
            "text_column": self.text_column,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetBlockInfo":
        """Create from dictionary."""
        return cls(
            dataset_name=data["dataset_name"],
            total_samples=data["total_samples"],
            total_blocks=data["total_blocks"],
            block_size=data["block_size"],
            blocks=[BlockMetadata.from_dict(b) for b in data["blocks"]],
            text_column=data.get("text_column"),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        )
    
    def save_manifest(self) -> Path:
        """Save manifest file to the output directory."""
        if not self.output_dir:
            raise ValueError("output_dir must be set to save manifest")
        
        manifest_path = self.output_dir / "block_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved block manifest to {manifest_path}")
        return manifest_path
    
    @classmethod
    def load_manifest(cls, manifest_path: Path) -> "DatasetBlockInfo":
        """Load manifest from file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class StreamingBlockWriter:
    """
    Writes streaming dataset to disk in 100k blocks as data arrives.
    
    This is the primary mechanism for downloading datasets:
    - Streams samples from HuggingFace
    - Writes blocks to disk as they fill up
    - Enables resumable downloads
    - Memory efficient: only holds one block at a time
    """
    
    def __init__(
        self,
        output_dir: Path,
        dataset_name: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize streaming block writer.
        
        Args:
            output_dir: Directory to write blocks to
            dataset_name: Name of the dataset
            block_size: Number of samples per block
            log_callback: Optional callback for logging progress
        """
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.block_size = block_size
        self.log = log_callback or (lambda x: None)
        
        # Create blocks subdirectory
        self.blocks_dir = self.output_dir / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.current_block: List[Dict[str, Any]] = []
        self.current_block_id = 0
        self.total_samples = 0
        self.blocks: List[BlockMetadata] = []
        self.text_column: Optional[str] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._finalized = False
        
        logger.info(f"StreamingBlockWriter initialized: output={output_dir}, block_size={block_size}")
    
    def add_sample(self, sample: Dict[str, Any]) -> Optional[BlockMetadata]:
        """
        Add a sample to the current block.
        
        Returns:
            BlockMetadata if a block was flushed to disk, None otherwise
        """
        if self._finalized:
            raise RuntimeError("Writer has been finalized, cannot add more samples")
        
        with self._lock:
            self.current_block.append(sample)
            self.total_samples += 1
            
            # Detect text column from first sample
            if self.text_column is None and sample:
                text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
                for col in text_columns:
                    if col in sample:
                        self.text_column = col
                        logger.debug(f"Detected text column: {col}")
                        break
                if self.text_column is None and sample:
                    self.text_column = list(sample.keys())[0]
                    logger.debug(f"Using first column as text: {self.text_column}")
            
            # Flush block if full
            if len(self.current_block) >= self.block_size:
                return self._flush_current_block()
        
        return None
    
    def _flush_current_block(self) -> Optional[BlockMetadata]:
        """Flush current block to disk. Assumes lock is held."""
        if not self.current_block:
            return None
        
        block_id = self.current_block_id
        start_index = block_id * self.block_size
        sample_count = len(self.current_block)
        
        # Determine block file path
        block_file = self.blocks_dir / f"block_{block_id:05d}.jsonl"
        
        try:
            # Write block as JSONL for efficiency
            with open(block_file, "w", encoding="utf-8") as f:
                for sample in self.current_block:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            size_bytes = block_file.stat().st_size
            
            metadata = BlockMetadata(
                block_id=block_id,
                sample_count=sample_count,
                start_index=start_index,
                end_index=start_index + sample_count,
                file_path=block_file,
                size_bytes=size_bytes,
                is_complete=True,
            )
            
            self.blocks.append(metadata)
            self.current_block_id += 1
            self.current_block = []
            
            size_mb = size_bytes / (1024 * 1024)
            self.log(f"   📦 Block {block_id} saved: {sample_count:,} samples ({size_mb:.1f} MB)")
            logger.info(f"Flushed block {block_id}: {sample_count} samples, {size_mb:.1f} MB")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to flush block {block_id}: {e}")
            self.log(f"   ❌ Failed to save block {block_id}: {e}")
            return None
    
    def finalize(self) -> DatasetBlockInfo:
        """
        Finalize the dataset, flushing any remaining samples.
        
        Returns:
            DatasetBlockInfo with complete block information
        """
        with self._lock:
            if self._finalized:
                raise RuntimeError("Writer has already been finalized")
            
            # Flush any remaining samples
            if self.current_block:
                self._flush_current_block()
            
            self._finalized = True
            
            info = DatasetBlockInfo(
                dataset_name=self.dataset_name,
                total_samples=self.total_samples,
                total_blocks=len(self.blocks),
                block_size=self.block_size,
                blocks=self.blocks,
                text_column=self.text_column,
                output_dir=self.output_dir,
            )
            
            # Save manifest
            try:
                info.save_manifest()
            except Exception as e:
                logger.warning(f"Failed to save manifest: {e}")
            
            logger.info(f"Finalized dataset: {self.total_samples} samples in {len(self.blocks)} blocks")
            self.log(f"   ✅ Dataset saved: {self.total_samples:,} samples in {len(self.blocks)} blocks")
            
            return info
    
    @property
    def samples_in_current_block(self) -> int:
        """Number of samples currently buffered."""
        return len(self.current_block)


def process_raw_dataset_to_blocks(
    input_path: Path,
    output_dir: Path,
    dataset_name: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    log_callback: Optional[Callable[[str], None]] = None,
) -> DatasetBlockInfo:
    """
    Process a raw downloaded dataset into 100k blocks.
    
    This post-processes datasets that weren't downloaded in block format.
    Supports:
    - Arrow datasets (HuggingFace format)
    - JSONL files
    - Plain text files (one sample per line)
    
    Args:
        input_path: Path to the raw dataset (directory or file)
        output_dir: Directory to write blocks to
        dataset_name: Name of the dataset
        block_size: Samples per block (default 100k)
        log_callback: Optional logging callback
        
    Returns:
        DatasetBlockInfo with block structure
    """
    log = log_callback or (lambda x: None)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    log(f"📦 Processing dataset into {block_size:,}-sample blocks...")
    logger.info(f"Processing raw dataset to blocks: {input_path} -> {output_dir}")
    
    writer = StreamingBlockWriter(output_dir, dataset_name, block_size, log)
    
    # Detect input format and iterate samples
    if input_path.is_dir():
        # Could be Arrow dataset or directory of files
        if (input_path / "dataset_info.json").exists() or (input_path / "state.json").exists():
            # HuggingFace Arrow dataset
            _process_arrow_dataset(input_path, writer, log)
        else:
            # Directory of text files
            _process_directory(input_path, writer, log)
    elif input_path.suffix == ".jsonl":
        _process_jsonl_file(input_path, writer, log)
    elif input_path.suffix in (".txt", ".text"):
        _process_text_file(input_path, writer, log)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    return writer.finalize()


def _process_arrow_dataset(
    dataset_path: Path,
    writer: StreamingBlockWriter,
    log: Callable[[str], None],
):
    """Process HuggingFace Arrow dataset."""
    try:
        from datasets import Dataset
        
        log(f"   📂 Loading Arrow dataset from {dataset_path}...")
        ds = Dataset.load_from_disk(str(dataset_path))
        
        total = len(ds)
        log(f"   📊 Found {total:,} samples")
        
        for i, sample in enumerate(ds):
            writer.add_sample(dict(sample))
            
            # Progress every 10k samples
            if (i + 1) % 10000 == 0:
                log(f"   ⏳ Processed {i + 1:,}/{total:,} samples...")
                
    except Exception as e:
        logger.error(f"Failed to process Arrow dataset: {e}")
        raise


def _process_jsonl_file(
    file_path: Path,
    writer: StreamingBlockWriter,
    log: Callable[[str], None],
):
    """Process JSONL file."""
    log(f"   📄 Processing JSONL file: {file_path}")
    
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    writer.add_sample(sample)
                    count += 1
                    
                    if count % 10000 == 0:
                        log(f"   ⏳ Processed {count:,} samples...")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    
    log(f"   📊 Processed {count:,} samples from JSONL")


def _process_text_file(
    file_path: Path,
    writer: StreamingBlockWriter,
    log: Callable[[str], None],
):
    """Process plain text file (one sample per line)."""
    log(f"   📄 Processing text file: {file_path}")
    
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                writer.add_sample({"text": line})
                count += 1
                
                if count % 10000 == 0:
                    log(f"   ⏳ Processed {count:,} samples...")
    
    log(f"   📊 Processed {count:,} samples from text file")


def _process_directory(
    dir_path: Path,
    writer: StreamingBlockWriter,
    log: Callable[[str], None],
):
    """Process directory of text files."""
    log(f"   📂 Processing directory: {dir_path}")
    
    # Find all text files
    text_files = list(dir_path.glob("**/*.txt")) + list(dir_path.glob("**/*.text"))
    jsonl_files = list(dir_path.glob("**/*.jsonl"))
    
    log(f"   📊 Found {len(text_files)} text files, {len(jsonl_files)} JSONL files")
    
    for f in text_files:
        _process_text_file(f, writer, log)
    
    for f in jsonl_files:
        _process_jsonl_file(f, writer, log)


def verify_block_structure(manifest_path: Path) -> Tuple[bool, str]:
    """
    Verify that a block-structured dataset is valid.
    
    Args:
        manifest_path: Path to block_manifest.json
        
    Returns:
        (is_valid, message)
    """
    try:
        info = DatasetBlockInfo.load_manifest(manifest_path)
        
        errors = []
        total_verified = 0
        
        for block in info.blocks:
            if not block.file_path.exists():
                errors.append(f"Block {block.block_id} file missing: {block.file_path}")
                continue
            
            # Count lines in JSONL
            with open(block.file_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            
            if line_count != block.sample_count:
                errors.append(
                    f"Block {block.block_id} count mismatch: "
                    f"expected {block.sample_count}, got {line_count}"
                )
            
            total_verified += line_count
        
        if errors:
            return False, "\n".join(errors)
        
        if total_verified != info.total_samples:
            return False, f"Total sample count mismatch: expected {info.total_samples}, got {total_verified}"
        
        return True, f"Valid: {info.total_blocks} blocks, {info.total_samples:,} samples"
        
    except Exception as e:
        return False, f"Verification failed: {e}"


def get_block_info(dataset_path: Path) -> Optional[DatasetBlockInfo]:
    """
    Get block info for a dataset if it has block structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        DatasetBlockInfo if block manifest exists, None otherwise
    """
    manifest_path = dataset_path / "block_manifest.json"
    if manifest_path.exists():
        try:
            return DatasetBlockInfo.load_manifest(manifest_path)
        except Exception as e:
            logger.warning(f"Failed to load block manifest: {e}")
    return None


def estimate_block_count(total_samples: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Calculate number of blocks needed for a dataset."""
    return (total_samples + block_size - 1) // block_size
