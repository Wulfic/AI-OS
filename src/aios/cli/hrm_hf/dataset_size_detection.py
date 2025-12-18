"""Dataset size detection for epoch tracking.

Provides utilities to detect the total number of samples in a dataset for
accurate epoch tracking and progress reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import warnings


def detect_dataset_size(
    dataset_file: str,
    ascii_only: bool = False,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Detect total samples, samples per block, and total blocks for a dataset.
    
    Args:
        dataset_file: Path to dataset file, directory, archive, or HF dataset URI
        ascii_only: Whether to filter for ASCII-only lines
    
    Returns:
        Tuple of (dataset_total_samples, samples_per_block, total_blocks)
        Returns (None, None, None) if detection fails
    
    Terminology:
    - Dataset: Complete dataset (could be millions of samples)
    - Block: Downloaded/loaded chunk (e.g., 100k samples from HF streaming)
    - Epoch: One complete pass through ALL blocks in dataset
    
    Detection strategy:
    - Text files: Count lines (single block)
    - HuggingFace datasets: Use dataset length, block size for streaming
    - HuggingFace streaming: Detect total size, use block-based loading
    - Archives/directories: Sample-based estimation
    """
    try:
        # HuggingFace Hub streaming format
        if dataset_file.startswith("hf://"):
            return _detect_hf_streaming_size(dataset_file, ascii_only)
        
        p = Path(dataset_file)
        
        # Directory (may be HF dataset or plain text files)
        if p.is_dir():
            return _detect_directory_size(p, ascii_only)
        
        # Single file
        if p.exists() and p.is_file():
            return _detect_file_size(p, ascii_only)
        
        # Unknown/invalid path
        warnings.warn(f"Could not detect dataset size: {dataset_file} not found or invalid")
        return (None, None, None)
        
    except Exception as e:
        warnings.warn(f"Error detecting dataset size for {dataset_file}: {e}")
        return (None, None, None)


def _detect_file_size(p: Path, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Detect size of a single text file."""
    from aios.data.datasets.archive_readers import _is_archive
    
    # Archive file - use sampling estimation
    if _is_archive(p):
        return _detect_archive_size(p, ascii_only)
    
    # Plain text file - count lines
    try:
        line_count = 0
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                if ascii_only:
                    try:
                        stripped.encode("ascii")
                    except UnicodeEncodeError:
                        continue
                
                line_count += 1
        
        # For local text files, all samples are in one "block"
        return (line_count, line_count, 1)
        
    except Exception as e:
        warnings.warn(f"Error counting lines in {p}: {e}")
        return (None, None, None)


def _detect_archive_size(p: Path, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Estimate size of archive by sampling.
    
    Archives are treated as chunked datasets since we can't easily count all entries.
    We sample and estimate based on the sample.
    """
    try:
        from aios.data.datasets.archive_readers import read_archive_text_lines
        
        # Sample 5000 lines to estimate
        sample_size = 5000
        sample_lines = read_archive_text_lines(p, max_lines=sample_size)
        
        if ascii_only:
            sample_lines = [
                ln for ln in sample_lines
                if _is_ascii(ln)
            ]
        
        if not sample_lines:
            return (None, None, None)
        
        # If we got fewer lines than requested, we have the full dataset
        if len(sample_lines) < sample_size:
            total_samples = len(sample_lines)
            return (total_samples, total_samples, 1)
        
        # Otherwise, estimate using chunks
        # Treat each 5000-line sample as a "chunk"
        samples_per_chunk = sample_size
        # Estimate 20 chunks (100K samples) as a reasonable default
        estimated_total = samples_per_chunk * 20
        total_chunks = 20
        
        return (estimated_total, samples_per_chunk, total_chunks)
        
    except Exception as e:
        warnings.warn(f"Error estimating archive size for {p}: {e}")
        return (None, None, None)


def _detect_directory_size(p: Path, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Detect size of directory (preprocessed, HF dataset, or text files)."""
    # Check if this is a preprocessed dataset (block structure)
    info_file = p / "dataset_info.json"
    block_0 = p / "block_0" / "samples.txt"
    
    if info_file.exists() and block_0.exists():
        try:
            import json
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            
            total_samples = info.get("total_samples")
            samples_per_block = info.get("samples_per_block")
            total_blocks = info.get("total_blocks")
            
            if all([total_samples, samples_per_block, total_blocks]):
                # Preprocessed dataset detected
                return (total_samples, samples_per_block, total_blocks)
        except Exception:
            pass
    
    # Check if this is a HuggingFace dataset directory
    is_hf_dataset = False
    try:
        if info_file.exists() or (p / "data").is_dir() or any(p.glob("*.arrow")):
            is_hf_dataset = True
    except Exception:
        pass
    
    if is_hf_dataset:
        return _detect_hf_local_size(p, ascii_only)
    
    # Plain directory with text files - count lines across all files
    return _detect_text_directory_size(p, ascii_only)


def _detect_hf_local_size(p: Path, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Detect size of local HuggingFace dataset."""
    try:
        from datasets import load_from_disk
        
        dataset = load_from_disk(str(p))
        total_samples = len(dataset)
        
        # HF datasets are typically not chunked when loaded from disk
        # Treat entire dataset as one chunk
        return (total_samples, total_samples, 1)
        
    except ImportError:
        warnings.warn("datasets library not installed. Cannot detect HuggingFace dataset size.")
        return (None, None, None)
    except Exception as e:
        warnings.warn(f"Error detecting HuggingFace dataset size for {p}: {e}")
        return (None, None, None)


def _detect_hf_streaming_size(hf_uri: str, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Detect size of HuggingFace Hub streaming dataset.
    
    For streaming datasets, we download in blocks (e.g., 100k samples per block).
    Each block is then subdivided for training.
    """
    try:
        from datasets import load_dataset
        
        # Parse hf://dataset_path[:config][:split]
        hf_path = hf_uri[5:]  # Remove 'hf://' prefix
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
        
        # For streaming datasets, we download in blocks
        # Block size: 100k samples (reasonable download chunk for large datasets)
        samples_per_block = 100000
        
        # Try to get dataset info for total size
        total_samples = None
        try:
            # Some streaming datasets have info with num_rows
            if hasattr(dataset, 'info') and hasattr(dataset.info, 'splits'):
                split_info = dataset.info.splits.get(split)
                if split_info and hasattr(split_info, 'num_rows'):
                    total_samples = split_info.num_rows
        except Exception:
            pass
        
        # If we can't get exact size, estimate based on a sample
        if total_samples is None:
            # Sample first 1000 items to check if dataset is small
            sample_count = 0
            try:
                for item in dataset:
                    sample_count += 1
                    if sample_count >= 1000:
                        break
                
                # If we got fewer than 1000, dataset might be small
                if sample_count < 1000:
                    total_samples = sample_count
                    # Small dataset - use smaller block size
                    samples_per_block = min(sample_count, 10000)
                else:
                    # Estimate a large dataset - could be millions
                    # Conservative estimate: 1 million samples
                    total_samples = 1000000
            except Exception:
                # Default estimate for unknown datasets
                total_samples = 1000000
        
        # Calculate total blocks
        total_blocks = (total_samples + samples_per_block - 1) // samples_per_block
        
        return (total_samples, samples_per_block, total_blocks)
        
    except ImportError:
        warnings.warn("datasets library not installed. Cannot detect HuggingFace streaming dataset size.")
        return (None, None, None)
    except Exception as e:
        warnings.warn(f"Error detecting HuggingFace streaming dataset size for {hf_uri}: {e}")
        return (None, None, None)


def _detect_text_directory_size(p: Path, ascii_only: bool) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Count lines across all text files in directory."""
    from aios.data.datasets.constants import TEXT_EXTS
    
    try:
        total_lines = 0
        
        for fp in p.rglob("*"):
            if not fp.is_file():
                continue
            
            if fp.suffix.lower() not in TEXT_EXTS:
                continue
            
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        
                        if ascii_only:
                            try:
                                stripped.encode("ascii")
                            except UnicodeEncodeError:
                                continue
                        
                        total_lines += 1
            except Exception:
                continue
        
        if total_lines == 0:
            return (None, None, None)
        
        # Directory with text files = one big "chunk"
        return (total_lines, total_lines, 1)
        
    except Exception as e:
        warnings.warn(f"Error counting lines in directory {p}: {e}")
        return (None, None, None)


def _is_ascii(s: str) -> bool:
    """Check if string is ASCII-only."""
    try:
        s.encode("ascii")
        return True
    except (UnicodeEncodeError, AttributeError):
        return False


def is_epoch_complete(
    blocks_processed: str,
    total_blocks: int,
) -> bool:
    """Check if all blocks have been processed (epoch is complete).
    
    Args:
        blocks_processed: Comma-separated string of block indices (e.g., "0,3,7,1,5,2,4,6")
        total_blocks: Total number of blocks in dataset
    
    Returns:
        True if all blocks have been visited, False otherwise
    """
    if total_blocks <= 0:
        return False
    
    if not blocks_processed or blocks_processed.strip() == "":
        return False
    
    try:
        # Parse block indices
        processed_set = set(int(idx.strip()) for idx in blocks_processed.split(",") if idx.strip())
        
        # Check if all blocks (0 to total_blocks-1) have been processed
        required_blocks = set(range(total_blocks))
        
        return required_blocks.issubset(processed_set)
        
    except Exception:
        return False


def add_block_to_processed(
    blocks_processed: str,
    block_index: int,
) -> str:
    """Add a block index to the processed blocks string.
    
    Args:
        blocks_processed: Current comma-separated string of block indices
        block_index: New block index to add
    
    Returns:
        Updated comma-separated string with new block added (if not already present)
    """
    if not blocks_processed or blocks_processed.strip() == "":
        return str(block_index)
    
    try:
        # Parse existing blocks
        processed_set = set(int(idx.strip()) for idx in blocks_processed.split(",") if idx.strip())
        
        # Add new block
        processed_set.add(block_index)
        
        # Return as sorted comma-separated string
        return ",".join(str(idx) for idx in sorted(processed_set))
        
    except Exception:
        # If parsing fails, append to end
        return f"{blocks_processed},{block_index}"
