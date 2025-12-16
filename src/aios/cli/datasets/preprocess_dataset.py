"""Dataset Preprocessing Utility

Converts downloaded datasets into a block-based structure for efficient training.

Structure created:
dataset_name/
├── raw/              # Original downloaded files (preserved)
├── block_0/          # First 100k samples
│   └── samples.txt   # One sample per line
├── block_1/          # Next 100k samples
│   └── samples.txt
├── ...
└── dataset_info.json # Metadata (total samples, blocks, etc.)

This preprocessing enables:
- Fast dataset size detection (read metadata file)
- Efficient block loading during training
- Progress tracking with known block counts
- Better performance on network drives
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List, Tuple
import shutil


def preprocess_dataset(
    dataset_path: str | Path,
    samples_per_block: int = 100000,
    ascii_only: bool = False,
    overwrite: bool = False,
) -> Tuple[int, int, int]:
    """Preprocess a dataset directory into block structure.
    
    Args:
        dataset_path: Path to dataset directory (containing text files or raw downloads)
        samples_per_block: Number of samples per block (default: 100k)
        ascii_only: Filter to ASCII-only text
        overwrite: Whether to overwrite existing preprocessed structure
    
    Returns:
        Tuple of (total_samples, samples_per_block, total_blocks)
        
    Raises:
        ValueError: If dataset_path is invalid or empty
        FileExistsError: If preprocessed structure exists and overwrite=False
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset path must be a directory: {dataset_path}")
    
    # Check for existing preprocessed structure
    info_file = dataset_path / "dataset_info.json"
    if info_file.exists() and not overwrite:
        # Already preprocessed, load metadata
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            print(f"✓ Dataset already preprocessed: {info['total_samples']:,} samples in {info['total_blocks']} blocks")
            return (info["total_samples"], info["samples_per_block"], info["total_blocks"])
        except Exception:
            if not overwrite:
                raise FileExistsError(f"Preprocessed structure exists but metadata is invalid. Use overwrite=True to rebuild.")
    
    print(f"📦 Preprocessing dataset: {dataset_path.name}")
    print(f"   Block size: {samples_per_block:,} samples")
    
    # Step 1: Move raw files to raw/ subdirectory if not already there
    raw_dir = dataset_path / "raw"
    if not raw_dir.exists():
        print("   Moving raw files to raw/ subdirectory...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Move all text files and dataset files to raw/
        moved_count = 0
        for item in dataset_path.iterdir():
            if item.name == "raw":
                continue
            if item.is_file() or (item.is_dir() and item.name in ["data", ".arrow"]):
                try:
                    shutil.move(str(item), str(raw_dir / item.name))
                    moved_count += 1
                except Exception as e:
                    print(f"   Warning: Could not move {item.name}: {e}")
        
        if moved_count > 0:
            print(f"   ✓ Moved {moved_count} items to raw/")
    
    # Step 2: Read all text samples from raw directory
    print("   Reading samples from raw files...")
    samples = _read_all_samples(raw_dir, ascii_only)
    
    if not samples:
        raise ValueError(f"No text samples found in {raw_dir}")
    
    total_samples = len(samples)
    total_blocks = (total_samples + samples_per_block - 1) // samples_per_block
    
    print(f"   ✓ Found {total_samples:,} samples")
    print(f"   Creating {total_blocks} blocks...")
    
    # Step 3: Create block directories and split samples
    for block_id in range(total_blocks):
        block_dir = dataset_path / f"block_{block_id}"
        block_dir.mkdir(parents=True, exist_ok=True)
        
        start_idx = block_id * samples_per_block
        end_idx = min(start_idx + samples_per_block, total_samples)
        block_samples = samples[start_idx:end_idx]
        
        # Write block samples
        block_file = block_dir / "samples.txt"
        with open(block_file, "w", encoding="utf-8") as f:
            for sample in block_samples:
                f.write(sample.strip() + "\n")
        
        print(f"   ✓ Block {block_id}: {len(block_samples):,} samples")
    
    # Step 4: Create metadata file
    metadata = {
        "dataset_name": dataset_path.name,
        "total_samples": total_samples,
        "samples_per_block": samples_per_block,
        "total_blocks": total_blocks,
        "ascii_only": ascii_only,
        "preprocessed_by": "AI-OS dataset preprocessor",
        "structure": "block_N/samples.txt format"
    }
    
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Created metadata file")
    print(f"✅ Preprocessing complete!")
    print(f"   Total: {total_samples:,} samples in {total_blocks} blocks")
    
    return (total_samples, samples_per_block, total_blocks)


def _read_all_samples(directory: Path, ascii_only: bool) -> List[str]:
    """Read all text samples from a directory.
    
    Supports:
    - Plain text files (.txt)
    - HuggingFace datasets (load_from_disk)
    - JSON/JSONL files with text fields
    """
    samples = []
    
    # Try loading as HuggingFace dataset first
    if _is_hf_dataset(directory):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(str(directory))
            
            # Find text column
            text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
            found_column = None
            
            for col in text_columns:
                if col in dataset.column_names:
                    found_column = col
                    break
            
            if not found_column and dataset.column_names:
                found_column = dataset.column_names[0]
            
            if found_column:
                print(f"   Loading HuggingFace dataset (column: {found_column})...")
                for item in dataset:
                    text = str(item.get(found_column, "")).strip()
                    if text:
                        if not ascii_only or _is_ascii(text):
                            samples.append(text)
                
                return samples
        except ImportError:
            print("   Warning: datasets library not available, falling back to file reading")
        except Exception as e:
            print(f"   Warning: Could not load as HF dataset: {e}")
    
    # Fallback: Read text files
    print("   Reading text files...")
    from aios.data.datasets.constants import TEXT_EXTS
    
    file_count = 0
    for filepath in directory.rglob("*"):
        if not filepath.is_file():
            continue
        
        if filepath.suffix.lower() not in TEXT_EXTS:
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        if not ascii_only or _is_ascii(text):
                            samples.append(text)
            
            file_count += 1
            if file_count % 100 == 0:
                print(f"   Processed {file_count} files, {len(samples):,} samples so far...")
                
        except Exception as e:
            print(f"   Warning: Could not read {filepath.name}: {e}")
            continue
    
    return samples


def _is_hf_dataset(directory: Path) -> bool:
    """Check if directory contains a HuggingFace dataset."""
    return (
        (directory / "dataset_info.json").exists() or
        (directory / "data").is_dir() or
        any(directory.glob("*.arrow"))
    )


def _is_ascii(text: str) -> bool:
    """Check if text is ASCII-only."""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def is_preprocessed(dataset_path: str | Path) -> bool:
    """Check if a dataset has been preprocessed into block structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if dataset has valid preprocessed structure
    """
    dataset_path = Path(dataset_path)
    info_file = dataset_path / "dataset_info.json"
    
    if not info_file.exists():
        return False
    
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        # Check for required fields
        required_fields = ["total_samples", "samples_per_block", "total_blocks"]
        if not all(field in info for field in required_fields):
            return False
        
        # Check that at least block_0 exists
        block_0 = dataset_path / "block_0" / "samples.txt"
        return block_0.exists()
        
    except Exception:
        return False


def get_preprocessed_info(dataset_path: str | Path) -> Optional[dict]:
    """Get metadata from a preprocessed dataset.
    
    Args:
        dataset_path: Path to preprocessed dataset directory
        
    Returns:
        Metadata dict or None if not preprocessed
    """
    dataset_path = Path(dataset_path)
    info_file = dataset_path / "dataset_info.json"
    
    if not info_file.exists():
        return None
    
    try:
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess downloaded datasets into block-based structure"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=100000,
        help="Samples per block (default: 100000)"
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Filter to ASCII-only text"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed structure"
    )
    
    args = parser.parse_args()
    
    try:
        total_samples, samples_per_block, total_blocks = preprocess_dataset(
            dataset_path=args.dataset_path,
            samples_per_block=args.block_size,
            ascii_only=args.ascii_only,
            overwrite=args.overwrite
        )
        
        print(f"\n✅ Success!")
        print(f"   Samples: {total_samples:,}")
        print(f"   Blocks: {total_blocks}")
        print(f"   Per block: {samples_per_block:,}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
