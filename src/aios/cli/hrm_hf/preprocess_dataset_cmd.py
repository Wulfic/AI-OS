"""Preprocess dataset command for hrm-hf CLI."""

from __future__ import annotations

from pathlib import Path


def preprocess_dataset_cmd(
    dataset_path: str,
    block_size: int = 100000,
    ascii_only: bool = False,
    overwrite: bool = False,
) -> None:
    """Preprocess a downloaded dataset into block-based structure.
    
    Args:
        dataset_path: Path to dataset directory
        block_size: Samples per block (default: 100000)
        ascii_only: Filter to ASCII-only text
        overwrite: Overwrite existing preprocessed structure
    """
    from ..datasets.preprocess_dataset import preprocess_dataset
    
    try:
        print(f"📦 Preprocessing dataset: {dataset_path}")
        print(f"   Block size: {block_size:,} samples per block")
        if ascii_only:
            print(f"   Filtering: ASCII-only text")
        if overwrite:
            print(f"   Mode: Overwrite existing")
        print()
        
        total_samples, samples_per_block, total_blocks = preprocess_dataset(
            dataset_path=dataset_path,
            samples_per_block=block_size,
            ascii_only=ascii_only,
            overwrite=overwrite,
        )
        
        print()
        print("=" * 60)
        print("✅ Preprocessing Complete!")
        print("=" * 60)
        print(f"Dataset: {Path(dataset_path).name}")
        print(f"Total Samples: {total_samples:,}")
        print(f"Blocks Created: {total_blocks}")
        print(f"Samples per Block: {samples_per_block:,}")
        print()
        print("The dataset is now optimized for training:")
        print(f"  • Fast size detection (metadata file)")
        print(f"  • Efficient block loading")
        print(f"  • Progress tracking with known block counts")
        print()
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        raise
