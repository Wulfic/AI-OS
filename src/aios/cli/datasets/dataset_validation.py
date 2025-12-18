"""Dataset validation and auto-preprocessing for training.

This module ensures datasets are properly preprocessed before training starts.
For downloaded datasets, it checks if they've been converted to block structure,
and automatically preprocesses them if needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_and_preprocess_dataset(
    dataset_file: str,
    samples_per_block: int = 100000,
    ascii_only: bool = False,
    auto_preprocess: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Check if a dataset is properly preprocessed and preprocess if needed.
    
    This function is called before training starts to ensure the dataset is in
    the correct block format for efficient training and progress tracking.
    
    Args:
        dataset_file: Path to dataset file or directory
        samples_per_block: Number of samples per block (default: 100k)
        ascii_only: Filter to ASCII-only text
        auto_preprocess: Automatically preprocess if not already done
        
    Returns:
        Tuple of (needs_preprocessing, error_message)
        - (False, None): Dataset is ready for training
        - (True, None): Dataset was auto-preprocessed successfully
        - (True, error_msg): Preprocessing needed but failed
        
    Dataset types handled:
    - HuggingFace streaming (hf://...): No preprocessing needed, skip check
    - Single text files: No preprocessing needed (treated as one block)
    - Directories: Check for block structure, preprocess if needed
    - Archives: No preprocessing (read-only format)
    """
    # Skip preprocessing check for special formats
    if dataset_file.startswith("hf://"):
        logger.debug(f"Skipping preprocessing check for HuggingFace streaming dataset: {dataset_file}")
        return (False, None)
    
    # Normalize path separators for cross-platform compatibility
    dataset_file = str(Path(dataset_file).as_posix()) if '\\' in dataset_file else dataset_file
    dataset_path = Path(dataset_file)
    
    # Check if path exists at all
    if not dataset_path.exists():
        # Try with normalized path separators
        normalized_path = Path(str(dataset_file).replace('/', os.sep).replace('\\', os.sep))
        if normalized_path.exists():
            dataset_path = normalized_path
        else:
            logger.error(f"Dataset path does not exist: {dataset_file}")
            return (True, f"Dataset path not found: {dataset_file}")
    
    # Single files don't need preprocessing
    if dataset_path.is_file():
        logger.debug(f"Single file dataset, no preprocessing needed: {dataset_file}")
        return (False, None)
    
    # Directory check - this is where preprocessing matters
    if not dataset_path.is_dir():
        logger.warning(f"Dataset path exists but is not a file or directory: {dataset_file}")
        return (True, f"Dataset path is not a valid file or directory: {dataset_file}")
    
    # Check if already preprocessed
    from .preprocess_dataset import is_preprocessed, get_preprocessed_info
    
    if is_preprocessed(dataset_path):
        info = get_preprocessed_info(dataset_path)
        if info:
            logger.info(
                f"✓ Dataset already preprocessed: {info['total_samples']:,} samples "
                f"in {info['total_blocks']} blocks"
            )
        return (False, None)
    
    # Dataset is a directory but not preprocessed
    logger.info(f"📦 Dataset not preprocessed: {dataset_path}")
    
    if not auto_preprocess:
        msg = (
            f"Dataset '{dataset_path.name}' has not been preprocessed into block structure. "
            f"Run: aios hrm-hf preprocess-dataset {dataset_path}"
        )
        logger.warning(msg)
        return (True, msg)
    
    # Auto-preprocess the dataset
    logger.info(f"🔄 Auto-preprocessing dataset: {dataset_path}")
    print(f"\n{'='*60}")
    print(f"📦 Dataset preprocessing required")
    print(f"   Path: {dataset_path}")
    print(f"   Block size: {samples_per_block:,} samples per block")
    if ascii_only:
        print(f"   Filter: ASCII-only text")
    print(f"{'='*60}\n")
    
    try:
        from .preprocess_dataset import preprocess_dataset
        
        total_samples, actual_samples_per_block, total_blocks = preprocess_dataset(
            dataset_path=dataset_path,
            samples_per_block=samples_per_block,
            ascii_only=ascii_only,
            overwrite=False,
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Preprocessing complete!")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total blocks: {total_blocks}")
        print(f"   Samples per block: {actual_samples_per_block:,}")
        print(f"{'='*60}\n")
        
        logger.info(
            f"Successfully preprocessed dataset: {total_samples:,} samples "
            f"in {total_blocks} blocks"
        )
        return (True, None)
        
    except Exception as e:
        error_msg = f"Failed to preprocess dataset: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"\n❌ Error: {error_msg}\n")
        return (True, error_msg)


def should_preprocess_dataset(dataset_file: str) -> bool:
    """Quick check if a dataset should be preprocessed.
    
    Args:
        dataset_file: Path to dataset file or directory
        
    Returns:
        True if dataset should be preprocessed (is a directory and not already preprocessed)
    """
    if dataset_file.startswith("hf://"):
        return False
    
    dataset_path = Path(dataset_file)
    
    if not dataset_path.is_dir():
        return False
    
    from .preprocess_dataset import is_preprocessed
    return not is_preprocessed(dataset_path)


def validate_dataset_for_training(
    dataset_file: str,
    samples_per_block: int = 100000,
    ascii_only: bool = False,
) -> None:
    """Validate and preprocess dataset before training starts.
    
    This is the main entry point called by training code to ensure
    the dataset is ready for training.
    
    Args:
        dataset_file: Path to dataset file or directory
        samples_per_block: Number of samples per block
        ascii_only: Filter to ASCII-only text
        
    Raises:
        ValueError: If dataset validation/preprocessing fails
    """
    was_preprocessed, error = check_and_preprocess_dataset(
        dataset_file=dataset_file,
        samples_per_block=samples_per_block,
        ascii_only=ascii_only,
        auto_preprocess=True,
    )
    
    if error:
        raise ValueError(error)
    
    if was_preprocessed:
        logger.info("Dataset was auto-preprocessed and is now ready for training")
    else:
        logger.debug("Dataset validation passed, ready for training")
