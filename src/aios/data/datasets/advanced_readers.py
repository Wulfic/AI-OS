"""Advanced dataset reading with HuggingFace support, streaming, and caching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import io

from .constants import TEXT_EXTS
from .readers import read_text_lines_sample
from .archive_readers import _is_archive, read_archive_text_lines

logger = logging.getLogger(__name__)


def read_text_lines_sample_any(path: str | Path, max_lines: int = 1000, cycle: int = 0) -> List[str]:
    """Read up to max_lines lines from a dataset path that can be a file, directory, or archive.

    Supported archives (best-effort): zip, tar(.gz/.bz2/.xz), gz (single file), and optionally rar/7z if libraries are installed.
    Supports HuggingFace datasets via .arrow files or dataset_info.json.
    
    Special format: hf://<dataset_path>[:config][:split] for streaming from HuggingFace Hub.
    Example: hf://wikitext:wikitext-2-raw-v1:train
    """
    logger.debug(f"read_text_lines_sample_any: path={path}, max_lines={max_lines}, cycle={cycle}")
    
    # Check for HuggingFace Hub streaming format
    if isinstance(path, str) and path.startswith("hf://"):
        logger.info(f"Loading dataset from HuggingFace Hub: {path}")
        return _read_hf_streaming(path, max_lines, cycle)
    
    p = Path(path)
    try:
        if p.is_dir():
            logger.info(f"Reading from directory: {p}")
            return _read_directory(p, max_lines)
        
        if not p.exists():
            logger.warning(f"Dataset path does not exist: {p}")
            return []
        
        if not _is_archive(p):
            # If CSV, prefer treating each row as a JSON-ish line only when asked; otherwise fallback to raw lines
            if p.suffix.lower() == ".csv":
                logger.info(f"Reading CSV file: {p}")
                # Return raw lines so CLI can re-parse with proper column selection
                return read_text_lines_sample(p, max_lines=max_lines)
            logger.debug(f"Reading text file: {p}")
            return read_text_lines_sample(p, max_lines=max_lines)

        # Archive handling
        logger.info(f"Reading archive file: {p} ({p.suffix})")
        return read_archive_text_lines(p, max_lines)
        
    except Exception as e:
        logger.error(f"Failed to read dataset from {path}: {e}")
        return []


def _read_hf_streaming(path: str, max_lines: int, cycle: int) -> List[str]:
    """Read from HuggingFace Hub using streaming."""
    # Coordinate with stream manager to avoid conflicts
    stream_mgr = None
    dataset_id = None
    try:
        from ..stream_manager import get_stream_manager
        stream_mgr = get_stream_manager()
    except ImportError:
        pass  # Stream manager not available
    
    # Get streaming cache
    chunk_cache = None
    try:
        from ..streaming_cache import get_cache
        chunk_cache = get_cache()
    except ImportError:
        pass  # Cache not available
    
    try:
        from datasets import load_dataset
        
        # Parse hf://dataset_path[:config][:split]
        hf_path = path[5:]  # Remove 'hf://' prefix
        parts = hf_path.split(":")
        
        dataset_path = parts[0]
        config = parts[1] if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 else "train"
        dataset_id = dataset_path
        
        logger.info(
            f"HuggingFace streaming: dataset={dataset_path}, "
            f"config={config}, split={split}, cycle={cycle}"
        )
        
        # Calculate chunk index based on cycle for linear progression through dataset
        # chunk_index progresses linearly (0,1,2,3,...) to support ChunkTracker resume
        # cycle=0 uses chunk 0, cycle=1 uses chunk 1, cycle=25 uses chunk 25, etc.
        chunk_index = cycle
        
        # Check if cache exists for a different chunk size - if so, clear old caches
        if chunk_cache and cycle == 0:  # Only check on first cycle to avoid repeated checks
            # Check if any cached chunks exist with different max_lines
            try:
                has_different_size = chunk_cache.has_chunks_with_different_size(
                    dataset_path=dataset_path,
                    config=config,
                    split=split,
                    current_max_lines=max_lines
                )
                if has_different_size:
                    cleared_count = chunk_cache.clear_dataset_cache_except_size(
                        dataset_path=dataset_path,
                        config=config,
                        split=split,
                        keep_max_lines=max_lines
                    )
                    if cleared_count > 0:
                        logger.info(f"Cleared {cleared_count} old chunks with different chunk size for {dataset_path}")
            except Exception as e:
                logger.debug(f"Cache check failed: {e}")
        
        # Try to get from cache first
        if chunk_cache:
            cached_lines = chunk_cache.get_cached_chunk(
                dataset_path=dataset_path,
                config=config,
                split=split,
                chunk_index=chunk_index,
                max_age_hours=72.0,  # Cache valid for 3 days
                max_lines=max_lines  # Include chunk size in cache key
            )
            if cached_lines:
                try:
                    # Calculate block/chunk info for display (using standard 100k samples/block)
                    samples_per_block = 100000
                    chunks_per_block = samples_per_block // max_lines
                    block_id = chunk_index // chunks_per_block
                    chunk_in_block = chunk_index % chunks_per_block
                    logger.info(f"Using cached chunk for {dataset_path} (Block {block_id}, Chunk {chunk_in_block}, {len(cached_lines)} lines)")
                except Exception:
                    pass  # Ignore encoding errors in print
                # Return the requested number of lines from cache
                return cached_lines[:max_lines]
        
        logger.info(f"Downloading chunk {chunk_index} from HuggingFace Hub...")
        
        # Register training with stream manager
        if stream_mgr:
            can_proceed, reason = stream_mgr.can_train(dataset_path)
            if can_proceed:
                success, msg = stream_mgr.register_training(dataset_path)
                if success and "pause" in reason.lower():
                    logger.info(f"Training registered: {reason}")
        
        # Load with streaming to avoid downloading entire dataset
        dataset = load_dataset(
            dataset_path,
            name=config,
            split=split,
            streaming=True
        )
        
        # Try common text column names
        text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body', 'input', 'output']
        found_column = None
        
        # Peek at first item to find text column
        try:
            first_item = next(iter(dataset))
            if isinstance(first_item, dict):
                # Check for standard text columns
                for col in text_columns:
                    if col in first_item and first_item[col]:
                        found_column = col
                        break
                # If no standard column, use first string-like column
                if found_column is None:
                    for key, value in first_item.items():
                        if isinstance(value, str) and value.strip():
                            found_column = key
                            break
            logger.debug(f"Using text column: {found_column}")
        except Exception as e:
            logger.warning(f"Failed to detect text column: {e}")
            found_column = "text"  # Default fallback
        
        # Extract lines with limit
        out: List[str] = []
        if found_column:
            for item in dataset:
                if len(out) >= max_lines:
                    break
                try:
                    if isinstance(item, dict):
                        text = item.get(found_column, "")  # type: ignore
                    else:
                        text = item[found_column]  # type: ignore
                    if text and str(text).strip():
                        out.append(str(text).strip())
                except Exception:
                    continue
        
        logger.info(f"Downloaded {len(out)} lines from HuggingFace Hub")
        
        # Cache the downloaded chunk for future use
        if chunk_cache and out:
            try:
                chunk_cache.save_chunk(
                    dataset_path=dataset_path,
                    config=config,
                    split=split,
                    chunk_index=chunk_index,
                    lines=out,
                    max_lines=max_lines  # Include chunk size in cache key
                )
                logger.debug(f"Saved chunk to cache: chunk {chunk_index}, {len(out)} lines")
            except Exception as e:
                logger.warning(f"Failed to cache chunk: {e}")
        
        # Unregister training
        if stream_mgr and dataset_id:
            stream_mgr.unregister_training(dataset_id)
        
        return out
        
    except ImportError:
        logger.error("datasets library not installed. Cannot stream from HuggingFace.")
        logger.warning("Install datasets library with: pip install datasets")
        # Unregister training on error
        if stream_mgr and dataset_id:
            stream_mgr.unregister_training(dataset_id)
        return []
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset {path}: {e}")
        # Unregister training on error
        if stream_mgr and dataset_id:
            stream_mgr.unregister_training(dataset_id)
        return []


def _read_directory(p: Path, max_lines: int) -> List[str]:
    """Read text from a directory (supports HuggingFace datasets or plain text files)."""
    # Check if this is a HuggingFace dataset directory
    # HF datasets have dataset_info.json, data/ directory, or .arrow files
    is_hf_dataset = False
    try:
        if (p / "dataset_info.json").exists() or (p / "data").is_dir() or any(p.glob("*.arrow")):
            is_hf_dataset = True
            logger.info(f"Detected HuggingFace dataset directory: {p}")
    except Exception:
        pass
    
    # Try loading as HuggingFace dataset first
    if is_hf_dataset:
        try:
            from datasets import load_from_disk
            logger.info(f"Loading HuggingFace dataset from disk: {p}")
            dataset = load_from_disk(str(p))
            
            # Extract text from the dataset
            out: List[str] = []
            
            # Try common text column names
            text_columns = ['text', 'content', 'sentence', 'document', 'article', 'body']
            found_column = None
            
            # Check which columns exist
            if hasattr(dataset, 'column_names'):
                available_columns = dataset.column_names
                logger.debug(f"Available columns: {available_columns}")
                for col in text_columns:
                    if col in available_columns:
                        found_column = col
                        break
                
                # If no standard text column, use the first string column
                if found_column is None and len(available_columns) > 0:
                    try:
                        found_column = str(list(available_columns)[0])
                    except Exception:
                        pass
            
            logger.info(f"Using column '{found_column}' for text extraction")
            
            # Extract lines from dataset
            if found_column:
                count = 0
                for item in dataset:
                    if count >= max_lines:
                        break
                    try:
                        # Handle dict-like items
                        if hasattr(item, 'get'):
                            text = item.get(found_column, '')  # type: ignore
                        elif hasattr(item, '__getitem__'):
                            text = item[found_column]  # type: ignore
                        else:
                            continue
                            
                        if text and str(text).strip():
                            out.append(str(text).strip())
                            count += 1
                    except Exception:
                        continue
                
                if out:
                    logger.info(f"Extracted {len(out)} lines from HuggingFace dataset")
                    return out
        except ImportError:
            logger.debug("datasets library not available, falling back to file scanning")
            # datasets library not available, fall back to file scanning
            pass
        except Exception as e:
            logger.warning(f"HF dataset loading failed, falling back to file scanning: {e}")
            # HF dataset loading failed, fall back to file scanning
            pass
    
    # Fallback: Scan text-like files in directory
    logger.info(f"Scanning directory for text files: {p}")
    out: List[str] = []
    file_count = 0
    for fp in p.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in TEXT_EXTS:
            file_count += 1
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        out.append(line.strip())
                        if len(out) >= max_lines:
                            logger.info(f"Read {len(out)} lines from {file_count} text files")
                            return out
            except Exception as e:
                logger.debug(f"Failed to read file {fp}: {e}")
                continue
    
    logger.info(f"Read {len(out)} lines from {file_count} text files")
    return out
