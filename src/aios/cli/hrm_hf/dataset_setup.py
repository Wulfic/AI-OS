"""Dataset and model configuration setup."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Any

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def setup_dataset_mode(
    config: "TrainingConfig",
    lines: list[str],
    log_fn,
    rank: int = 0,
    world_size: int = 1,
    tok: Optional[Any] = None,
) -> Tuple[Any, Any, Optional[Any], int]:
    """Setup dataset for streaming or eager loading.
    
    Args:
        config: Training configuration
        lines: List of text lines
        log_fn: Logging function
        rank: DDP rank (default: 0)
        world_size: DDP world size (default: 1)
    
    Returns:
        Tuple of (input_ids, labels, streaming_dataset, N)
    """
    from .encoding import encode_lines as _encode_lines_helper
    from ..hrm_hf_utils import load_tokenizer as _load_tokenizer_helper
    
    # Use provided tokenizer (from brain metadata) when available to ensure consistency
    if tok is None:
        tok = _load_tokenizer_helper(config.model)
    num_lines = len(lines)
    max_seq_len = config.max_seq_len
    batch_size = config.batch_size
    
    # Estimate memory usage
    estimated_dataset_gb = (num_lines * max_seq_len * 4 * 2) / (1024 ** 3)
    # Use streaming only if dataset is large (no hardcoded context length limit)
    use_streaming = estimated_dataset_gb > 4.0
    
    streaming_dataset = None
    
    if use_streaming:
        log_fn({
            "dataset_loading": "streaming",
            "reason": f"Dataset would use {estimated_dataset_gb:.2f} GB (>4.0 GB threshold)",
            "num_samples": num_lines,
            "max_seq_len": max_seq_len,
        })
        
        from .streaming_dataset import create_streaming_dataset
        
        shuffle_mode = not config.linear_dataset
        start_offset = config.dataset_start_offset if config.linear_dataset else 0
        
        streaming_dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tok,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=shuffle_mode,
            epoch=0,
            start_offset=start_offset,
            rank=rank,
            world_size=world_size,
        )
        input_ids = None
        labels = None
        N = len(lines)
        
        mode_str = "linear (sequential)" if config.linear_dataset else "shuffled"
        log_fn({
            "dataset_mode": "streaming",
            "progression": mode_str,
            "initial_samples": N,
            "epoch": 0,
            "start_offset": start_offset if config.linear_dataset else 0
        })
    else:
        log_fn({
            "dataset_loading": "eager (all in memory)",
            "estimated_gb": f"{estimated_dataset_gb:.2f}",
            "num_samples": num_lines,
        })
        input_ids, labels = _encode_lines_helper(tok, lines, max_seq_len)
        
        # Early validation: ensure token IDs are within tokenizer vocab (catch mismatch ASAP)
        try:
            max_token_id = int(input_ids.max().item()) if hasattr(input_ids, 'numel') and input_ids.numel() > 0 else -1
        except Exception:
            max_token_id = -1
        # Use len(tok) to include special tokens, not just vocab_size
        actual_vocab_size = len(tok) if hasattr(tok, '__len__') else getattr(tok, 'vocab_size', 0)
        if actual_vocab_size > 0 and max_token_id >= actual_vocab_size:
            log_fn({
                "error": "tokenizer_mismatch_detected",
                "max_token_id": max_token_id,
                "tokenizer_vocab_size": actual_vocab_size,
                "hint": "Your dataset was likely tokenized with a different tokenizer than the model. Ensure the dataset/tokenizer matches the brain's tokenizer.",
            })
            raise RuntimeError(
                f"Token IDs exceed tokenizer vocab (max_id={max_token_id} >= vocab_size={actual_vocab_size}). "
                "Please use the same tokenizer for dataset and model (brain metadata)."
            )
        
        # DDP: Shard dataset across ranks for eager loading
        if world_size > 1:
            total_samples = input_ids.shape[0]
            samples_per_rank = total_samples // world_size
            rank_start = rank * samples_per_rank
            rank_end = rank_start + samples_per_rank if rank < world_size - 1 else total_samples
            
            input_ids = input_ids[rank_start:rank_end]
            labels = labels[rank_start:rank_end]
            
            log_fn({
                "ddp_eager_sharding": "enabled",
                "rank": rank,
                "world_size": world_size,
                "rank_samples": input_ids.shape[0],
                "total_samples": total_samples,
                "note": "Each GPU processes different data samples"
            })
        
        N = input_ids.shape[0]
    
    return input_ids, labels, streaming_dataset, N


def calculate_warmup_and_coverage(config: "TrainingConfig", num_lines: int, log_fn) -> int:
    """Calculate warmup steps and validate dataset coverage.
    
    Returns:
        Number of warmup steps
    """
    # Calculate warmup: 10% of total steps or 200, whichever is smaller
    warmup_steps = min(200, max(10, config.steps // 10))
    
    # Validate coverage
    expected_samples_per_epoch = config.steps * config.batch_size
    coverage_pct = (expected_samples_per_epoch / num_lines) * 100 if num_lines > 0 else 0
    
    if coverage_pct < 50 and num_lines > 100:
        log_fn({
            "warning": "low_dataset_coverage",
            "dataset_samples": num_lines,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "samples_per_epoch": expected_samples_per_epoch,
            "coverage_percent": round(coverage_pct, 1),
            "recommendation": f"Increase --steps to {num_lines // config.batch_size} for full dataset coverage",
            "impact": "Low coverage means the model sees only a small fraction of your data each epoch",
        })
    
    return warmup_steps


def setup_epoch_tracking(config: "TrainingConfig", log_fn) -> None:
    """Initialize or restore epoch tracking."""
    from .dataset_size_detection import detect_dataset_size
    
    if config.dataset_total_samples is None:
        # Validate dataset_file before attempting detection
        dataset_file = getattr(config, "dataset_file", None)
        if not isinstance(dataset_file, str) or not dataset_file:
            log_fn({
                "epoch_tracking": "disabled",
                "reason": "No dataset_file provided",
                "note": "Provide --dataset-file to enable epoch tracking",
            })
            return
        
        # First time training - detect dataset size
        dataset_total_samples, samples_per_block, total_blocks = detect_dataset_size(
            dataset_file=dataset_file,
            ascii_only=config.ascii_only,
        )
        
        if dataset_total_samples is not None:
            config.dataset_total_samples = dataset_total_samples
            config.samples_per_block = samples_per_block
            config.total_blocks = total_blocks
            
            # Calculate chunks per block for GUI display
            dataset_chunk_size = getattr(config, 'dataset_chunk_size', 4000)
            chunks_per_block = max(1, (int(samples_per_block) + int(dataset_chunk_size) - 1) // int(dataset_chunk_size))
            
            log_fn({
                "epoch_tracking": "initialized",
                "dataset_total_samples": dataset_total_samples,
                "samples_per_block": samples_per_block,
                "total_blocks": total_blocks,
                "dataset_chunk_size": dataset_chunk_size,
                "chunks_per_block": chunks_per_block,
                "note": "Epoch = one complete pass through all dataset blocks",
            })
        else:
            log_fn({
                "epoch_tracking": "disabled",
                "reason": "Could not detect dataset size",
                "note": "Epoch tracking requires known dataset size",
            })
    else:
        # Resuming from checkpoint
        # Calculate chunks per block for GUI display
        dataset_chunk_size = getattr(config, 'dataset_chunk_size', 4000)
        chunks_per_block = max(1, (int(config.samples_per_block) + int(dataset_chunk_size) - 1) // int(dataset_chunk_size)) if config.samples_per_block else None
        
        log_fn({
            "epoch_tracking": "restored",
            "dataset_total_samples": config.dataset_total_samples,
            "samples_per_block": config.samples_per_block,
            "total_blocks": config.total_blocks,
            "current_epoch": config.current_epoch,
            "samples_processed_this_epoch": config.samples_processed_this_epoch,
            "dataset_chunk_size": dataset_chunk_size,
            "chunks_per_block": chunks_per_block,
        })
