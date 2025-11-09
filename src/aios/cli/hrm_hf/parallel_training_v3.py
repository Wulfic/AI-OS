"""
Parallel Independent Training V3 - Block/Chunk Distribution System
Properly distributes dataset blocks and chunks to GPUs without duplication.

Architecture:
- **Block**: Large dataset chunk (default 100k items) streamed from HuggingFace or loaded from file
- **Chunk**: Subdivision of a block (e.g., 100 items) that gets processed by one GPU
- **Step**: One training iteration (forward + backward pass processing batch_size items)
- **Parallel Processing**: Each GPU works independently. --steps N means EACH GPU does N steps.
- **Global Step Counter**: Running total of all steps across all GPUs (for statistics only)

Example with 100k block, chunk_size=100, batch_size=1, 2 GPUs, --steps 100:
  - Block 0: 100,000 items split into 1,000 chunks of 100 items each
  - GPU 0: Claims chunks 0, 2, 4... processes 100 steps
  - GPU 1: Claims chunks 1, 3, 5... processes 100 steps
  - Global step counter: 200 total steps

Features:
- Downloads HF datasets in blocks (default 100k items)
- Distributes unique chunks to each GPU from current block
- Tracks which chunks have been trained (no duplication)
- Supports stopping conditions: steps, stop_after_block, stop_on_epoch
- Handles iterate mode for continuous training
- Detects epoch completion across all blocks
"""

from __future__ import annotations

import os
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

# Set HF_HOME before transformers imports
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)

import torch
import torch.amp
from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

# Import infrastructure
from ..hrm_hf_utils import load_tokenizer as _load_tokenizer
from .model_building import (
    calculate_vocab_size,
    build_model_config,
    build_model,
)
from .optimizer_setup import create_optimizer
from .encoding import adjust_tokenizer_padding, encode_lines
from .memory_optimization import configure_chunking
from .config_validation import auto_adjust_moe_learning_rate
from .block_manager import BlockManager, DataBlock
from .chunk_tracker import ChunkTracker


def train_gpu_worker(
    config: "TrainingConfig",
    gpu_id: int,
    checkpoint_dir: Path,
    tokenizer,
    vocab_size: int,
    block_manager: BlockManager,
    chunk_tracker: ChunkTracker,
    write_jsonl,
    device_index: int = 0,
    stop_event: Optional[Any] = None,
    graceful_stop_event: Optional[Any] = None,
) -> str:
    """GPU worker that trains on unique chunks from blocks.
    
    Each GPU:
    1. Claims an untrained chunk from the current block
    2. Trains on that chunk
    3. Reports progress to ChunkTracker
    4. Repeats until stopping condition met
    
    Args:
        config: Training configuration
        gpu_id: GPU device ID (logical ID for display/logging)
        checkpoint_dir: Directory to save checkpoints
        tokenizer: Tokenizer for encoding
        vocab_size: Vocabulary size
        block_manager: Manages dataset blocks
        chunk_tracker: Tracks training progress
        write_jsonl: Function to write JSONL metrics
        device_index: PyTorch device index (remapped by CUDA_VISIBLE_DEVICES)
        stop_event: Event to signal immediate stop
        graceful_stop_event: Event to signal graceful stop (finish current chunk)
        
    Returns:
        Path to final checkpoint
    """
    import signal
    import sys
    
    # Install signal handler to catch termination
    def signal_handler(signum, frame):
        try:
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        except:
            sig_name = str(signum)
        print(f"\n[GPU {gpu_id}] !!!!! SIGNAL RECEIVED: {sig_name} ({signum}) !!!!!", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # Don't exit immediately - let the process finish gracefully
        # sys.exit(1)
    
    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
        print(f"[GPU {gpu_id}] Signal handlers installed", flush=True)
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to install signal handlers: {e}", flush=True)
    
    # CRITICAL: Use the actual GPU ID (gpu_id), not the device_index
    # When CUDA_VISIBLE_DEVICES is not set, we need to select the physical GPU
    # device_index is only used when CUDA_VISIBLE_DEVICES successfully remaps devices
    
    # Check if CUDA_VISIBLE_DEVICES was set
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # CUDA_VISIBLE_DEVICES is set - use remapped index
        device = torch.device(f'cuda:{device_index}')
        print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES={cuda_visible}, using remapped index {device_index}")
    else:
        # CUDA_VISIBLE_DEVICES not set - use actual GPU ID
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES not set, using physical GPU {gpu_id}")
    
    torch.cuda.set_device(device)
    
    # Verify which GPU we're actually using
    actual_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(actual_device)
    print(f"[GPU {gpu_id}] Active CUDA device: {actual_device} ({device_name})")
    
    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] Starting Training Worker")
    print(f"{'='*60}\n")
    
    # Build model
    model_config = build_model_config(
        config=config,
        vocab_size=vocab_size,
        log_fn=print
    )
    
    model = build_model(
        config=model_config,
        student_init=config.student_init,
        log_fn=print
    )
    model = model.to(device)
    
    # Clear CUDA cache before loading optimizer to maximize available memory
    torch.cuda.empty_cache()
    
    # Auto-adjust learning rate for MoE if enabled
    # CRITICAL: Must happen before optimizer creation
    adjusted_lr = auto_adjust_moe_learning_rate(config, print)
    if adjusted_lr != config.lr:
        print(f"[GPU {gpu_id}] Learning rate adjusted: {config.lr} -> {adjusted_lr}")
        # Create a copy of config with adjusted LR for this GPU worker
        config.lr = adjusted_lr
    
    # Build optimizer
    optimizer = create_optimizer(
        model=model,
        config=config,
        use_deepspeed_optimizer=False,
        log_fn=print
    )
    
    # Configure chunking
    segment_rollout, use_chunking, final_chunk_size = configure_chunking(
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
        use_chunked_training=config.use_chunked_training,
        gradient_checkpointing=config.gradient_checkpointing,
        use_cpu_offload=config.use_cpu_offload,
        log_fn=print
    )
    
    # Setup AMP
    use_amp = config.use_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)  # type: ignore[attr-defined]
    
    # Training state
    model.train()
    total_steps_this_gpu = 0  # Optimizer steps (after gradient accumulation)
    total_micro_batches_this_gpu = 0  # True training steps (all forward/backward passes)
    current_block_id = 0
    graceful_stop_pending = False  # Track graceful stop request
    
    print(f"[GPU {gpu_id}] Starting training loop...")
    
    try:
        while True:
            # Check for immediate stop event
            if stop_event and stop_event.is_set():
                print(f"[GPU {gpu_id}] Immediate stop requested - terminating now", flush=True)
                break
            
            # Check for graceful stop event - set flag to stop after current chunk
            if graceful_stop_event and graceful_stop_event.is_set() and not graceful_stop_pending:
                graceful_stop_pending = True
                print(f"[GPU {gpu_id}] Graceful stop requested - will finish current chunk then exit", flush=True)
            
            # If graceful stop is pending and we're between chunks, exit now
            if graceful_stop_pending:
                print(f"[GPU {gpu_id}] Graceful stop honored - exiting between chunks", flush=True)
                break
            
            # No total step limit check here - iterate mode continues indefinitely
            # --steps parameter controls steps PER CHUNK, not total
            # Training stops when:
            #   1. Manual stop (immediate or graceful)
            #   2. stop_after_block is enabled and block completes
            #   3. stop_after_epoch is enabled and epoch completes
            #   4. iterate is disabled and dataset exhausted
            
            # Get current block with progress feedback
            def progress_callback(msg: str) -> None:
                """Print progress updates from block manager."""
                print(msg)
            
            block = block_manager.get_block(current_block_id, progress_callback=progress_callback)
            
            if block is None:
                # No more blocks - epoch complete
                print(f"[GPU {gpu_id}] Block {current_block_id} not available (end of dataset)")
                
                # Check if epoch complete
                total_blocks = block_manager.get_total_blocks()
                
                # Update chunk_tracker with total blocks if detected for first time
                if total_blocks and chunk_tracker.total_blocks_in_dataset is None:
                    chunk_tracker.total_blocks_in_dataset = total_blocks
                    print(f"[GPU {gpu_id}] Detected {total_blocks} total blocks in dataset")
                    
                    # Log to GUI
                    write_jsonl({
                        "event": "blocks_detected",
                        "total_blocks": total_blocks,
                        "current_block": current_block_id,
                    })
                
                if total_blocks and chunk_tracker.check_epoch_complete(total_blocks):
                    print(f"[GPU {gpu_id}] Epoch {chunk_tracker.current_epoch} complete!")
                    
                    # Iterate mode takes priority - continue training
                    if config.iterate:
                        # Check if stop_after_epoch is set (one-time stop flag)
                        if config.stop_after_epoch:
                            print(f"[GPU {gpu_id}] Epoch complete + stop_after_epoch set, stopping")
                            break
                        
                        print(f"[GPU {gpu_id}] Iterate mode enabled, starting new epoch")
                        chunk_tracker.start_new_epoch()
                        block_manager.reset()
                        current_block_id = 0
                        continue
                    else:
                        print(f"[GPU {gpu_id}] Iterate mode disabled, stopping after epoch")
                        break
                else:
                    # Dataset exhausted but epoch not complete
                    print(f"[GPU {gpu_id}] Dataset exhausted (epoch not complete), stopping")
                    break
            
            # Check for graceful stop BEFORE claiming next chunk
            if graceful_stop_event and graceful_stop_event.is_set():
                print(f"[GPU {gpu_id}] Graceful stop detected before claiming next chunk - exiting cleanly", flush=True)
                break
            
            # Get next untrained chunk from this block
            chunk_size = config.dataset_chunk_size
            total_chunks_in_block = block.chunk_count(chunk_size)
            
            chunk_id = chunk_tracker.get_next_untrained_chunk(
                block_id=current_block_id,
                total_chunks_in_block=total_chunks_in_block,
                gpu_id=gpu_id
            )
            
            if chunk_id is None:
                # Block exhausted, check if we should stop
                print(f"[GPU {gpu_id}] Block {current_block_id} fully trained")
                chunk_tracker.mark_block_complete(current_block_id)
                
                # Free the block from memory now that it's complete
                # This prevents multiple blocks accumulating in memory
                block_manager.free_block(current_block_id)
                
                if config.stop_after_block:
                    print(f"[GPU {gpu_id}] stop_after_block enabled, stopping")
                    break
                
                # Move to next block
                current_block_id += 1
                continue
            
            # Check for graceful stop AGAIN after claiming chunk (catch race condition)
            # Event might have been set between the check above and chunk claim
            if graceful_stop_event and graceful_stop_event.is_set():
                print(f"[GPU {gpu_id}] Graceful stop detected after claiming chunk {chunk_id} - exiting without training it", flush=True)
                # Note: Chunk was claimed but not trained, so it will remain available for next session
                break
            
            # Load ONLY this chunk (e.g., 100 samples) instead of full block (100k samples)
            # This is the key memory optimization - 1000x reduction
            chunk_samples = block_manager.get_chunk(current_block_id, chunk_id, chunk_size)
            
            if not chunk_samples:
                print(f"[GPU {gpu_id}] Warning: Empty chunk {chunk_id} in block {current_block_id}")
                continue
            
            # Calculate number of training steps needed for this chunk
            # Account for gradient accumulation when calculating steps
            grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
            effective_batch_size = config.batch_size * grad_accum_steps
            
            # Limit to config.steps (optimizer steps per chunk, not micro-batches)
            max_steps_this_chunk = config.steps
            # Calculate optimizer steps (not micro-batch steps)
            optimizer_steps_in_chunk = (len(chunk_samples) + effective_batch_size - 1) // effective_batch_size
            steps_to_train = min(optimizer_steps_in_chunk, max_steps_this_chunk)
            
            print(f"\n[GPU {gpu_id}] > Training Block {current_block_id} Chunk {chunk_id}: "
                  f"{len(chunk_samples)} samples, training {steps_to_train} optimizer steps "
                  f"(batch_size={config.batch_size}, grad_accum={grad_accum_steps}, effective_batch={effective_batch_size})")
            
            # Train on chunk samples (limited by config.steps)
            samples_trained = 0
            chunk_losses = []
            step_in_chunk = 0  # Counts optimizer steps, not micro-batches
            micro_batch_count = 0  # Counts micro-batches for gradient accumulation
            
            try:
                for batch_start in range(0, len(chunk_samples), config.batch_size):
                    # Check for immediate stop during training
                    if stop_event and stop_event.is_set():
                        print(f"[GPU {gpu_id}] Immediate stop requested during chunk - terminating now", flush=True)
                        graceful_stop_pending = False  # Override graceful stop for immediate exit
                        break
                    
                    # Check for graceful stop during chunk training
                    # Set flag but DON'T break - let the entire chunk complete
                    if graceful_stop_event and graceful_stop_event.is_set() and not graceful_stop_pending:
                        graceful_stop_pending = True
                        print(f"[GPU {gpu_id}] Graceful stop detected - will finish current chunk then exit", flush=True)
                    
                    # Stop if we've reached the per-chunk step limit
                    if step_in_chunk >= steps_to_train:
                        print(f"[GPU {gpu_id}] Reached per-chunk step limit ({steps_to_train}), moving to next chunk", flush=True)
                        break
                    
                    batch_end = min(batch_start + config.batch_size, len(chunk_samples))
                    batch_lines = chunk_samples[batch_start:batch_end]
                    
                    if not batch_lines:
                        continue
                    
                    # Encode batch
                    input_ids, labels = encode_lines(tokenizer, batch_lines, config.max_seq_len)
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    
                    # Prepare batch
                    batch = {
                        'inputs': input_ids,
                        'targets': labels,
                        'puzzle_identifiers': torch.arange(len(batch_lines), device=device)
                    }
                    
                    # Forward pass
                    # Only zero gradients at the start of accumulation cycle
                    is_accumulation_start = (micro_batch_count % grad_accum_steps == 0)
                    if is_accumulation_start:
                        optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda', enabled=use_amp):  # type: ignore[attr-defined]
                        loss, metrics = segment_rollout(
                            model=model,
                            batch=batch,
                            max_segments=config.halt_max_steps,
                            epsilon=0.0
                        )
                    
                    # Scale loss by accumulation steps for proper gradient averaging
                    scaled_loss = loss / grad_accum_steps
                    
                    # Backward pass
                    if use_amp:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    
                    micro_batch_count += 1
                    total_micro_batches_this_gpu += 1  # Track total true steps across all chunks
                    chunk_losses.append(loss.item())  # Store unscaled loss for logging
                    samples_trained += len(batch_lines)
                    
                    # Only step optimizer after accumulating gradients
                    is_accumulation_end = (micro_batch_count % grad_accum_steps == 0)
                    if is_accumulation_end or batch_start + config.batch_size >= len(chunk_samples):
                        # This is an optimizer step
                        if use_amp:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                            
                            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                                scaler.step(optimizer)
                            scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                            
                            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                                optimizer.step()
                        
                        total_steps_this_gpu += 1
                        step_in_chunk += 1
                        
                        # Report progress every optimizer step or at end of chunk
                        # Only report AFTER completing an optimizer step to avoid duplicate prints during gradient accumulation
                        if step_in_chunk >= steps_to_train or batch_start + config.batch_size >= len(chunk_samples) or True:
                            avg_loss = sum(chunk_losses[-min(5*grad_accum_steps, len(chunk_losses)):]) / min(5*grad_accum_steps, len(chunk_losses))
                            print(f"[GPU {gpu_id}] Optimizer Step: {step_in_chunk}/{steps_to_train} | "
                                  f"Block {current_block_id} Chunk {chunk_id} | Loss={avg_loss:.4f}")
            
            except Exception as batch_error:
                # Catch any exception during batch processing to prevent silent crashes
                print(f"\n[ERROR] GPU {gpu_id} EXCEPTION during batch processing: {batch_error}", flush=True)
                import traceback
                traceback.print_exc()
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                # Re-raise to trigger proper cleanup
                raise
            
            # Mark chunk complete
            # Mark chunk complete
            print(f"[GPU {gpu_id}] Batch loop done for chunk {chunk_id} (graceful_stop={graceful_stop_pending}, steps={step_in_chunk})", flush=True)
            
            avg_chunk_loss = sum(chunk_losses) / len(chunk_losses) if chunk_losses else 0.0
            chunk_tracker.mark_chunk_complete(
                block_id=current_block_id,
                chunk_id=chunk_id,
                gpu_id=gpu_id,
                step=total_steps_this_gpu,
                samples_trained=samples_trained,
                true_steps=micro_batch_count  # Pass the number of micro-batches processed in this chunk
            )
            
            # Log progress stats to GUI
            stats = chunk_tracker.get_progress_stats()
            
            # Get current total blocks if detected
            total_blocks = block_manager.get_total_blocks()
            if total_blocks and chunk_tracker.total_blocks_in_dataset is None:
                chunk_tracker.total_blocks_in_dataset = total_blocks
            
            write_jsonl({
                "event": "chunk_complete",
                "gpu_id": gpu_id,
                "block_id": current_block_id,
                "chunk_id": chunk_id,
                "step": total_steps_this_gpu,
                "loss": avg_chunk_loss,
                "session_steps": stats.get("session_steps", total_steps_this_gpu),  # Current session optimizer steps
                "session_true_steps": stats.get("session_true_steps", 0),  # Current session true steps (micro-batches)
                "total_true_steps": stats.get("total_true_steps", 0),  # All-time total true steps (all sessions)
                "total_gpu_steps": stats["total_gpu_steps"],  # Historical max across all chunks/sessions
                "total_chunks_trained": stats["total_chunks_trained"],
                "blocks_completed": stats["blocks_completed"],
                "current_epoch": stats["current_epoch"],
                "total_blocks": stats.get("total_blocks_in_dataset"),
            })
            
            print(f"[GPU {gpu_id}] * Completed Block {current_block_id} Chunk {chunk_id}: "
                  f"{step_in_chunk} optimizer steps (limit: {steps_to_train}) | "
                  f"{micro_batch_count} micro-batches | "
                  f"GPU Total: {total_steps_this_gpu} opt steps ({total_micro_batches_this_gpu} true steps) | "
                  f"Avg Loss: {avg_chunk_loss:.4f}")
            
            # If immediate stop was triggered during chunk, break outer loop
            if stop_event and stop_event.is_set():
                print(f"[GPU {gpu_id}] Immediate stop confirmed - exiting", flush=True)
                break
            
            # Check for graceful stop after chunk completion
            if graceful_stop_pending:
                print(f"[GPU {gpu_id}] Graceful stop honored after chunk completion", flush=True)
                break
            
            # If iterate is disabled, stop after training one chunk
            if not config.iterate:
                print(f"[GPU {gpu_id}] Iterate mode disabled - stopping after one chunk", flush=True)
                break
        
        # Save final checkpoint
        print(f"[GPU {gpu_id}] Saving final checkpoint...", flush=True)
        
        # Ensure checkpoint directory exists (in case worker process doesn't have it)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"gpu{gpu_id}_final.safetensors"
        
        from safetensors.torch import save_file as save_safetensors
        state_dict = model.state_dict()
        save_safetensors(state_dict, str(checkpoint_path))
        
        # Release state_dict to avoid file handle issues on Windows
        del state_dict
        
        print(f"[GPU {gpu_id}] Training complete! Steps: {total_steps_this_gpu}", flush=True)
        
    except Exception as e:
        print(f"\n[ERROR] GPU {gpu_id} training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        # Cleanup
        del model
        del optimizer
        torch.cuda.empty_cache()
    
    return str(checkpoint_path)


def merge_checkpoints(checkpoint_paths: list[str], output_path: str) -> None:
    """Merge multiple checkpoints by averaging weights."""
    import gc
    import time
    import platform
    
    print(f"\n{'='*60}")
    print(f"[MERGE] Merging {len(checkpoint_paths)} checkpoints...")
    print(f"{'='*60}")
    
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
    
    # Load all checkpoints
    state_dicts = []
    for p in checkpoint_paths:
        try:
            state_dict = load_safetensors(p, device='cpu')
            state_dicts.append(state_dict)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    
    if not state_dicts:
        raise ValueError("No valid checkpoints to merge")
    
    # Average weights
    merged_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        stacked = torch.stack(tensors)
        merged_state[key] = stacked.mean(dim=0)
    
    # Release source checkpoint file handles (critical on Windows)
    del state_dicts
    del tensors
    del stacked
    gc.collect()
    
    # Windows-specific: Wait for file handles to be released
    if platform.system() == 'Windows':
        time.sleep(0.5)
    
    # Save merged checkpoint
    save_safetensors(merged_state, output_path)
    print(f"[OK] Merged checkpoint saved: {output_path}\n")


def run_parallel_training_v3(
    config: "TrainingConfig",
    stop_event: Optional[Any] = None,
    graceful_stop_event: Optional[Any] = None,
) -> None:
    """Main orchestrator for block/chunk-based parallel training.
    
    Features:
    - Loads HF datasets in blocks (100k samples)
    - Distributes unique chunks to each GPU
    - Tracks training progress (no duplication)
    - Supports all stopping conditions
    - Handles iterate mode and epoch detection
    
    Args:
        config: Training configuration
        stop_event: Multiprocessing event for immediate stop
        graceful_stop_event: Multiprocessing event for graceful stop (finish chunk)
    """
    
    print("\n" + "="*60)
    print("PARALLEL TRAINING V3 - BLOCK/CHUNK DISTRIBUTION")
    print("="*60)
    print("Features:")
    print("  • Block-based dataset streaming (100k items/block)")
    print("  • Unique chunk distribution per GPU")
    print("  • Progress tracking (no duplicate training)")
    print("  • Stopping conditions: steps/block/epoch")
    print("  • Iterate mode support")
    print("="*60 + "\n")
    
    # Setup write_jsonl for GUI metrics logging
    from .training_helpers import write_jsonl as _write_jsonl_helper
    log_file = config.log_file if hasattr(config, 'log_file') else None
    
    def write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(
            log_file=log_file,
            payload=payload,
            is_distributed=False,
            rank_id=0
        )
    
    # Setup directories
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "parallel_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer - check brain.json in save_dir even if student_init is None (fresh start)
    from .brain_metadata import load_brain_metadata, extract_tokenizer_from_metadata
    brain_metadata = load_brain_metadata(
        student_init=config.student_init,
        log_fn=print,
        save_dir=str(save_dir)  # Fallback to brain directory for fresh starts
    )
    tokenizer_path = extract_tokenizer_from_metadata(brain_metadata, config.model)
    
    print(f"[LOAD] Loading tokenizer from: {tokenizer_path}")
    tokenizer = _load_tokenizer(tokenizer_path)
    adjust_tokenizer_padding(tokenizer)
    vocab_size = calculate_vocab_size(tokenizer, print)
    print(f"[OK] Vocabulary size: {vocab_size:,}\n")
    
    # Initialize BlockManager
    from aios.data.datasets import read_text_lines_sample_any
    
    if not config.dataset_file:
        raise ValueError("dataset_file is required for parallel training")
    
    print(f"[INIT] Initializing BlockManager (async)...")
    print(f"   Dataset: {config.dataset_file}")
    print(f"   Items per block: {config.samples_per_block:,}")
    print(f"   Chunk size: {config.dataset_chunk_size:,}")
    
    # Calculate total chunks per block
    chunks_per_block = config.samples_per_block // config.dataset_chunk_size
    print(f"   Chunks per block: {chunks_per_block:,}")
    
    # Create BlockManager - initialization is now lazy (non-blocking)
    # Blocks will be loaded on-demand when training starts
    block_manager = BlockManager(
        dataset_path=config.dataset_file,
        samples_per_block=config.samples_per_block,
        dataset_chunk_size=config.dataset_chunk_size,
        ascii_only=config.ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
        enable_prefetch=True,  # Lightweight metadata prefetch for next block
    )
    print(f"[OK] BlockManager initialized (ready for lazy loading)")
    print(f"[INFO] Blocks will be loaded on-demand during training\n")
    
    # Get total blocks if already detected (for local datasets)
    total_blocks_detected = block_manager.get_total_blocks()
    if total_blocks_detected:
        print(f"[INFO] Detected {total_blocks_detected} total blocks in dataset")
    
    # Initialize ChunkTracker in brain directory for proper resume detection
    state_file = save_dir / "chunk_tracker_state.json"
    print(f"[INIT] Initializing ChunkTracker...")
    print(f"   Save directory (config.save_dir): {config.save_dir}")
    print(f"   Save directory (local save_dir var): {save_dir}")
    print(f"   Bundle directory: {config.bundle_dir}")
    print(f"   Brain name: {config.brain_name}")
    print(f"   State file: {state_file}")
    print(f"   Note: Checkpoint will be saved to: {save_dir / 'actv1_student.safetensors'}")
    
    chunk_tracker = ChunkTracker(state_file=state_file)
    
    # Set total blocks in ChunkTracker if detected
    if total_blocks_detected:
        chunk_tracker.total_blocks_in_dataset = total_blocks_detected
        print(f"[INFO] ChunkTracker configured with {total_blocks_detected} total blocks")
        
        # Log to GUI for progress tracking
        write_jsonl({
            "epoch_tracking": "initialized",
            "total_blocks": total_blocks_detected,
            "samples_per_block": config.samples_per_block,
            "dataset_chunk_size": config.dataset_chunk_size,
            "chunks_per_block": chunks_per_block,
            "note": "Block-based parallel training enabled"
        })
    else:
        # Log that blocks will be detected during training
        write_jsonl({
            "epoch_tracking": "disabled",
            "reason": "Total blocks not yet detected (HF streaming or first-time local dataset)",
            "samples_per_block": config.samples_per_block,
            "dataset_chunk_size": config.dataset_chunk_size,
            "chunks_per_block": chunks_per_block,
            "note": "Blocks will be detected during training"
        })
    
    # Show resumed state if applicable
    stats = chunk_tracker.get_progress_stats()
    if stats['total_chunks_trained'] > 0:
        print(f"[RESUME] Resuming from previous state:")
        print(f"   Chunks trained: {stats['total_chunks_trained']}")
        print(f"   Epoch: {stats['current_epoch']}")
        print(f"   Items trained: {stats['total_samples_trained']:,}")
    
    print(f"[OK] ChunkTracker initialized\n")
    
    # Parse cuda_ids
    cuda_ids = config.cuda_ids
    if isinstance(cuda_ids, str):
        cuda_ids = [int(x.strip()) for x in cuda_ids.split(',')]
    else:
        cuda_ids = list(cuda_ids) if cuda_ids else [0]
    
    num_gpus = len(cuda_ids)
    
    print(f"[PLAN] Training Plan:")
    print(f"   GPUs: {cuda_ids} ({num_gpus} GPUs)")
    print(f"   Steps PER CHUNK: {config.steps} (each chunk trains for this many steps)")
    print(f"   Stop after block: {config.stop_after_block}")
    print(f"   Stop after epoch: {config.stop_after_epoch}")
    print(f"   Iterate mode: {'ENABLED (continuous training)' if config.iterate else 'DISABLED (stop after dataset exhausted)'}")
    print()
    
    # Events are passed as arguments from the caller (GUI with multiprocessing.Manager)
    # No need to create them here
    
    # Launch GPU workers
    print(f"[PARALLEL] Launching {num_gpus} GPU workers...")
    threads = []
    checkpoint_paths_list: list[str | None] = [None] * num_gpus
    exceptions: list[Exception | None] = [None] * num_gpus
    
    def worker_wrapper(idx, gpu_id):
        """Wrapper to catch exceptions."""
        try:
            ckpt_path = train_gpu_worker(
                config=config,
                gpu_id=gpu_id,
                checkpoint_dir=checkpoint_dir,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                block_manager=block_manager,
                chunk_tracker=chunk_tracker,
                write_jsonl=write_jsonl,
                device_index=idx,  # Use index for PyTorch (0, 1, 2...) not original GPU ID
                stop_event=stop_event,
                graceful_stop_event=graceful_stop_event,
            )
            checkpoint_paths_list[idx] = ckpt_path
        except Exception as e:
            exceptions[idx] = e
            print(f"\n[ERROR] GPU {gpu_id} worker failed: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        # Launch all workers
        for i, gpu_id in enumerate(cuda_ids):
            t = threading.Thread(target=worker_wrapper, args=(i, gpu_id), daemon=False)
            threads.append(t)
            t.start()
            print(f"   [OK] Launched GPU {gpu_id} worker")
        
        print(f"\n[PARALLEL] All {num_gpus} workers running...\n")
        
        # Wait for completion
        for i, t in enumerate(threads):
            t.join()
            exc = exceptions[i]
            if exc is not None:
                raise exc
        
        print(f"\n[OK] All {num_gpus} workers completed!")
        
        # Save final tracker state
        chunk_tracker.save()
        
    except Exception as e:
        print(f"\n[FATAL] Parallel training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Merge checkpoints to expected resume location
    final_checkpoint_path = save_dir / "actv1_student.safetensors"
    valid_checkpoints = [str(p) for p in checkpoint_paths_list if p is not None]
    
    if len(valid_checkpoints) > 0:
        merge_checkpoints(valid_checkpoints, str(final_checkpoint_path))
        print(f"[OK] Merged checkpoint saved to: {final_checkpoint_path}")
    else:
        print("[WARN] No valid checkpoints to merge")
    
    # Print final stats
    final_stats = chunk_tracker.get_progress_stats()
    
    # Log final stats to GUI
    write_jsonl({
        "event": "training_complete",
        "session_steps": final_stats.get('session_steps', 0),
        "session_true_steps": final_stats.get('session_true_steps', 0),  # Aggregated true steps
        "total_gpu_steps": final_stats['total_gpu_steps'],
        "total_true_steps": final_stats.get('total_true_steps', 0),  # All-time aggregated true steps
        "total_samples_trained": final_stats['total_samples_trained'],
        "total_chunks_trained": final_stats['total_chunks_trained'],
        "session_chunks_trained": final_stats.get('session_chunks_trained', 0),
        "blocks_completed": final_stats['blocks_completed'],
        "current_epoch": final_stats['current_epoch'],
    })
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Statistics:")
    print(f"   Total optimizer steps (all GPUs): {final_stats['total_gpu_steps']}")
    print(f"   Total true steps (all GPUs): {final_stats.get('total_true_steps', 0)}")
    print(f"   Session true steps: {final_stats.get('session_true_steps', 0)}")
    print(f"   Items trained: {final_stats['total_samples_trained']:,}")
    print(f"   Chunks trained: {final_stats['total_chunks_trained']}")
    print(f"   Blocks completed: {final_stats['blocks_completed']}")
    print(f"   Current epoch: {final_stats['current_epoch']}")
    print(f"   Checkpoint: {final_checkpoint_path}")
    print(f"   Resume: Training can be resumed from this checkpoint")
    print("="*60 + "\n")
