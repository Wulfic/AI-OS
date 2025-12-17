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

import json
import os
import threading
import time
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
    stop_ack_event: Optional[Any] = None,
    graceful_stop_ack_event: Optional[Any] = None,
    tracker_gpu_id: Optional[int] = None,
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
        stop_ack_event: Event set when immediate stop is observed
        graceful_stop_ack_event: Event set when graceful stop is observed
        
    Returns:
        Path to final checkpoint
    """
    import signal
    import sys

    tracker_id = tracker_gpu_id if tracker_gpu_id is not None else gpu_id

    # Determine if we should emit progress updates to a shared file (DDP spawn path)
    progress_file: Optional[str] = None
    try:
        candidate = os.environ.get("AIOS_DDP_PROGRESS_FILE")
        if candidate:
            ddp_rank = None
            ddp_initialized = False
            try:
                import torch.distributed as dist  # type: ignore
                if dist.is_available() and dist.is_initialized():
                    ddp_rank = dist.get_rank()
                    ddp_initialized = True
            except Exception:
                ddp_initialized = False
                ddp_rank = None

            if ddp_initialized:
                if ddp_rank == 0:
                    progress_file = candidate
            else:
                # Fallback: allow tracker_id 0 to report when DDP not detected
                if tracker_id == 0:
                    progress_file = candidate
    except Exception:
        progress_file = None

    def _write_progress(
        *,
        step: int,
        total_steps: int,
        loss: Optional[float],
        block_id: int,
        chunk_id: int,
        global_optimizer_steps: int,
        samples_trained: int,
    ) -> None:
        if not progress_file:
            return
        try:
            payload = {
                "step": int(max(0, step)),
                "total_steps": int(max(1, total_steps)),
                "loss": float(loss) if loss is not None else None,
                "block_id": int(block_id),
                "chunk_id": int(chunk_id),
                "global_optimizer_steps": int(max(0, global_optimizer_steps)),
                "samples_trained": int(max(0, samples_trained)),
                "timestamp": time.time(),
            }
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump(payload, pf)
        except Exception:
            pass
    
    # Install signal handler to catch termination
    def signal_handler(signum, frame):
        try:
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        except:
            sig_name = str(signum)
        print(f"\n[GPU {tracker_id}] !!!!! SIGNAL RECEIVED: {sig_name} ({signum}) !!!!!", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        # Don't exit immediately - let the process finish gracefully
        # sys.exit(1)
    
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGBREAK'):  # Windows
                signal.signal(signal.SIGBREAK, signal_handler)
            print(f"[GPU {tracker_id}] Signal handlers installed", flush=True)
        except Exception as e:
            print(f"[GPU {tracker_id}] Failed to install signal handlers: {e}", flush=True)
    else:
        print(
            f"[GPU {tracker_id}] Skipping signal handlers (thread {threading.current_thread().name})",
            flush=True,
        )
    
    def _ack_immediate_stop() -> None:
        if stop_ack_event is None:
            return
        try:
            if not stop_ack_event.is_set():
                stop_ack_event.set()
        except Exception:
            pass

    def _ack_graceful_stop() -> None:
        if graceful_stop_ack_event is None:
            return
        try:
            if not graceful_stop_ack_event.is_set():
                graceful_stop_ack_event.set()
        except Exception:
            pass

    # CRITICAL: Use the actual GPU ID (gpu_id), not the device_index
    # When CUDA_VISIBLE_DEVICES is not set, we need to select the physical GPU
    # device_index is only used when CUDA_VISIBLE_DEVICES successfully remaps devices
    
    # Check if CUDA_VISIBLE_DEVICES was set
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # CUDA_VISIBLE_DEVICES is set - use remapped index
        device = torch.device(f'cuda:{device_index}')
        print(f"[GPU {tracker_id}] CUDA_VISIBLE_DEVICES={cuda_visible}, using remapped index {device_index}")
    else:
        # CUDA_VISIBLE_DEVICES not set - use actual GPU ID
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU {tracker_id}] CUDA_VISIBLE_DEVICES not set, using physical GPU {gpu_id}")
    
    torch.cuda.set_device(device)
    
    # Verify which GPU we're actually using
    actual_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(actual_device)
    print(f"[GPU {tracker_id}] Active CUDA device: {actual_device} ({device_name})")
    
    print(f"\n{'='*60}")
    print(f"[GPU {tracker_id}] Starting Training Worker")
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
        print(f"[GPU {tracker_id}] Learning rate adjusted: {config.lr} -> {adjusted_lr}")
        # Create a copy of config with adjusted LR for this GPU worker
        config.lr = adjusted_lr
    
    # Build optimizer
    optimizer = create_optimizer(
        model=model,
        config=config,
        use_deepspeed_optimizer=False,
        log_fn=print
    )

    # Optional: Adaptive LR scheduler (enables warmup in this path too).
    adaptive_lr_scheduler = None
    warmup_steps = 0
    base_lr = float(getattr(config, "lr", 0.0) or 0.0)
    if getattr(config, "auto_adjust_lr", False) and base_lr > 0:
        try:
            from .adaptive_lr import AdaptiveLRScheduler, build_adaptive_lr_config
            from pathlib import Path
            import json as _json

            use_moe_flag = bool(getattr(config, "use_moe", False))

            override_dict = {}
            if getattr(config, "adaptive_lr_debug_level", None) is not None:
                override_dict["debug_level"] = int(getattr(config, "adaptive_lr_debug_level"))
            if getattr(config, "adaptive_lr_emit_window_summary", None) is not None:
                override_dict["emit_window_summary"] = bool(getattr(config, "adaptive_lr_emit_window_summary"))
            if getattr(config, "adaptive_lr_window_summary_every", None) is not None:
                override_dict["window_summary_every"] = int(getattr(config, "adaptive_lr_window_summary_every"))

            cfg_lr = build_adaptive_lr_config(
                base_lr=base_lr,
                steps=int(getattr(config, "steps", 0) or 0) or None,
                use_moe=use_moe_flag,
                config_path=getattr(config, "adaptive_lr_config", None),
                override_dict=override_dict or None,
            )

            # Per-GPU state file to avoid contention
            state_path = getattr(config, "adaptive_lr_state_path", None)
            if not state_path:
                state_path = str(Path(checkpoint_dir) / f"adaptive_lr_state_gpu{tracker_id}.json")

            # Ensure all adaptive-lr JSONL events are attributed to the worker.
            def _adaptive_lr_log(payload: dict) -> None:
                try:
                    if isinstance(payload, dict) and "gpu_id" not in payload:
                        payload = {"gpu_id": tracker_id, **payload}
                    write_jsonl(payload)
                except Exception:
                    pass

            restored = False
            try:
                p_state = Path(state_path)
                p_state.parent.mkdir(parents=True, exist_ok=True)
                if bool(getattr(config, "resume", False)) and not bool(getattr(config, "adaptive_lr_reset_state", False)) and p_state.exists():
                    state_obj = _json.loads(p_state.read_text(encoding="utf-8"))
                    adaptive_lr_scheduler = AdaptiveLRScheduler.from_state_dict(
                        optimizer=optimizer,
                        state=state_obj,
                        log_fn=_adaptive_lr_log,
                        state_path=str(p_state),
                    )
                    # Prefer freshly-resolved cfg_lr while keeping restored stats.
                    try:
                        if adaptive_lr_scheduler is not None:
                            if int(getattr(adaptive_lr_scheduler.config, "window_size", 0) or 0) != int(getattr(cfg_lr, "window_size", 0) or 0):
                                old_vals = list(getattr(adaptive_lr_scheduler, "loss_window", []) or [])
                                from collections import deque as _deque
                                adaptive_lr_scheduler.loss_window = _deque(old_vals[-int(cfg_lr.window_size):], maxlen=int(cfg_lr.window_size))  # type: ignore[attr-defined]
                            adaptive_lr_scheduler.config = cfg_lr  # type: ignore[assignment]
                    except Exception:
                        pass
                    restored = True
                    write_jsonl(
                        {
                            "event": "adaptive_lr_state_restored",
                            "gpu_id": tracker_id,
                            "path": str(p_state),
                            "window_index": int(getattr(adaptive_lr_scheduler, "window_index", 0) or 0),
                            "step": int(getattr(adaptive_lr_scheduler, "total_observations", 0) or 0),
                            "lr": float(getattr(adaptive_lr_scheduler, "current_lr", base_lr) or base_lr),
                            "mode_requested": str(getattr(adaptive_lr_scheduler, "_mode_requested", "")),
                            "mode_active": str(getattr(adaptive_lr_scheduler, "_mode", "")),
                        }
                    )
            except Exception as _e_state:
                try:
                    write_jsonl({"event": "adaptive_lr_state_restore_failed", "gpu_id": tracker_id, "error": str(_e_state), "path": state_path})
                except Exception:
                    pass

            if adaptive_lr_scheduler is None:
                adaptive_lr_scheduler = AdaptiveLRScheduler(
                    optimizer=optimizer,
                    config=cfg_lr,
                    use_moe=use_moe_flag,
                    log_fn=_adaptive_lr_log,
                    state_path=str(state_path),
                )

            # Warmup matches the single/DDD HRM HF path: 10% of configured steps, capped.
            try:
                warmup_steps = min(200, max(10, int(getattr(config, "steps", 0) or 0) // 10))
            except Exception:
                warmup_steps = 10

            write_jsonl({
                "event": "adaptive_lr_enabled",
                "gpu_id": tracker_id,
                "base_lr": base_lr,
                "warmup_steps": warmup_steps,
                "window_size": cfg_lr.window_size,
                "lr_min": cfg_lr.lr_min,
                "lr_max": cfg_lr.lr_max,
                "use_moe": use_moe_flag,
                "config_path": getattr(config, "adaptive_lr_config", None),
                "debug_level": int(getattr(cfg_lr, "debug_level", 0) or 0),
                "emit_window_summary": bool(getattr(cfg_lr, "emit_window_summary", False) or int(getattr(cfg_lr, "debug_level", 0) or 0) >= 2),
                "window_summary_every": int(getattr(cfg_lr, "window_summary_every", 1) or 1),
                "state_path": str(state_path),
                "state_restored": bool(restored),
            })
        except Exception as _e:
            try:
                write_jsonl({"event": "adaptive_lr_init_failed", "gpu_id": tracker_id, "error": str(_e)})
            except Exception:
                pass
    
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
    total_samples_trained_this_gpu = 0  # Steps = samples/rows trained (for display)
    current_block_id = config.start_block_id  # Start from specified block
    graceful_stop_pending = False  # Track graceful stop request
    
    if config.start_block_id > 0 or config.start_chunk_id > 0:
        print(f"[GPU {tracker_id}] Starting from Block {config.start_block_id}, Chunk {config.start_chunk_id}")
    
    print(f"[GPU {tracker_id}] Starting training loop...")
    
    try:
        while True:
            # Check for immediate stop event
            if stop_event and stop_event.is_set():
                _ack_immediate_stop()
                print(f"[GPU {tracker_id}] Immediate stop requested - terminating now", flush=True)
                break
            
            # Check for graceful stop event - set flag to stop after current chunk
            if graceful_stop_event and graceful_stop_event.is_set() and not graceful_stop_pending:
                _ack_graceful_stop()
                graceful_stop_pending = True
                print(f"[GPU {tracker_id}] Graceful stop requested - will finish current chunk then exit", flush=True)
            
            # If graceful stop is pending and we're between chunks, exit now
            if graceful_stop_pending:
                print(f"[GPU {tracker_id}] Graceful stop honored - exiting between chunks", flush=True)
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
                print(f"[GPU {tracker_id}] Block {current_block_id} not available (end of dataset)")
                
                # Check if epoch complete
                total_blocks = block_manager.get_total_blocks()
                
                # Update chunk_tracker with total blocks if detected for first time
                if total_blocks and chunk_tracker.total_blocks_in_dataset is None:
                    chunk_tracker.total_blocks_in_dataset = total_blocks
                    chunk_tracker.save()  # Persist immediately so resume dialog can see it
                    print(f"[GPU {tracker_id}] Detected {total_blocks} total blocks in dataset")
                    
                    # Log to GUI
                    write_jsonl({
                        "event": "blocks_detected",
                        "total_blocks": total_blocks,
                        "current_block": current_block_id,
                    })
                
                if total_blocks and chunk_tracker.check_epoch_complete(total_blocks):
                    print(f"[GPU {tracker_id}] Epoch {chunk_tracker.current_epoch} complete!")
                    
                    # Iterate mode takes priority - continue training
                    if config.iterate:
                        # Check if stop_after_epoch is set (one-time stop flag)
                        if config.stop_after_epoch:
                            print(f"[GPU {tracker_id}] Epoch complete + stop_after_epoch set, stopping")
                            break
                        
                        print(f"[GPU {tracker_id}] Iterate mode enabled, starting new epoch")
                        chunk_tracker.start_new_epoch()
                        block_manager.reset()
                        current_block_id = config.start_block_id  # Respect start position across epochs
                        continue
                    else:
                        print(f"[GPU {tracker_id}] Iterate mode disabled, stopping after epoch")
                        break
                else:
                    # Dataset exhausted but epoch not complete
                    print(f"[GPU {tracker_id}] Dataset exhausted (epoch not complete), stopping")
                    break
            
            # Check for graceful stop BEFORE claiming next chunk
            if graceful_stop_event and graceful_stop_event.is_set():
                _ack_graceful_stop()
                print(f"[GPU {tracker_id}] Graceful stop detected before claiming next chunk - exiting cleanly", flush=True)
                break
            
            # Get next untrained chunk from this block
            chunk_size = config.dataset_chunk_size
            total_chunks_in_block = block.chunk_count(chunk_size)
            
            chunk_id = chunk_tracker.get_next_untrained_chunk(
                block_id=current_block_id,
                total_chunks_in_block=total_chunks_in_block,
                gpu_id=tracker_id,
            )
            if chunk_id is None:
                # Block exhausted, check if we should stop
                print(f"[GPU {tracker_id}] Block {current_block_id} fully trained")
                chunk_tracker.mark_block_complete(current_block_id)
                
                # Free the block from memory now that it's complete
                # This prevents multiple blocks accumulating in memory
                block_manager.free_block(current_block_id)
                
                if config.stop_after_block:
                    print(f"[GPU {tracker_id}] stop_after_block enabled, stopping")
                    break
                
                # Move to next block
                current_block_id += 1
                continue
            
            # Check for graceful stop AGAIN after claiming chunk (catch race condition)
            # Event might have been set between the check above and chunk claim
            if graceful_stop_event and graceful_stop_event.is_set():
                _ack_graceful_stop()
                print(f"[GPU {tracker_id}] Graceful stop detected after claiming chunk {chunk_id} - exiting without training it", flush=True)
                # Note: Chunk was claimed but not trained, so it will remain available for next session
                break
            
            # Load ONLY this chunk (e.g., 100 samples) instead of full block (100k samples)
            # This is the key memory optimization - 1000x reduction
            chunk_samples = block_manager.get_chunk(current_block_id, chunk_id, chunk_size)
            
            if not chunk_samples:
                print(f"[GPU {tracker_id}] Warning: Empty chunk {chunk_id} in block {current_block_id}")
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
            
            print(f"\n[GPU {tracker_id}] > Training Block {current_block_id} Chunk {chunk_id}: "
                  f"{len(chunk_samples)} samples, training {steps_to_train} optimizer steps "
                  f"(batch_size={config.batch_size}, grad_accum={grad_accum_steps}, effective_batch={effective_batch_size})")
            
            # Train on chunk samples (limited by config.steps)
            samples_trained = 0
            chunk_losses = []
            step_in_chunk = 0  # Counts optimizer steps, not micro-batches
            micro_batch_count = 0  # Counts micro-batches for gradient accumulation

            if progress_file:
                _write_progress(
                    step=0,
                    total_steps=max(1, steps_to_train),
                    loss=None,
                    block_id=current_block_id,
                    chunk_id=chunk_id,
                    global_optimizer_steps=total_steps_this_gpu,
                    samples_trained=0,
                )
            
            try:
                for batch_start in range(0, len(chunk_samples), config.batch_size):
                    # Check for immediate stop during training
                    if stop_event and stop_event.is_set():
                        _ack_immediate_stop()
                        print(f"[GPU {tracker_id}] Immediate stop requested during chunk - terminating now", flush=True)
                        graceful_stop_pending = False  # Override graceful stop for immediate exit
                        break
                    
                    # Check for graceful stop during chunk training
                    # Set flag but DON'T break - let the entire chunk complete
                    if graceful_stop_event and graceful_stop_event.is_set() and not graceful_stop_pending:
                        _ack_graceful_stop()
                        graceful_stop_pending = True
                        print(f"[GPU {tracker_id}] Graceful stop detected - will finish current chunk then exit", flush=True)
                    
                    # Stop if we've reached the per-chunk step limit
                    if step_in_chunk >= steps_to_train:
                        print(f"[GPU {tracker_id}] Reached per-chunk step limit ({steps_to_train}), moving to next chunk", flush=True)
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
                    samples_in_batch = len(batch_lines)
                    samples_trained += samples_in_batch
                    total_samples_trained_this_gpu += samples_in_batch  # Track cumulative samples (steps)
                    chunk_losses.append(loss.item())  # Store unscaled loss for logging
                    
                    # Only step optimizer after accumulating gradients
                    is_accumulation_end = (micro_batch_count % grad_accum_steps == 0)
                    if is_accumulation_end or batch_start + config.batch_size >= len(chunk_samples):
                        # Warmup (per optimizer step) when adaptive LR is enabled.
                        if adaptive_lr_scheduler is not None and warmup_steps > 0 and base_lr > 0:
                            try:
                                # total_steps_this_gpu counts completed optimizer steps so far.
                                next_step_idx = total_steps_this_gpu + 1
                                if total_steps_this_gpu < warmup_steps:
                                    warmup_factor = next_step_idx / warmup_steps
                                    current_lr = base_lr * warmup_factor
                                    for pg in optimizer.param_groups:
                                        pg["lr"] = float(current_lr)
                                elif total_steps_this_gpu == warmup_steps:
                                    for pg in optimizer.param_groups:
                                        pg["lr"] = float(base_lr)
                                    write_jsonl({"event": "warmup_complete", "gpu_id": tracker_id, "step": total_steps_this_gpu, "lr": base_lr})
                            except Exception:
                                pass

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

                        # Adaptive LR: observe once per optimizer step after warmup.
                        if adaptive_lr_scheduler is not None and base_lr > 0 and (warmup_steps <= 0 or total_steps_this_gpu >= warmup_steps):
                            try:
                                lookback = min(len(chunk_losses), max(1, int(grad_accum_steps)))
                                obs_loss = sum(chunk_losses[-lookback:]) / max(1, lookback)
                                adaptive_lr_scheduler.observe(float(obs_loss))
                            except Exception:
                                pass
                        
                        # Report progress every optimizer step or at end of chunk
                        # Only report AFTER completing an optimizer step to avoid duplicate prints during gradient accumulation
                        if step_in_chunk >= steps_to_train or batch_start + config.batch_size >= len(chunk_samples) or True:
                            avg_loss = sum(chunk_losses[-min(5*grad_accum_steps, len(chunk_losses)):]) / min(5*grad_accum_steps, len(chunk_losses))
                            print(f"[GPU {tracker_id}] Optimizer Step: {step_in_chunk}/{steps_to_train} | "
                                  f"Block {current_block_id} Chunk {chunk_id} | Loss={avg_loss:.4f}")

                            # Emit real-time step update for GUI (per-GPU, GUI will aggregate)
                            write_jsonl({
                                "event": "step",
                                "gpu_id": tracker_id,
                                "gpu_session_steps": total_samples_trained_this_gpu,  # This GPU's steps = samples trained
                                "gpu_optimizer_steps": total_steps_this_gpu,  # This GPU's optimizer steps
                                "loss": avg_loss,
                                "block_id": current_block_id,
                                "chunk_id": chunk_id,
                            })

                            if progress_file:
                                _write_progress(
                                    step=step_in_chunk,
                                    total_steps=max(1, steps_to_train),
                                    loss=avg_loss,
                                    block_id=current_block_id,
                                    chunk_id=chunk_id,
                                    global_optimizer_steps=total_steps_this_gpu,
                                    samples_trained=total_samples_trained_this_gpu,
                                )
            
            except Exception as batch_error:
                # Catch any exception during batch processing to prevent silent crashes
                print(f"\n[ERROR] GPU {tracker_id} EXCEPTION during batch processing: {batch_error}", flush=True)
                import traceback
                traceback.print_exc()
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                # Re-raise to trigger proper cleanup
                raise
            
            # Mark chunk complete
            # Mark chunk complete
            print(f"[GPU {tracker_id}] Batch loop done for chunk {chunk_id} (graceful_stop={graceful_stop_pending}, steps={step_in_chunk})", flush=True)
            
            avg_chunk_loss = sum(chunk_losses) / len(chunk_losses) if chunk_losses else 0.0
            chunk_tracker.mark_chunk_complete(
                block_id=current_block_id,
                chunk_id=chunk_id,
                gpu_id=tracker_id,
                optimizer_step=total_steps_this_gpu,
                steps=samples_trained  # Steps = samples/rows trained in this chunk
            )
            
            # Log progress stats to GUI
            stats = chunk_tracker.get_progress_stats()
            
            # Get current total blocks if detected
            total_blocks = block_manager.get_total_blocks()
            if total_blocks and chunk_tracker.total_blocks_in_dataset is None:
                chunk_tracker.total_blocks_in_dataset = total_blocks
            
            write_jsonl({
                "event": "chunk_complete",
                "gpu_id": tracker_id,
                "block_id": current_block_id,
                "chunk_id": chunk_id,
                "optimizer_step": total_steps_this_gpu,
                "loss": avg_chunk_loss,
                "session_steps": stats.get("session_steps", 0),  # Current session steps (samples trained)
                "total_steps": stats.get("total_steps", 0),  # All-time total steps
                "total_optimizer_steps": stats["total_optimizer_steps"],  # Historical max optimizer steps
                "total_chunks_trained": stats["total_chunks_trained"],
                "blocks_completed": stats["blocks_completed"],
                "current_epoch": stats["current_epoch"],
                "total_blocks": stats.get("total_blocks_in_dataset"),
            })
            
            print(f"[GPU {tracker_id}] * Completed Block {current_block_id} Chunk {chunk_id}: "
                  f"{step_in_chunk} optimizer steps (limit: {steps_to_train}) | "
                  f"{samples_trained} samples trained | "
                  f"GPU Total: {total_steps_this_gpu} opt steps ({total_samples_trained_this_gpu} steps) | "
                  f"Avg Loss: {avg_chunk_loss:.4f}")

            if progress_file:
                _write_progress(
                    step=step_in_chunk,
                    total_steps=max(1, steps_to_train),
                    loss=avg_chunk_loss,
                    block_id=current_block_id,
                    chunk_id=chunk_id,
                    global_optimizer_steps=total_steps_this_gpu,
                    samples_trained=total_samples_trained_this_gpu,
                )
            
            # If immediate stop was triggered during chunk, break outer loop
            if stop_event and stop_event.is_set():
                _ack_immediate_stop()
                print(f"[GPU {tracker_id}] Immediate stop confirmed - exiting", flush=True)
                break
            
            # Check for graceful stop after chunk completion
            if graceful_stop_pending:
                print(f"[GPU {tracker_id}] Graceful stop honored after chunk completion", flush=True)
                break
            
            # If iterate is disabled, stop after training one chunk
            if not config.iterate:
                print(f"[GPU {tracker_id}] Iterate mode disabled - stopping after one chunk", flush=True)
                break
        
        # Save final checkpoint
        print(f"[GPU {tracker_id}] Saving final checkpoint...", flush=True)
        
        # Ensure checkpoint directory exists (in case worker process doesn't have it)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"gpu{tracker_id}_final.safetensors"
        
        from safetensors.torch import save_file as save_safetensors
        state_dict = model.state_dict()
        save_safetensors(state_dict, str(checkpoint_path))
        
        # Release state_dict to avoid file handle issues on Windows
        del state_dict
        
        print(f"[GPU {tracker_id}] Training complete! Steps: {total_steps_this_gpu}", flush=True)
        
    except Exception as e:
        print(f"\n[ERROR] GPU {tracker_id} training failed: {e}", flush=True)
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


def _run_parallel_training_v3_ddp(
    *,
    config: "TrainingConfig",
    checkpoint_dir: Path,
    tokenizer,
    vocab_size: int,
    block_manager: BlockManager,
    chunk_tracker: ChunkTracker,
    write_jsonl,
    final_checkpoint_path: Path,
) -> None:
    import os
    from pathlib import Path as _Path

    import torch

    try:
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("DDP requested but torch.distributed is unavailable") from exc

    if not dist.is_initialized():
        raise RuntimeError("DDP mode requested but torch.distributed is not initialized")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Always prefer LOCAL_RANK for device selection; CUDA_VISIBLE_DEVICES remaps indices per process
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_index = local_rank if device_count > 0 else 0
    if device_count > 0 and device_index >= device_count:
        device_index = device_index % device_count

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device_index)
        except RuntimeError as err:
            print(f"[DDP] Warning: failed to set CUDA device {device_index}: {err}")
        physical_device = torch.cuda.current_device()
    else:
        physical_device = 0

    # Each rank runs the standard worker, but shares tracker state via disk locks
    ckpt_path = train_gpu_worker(
        config=config,
        gpu_id=int(physical_device),
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        block_manager=block_manager,
        chunk_tracker=chunk_tracker,
        write_jsonl=write_jsonl,
        device_index=device_index,
        stop_event=None,
        graceful_stop_event=None,
        stop_ack_event=None,
        graceful_stop_ack_event=None,
        tracker_gpu_id=rank,
    )

    # Persist tracker state changes before synchronization
    chunk_tracker.save()

    dist.barrier()

    local_ckpt = ckpt_path if ckpt_path and _Path(ckpt_path).exists() else None
    gathered: Optional[list[Optional[str]]] = [None] * world_size if rank == 0 else None
    dist.gather_object(local_ckpt, gathered, dst=0)

    if rank == 0:
        valid_paths = [p for p in gathered or [] if p]
        if valid_paths:
            merge_checkpoints(valid_paths, str(final_checkpoint_path))
            print(f"[DDP] Merged {len(valid_paths)} checkpoints into {final_checkpoint_path}")
        else:
            print("[DDP] Warning: No valid checkpoints gathered from ranks")

        # Refresh tracker state from disk to capture all ranks' updates
        chunk_tracker.refresh()
        final_stats = chunk_tracker.get_progress_stats()

        write_jsonl({
            "event": "training_complete",
            "session_steps": final_stats.get('session_steps', 0),
            "total_steps": final_stats.get('total_steps', 0),
            "total_optimizer_steps": final_stats['total_optimizer_steps'],
            "total_chunks_trained": final_stats['total_chunks_trained'],
            "session_chunks_trained": final_stats.get('session_chunks_trained', 0),
            "blocks_completed": final_stats['blocks_completed'],
            "current_epoch": final_stats['current_epoch'],
        })

        # Update brain.json with training_steps (similar to non-DDP path)
        save_dir = Path(config.save_dir)
        brain_json_path = save_dir / "brain.json"
        session_steps = final_stats.get('session_steps', 0)
        try:
            import json as _json
            import time as _t
            meta = {}
            if brain_json_path.exists():
                with brain_json_path.open("r", encoding="utf-8") as f:
                    meta = _json.load(f) or {}
            
            # Increment training_steps
            previous_steps = int(meta.get("training_steps", 0))
            total_steps = previous_steps + int(session_steps)
            meta["training_steps"] = total_steps
            
            # Track dataset history
            dataset_history = meta.get("dataset_history", [])
            if config.dataset_file:
                dataset_name = Path(config.dataset_file).name
                dataset_path = str(Path(config.dataset_file).resolve()) if not config.dataset_file.startswith("hf://") else config.dataset_file
                session_record = {
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "steps": int(session_steps),
                    "timestamp": float(_t.time()),
                }
                dataset_history.append(session_record)
                meta["dataset_history"] = dataset_history
            
            # Update last_session metadata
            meta["last_session"] = {
                "timestamp": float(_t.time()),
                "steps_completed": int(session_steps),
                "total_steps": int(total_steps),
                "stopped_early": False,
                "dataset_file": str(config.dataset_file) if config.dataset_file else None,
                "checkpoint_path": str(final_checkpoint_path),
            }
            meta["last_trained"] = float(_t.time())
            
            # Write updated brain.json
            with brain_json_path.open("w", encoding="utf-8") as f:
                _json.dump(meta, f, indent=2)
            
            print(f"[DDP] Updated brain.json: training_steps {previous_steps} -> {total_steps}")
        except Exception as e:
            print(f"[DDP] Warning: Failed to update brain.json: {e}")
        
        # Truncate metrics.jsonl to keep file size manageable
        try:
            log_file_path = Path(config.log_file) if config.log_file else save_dir / "metrics.jsonl"
            if log_file_path.exists():
                log_file_path.unlink()
                print(f"[DDP] Cleared metrics log: {log_file_path.name}")
        except Exception as e:
            print(f"[DDP] Warning: Failed to clear metrics log: {e}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE! (DDP mode)")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Total optimizer steps: {final_stats['total_optimizer_steps']}")
        print(f"Session steps: {final_stats.get('session_steps', 0)}")
        print(f"Total steps: {final_stats.get('total_steps', 0)}")
        print(f"Chunks trained: {final_stats['total_chunks_trained']}")
        print(f"Blocks completed: {final_stats['blocks_completed']}")
        print(f"Current epoch: {final_stats['current_epoch']}")
        print(f"Checkpoint: {final_checkpoint_path}")
        print("=" * 60 + "\n")

    dist.barrier()

def run_parallel_training_v3(
    config: "TrainingConfig",
    stop_event: Optional[Any] = None,
    graceful_stop_event: Optional[Any] = None,
    stop_ack_event: Optional[Any] = None,
    graceful_stop_ack_event: Optional[Any] = None,
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
        stop_ack_event: Event set when workers observe the immediate stop request
        graceful_stop_ack_event: Event set when workers observe the graceful stop request
    """
    
    zero_stage = str(getattr(config, "zero_stage", "none") or "none").lower()

    ddp_active = False
    ddp_rank = 0
    ddp_world = 1
    dist_mod = None
    if torch.distributed.is_available():
        try:
            import torch.distributed as dist_mod  # type: ignore
            if dist_mod.is_initialized():
                ddp_active = True
                ddp_rank = dist_mod.get_rank()
                ddp_world = dist_mod.get_world_size()
        except Exception:
            ddp_active = False
            dist_mod = None

    is_primary = (not ddp_active) or (ddp_rank == 0)

    if is_primary:
        print("\n" + "="*60)
        print("PARALLEL TRAINING V3 - BLOCK/CHUNK DISTRIBUTION")
        print("="*60)
        print("Features:")
        print("  • Block-based dataset streaming (100k items/block)")
        print("  • Unique chunk distribution per GPU")
        print("  • Progress tracking (no duplicate training)")
        print("  • Stopping conditions: steps/block/epoch")
        print("  • Iterate mode support")
        if zero_stage != "none":
            label = "ZeRO-3 + Inference" if zero_stage == "zero3" else f"ZeRO stage {zero_stage}"
            print(f"  • {label}: DeepSpeed configuration active (model sharded across GPUs)")
        if ddp_active:
            print(f"  • Distributed Data Parallel integration (world_size={ddp_world})")
        print("="*60 + "\n")
    
    # Setup write_jsonl for GUI metrics logging
    from .training_helpers import write_jsonl as _write_jsonl_helper
    log_file = config.log_file if hasattr(config, 'log_file') else None
    
    def write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(
            log_file=log_file,
            payload=payload,
            is_distributed=ddp_active,
            rank_id=ddp_rank,
        )
    
    # Setup directories
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "parallel_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------
    # Resume precedence + config source logging
    # --------------------------------------------------------------------
    # In this training path, checkpoint loading is controlled by `student_init`.
    # For resume runs, we make that explicit and deterministic:
    # - If `--resume` is set and a checkpoint exists in `save_dir`, we load from `save_dir`.
    # - If `--resume` is set but no checkpoint exists, we disable resume (fresh start).
    # - If `--resume` is NOT set, we start fresh (and we do NOT reuse chunk-tracker state).
    resume_requested = bool(getattr(config, "resume", False))
    original_student_init = getattr(config, "student_init", None)
    checkpoint_expected = save_dir / "actv1_student.safetensors"
    checkpoint_legacy = save_dir / "final_model.safetensors"
    checkpoint_exists = checkpoint_expected.exists() or checkpoint_legacy.exists()
    
    # Create dataset-specific state file path
    from .chunk_tracker import sanitize_dataset_name, get_dataset_state_file
    dataset_name_sanitized = sanitize_dataset_name(config.dataset_file) if config.dataset_file else "unknown"
    chunk_size = getattr(config, 'dataset_chunk_size', None)
    chunk_state_file = get_dataset_state_file(save_dir, dataset_name_sanitized, chunk_size)
    
    # Ensure parent directory exists
    chunk_state_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Legacy state file for backward compatibility
    legacy_state_file = save_dir / "chunk_tracker_state.json"

    effective_resume = bool(resume_requested and checkpoint_exists)
    if resume_requested and not checkpoint_exists:
        # Avoid a surprising "resume" that does not actually load weights.
        try:
            write_jsonl({
                "event": "resume_disabled",
                "reason": "checkpoint_missing",
                "save_dir": str(save_dir),
                "expected_checkpoint": str(checkpoint_expected),
                "legacy_checkpoint": str(checkpoint_legacy),
            })
        except Exception:
            pass
        if is_primary:
            print(f"[WARN] --resume requested but no checkpoint found in {save_dir}; starting fresh")
        try:
            setattr(config, "resume", False)
        except Exception:
            pass

    # If we are resuming, force checkpoint load from this brain directory.
    # This makes resume behavior consistent even if GUI/CLI did not set --student-init.
    resolved_student_init = original_student_init
    student_init_source = "cli"
    if effective_resume:
        resolved_student_init = str(save_dir)
        student_init_source = "resume(save_dir)"
        if original_student_init and str(original_student_init) != str(resolved_student_init):
            if is_primary:
                print(f"[INFO] Resume precedence: ignoring student_init={original_student_init!r} and loading from {save_dir}")
        try:
            setattr(config, "student_init", resolved_student_init)
        except Exception:
            pass

    # Chunk-tracker precedence: if not resuming, do not reuse historical chunk progress.
    chunk_state_action = "load" if effective_resume else "reset"
    chunk_state_existed = chunk_state_file.exists()
    if not effective_resume and chunk_state_existed:
        try:
            from datetime import datetime

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = chunk_state_file.with_name(f"{chunk_state_file.name}.bak_{ts}")
            try:
                chunk_state_file.rename(backup_path)
            except Exception:
                # If rename fails (e.g., cross-device), fall back to delete.
                chunk_state_file.unlink(missing_ok=True)  # type: ignore[arg-type]
                backup_path = None

            # Also remove lock file if present to avoid stale lock issues.
            try:
                lock_path = chunk_state_file.with_suffix(".lock")
                lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

            try:
                write_jsonl({
                    "event": "chunk_tracker_state_reset",
                    "reason": "resume_disabled_or_not_requested",
                    "path": str(chunk_state_file),
                    "backup": str(backup_path) if backup_path else None,
                })
            except Exception:
                pass
            if is_primary:
                msg = f"[INFO] Starting fresh: cleared prior chunk-tracker state ({chunk_state_file})"
                if backup_path:
                    msg += f" -> {backup_path.name}"
                print(msg)
        except Exception:
            pass

    # Emit a single structured snapshot of where key settings came from.
    try:
        write_jsonl({
            "event": "config_source",
            "training_path": "parallel_training_v3",
            "resume": {
                "requested": bool(resume_requested),
                "effective": bool(effective_resume),
                "checkpoint_exists": bool(checkpoint_exists),
                "expected_checkpoint": str(checkpoint_expected),
                "legacy_checkpoint": str(checkpoint_legacy),
            },
            "student_init": {
                "original": str(original_student_init) if original_student_init else None,
                "resolved": str(resolved_student_init) if resolved_student_init else None,
                "source": student_init_source,
            },
            "chunk_tracker_state": {
                "path": str(chunk_state_file),
                "existed": bool(chunk_state_existed),
                "action": chunk_state_action,
            },
        })
    except Exception:
        pass
    
    # Load tokenizer - check brain.json in save_dir even if student_init is None (fresh start)
    from .brain_metadata import load_brain_metadata, extract_tokenizer_from_metadata
    brain_metadata = load_brain_metadata(
        student_init=config.student_init,
        log_fn=print,
        save_dir=str(save_dir)  # Fallback to brain directory for fresh starts
    )
    tokenizer_path = extract_tokenizer_from_metadata(brain_metadata, config.model)
    
    if is_primary:
        print(f"[LOAD] Loading tokenizer from: {tokenizer_path}")
    tokenizer = _load_tokenizer(tokenizer_path)
    adjust_tokenizer_padding(tokenizer)
    vocab_size = calculate_vocab_size(tokenizer, print)
    if is_primary:
        print(f"[OK] Vocabulary size: {vocab_size:,}\n")
    
    # Initialize BlockManager
    from aios.data.datasets import read_text_lines_sample_any
    
    if not config.dataset_file:
        raise ValueError("dataset_file is required for parallel training")
    
    # Validate and preprocess dataset if needed (before initializing BlockManager)
    if is_primary:
        from aios.cli.datasets.dataset_validation import validate_dataset_for_training
        try:
            validate_dataset_for_training(
                dataset_file=config.dataset_file,
                samples_per_block=config.samples_per_block,
                ascii_only=config.ascii_only,
            )
        except ValueError as e:
            print(f"\n❌ Dataset validation failed: {e}\n")
            write_jsonl({"started": False, "error": f"dataset_validation_failed: {e}"})
            raise RuntimeError(f"Dataset validation failed: {e}")
    
    if is_primary:
        print(f"[INIT] Initializing BlockManager (async)...")
        print(f"   Dataset: {config.dataset_file}")
        print(f"   Items per block: {config.samples_per_block:,}")
        print(f"   Chunk size: {config.dataset_chunk_size:,}")
    
    # Calculate total chunks per block
    chunks_per_block = config.samples_per_block // config.dataset_chunk_size
    if is_primary:
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
    if is_primary:
        print(f"[OK] BlockManager initialized (ready for lazy loading)")
        print(f"[INFO] Blocks will be loaded on-demand during training\n")
    
    # Start pre-downloading first 5 blocks in background for HF streaming datasets
    if config.dataset_file and isinstance(config.dataset_file, str) and config.dataset_file.startswith("hf://"):
        block_manager.start_predownload(num_blocks=5)
        if is_primary:
            print(f"[INFO] Started background pre-download of first 5 blocks\n")
    
    # Get total blocks if already detected (for local datasets)
    total_blocks_detected = block_manager.get_total_blocks()
    if total_blocks_detected and is_primary:
        print(f"[INFO] Detected {total_blocks_detected} total blocks in dataset")
    
    # Initialize ChunkTracker in brain directory for proper resume detection
    state_file = chunk_state_file
    if is_primary:
        print(f"[INIT] Initializing ChunkTracker...")
        print(f"   Save directory (config.save_dir): {config.save_dir}")
        print(f"   Save directory (local save_dir var): {save_dir}")
        print(f"   Bundle directory: {config.bundle_dir}")
        print(f"   Brain name: {config.brain_name}")
        print(f"   State file: {state_file}")
        print(f"   Note: Checkpoint will be saved to: {save_dir / 'actv1_student.safetensors'}")
    
    # Extract brain_id from metadata for per-brain tracking
    brain_id = brain_metadata.get("brain_id") if brain_metadata else None
    # Extract dataset name from config
    dataset_name = None
    if config.dataset_file:
        dataset_name = Path(config.dataset_file).stem if not config.dataset_file.startswith("hf://") else config.dataset_file.replace("hf://", "").replace("/", "_")
    
    if is_primary and brain_id:
        print(f"[INFO] Brain ID: {brain_id}")
    if is_primary and dataset_name:
        print(f"[INFO] Dataset: {dataset_name}")
    
    chunk_tracker = ChunkTracker(
        state_file=state_file,
        brain_id=brain_id,
        dataset_name=dataset_name,
        start_block_id=config.start_block_id,
        start_chunk_id=config.start_chunk_id
    )
    
    # Set total blocks in ChunkTracker if detected
    if total_blocks_detected:
        chunk_tracker.total_blocks_in_dataset = total_blocks_detected
        if is_primary:
            print(f"[INFO] ChunkTracker configured with {total_blocks_detected} total blocks")
        
        # Log to GUI for progress tracking
        write_jsonl({
            "epoch_tracking": "initialized",
            "total_blocks": total_blocks_detected,
            "samples_per_block": config.samples_per_block,
            "dataset_chunk_size": config.dataset_chunk_size,
            "chunks_per_block": chunks_per_block,
            "note": "Block-based parallel training enabled",
            "zero_stage": zero_stage,
        })
    else:
        # Log that blocks will be detected during training
        write_jsonl({
            "epoch_tracking": "disabled",
            "reason": "Total blocks not yet detected (HF streaming or first-time local dataset)",
            "samples_per_block": config.samples_per_block,
            "dataset_chunk_size": config.dataset_chunk_size,
            "chunks_per_block": chunks_per_block,
            "note": "Blocks will be detected during training",
            "zero_stage": zero_stage,
        })
    
    # Show resumed state if applicable
    stats = chunk_tracker.get_progress_stats()
    if stats['total_chunks_trained'] > 0 and is_primary:
        print(f"[RESUME] Resuming from previous state:")
        print(f"   Chunks trained: {stats['total_chunks_trained']}")
        print(f"   Epoch: {stats['current_epoch']}")
        print(f"   Steps trained: {stats['total_steps']:,}")
    
    if is_primary:
        print(f"[OK] ChunkTracker initialized\n")
    
    # Parse cuda_ids from config
    cuda_ids = config.cuda_ids
    if isinstance(cuda_ids, str):
        cuda_ids = [int(x.strip()) for x in cuda_ids.split(',')]
    else:
        cuda_ids = list(cuda_ids) if cuda_ids else [0]
    
    # Validate requested GPU IDs against what PyTorch can actually see
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Detected no visible CUDA devices; install a compatible GPU or rerun with --device cpu."
        )

    visible_device_count = torch.cuda.device_count()
    if visible_device_count <= 0:
        raise RuntimeError(
            "No CUDA devices are visible to PyTorch. Ensure GPU drivers are installed or adjust --cuda-ids/--device options."
        )

    # When CUDA_VISIBLE_DEVICES is set, PyTorch remaps device indices to 0, 1, 2, ...
    # For example, if CUDA_VISIBLE_DEVICES="1,2", then physical GPU 1 becomes cuda:0 and GPU 2 becomes cuda:1
    # We need to use these remapped indices (0, 1, ...) instead of the original GPU IDs
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        # CUDA_VISIBLE_DEVICES is set - use remapped indices (0, 1, 2, ...) up to visible_device_count
        num_requested = len(cuda_ids)
        if num_requested > visible_device_count:
            raise RuntimeError(
                f"Requested {num_requested} GPU(s) but only {visible_device_count} device(s) are visible via CUDA_VISIBLE_DEVICES={cuda_visible_devices}. "
                "Adjust GPU selection in Resources tab."
            )
        # Use remapped indices: 0, 1, 2, ... instead of original GPU IDs
        original_cuda_ids = cuda_ids
        cuda_ids = list(range(num_requested))
        if is_primary and original_cuda_ids != cuda_ids:
            print(f"[GPU] Remapped GPU IDs {original_cuda_ids} → {cuda_ids} (CUDA_VISIBLE_DEVICES={cuda_visible_devices})")
    else:
        # No CUDA_VISIBLE_DEVICES - validate original IDs directly
        invalid_ids = [gpu for gpu in cuda_ids if gpu < 0 or gpu >= visible_device_count]
        if invalid_ids:
            raise RuntimeError(
                f"Requested CUDA device IDs {invalid_ids} but only {visible_device_count} device(s) are visible. Set --cuda-ids accordingly."
            )

    num_gpus = len(cuda_ids)
    
    if is_primary:
        print(f"[PLAN] Training Plan:")
        print(f"   GPUs: {cuda_ids} ({num_gpus} GPUs)")
        print(f"   Steps PER CHUNK: {config.steps} (each chunk trains for this many steps)")
        print(f"   Stop after block: {config.stop_after_block}")
        print(f"   Stop after epoch: {config.stop_after_epoch}")
        print(f"   Iterate mode: {'ENABLED (continuous training)' if config.iterate else 'DISABLED (stop after dataset exhausted)'}")
        print()
    
    final_checkpoint_path = save_dir / "actv1_student.safetensors"

    if ddp_active:
        _run_parallel_training_v3_ddp(
            config=config,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            block_manager=block_manager,
            chunk_tracker=chunk_tracker,
            write_jsonl=write_jsonl,
            final_checkpoint_path=final_checkpoint_path,
        )
        return

    # Events are passed as arguments from the caller (GUI with multiprocessing.Manager)
    # No need to create them here
    
    # Launch GPU workers
    if is_primary:
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
                stop_ack_event=stop_ack_event,
                graceful_stop_ack_event=graceful_stop_ack_event,
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
            if is_primary:
                print(f"   [OK] Launched GPU {gpu_id} worker")
        
        if is_primary:
            print(f"\n[PARALLEL] All {num_gpus} workers running...\n")
        
        # Wait for completion
        for i, t in enumerate(threads):
            t.join()
            exc = exceptions[i]
            if exc is not None:
                raise exc
        
        if is_primary:
            print(f"\n[OK] All {num_gpus} workers completed!")
        
        # Save final tracker state
        chunk_tracker.save()
        
    except Exception as e:
        if is_primary:
            print(f"\n[FATAL] Parallel training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Merge checkpoints to expected resume location
    valid_checkpoints = [str(p) for p in checkpoint_paths_list if p is not None]
    
    if len(valid_checkpoints) > 0:
        merge_checkpoints(valid_checkpoints, str(final_checkpoint_path))
        if is_primary:
            print(f"[OK] Merged checkpoint saved to: {final_checkpoint_path}")
    else:
        if is_primary:
            print("[WARN] No valid checkpoints to merge")
    
    # Print final stats
    final_stats = chunk_tracker.get_progress_stats()
    
    # Log final stats to GUI
    write_jsonl({
        "event": "training_complete",
        "session_steps": final_stats.get('session_steps', 0),
        "total_steps": final_stats.get('total_steps', 0),
        "total_optimizer_steps": final_stats['total_optimizer_steps'],
        "total_chunks_trained": final_stats['total_chunks_trained'],
        "session_chunks_trained": final_stats.get('session_chunks_trained', 0),
        "blocks_completed": final_stats['blocks_completed'],
        "current_epoch": final_stats['current_epoch'],
        "zero_stage": zero_stage,
    })
    
    # Update brain.json with training_steps (similar to finalization.py)
    if is_primary:
        brain_json_path = save_dir / "brain.json"
        session_steps = final_stats.get('session_steps', 0)
        try:
            import json as _json
            import time as _t
            meta = {}
            if brain_json_path.exists():
                with brain_json_path.open("r", encoding="utf-8") as f:
                    meta = _json.load(f) or {}
            
            # Increment training_steps
            previous_steps = int(meta.get("training_steps", 0))
            total_steps = previous_steps + int(session_steps)
            meta["training_steps"] = total_steps
            
            # Track dataset history
            dataset_history = meta.get("dataset_history", [])
            if config.dataset_file:
                dataset_name = Path(config.dataset_file).name
                dataset_path = str(Path(config.dataset_file).resolve()) if not config.dataset_file.startswith("hf://") else config.dataset_file
                session_record = {
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "steps": int(session_steps),
                    "timestamp": float(_t.time()),
                }
                dataset_history.append(session_record)
                meta["dataset_history"] = dataset_history
            
            # Update last_session metadata
            meta["last_session"] = {
                "timestamp": float(_t.time()),
                "steps_completed": int(session_steps),
                "total_steps": int(total_steps),
                "stopped_early": False,
                "dataset_file": str(config.dataset_file) if config.dataset_file else None,
                "checkpoint_path": str(final_checkpoint_path),
            }
            meta["last_trained"] = float(_t.time())
            
            # Write updated brain.json
            with brain_json_path.open("w", encoding="utf-8") as f:
                _json.dump(meta, f, indent=2)
            
            print(f"[OK] Updated brain.json: training_steps {previous_steps} -> {total_steps}")
        except Exception as e:
            print(f"[WARN] Failed to update brain.json: {e}")
        
        # Truncate metrics.jsonl to keep file size manageable
        # Since training_steps is now persisted in brain.json, we can safely clear this
        try:
            log_file_path = Path(config.log_file) if config.log_file else save_dir / "metrics.jsonl"
            if log_file_path.exists():
                log_file_path.unlink()
                print(f"[OK] Cleared metrics log: {log_file_path.name}")
        except Exception as e:
            print(f"[WARN] Failed to clear metrics log: {e}")
    
    if is_primary:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final Statistics:")
        print(f"   Total optimizer steps (all GPUs): {final_stats['total_optimizer_steps']}")
        print(f"   Session steps: {final_stats.get('session_steps', 0)}")
        print(f"   Total steps: {final_stats.get('total_steps', 0)}")
        print(f"   Chunks trained: {final_stats['total_chunks_trained']}")
        print(f"   Blocks completed: {final_stats['blocks_completed']}")
        print(f"   Current epoch: {final_stats['current_epoch']}")
        if zero_stage != "none":
            print(f"   ZeRO stage: {zero_stage}")
        print(f"   Checkpoint: {final_checkpoint_path}")
        print(f"   Resume: Training can be resumed from this checkpoint")
        print("="*60 + "\n")
