"""Training loop and iteration management."""
from __future__ import annotations

import logging
import typer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Callable

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

from .training_logic import train_epoch as _train_epoch_helper

logger = logging.getLogger(__name__)


def execute_training_epoch(
    model: Any,
    segment_rollout: Callable,
    opt: Any,
    device_obj: Any,
    dml_device: Any,
    input_ids: Any,
    labels: Any,
    config: "TrainingConfig",
    streaming_dataset: Any,
    tokenizer: Any,
    lines: list[str],
    scaler: Any,
    deepspeed_engine: Any,
    inference_manager: Any,
    warmup_steps: int,
    base_lr: float,
    step_offset: int,
    is_distributed: bool,
    world_sz: int,
    write_jsonl: Callable,
    should_stop: Callable,
    adaptive_lr_scheduler: Any = None,
) -> Tuple[int, bool, int, Optional[str]]:
    """Execute a single training epoch.
    
    Returns:
        Tuple of (steps_done, stopped_early, batch_size, stop_reason)
    """
    logger.info(f"Starting training epoch: batch_size={config.batch_size}, steps={config.steps}, device={device_obj}")
    logger.debug(f"Training config: use_amp={config.use_amp}, distributed={is_distributed}, world_size={world_sz}")
    
    # No periodic evaluation - only runs at end
    def _maybe_eval():
        pass
    
    s_done, early, new_bs, stop_reason = _train_epoch_helper(
        model_student=model,
        segment_rollout=segment_rollout,
        opt=opt,
        device_obj=device_obj,
        dml_device=dml_device,
        input_ids=input_ids,
        labels=labels,
        batch_size=config.batch_size,
        steps=config.steps,
        halt_max_steps=config.halt_max_steps,
        sys_mem_cap_pct=config.sys_mem_cap_pct,
        dev=str(device_obj).split(':')[0],
        is_distributed=is_distributed,
        world_sz=world_sz,
        stop_file=config.stop_file,
        write_jsonl=write_jsonl,
        should_stop=should_stop,
        write_last_safe_batches=lambda train_bs: None,  # Placeholder
        eval_maybe=_maybe_eval,
        gpu_util_target=0,  # Disabled
        cpu_util_target=0,  # Disabled
        gpu_util_mode="duty",
        gpu_util_poll_ms=50,
        streaming_dataset=streaming_dataset,
        tokenizer=tokenizer,
        lines=lines if streaming_dataset else None,
        use_amp=config.use_amp,
        scaler=scaler,
        deepspeed_engine=deepspeed_engine,
        inference_manager=inference_manager,
        hot_reload_steps=config.hot_reload_steps if inference_manager else 0,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        adaptive_lr_scheduler=adaptive_lr_scheduler,
        stop_after_epoch=config.stop_after_epoch,
        step_offset=step_offset,
        config=config,
    )
    
    logger.info(f"Training epoch completed: steps_done={s_done}, stopped_early={early}, new_batch_size={new_bs}, stop_reason={stop_reason}")
    
    return s_done, early, new_bs, stop_reason


def run_iterate_mode(
    config: "TrainingConfig",
    lines: list[str],
    resume_cycle: int,
    execute_epoch: Callable,
    load_new_lines: Callable,
    train_with_lines: Callable,
    log_fn,
    write_jsonl: Callable,
    checkpoint_saver = None,
    chunk_tracker: object | None = None,
) -> Tuple[int, bool, Optional[str], int]:
    """Run training in iterate mode (multiple cycles with data refresh).
    
    Tracks samples across cycles to implement proper block/epoch completion:
    - CHUNK: 4k samples loaded per cycle (one call to load_new_lines)
    - BLOCK: 100k samples (25 chunks) 
    - EPOCH: All blocks in complete dataset
    
    Stop conditions:
    - stop_after_block: Continue until current block (100k samples) complete
    - stop_after_epoch: Continue until all blocks in dataset processed
    
    Returns:
        Tuple of (steps_done, stopped_early, last_stop_reason, final_cycle)
    """
    logger.info(f"Starting iterate mode training: resume_cycle={resume_cycle}")
    
    cycle = resume_cycle
    steps_done = 0
    stopped_early = False
    last_stop_reason = None
    
    # Track samples for block/epoch completion
    # CRITICAL: Initialize from config OR default to 0, then keep config.current_block_samples in sync
    if not hasattr(config, 'current_block_samples') or config.current_block_samples is None:
        config.current_block_samples = 0
    samples_this_block = config.current_block_samples
    samples_per_block = config.samples_per_block if hasattr(config, 'samples_per_block') and config.samples_per_block else 100000
    dataset_chunk_size = getattr(config, 'dataset_chunk_size', 4000)
    chunks_per_block = max(1, (int(samples_per_block) + int(dataset_chunk_size) - 1) // int(dataset_chunk_size))
    # Track a logical current block id for single-GPU iterate mode
    current_block_id = getattr(config, 'current_block_id', 0)
    
    log_fn({
        "iterate_mode": "starting",
        "initial_cycle": cycle,
        "resumed": resume_cycle > 0,
        "samples_per_block": samples_per_block,
        "current_block_samples": samples_this_block,
    })
    
    try:
        while True:
            log_fn({
                "iterate_cycle": cycle,
                "status": "starting",
                "note": "Checking if cycle needs processing"
            })
            
            # If linear dataset mode and tracker enabled, check if this chunk is already trained BEFORE loading data
            chunk_already_trained = False
            chunk_id_in_block = None
            if getattr(config, 'linear_dataset', False) and chunk_tracker is not None:
                try:
                    # Compute which chunk within the current block we're about to train
                    chunk_id_in_block = (samples_this_block // int(dataset_chunk_size)) % chunks_per_block
                    # Get rank/gpu id from config if available (for DDP/parallel compatibility)
                    gpu_rank = getattr(config, '_tracker_rank_id', 0)
                    if hasattr(chunk_tracker, 'claim_chunk'):
                        # CRITICAL: Check return value - False means already trained
                        can_claim = chunk_tracker.claim_chunk(current_block_id, int(chunk_id_in_block), gpu_id=gpu_rank)
                        if not can_claim:
                            chunk_already_trained = True
                            log_fn({
                                "event": "chunk_skipped",
                                "reason": "already_trained",
                                "cycle": cycle,
                                "block_id": current_block_id,
                                "chunk_id": chunk_id_in_block,
                                "note": "Chunk already trained in previous session - skipping data load and training"
                            })
                except Exception:
                    pass
            
            # Only load data if chunk needs training
            new_lines = []
            if not chunk_already_trained:
                log_fn({
                    "iterate_cycle": cycle,
                    "status": "loading_data",
                    "note": "Loading fresh dataset for this cycle"
                })
                
                # Load new dataset for this cycle
                new_lines = load_new_lines(cycle=cycle)
                if not new_lines:
                    log_fn({
                        "iterate_cycle": cycle,
                        "status": "no_data",
                        "action": "stopping"
                    })
                    break
                
                # Update training data
                train_with_lines(new_lines)
                
                log_fn({
                    "iterate_cycle": cycle,
                    "dataset_samples": len(new_lines),
                    "status": "training"
                })
            else:
                # For skipped chunks, we need to know chunk size for progress tracking
                # Use dataset_chunk_size as default
                log_fn({
                    "iterate_cycle": cycle,
                    "status": "skipped",
                    "note": f"Skipping cycle {cycle} (already trained)"
                })

            # Run training epoch only if chunk not already trained
            if not chunk_already_trained:
                epoch_steps, early, _, stop_reason = execute_epoch()
                steps_done += epoch_steps
            else:
                # Skip training but still count as completed
                epoch_steps = 0
                early = False
                stop_reason = None
                # Restore steps from tracker
                try:
                    if chunk_tracker is not None and hasattr(chunk_tracker, 'get_progress_stats'):
                        stats = chunk_tracker.get_progress_stats()
                        steps_done = stats.get("total_gpu_steps", steps_done)
                except Exception:
                    pass
            
            stopped_early = early
            last_stop_reason = stop_reason
            
            # Track samples for block completion
            if chunk_already_trained:
                # For skipped chunks, use dataset_chunk_size
                chunk_samples = int(dataset_chunk_size)
            else:
                chunk_samples = len(new_lines)
            samples_this_block += chunk_samples
            
            # CRITICAL: Keep config.current_block_samples in sync IMMEDIATELY
            config.current_block_samples = samples_this_block

            # If linear dataset mode and tracker enabled, mark chunk complete after training
            # Don't mark if chunk was already trained (skipped)
            if getattr(config, 'linear_dataset', False) and chunk_tracker is not None and not chunk_already_trained:
                try:
                    # Use already calculated chunk_id_in_block if available, otherwise calculate
                    if chunk_id_in_block is None:
                        chunk_id_in_block = ((samples_this_block - chunk_samples) // int(dataset_chunk_size)) % chunks_per_block
                    gpu_rank = getattr(config, '_tracker_rank_id', 0)
                    is_rank_0 = getattr(config, '_tracker_is_rank_0', True)
                    if hasattr(chunk_tracker, 'mark_chunk_complete'):
                        chunk_tracker.mark_chunk_complete(
                            block_id=int(current_block_id),
                            chunk_id=int(chunk_id_in_block),
                            gpu_id=gpu_rank,
                            step=int(steps_done),
                            samples_trained=int(chunk_samples),
                        )
                        # Emit parallel-compatible stats payload (rank 0 only to avoid duplicates)
                        if is_rank_0:
                            if hasattr(chunk_tracker, 'get_progress_stats'):
                                stats = chunk_tracker.get_progress_stats()  # type: ignore[assignment]
                            else:
                                stats = {
                                    "total_gpu_steps": int(steps_done),
                                    "total_chunks_trained": 0,
                                    "blocks_completed": 0,
                                    "current_epoch": 0,
                                }
                            write_jsonl({
                                "event": "chunk_complete",
                                "gpu_id": gpu_rank,
                                "block_id": int(current_block_id),
                                "chunk_id": int(chunk_id_in_block),
                                "step": int(steps_done),
                                "loss": None,
                                "lr": None,
                                "adaptive_lr_mode_requested": None,
                                "adaptive_lr_mode_active": None,
                                "total_gpu_steps": stats.get("total_gpu_steps", int(steps_done)),
                                "total_chunks_trained": stats.get("total_chunks_trained", 0),
                                "blocks_completed": stats.get("blocks_completed", 0),
                                "current_epoch": stats.get("current_epoch", 0),
                            })
                except Exception:
                    pass
            
            write_jsonl({
                "event": "cycle_complete",
                "cycle": cycle,
                "steps_this_cycle": epoch_steps,
                "total_steps": steps_done,
                "stopped_early": early,
                "stop_reason": stop_reason,
                "chunk_samples": chunk_samples,
                "block_progress": samples_this_block,
                "block_size": samples_per_block,
            })
            
            # Save checkpoint after each chunk (CRITICAL for resume capability)
            if checkpoint_saver:
                checkpoint_saver.update_progress(steps_done, cycle)
                checkpoint_saver.save_checkpoint(
                    reason="chunk_complete",
                    step=steps_done,
                    cycle=cycle,
                )
            
            # Save chunk tracker state to disk (sync with checkpoint)
            if chunk_tracker is not None:
                try:
                    chunk_tracker.save()
                    log_fn({"chunk_tracker": "state_saved", "cycle": cycle, "steps": steps_done})
                except Exception as e:
                    log_fn({"chunk_tracker": "save_failed", "error": str(e)})
            
            # Check for external stop conditions (stop file, halt, etc.)
            if early and stop_reason not in ["stop_after_block", "stop_after_epoch"]:
                log_fn({
                    "iterate_mode": "stopped",
                    "cycle": cycle,
                    "reason": stop_reason or "early_stop",
                    "total_steps": steps_done
                })
                break
            
            # Check block completion (one block = samples_per_block)
            block_complete = samples_this_block >= samples_per_block
            
            if block_complete:
                # If tracker enabled, mark the block as complete and advance logical block id
                if getattr(config, 'linear_dataset', False) and chunk_tracker is not None:
                    try:
                        if hasattr(chunk_tracker, 'mark_block_complete'):
                            chunk_tracker.mark_block_complete(int(current_block_id))
                        # advance logical block id for subsequent cycles
                        current_block_id += 1
                        setattr(config, 'current_block_id', int(current_block_id))
                        # reset per-block sample counter persisted in config if present
                        if hasattr(config, 'current_block_samples'):
                            config.current_block_samples = 0
                    except Exception:
                        pass
                
                # Update config for checkpoint saving
                if hasattr(config, 'current_block_samples'):
                    config.current_block_samples = 0
                
                samples_this_block = 0  # Reset for next block
                config.current_block_samples = 0  # Keep config in sync
                
                # Stop if stop_after_block is enabled
                if config.stop_after_block:
                    log_fn({
                        "iterate_mode": "stopped",
                        "cycle": cycle,
                        "reason": "stop_after_block",
                        "total_steps": steps_done,
                        "note": "Completed one block (100k samples)"
                    })
                    stopped_early = True
                    last_stop_reason = "stop_after_block"
                    break
                
                # For epoch completion, would check if all blocks visited
                # (requires tracking blocks_processed_this_epoch)
                # For now, stop_after_epoch with iterate means stop after one block
                if config.stop_after_epoch:
                    log_fn({
                        "iterate_mode": "stopped",
                        "cycle": cycle,
                        "reason": "stop_after_epoch",
                        "total_steps": steps_done,
                        "note": "Completed block (epoch tracking needs full implementation)"
                    })
                    stopped_early = True
                    last_stop_reason = "stop_after_epoch"
                    break
            
            # Note: config.current_block_samples already updated immediately after adding chunk_samples
            # No need to update again here
            
            cycle += 1
            log_fn({
                "iterate_cycle": cycle - 1,
                "status": "complete",
                "next_cycle": cycle
            })
            
    except typer.Exit as e:
        logger.warning(f"User exit requested during iterate mode: cycle={cycle}, exit_code={e.exit_code}")
        log_fn({
            "iterate_mode": "user_exit",
            "cycle": cycle,
            "exit_code": e.exit_code,
            "note": "Stop requested - will continue to finalization"
        })
        # Don't re-raise - let training continue to finalization
        # The stopped_early flag and stop_reason will handle this
        pass
    except Exception as e:
        logger.error(f"Error in iterate mode training: cycle={cycle}, error={e}", exc_info=True)
        log_fn({
            "iterate_mode": "error",
            "cycle": cycle,
            "error": str(e),
            "traceback": str(__import__('traceback').format_exc()),
        })
        # Don't re-raise - let training continue to finalization
        # Even if there's an error, we should save progress
        pass
    
    logger.info(f"Iterate mode training completed: cycles={cycle}, steps={steps_done}, stopped_early={stopped_early}, reason={last_stop_reason}")
    
    return steps_done, stopped_early, last_stop_reason, cycle


def run_single_epoch_mode(
    execute_epoch: Callable,
    log_fn,
    checkpoint_saver = None,
    steps_done: int = 0,
) -> Tuple[int, bool, Optional[str]]:
    """Run training for a single epoch (non-iterate mode).
    
    Returns:
        Tuple of (steps_done, stopped_early, last_stop_reason)
    """
    logger.info("Starting single epoch training mode")
    
    steps_done, stopped_early, _, last_stop_reason = execute_epoch()
    
    logger.info(f"Single epoch training completed: steps={steps_done}, stopped_early={stopped_early}, reason={last_stop_reason}")
    
    log_fn({
        "training_mode": "single_epoch",
        "steps_completed": steps_done,
        "stopped_early": stopped_early,
        "stop_reason": last_stop_reason
    })
    
    # Save checkpoint after epoch
    if checkpoint_saver:
        checkpoint_saver.update_progress(steps_done, 1)
        checkpoint_saver.save_checkpoint(
            reason="epoch_complete",
            step=steps_done,
            cycle=1,
        )
    
    return steps_done, stopped_early, last_stop_reason
