"""
Training Executor Module for ACT-V1 Training

Handles the execution of the training loop:
- Streaming dataset recreation for iterate mode
- Training epoch execution wrapper
- Iterate mode loop with cycle management
- Stop file checking and early stopping
- Data reloading between cycles
- Dataset verification and statistics
- Error handling for training loop

This module orchestrates the actual training process after all setup is complete.
"""

from typing import Any, Optional, Callable
import time
import typer


def execute_training(
    model_student: Any,
    optimizer: Any,
    segment_rollout: Callable,
    device_obj: Any,
    dml_device: Any,
    input_ids: Optional[Any],
    labels: Optional[Any],
    batch_size: int,
    steps: int,
    halt_max_steps: int,
    sys_mem_cap_pct: float,
    dev: str,
    is_distributed: bool,
    world_sz: int,
    stop_file: Optional[str],
    write_jsonl: Callable,
    should_stop: Callable,
    load_or_generate_lines: Callable,
    lines: list[str],
    streaming_dataset: Optional[Any],
    use_streaming: bool,
    cycle_count: int,
    tokenizer: Any,
    max_seq_len: int,
    use_amp: bool,
    scaler: Optional[Any],
    deepspeed_engine: Optional[Any],
    inference_manager: Optional[Any],
    hot_reload_steps: int,
    warmup_steps: int,
    base_lr: float,
    config: Any,  # TrainingConfig
    step_offset: int,
    iterate: bool,
    resume_cycle: int,
) -> dict[str, Any]:
    """
    Execute the training loop (single epoch or iterate mode).
    
    This function handles:
    1. Streaming dataset recreation for iterate mode
    2. Training epoch execution
    3. Iterate mode loop with cycle management
    4. Stop file checking between cycles
    5. Data reloading and verification
    6. Error handling and graceful degradation
    
    Args:
        model_student: The training model
        optimizer: Training optimizer
        segment_rollout: Chunked training function
        device_obj: PyTorch device
        dml_device: DirectML device (if applicable)
        input_ids, labels: Training data tensors (None for streaming)
        batch_size: Training batch size
        steps: Number of training steps per epoch
        halt_max_steps: Maximum adaptive computation steps
        sys_mem_cap_pct: System memory cap percentage
        dev: Device string
        is_distributed: Whether using distributed training
        world_sz: World size for distributed training
        stop_file: Path to stop file
        write_jsonl: Function to write JSONL logs
        should_stop: Function to check stop file
        load_or_generate_lines: Function to load new dataset lines
        lines: Current dataset lines
        streaming_dataset: Streaming dataset object (if applicable)
        use_streaming: Whether using streaming mode
        cycle_count: Initial cycle count
        tokenizer: Tokenizer object
        max_seq_len: Maximum sequence length
        use_amp: Whether using mixed precision
        scaler: AMP GradScaler
        deepspeed_engine: DeepSpeed engine (if using ZeRO)
        inference_manager: Multi-GPU inference manager
        hot_reload_steps: Frequency of hot-reloading for inference
        warmup_steps: Learning rate warmup steps
        base_lr: Base learning rate
        config: Training configuration
        step_offset: Starting step offset (for resume)
        iterate: Whether using iterate mode
        resume_cycle: Resume cycle number
        
    Returns:
        Dictionary containing training results:
        - steps_done: Total steps completed
        - cycle: Final cycle number
        - stopped_early: Whether training stopped early
        - last_stop_reason: Reason for stopping
        - batch_size: Final batch size
    """
    
    # Import helper functions
    from ..helpers import _write_last_safe_batches_helper, _train_epoch_helper
    
    # ============================================================================
    # Streaming Dataset Recreation Helper
    # ============================================================================
    def _train_with_lines(lines_in: list[str]) -> None:
        """Recreate streaming dataset with new lines for iterate mode."""
        nonlocal input_ids, labels, streaming_dataset, cycle_count
        
        if use_streaming:
            # CRITICAL: Recreate streaming dataset with new lines for iterate mode
            from ..streaming_dataset import create_streaming_dataset
            
            # Determine shuffle mode and start offset
            shuffle_mode = not config.linear_dataset
            start_offset = config.dataset_start_offset if config.linear_dataset else 0
            
            streaming_dataset = create_streaming_dataset(
                lines=lines_in,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                shuffle=shuffle_mode,
                epoch=cycle_count,
                start_offset=start_offset,
            )
            cycle_count += 1
        else:
            # Import encoding helper
            from ..helpers import _encode_lines_helper
            input_ids, labels = _encode_lines_helper(tokenizer, lines_in, max_seq_len)
    
    # ============================================================================
    # Streaming Dataset Initial Creation
    # ============================================================================
    if use_streaming:
        from ..streaming_dataset import create_streaming_dataset
        
        shuffle_mode = not config.linear_dataset
        start_offset = config.dataset_start_offset if config.linear_dataset else 0
        
        streaming_dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=shuffle_mode,
            epoch=0,
            start_offset=start_offset,
        )
        N = len(lines)
        
        mode_str = "linear (sequential)" if config.linear_dataset else "shuffled"
        print({
            "dataset_mode": "streaming",
            "progression": mode_str,
            "initial_samples": N,
            "epoch": 0,
            "start_offset": start_offset if config.linear_dataset else 0
        })
    else:
        if input_ids is not None:
            N = input_ids.shape[0]
        else:
            N = len(lines)
        streaming_dataset = None
    
    # ============================================================================
    # GPU/CPU Utilization Targets
    # ============================================================================
    import os
    try:
        _gpu_util_target = os.environ.get("AIOS_GPU_UTIL_TARGET")
        gpu_util_target = int(_gpu_util_target) if _gpu_util_target is not None else 0
    except Exception:
        gpu_util_target = 0
    
    try:
        _cpu_util_target = os.environ.get("AIOS_CPU_UTIL_TARGET")
        cpu_util_target = int(_cpu_util_target) if _cpu_util_target is not None else 0
    except Exception:
        cpu_util_target = 0
    
    try:
        gpu_util_mode = str(os.environ.get("AIOS_GPU_UTIL_MODE") or "duty").lower()
        gpu_util_poll_ms = int(os.environ.get("AIOS_GPU_UTIL_POLL_MS") or "50")
    except Exception:
        gpu_util_mode = "duty"
        gpu_util_poll_ms = 50
    
    # ============================================================================
    # Training State Variables
    # ============================================================================
    steps_done = 0
    stopped_early = False
    last_stop_reason: Optional[str] = None
    
    # ============================================================================
    # Training Epoch Wrapper
    # ============================================================================
    def _do_train_epoch():
        """Execute one training epoch."""
        nonlocal steps_done, batch_size, stopped_early, last_stop_reason
        
        # Periodic evaluation removed - only runs at end now
        def _maybe_eval():
            pass  # No-op during training
        
        s_done, early, new_bs, stop_reason = _train_epoch_helper(
            model_student=model_student,
            segment_rollout=segment_rollout,
            opt=optimizer,
            device_obj=device_obj,
            dml_device=dml_device,
            input_ids=input_ids,
            labels=labels,
            batch_size=batch_size,
            steps=steps,
            halt_max_steps=halt_max_steps,
            sys_mem_cap_pct=sys_mem_cap_pct,
            dev=dev,
            is_distributed=is_distributed,
            world_sz=world_sz,
            stop_file=stop_file,
            write_jsonl=write_jsonl,
            should_stop=should_stop,
            write_last_safe_batches=_write_last_safe_batches_helper,
            eval_maybe=_maybe_eval,
            gpu_util_target=gpu_util_target,
            cpu_util_target=cpu_util_target,
            gpu_util_mode=gpu_util_mode,
            gpu_util_poll_ms=gpu_util_poll_ms,
            streaming_dataset=streaming_dataset,
            tokenizer=tokenizer,
            lines=lines if use_streaming else None,
            use_amp=use_amp,
            scaler=scaler,
            deepspeed_engine=deepspeed_engine,
            inference_manager=inference_manager,
            hot_reload_steps=hot_reload_steps if inference_manager else 0,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
            stop_after_epoch=config.stop_after_epoch,
            step_offset=step_offset,
        )
        steps_done += s_done
        stopped_early = early
        batch_size = new_bs
        last_stop_reason = stop_reason
    
    # ============================================================================
    # Training Execution
    # ============================================================================
    # Initialize cycle for both modes
    cycle = resume_cycle if iterate else 0
    
    if not iterate:
        # Single epoch mode
        for _ in [0]:
            _do_train_epoch()
    else:
        # Iterate mode - multiple cycles
        print({
            "iterate_mode": "starting",
            "initial_cycle": cycle,
            "resumed": resume_cycle > 0
        })
        
        try:
            while True:
                # Check stop file FIRST before attempting any operations
                if stop_file and isinstance(stop_file, str):
                    try:
                        from pathlib import Path
                        if Path(stop_file).exists():
                            print({
                                "stopped": True,
                                "phase": "iterate_before_operations",
                                "cycle": int(cycle)
                            })
                            stopped_early = True
                            last_stop_reason = "stop_file"
                            break
                    except Exception:
                        pass
                
                # Check if training was stopped during epoch
                if stopped_early:
                    print({
                        "stopped": True,
                        "phase": "iterate_after_epoch",
                        "cycle": int(cycle)
                    })
                    break
                
                # Load new dataset chunk for this cycle
                try:
                    new_lines = load_or_generate_lines(cycle=cycle)
                    if not new_lines:
                        print({"started": False, "error": "no lines (iterate)"})
                        stopped_early = True
                        last_stop_reason = "no_data"
                        break
                except Exception as load_error:
                    print({
                        "data_load_error": str(load_error),
                        "cycle": cycle
                    })
                    stopped_early = True
                    last_stop_reason = "data_load_error"
                    break
                
                # Train with loaded lines
                try:
                    _train_with_lines(new_lines)
                except Exception as train_lines_error:
                    print({"train_with_lines_error": str(train_lines_error)})
                    stopped_early = True
                    last_stop_reason = "train_lines_error"
                    break
                
                try:
                    write_jsonl({"event": "iterate_cycle", "cycle": int(cycle)})
                except Exception:
                    pass
                
                print({
                    "debug": "before_do_train_epoch",
                    "cycle": cycle,
                    "stopped_early": stopped_early
                })
                _do_train_epoch()
                print({
                    "debug": "after_do_train_epoch",
                    "cycle": cycle,
                    "stopped_early": stopped_early
                })
                
                # Check if we need to break after epoch completes
                if stopped_early:
                    if last_stop_reason == "stop_after_epoch":
                        print({
                            "ITERATE_CYCLE_COMPLETE": True,
                            "cycle": cycle,
                            "stop_reason": last_stop_reason
                        })
                        stopped_early = False
                        last_stop_reason = None
                    else:
                        print({
                            "ITERATE_LOOP_STOPPED_EARLY_AFTER_EPOCH": True,
                            "cycle": cycle,
                            "stop_reason": last_stop_reason
                        })
                        break
                
                # Verify data variety after each cycle
                if use_streaming and streaming_dataset is not None:
                    try:
                        stats = streaming_dataset.get_sample_stats()
                        write_jsonl({"event": "dataset_stats", "cycle": int(cycle), **stats})
                        print({"cycle_dataset_verification": stats})
                    except Exception as e:
                        print({"dataset_stats_error": str(e)})
                
                print({
                    "debug": "before_cycle_increment",
                    "cycle": cycle,
                    "stopped_early": stopped_early
                })
                cycle += 1
                print({
                    "debug": "after_cycle_increment",
                    "cycle": cycle,
                    "stopped_early": stopped_early
                })
                
        except typer.Exit as e:
            # Catch typer.Exit and convert to stopped_early flag
            print({
                "iterate_loop_typer_exit": True,
                "exit_code": getattr(e, 'exit_code', None)
            })
            stopped_early = True
            last_stop_reason = "typer_exit"
            # Don't re-raise - allow finalization to run
        except Exception as e:
            # Catch any other exception and ensure finalization runs
            import traceback
            print({
                "iterate_loop_error": str(e),
                "traceback": str(traceback.format_exc())
            })
            stopped_early = True
            last_stop_reason = "exception"
            # Don't re-raise - allow finalization to run
    
    print({
        "EXITED_ITERATE_LOOP": True,
        "stopped_early": stopped_early,
        "steps_done": steps_done,
        "stop_reason": last_stop_reason
    })
    
    # ============================================================================
    # Return Training Results
    # ============================================================================
    return {
        "steps_done": steps_done,
        "cycle": cycle,
        "stopped_early": stopped_early,
        "last_stop_reason": last_stop_reason,
        "batch_size": batch_size,
    }
