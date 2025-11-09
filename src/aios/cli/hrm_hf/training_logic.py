from __future__ import annotations

from typing import Optional, Any, Callable, Dict


def average_gradients_if_distributed(model_student, *, is_distributed: bool, world_sz: int) -> None:
    """Average gradients across processes when torch.distributed is initialized.

    Safe to call when not distributed; it will no-op.
    """
    if not is_distributed:
        return
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        for p in model_student.parameters():
            if p.grad is None:
                continue
            try:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= float(max(1, world_sz))
            except Exception:
                pass
    except Exception:
        pass


def train_epoch(
    *,
    model_student,
    segment_rollout,
    opt,
    device_obj,
    dml_device,
    input_ids,
    labels,
    batch_size: int,
    steps: int,
    halt_max_steps: int,
    sys_mem_cap_pct: Optional[int],
    dev: str,
    is_distributed: bool,
    world_sz: int,
    stop_file: Optional[str],
    write_jsonl: Callable[[Dict[str, Any]], None],
    should_stop: Callable[[], bool],
    write_last_safe_batches: Callable[..., None],
    eval_maybe: Optional[Callable[[], None]] = None,
    gpu_util_target: int = 0,
    cpu_util_target: int = 0,
    gpu_util_mode: str = "duty",
    gpu_util_poll_ms: int = 50,
    streaming_dataset = None,  # NEW: Optional streaming dataset
    tokenizer = None,  # NEW: Tokenizer for streaming mode
    lines: Optional[list] = None,  # NEW: Raw lines for streaming mode
    use_amp: bool = False,  # NEW: Enable mixed precision training
    scaler = None,  # NEW: GradScaler for AMP
    deepspeed_engine = None,  # NEW: DeepSpeed engine if using ZeRO
    inference_manager = None,  # NEW: InferenceModelManager for multi-GPU inference
    hot_reload_steps: int = 0,  # NEW: Hot-reload frequency
    warmup_steps: int = 0,  # NEW: Number of warmup steps
    base_lr: float = 0.0,  # NEW: Base learning rate for warmup
    stop_after_epoch: bool = False,  # NEW: Stop training after epoch completes
    step_offset: int = 0,  # NEW: Resume from this step number
) -> tuple[int, bool, int, Optional[str]]:
    """Run a training epoch with OOM backoff and optional eval callback.

    Returns (steps_done, stopped_early, new_batch_size, stop_reason).
    
    Supports two modes:
    1. Eager loading: input_ids and labels are pre-materialized tensors
    2. Streaming: tokenize batches on-the-fly from lines using tokenizer
    
    Resume support:
    - step_offset: Start counting from this step (for display purposes)
    - Actual training still runs for 'steps' iterations
    """
    steps_done = 0
    stopped_early = False
    stop_reason: Optional[str] = None

    def _sys_mem_used_pct() -> Optional[float]:
        try:
            import psutil  # type: ignore
            v = psutil.virtual_memory()
            return float(v.percent)
        except Exception:
            return None

    try:
        import torch
    except Exception:
        raise

    # Determine if we're using streaming or eager loading
    use_streaming = streaming_dataset is not None and lines is not None and tokenizer is not None
    
    if use_streaming:
        assert lines is not None  # Type narrowing
        assert streaming_dataset is not None  # Type narrowing
        N = len(lines)
        # Create an iterator over the streaming dataset
        dataset_iter = iter(streaming_dataset)  # type: ignore
    else:
        assert input_ids is not None  # Type narrowing
        N = int(input_ids.shape[0])
        dataset_iter = None

    import time as __time
    import sys as __sys
    for step in range(int(max(1, steps))):
        # Force flush to ensure logs appear immediately
        print({"LOOP_ITERATION_START": step, "display_step": step_offset + step + 1}, flush=True)
        __sys.stdout.flush()
        
        # Calculate display step number (includes offset for resume)
        display_step = step_offset + step + 1
        
        # External stop gate (file-based)
        if stop_file and isinstance(stop_file, str):
            try:
                from pathlib import Path as _Path
                if _Path(stop_file).exists():
                    try:
                        write_jsonl({"event": "stopped", "phase": "train", "step": int(display_step)})
                    except Exception:
                        pass
                    print({"STOP_DETECTED": True, "phase": "train", "step": display_step, "will_break_from_loop": True})
                    stopped_early = True
                    stop_reason = "stop_file"
                    print({"STOP_BREAKING": "about to break from training loop"})
                    break
            except Exception as stop_check_error:
                print({"stop_check_exception": str(stop_check_error)})
                pass

        # CPU memory soft-cap backoff
        if dev == "cpu" and (sys_mem_cap_pct is not None) and batch_size > 1:
            used = _sys_mem_used_pct()
            if used is not None and used > float(sys_mem_cap_pct):
                batch_size = max(1, int(batch_size * 0.85))
                write_last_safe_batches(train_bs=int(batch_size))

        # Step attempt with CUDA OOM backoff
        attempt = 0
        max_attempts = 5  # Limit retry attempts
        throttle_gpu = False
        throttle_cpu = False
        # Initialize variables that may be used after loop exits
        loss = None
        metrics = None
        while True:
            if should_stop():
                # Set stopped_early flag and break cleanly to allow finalization
                stopped_early = True
                stop_reason = "external_stop"
                break
            _t0 = __time.perf_counter()
            
            # Learning rate warmup: gradually increase LR from 0 to base_lr over warmup_steps
            # This prevents early gradient explosions and loss spikes in MoE models
            if warmup_steps > 0 and base_lr > 0:
                if steps_done < warmup_steps:
                    warmup_factor = (steps_done + 1) / warmup_steps  # +1 to avoid starting at 0
                    current_lr = base_lr * warmup_factor
                    for param_group in opt.param_groups:
                        param_group['lr'] = current_lr
                elif steps_done == warmup_steps:
                    # Warmup complete, set to base LR and log
                    for param_group in opt.param_groups:
                        param_group['lr'] = base_lr
                    write_jsonl({"event": "warmup_complete", "step": steps_done, "lr": base_lr})
            
            try:
                # Get batch - either from streaming dataset or by sampling from pre-loaded tensors
                if use_streaming:
                    # Get next batch from streaming dataset
                    try:
                        assert dataset_iter is not None  # Type narrowing
                        assert streaming_dataset is not None  # Type narrowing
                        inp, tgt, puzzle_ids = next(dataset_iter)  # type: ignore
                    except StopIteration:
                        # Restart iterator (new epoch)
                        assert streaming_dataset is not None  # Type narrowing
                        dataset_iter = iter(streaming_dataset)  # type: ignore
                        inp, tgt, puzzle_ids = next(dataset_iter)  # type: ignore
                else:
                    # Sample random batch from pre-loaded tensors (original behavior)
                    assert input_ids is not None and labels is not None  # Type narrowing
                    try:
                        idx = torch.randint(0, N, (int(batch_size),))
                        inp = input_ids.index_select(0, idx)
                        tgt = labels.index_select(0, idx)
                        puzzle_ids = torch.zeros((inp.shape[0],), dtype=torch.int64)  # Create here for both paths
                    except RuntimeError as tensor_error:
                        # CUDA error or other tensor operation failure
                        print({"FATAL_TENSOR_ERROR": str(tensor_error), "step": steps_done}, flush=True)
                        # Force stop to trigger finalization
                        stopped_early = True
                        break
                
                # Clear CUDA cache before moving tensors to GPU
                if dev == "cuda" and attempt > 0:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                
                if dml_device is not None:
                    inp = inp.to(dml_device)
                    tgt = tgt.to(dml_device)
                else:
                    inp = inp.to(device_obj)
                    tgt = tgt.to(device_obj)
                
                # Create batch dict (puzzle_ids is now defined in both paths)
                batch = {
                    "inputs": inp,
                    "targets": tgt,
                    "puzzle_identifiers": puzzle_ids.to(dml_device or device_obj),
                }
                
                # Zero gradients (DeepSpeed doesn't need explicit zero_grad)
                if deepspeed_engine is None:
                    opt.zero_grad(set_to_none=True)
                
                # Use mixed precision if enabled (for CUDA only)
                # Using new torch.amp.autocast API (PyTorch 2.0+)
                if use_amp and dev == "cuda":
                    import torch
                    with torch.amp.autocast(device_type='cuda'):  # type: ignore[attr-defined]
                        loss, metrics = segment_rollout(model_student, batch, max_segments=int(halt_max_steps), epsilon=0.1)
                else:
                    loss, metrics = segment_rollout(model_student, batch, max_segments=int(halt_max_steps), epsilon=0.1)
                
                # ============================================================================
                # MoE Load Balancing Loss (CRITICAL FOR PROPER MoE TRAINING)
                # ============================================================================
                # Add auxiliary loss to encourage uniform expert usage and prevent expert collapse
                if hasattr(model_student, 'inner') and hasattr(model_student.inner, 'config'):
                    config = model_student.inner.config
                    if getattr(config, 'use_moe', False):
                        from aios.core.hrm_models.moe_layer import load_balancing_loss
                        
                        lb_loss_total = 0.0
                        lb_count = 0
                        
                        # Collect router logits from all MoE layers (H-level and L-level)
                        for level_name in ['H_level', 'L_level']:
                            if hasattr(model_student.inner, level_name):
                                level_module = getattr(model_student.inner, level_name)
                                for layer in level_module.layers:
                                    if hasattr(layer, 'mlp'):
                                        # Check for MoESwiGLU (has moe attribute)
                                        if hasattr(layer.mlp, 'moe'):
                                            moe = layer.mlp.moe
                                            if hasattr(moe, 'last_router_logits') and moe.last_router_logits is not None:
                                                lb_loss_total += load_balancing_loss(
                                                    moe.last_router_logits,
                                                    num_experts=config.num_experts
                                                )
                                                lb_count += 1
                                        # Check for direct MoELayer (has last_router_logits)
                                        elif hasattr(layer.mlp, 'last_router_logits') and layer.mlp.last_router_logits is not None:
                                            lb_loss_total += load_balancing_loss(
                                                layer.mlp.last_router_logits,
                                                num_experts=config.num_experts
                                            )
                                            lb_count += 1
                        
                        if lb_count > 0:
                            # Average across all MoE layers (lb_loss is a tensor)
                            lb_loss = lb_loss_total / lb_count
                            
                            # Get coefficient (default 0.05 after stability fix, configurable)
                            lb_coef = getattr(config, 'moe_load_balance_loss_coef', 0.05)
                            
                            # Add to main loss
                            loss = loss + lb_coef * lb_loss
                            
                            # Log for monitoring
                            if metrics is not None:
                                try:
                                    # lb_loss is always a tensor here
                                    metrics['lb_loss'] = float(lb_loss.detach().cpu().item()) if torch.is_tensor(lb_loss) else lb_loss
                                    metrics['lb_coef'] = lb_coef
                                    metrics['moe_layers'] = lb_count
                                except Exception:
                                    pass
                
                # NaN detection - skip batch if loss is NaN to prevent training collapse
                if loss is None or torch.isnan(loss).any() if torch.is_tensor(loss) else False or torch.isinf(loss).any() if torch.is_tensor(loss) else False:
                    # Use detach() to avoid autograd warnings
                    loss_val = float(loss.detach().cpu()) if loss is not None and torch.is_tensor(loss) else None
                    ce_val = float(metrics.get("ce").detach().cpu()) if metrics.get("ce") is not None and torch.is_tensor(metrics.get("ce")) else None
                    bce_halt = float(metrics.get("bce_halt").detach().cpu()) if metrics.get("bce_halt") is not None and torch.is_tensor(metrics.get("bce_halt")) else None
                    bce_continue = float(metrics.get("bce_continue").detach().cpu()) if metrics.get("bce_continue") is not None and torch.is_tensor(metrics.get("bce_continue")) else None
                    
                    print({
                        "event": "nan_detected",
                        "step": steps_done + 1,
                        "action": "skipping_batch",
                        "loss": loss_val,
                        "ce": ce_val,
                        "bce_halt": bce_halt,
                        "bce_continue": bce_continue,
                        "batch_size": batch["inputs"].shape[0],
                        "seq_len": batch["inputs"].shape[1],
                    })
                    
                    # Reset model state to prevent NaN propagation
                    if deepspeed_engine is None:
                        opt.zero_grad(set_to_none=True)
                    
                    continue  # Skip this batch and move to next one

                # Average scalar loss across processes BEFORE backward so gradients reflect global mean
                log_metrics_extra: Dict[str, Any] = {}
                if is_distributed:
                    try:
                        import torch.distributed as dist
                        if not (dist.is_available() and dist.is_initialized()):
                            raise RuntimeError("dist not initialized")
                        if loss is not None and hasattr(loss, "detach"):
                            _dl = loss.detach()
                            dist.all_reduce(_dl, op=dist.ReduceOp.SUM)
                            world_safe = float(max(1, world_sz))
                            try:
                                # Replace in-place operation with new tensor to preserve gradients
                                loss = loss * (1.0 / world_safe)
                            except Exception:
                                pass
                            try:
                                _avg_val = float((_dl / world_safe).cpu().item())
                            except Exception:
                                _avg_val = None
                            if _avg_val is not None:
                                log_metrics_extra["dist_avg_loss"] = _avg_val
                    except Exception:
                        pass

                # Use scaler for backward if AMP is enabled (but not with DeepSpeed)
                if deepspeed_engine is not None:
                    # DeepSpeed handles backward and optimizer step
                    deepspeed_engine.backward(loss)
                    deepspeed_engine.step()
                elif use_amp and scaler is not None and dev == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    average_gradients_if_distributed(model_student, is_distributed=is_distributed, world_sz=world_sz)
                    # More conservative gradient clipping for MoE stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                    # Detect NaN/Inf gradients and skip update if found
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print({"warning": "NaN/Inf gradients detected", "step": steps_done, "skipping": True})
                        opt.zero_grad(set_to_none=True)
                        scaler.update()  # Must update scaler even when skipping step to reset state
                    else:
                        scaler.step(opt)
                        scaler.update()
                else:
                    loss.backward()
                    average_gradients_if_distributed(model_student, is_distributed=is_distributed, world_sz=world_sz)
                    # More conservative gradient clipping for MoE stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)
                    # Detect NaN/Inf gradients and skip update if found
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print({"warning": "NaN/Inf gradients detected", "step": steps_done, "skipping": True})
                        opt.zero_grad(set_to_none=True)
                    else:
                        opt.step()
                    
                steps_done += 1
                
                # Hot-reload inference model if enabled and it's time
                if inference_manager is not None and hot_reload_steps > 0:
                    if steps_done % hot_reload_steps == 0:
                        try:
                            # Save current checkpoint for inference to load
                            # (Assuming checkpoint is saved automatically or we trigger it here)
                            inference_manager.reload_from_checkpoint("actv1_student.safetensors")
                        except Exception as e:
                            print({
                                "hot_reload": "failed",
                                "step": steps_done,
                                "error": str(e)
                            })
                
                # Track dataset position for streaming datasets
                if use_streaming and streaming_dataset is not None:
                    try:
                        current_pos = streaming_dataset.get_position()
                        log_metrics_extra['dataset_position'] = current_pos
                        log_metrics_extra['dataset_progress_pct'] = round((current_pos / streaming_dataset.num_samples) * 100, 2)
                    except Exception:
                        pass
                
                # Log performance metrics
                _t1 = __time.perf_counter()
                step_time = _t1 - _t0
                tokens_in_batch = batch["inputs"].shape[0] * batch["inputs"].shape[1]
                tokens_per_sec = tokens_in_batch / max(step_time, 1e-6)
                
                # Log memory and performance stats every few steps
                if steps_done % 5 == 0 or steps_done == 1:
                    try:
                        import torch
                        if dev == "cuda" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            allocated_gb = torch.cuda.memory_allocated(device_obj) / 1024**3
                            reserved_gb = torch.cuda.memory_reserved(device_obj) / 1024**3
                            max_allocated_gb = torch.cuda.max_memory_allocated(device_obj) / 1024**3
                            total_gb = torch.cuda.get_device_properties(device_obj).total_memory / 1024**3
                            
                            write_jsonl({
                                "step": steps_done,
                                "memory_gb": round(allocated_gb, 3),
                                "reserved_gb": round(reserved_gb, 3),
                                "peak_gb": round(max_allocated_gb, 3),
                                "total_gb": round(total_gb, 3),
                                "utilization_pct": round((allocated_gb / total_gb) * 100, 1),
                                "fragmentation_mb": round((reserved_gb - allocated_gb) * 1024, 1),
                                "tokens_per_sec": round(tokens_per_sec, 1),
                                "step_time_sec": round(step_time, 3),
                                "batch_size": batch_size,
                                "seq_len": batch["inputs"].shape[1],
                                **log_metrics_extra
                            })
                    except Exception as e:
                        # Don't fail training if logging fails
                        pass
                
                # Log expert usage statistics every 100 steps (MoE monitoring)
                if steps_done % 100 == 0:
                    try:
                        if hasattr(model_student, 'inner') and hasattr(model_student.inner, 'config'):
                            config = model_student.inner.config
                            if getattr(config, 'use_moe', False):
                                from aios.core.hrm_models.moe_layer import get_expert_usage_stats
                                
                                # Sample H-level first layer for expert usage stats
                                if hasattr(model_student.inner, 'H_level'):
                                    level_module = model_student.inner.H_level
                                    if len(level_module.layers) > 0:
                                        layer = level_module.layers[0]
                                        if hasattr(layer, 'mlp'):
                                            # Get router logits
                                            router_logits = None
                                            if hasattr(layer.mlp, 'moe') and hasattr(layer.mlp.moe, 'last_router_logits'):
                                                router_logits = layer.mlp.moe.last_router_logits
                                            elif hasattr(layer.mlp, 'last_router_logits'):
                                                router_logits = layer.mlp.last_router_logits
                                            
                                            if router_logits is not None:
                                                stats = get_expert_usage_stats(router_logits)
                                                write_jsonl({
                                                    "event": "expert_usage",
                                                    "step": steps_done,
                                                    "avg_routing_prob": [round(p, 4) for p in stats['avg_routing_prob']],
                                                    "token_counts": stats['token_counts'],
                                                    "total_tokens": stats['total_tokens'],
                                                })
                    except Exception:
                        # Don't fail training if monitoring fails
                        pass
                
                # Simple GPU utilization throttle: sleep to meet target duty cycle
                if (dev == "cuda") and (int(gpu_util_target) > 0) and (int(gpu_util_target) < 100):
                    try:
                        comp = max(1e-4, float(step_time))
                        target = max(1, int(gpu_util_target))
                        sleep_s = comp * (100.0 / float(target) - 1.0)
                        if sleep_s > 0:
                            throttle_gpu = True
                            __time.sleep(min(sleep_s, 0.5))
                    except Exception:
                        pass
                # CPU utilization throttle (duty-cycle approximation)
                if (dev == "cpu") and (int(cpu_util_target) > 0) and (int(cpu_util_target) < 100):
                    try:
                        _t1 = __time.perf_counter()
                        comp = max(1e-4, float(_t1 - _t0))
                        target = max(1, int(cpu_util_target))
                        sleep_s = comp * (100.0 / float(target) - 1.0)
                        if sleep_s > 0:
                            throttle_cpu = True
                            __time.sleep(min(sleep_s, 0.5))
                    except Exception:
                        pass
                break
            except RuntimeError as e:
                if ("out of memory" in str(e).lower()) and (dev == "cuda"):
                    # Clean up tensors explicitly - force garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear CUDA cache
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    
                    # Log memory stats
                    try:
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                            total = torch.cuda.get_device_properties(0).total_memory / 1e9
                            print({
                                "event": "oom_memory_stats",
                                "allocated_gb": round(allocated, 2),
                                "reserved_gb": round(reserved, 2),
                                "total_gb": round(total, 2),
                                "attempt": attempt + 1
                            })
                    except Exception:
                        pass
                    
                    attempt += 1
                    
                    # Check attempt limit
                    if attempt >= max_attempts:
                        print({
                            "event": "oom_max_attempts",
                            "attempts": attempt,
                            "final_batch_size": batch_size,
                            "suggestion": "Reduce max_seq_len or enable gradient checkpointing"
                        })
                        raise RuntimeError(
                            f"Out of memory after {attempt} attempts with batch_size={batch_size}. "
                            f"Try reducing max_seq_len or using gradient checkpointing."
                        ) from e
                    
                    decrease_factor = 0.7 if attempt == 1 else 0.5
                    new_bs = max(1, int(batch_size * decrease_factor))
                    
                    if new_bs < batch_size:
                        batch_size = new_bs
                        write_last_safe_batches(train_bs=int(batch_size))
                        try:
                            print({"event": "oom_backoff", "batch_size": int(batch_size), "attempt": int(attempt)})
                        except Exception:
                            pass
                        continue
                    else:
                        # batch_size is already 1 and still OOM
                        print({
                            "event": "oom_at_min_batch",
                            "batch_size": 1,
                            "suggestion": "Reduce max_seq_len, enable gradient checkpointing, or use smaller model"
                        })
                        raise RuntimeError(
                            f"Out of memory even at batch_size=1. "
                            f"Try reducing max_seq_len or using gradient checkpointing."
                        ) from e
                # Non-OOM or cannot recover
                raise

        # Check if stop was requested during batch processing
        if stopped_early:
            break

        # Logging after successful step
        try:
            cur_loss = None
            try:
                if loss is not None:
                    cur_loss = float(loss.detach().cpu().item())
            except Exception:
                cur_loss = None
            payload: Dict[str, Any] = {"event": "train", "step": int(step_offset + steps_done)}
            # Include throttle indicators to signal reaching target utilization
            try:
                payload["throttle_gpu"] = int(1 if throttle_gpu else 0)
                payload["throttle_cpu"] = int(1 if throttle_cpu else 0)
            except Exception:
                pass
            for k, v in (metrics or {}).items():
                try:
                    if hasattr(v, "detach"):
                        v = v.detach()
                    if hasattr(v, "item"):
                        v = float(v.item())
                except Exception:
                    pass
                if hasattr(v, "ndim") and getattr(v, "ndim", 0) == 0:
                    try:
                        import torch as _t
                        v = float(_t.tensor(v).cpu().item())
                    except Exception:
                        pass
                if isinstance(v, (int, float)):
                    payload[k] = v
            # Distributed average for scalar metrics
            if is_distributed:
                try:
                    import torch.distributed as dist
                    if not (dist.is_available() and dist.is_initialized()):
                        raise RuntimeError("dist not initialized")
                    world_safe = float(max(1, world_sz))
                    avg_keys = []
                    for mk, mv in list(payload.items()):
                        if mk in {"event", "step"}:
                            continue
                        if not isinstance(mv, (int, float)):
                            continue
                        avg_keys.append(mk)
                    import torch as _t
                    for mk in avg_keys:
                        try:
                            t = _t.tensor([float(payload[mk])], device=(dml_device or device_obj))
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)
                            payload[mk] = float((t / world_safe).item())
                        except Exception:
                            pass
                except Exception:
                    pass
            if cur_loss is not None:
                payload["loss"] = cur_loss
            print(payload)
            write_jsonl(payload)
            print({"STEP_COMPLETED": steps_done, "about_to_continue_loop": True}, flush=True)
        except Exception:
            pass

        if eval_maybe is not None:
            try:
                eval_maybe()
            except Exception:
                pass
    
    # Check if we should stop after epoch completion
    if stop_after_epoch and not stopped_early:
        try:
            write_jsonl({"event": "stopped_after_epoch", "steps_done": int(steps_done)})
            print({"stopped_after_epoch": True, "steps_done": steps_done})
            stopped_early = True
            stop_reason = "stop_after_epoch"
        except Exception:
            pass

    return steps_done, stopped_early, batch_size, stop_reason
