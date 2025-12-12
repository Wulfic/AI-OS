"""Training epoch implementation."""

from __future__ import annotations

import logging
from typing import Optional, Any, Callable, Dict

from .distributed import average_gradients_if_distributed
from .memory import sys_mem_used_pct

logger = logging.getLogger(__name__)


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
    streaming_dataset = None,
    tokenizer = None,
    lines: Optional[list] = None,
    use_amp: bool = False,
    scaler = None,
    deepspeed_engine = None,
    inference_manager = None,
    hot_reload_steps: int = 0,
    warmup_steps: int = 0,
    base_lr: float = 0.0,
    adaptive_lr_scheduler = None,
    stop_after_epoch: bool = False,
    step_offset: int = 0,
    config = None,
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
    logger.info(
        f"Starting training epoch: steps={steps}, batch_size={batch_size}, "
        f"device={dev}, distributed={is_distributed}, step_offset={step_offset}"
    )
    
    steps_done = 0
    stopped_early = False
    stop_reason: Optional[str] = None

    try:
        import torch
    except Exception:
        raise

    # Determine if we're using streaming or eager loading
    use_streaming = streaming_dataset is not None and lines is not None and tokenizer is not None
    
    if use_streaming:
        assert lines is not None
        assert streaming_dataset is not None
        N = len(lines)
        dataset_iter = iter(streaming_dataset)  # type: ignore
    else:
        assert input_ids is not None
        N = int(input_ids.shape[0])
        dataset_iter = None

    import time as __time
    import sys as __sys
    
    # Setup progress file for DDP monitoring (rank 0 only)
    progress_file = None
    try:
        import os
        if is_distributed and os.environ.get("AIOS_DDP_PROGRESS_FILE"):
            progress_file = os.environ.get("AIOS_DDP_PROGRESS_FILE")
    except Exception:
        pass
    
    # Verify if DDP is actually working (not just enabled)
    ddp_actually_working = False
    current_rank = 0
    if is_distributed:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                ddp_actually_working = True
                current_rank = dist.get_rank()
        except Exception:
            pass
    
    # Fallback: Check environment variables for rank (for spawned workers)
    if current_rank == 0 and not ddp_actually_working:
        try:
            import os
            env_rank = os.environ.get("AIOS_DDP_RANK") or os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
            if env_rank is not None:
                current_rank = int(env_rank)
                # If we have a rank env var, we're in a spawned worker
                if current_rank >= 0:
                    ddp_actually_working = True
        except Exception:
            pass
    
    # Log rank detection for debugging
    try:
        if is_distributed or ddp_actually_working:
            print({
                "ddp_status": "detected" if ddp_actually_working else "not_initialized",
                "current_rank": current_rank,
                "world_size": world_sz,
                "will_log_progress": current_rank == 0,
            }, flush=True)
    except Exception:
        pass
    
    # Set rank-specific random seed for data sampling diversity in DDP
    if ddp_actually_working:
        try:
            # Set different seed per rank to ensure different data samples
            torch.manual_seed(42 + current_rank * 1000 + steps_done)
        except Exception:
            pass
    
    # Track graceful stop flag outside loop
    graceful_stop_pending = False

    def _set_optimizer_lr(lr_value: float) -> None:
        """Set LR on whichever optimizer is active (DeepSpeed or torch optimizer)."""
        try:
            if deepspeed_engine is not None:
                ds_opt = getattr(deepspeed_engine, "optimizer", None)
                if ds_opt is not None and hasattr(ds_opt, "param_groups"):
                    for pg in ds_opt.param_groups:
                        pg["lr"] = float(lr_value)
                    return
        except Exception:
            pass

        try:
            if opt is not None and hasattr(opt, "param_groups"):
                for pg in opt.param_groups:
                    pg["lr"] = float(lr_value)
        except Exception:
            pass

    def _get_optimizer_lr() -> Optional[float]:
        """Best-effort read of current LR (first param group)."""
        try:
            if deepspeed_engine is not None:
                ds_opt = getattr(deepspeed_engine, "optimizer", None)
                if ds_opt is not None and hasattr(ds_opt, "param_groups"):
                    pgs = ds_opt.param_groups
                    if pgs:
                        return float(pgs[0].get("lr"))
        except Exception:
            pass
        try:
            if opt is not None and hasattr(opt, "param_groups"):
                pgs = opt.param_groups
                if pgs:
                    return float(pgs[0].get("lr"))
        except Exception:
            pass
        return None

    # Maintain a loss buffer for gradient-accumulation cycles so the adaptive LR
    # reacts to an averaged signal per optimizer step.
    micro_loss_buffer: list[float] = []
    
    for step in range(int(max(1, steps))):
        display_step = step_offset + step + 1
        
        # External stop gate (file-based)
        if stop_file and isinstance(stop_file, str):
            try:
                from pathlib import Path as _Path
                # Check for immediate STOP file
                if _Path(stop_file).exists():
                    try:
                        write_jsonl({"event": "stopped", "phase": "train", "step": int(display_step), "stop_type": "immediate"})
                    except Exception:
                        pass
                    print({"STOP_DETECTED": True, "phase": "train", "step": display_step, "will_break_from_loop": True})
                    stopped_early = True
                    stop_reason = "stop_file"
                    print({"STOP_BREAKING": "about to break from training loop"})
                    break
                
                # Check for graceful STOP file (finish current step then exit)
                # Set flag to stop after current step completes
                graceful_stop_file = _Path(stop_file).parent / "GRACEFUL_STOP"
                if graceful_stop_file.exists() and not graceful_stop_pending:
                    graceful_stop_pending = True
                    print({"GRACEFUL_STOP_DETECTED": True, "phase": "train", "step": display_step, "will_finish_current_step": True})
                    try:
                        write_jsonl({"event": "graceful_stop_requested", "phase": "train", "step": int(display_step), "stop_type": "graceful"})
                    except Exception:
                        pass
            except Exception as stop_check_error:
                print({"stop_check_exception": str(stop_check_error)})
                pass

        # CPU memory soft-cap backoff
        if dev == "cpu" and (sys_mem_cap_pct is not None) and batch_size > 1:
            used = sys_mem_used_pct()
            if used is not None and used > float(sys_mem_cap_pct):
                batch_size = max(1, int(batch_size * 0.85))
                write_last_safe_batches(train_bs=int(batch_size))

        # Step attempt with CUDA OOM backoff
        attempt = 0
        max_attempts = 5
        throttle_gpu = False
        throttle_cpu = False
        loss = None
        metrics = None
        while True:
            if should_stop():
                stopped_early = True
                stop_reason = "external_stop"
                break
            _t0 = __time.perf_counter()
            
            # Learning rate warmup
            if warmup_steps > 0 and base_lr > 0:
                if steps_done < warmup_steps:
                    warmup_factor = (steps_done + 1) / warmup_steps
                    current_lr = base_lr * warmup_factor
                    _set_optimizer_lr(current_lr)
                elif steps_done == warmup_steps:
                    _set_optimizer_lr(base_lr)
                    write_jsonl({"event": "warmup_complete", "step": steps_done, "lr": base_lr})
            
            try:
                # Get batch
                if use_streaming:
                    try:
                        assert dataset_iter is not None
                        assert streaming_dataset is not None
                        inp, tgt, puzzle_ids = next(dataset_iter)  # type: ignore
                    except StopIteration:
                        assert streaming_dataset is not None
                        dataset_iter = iter(streaming_dataset)  # type: ignore
                        inp, tgt, puzzle_ids = next(dataset_iter)  # type: ignore
                else:
                    assert input_ids is not None and labels is not None
                    try:
                        # In DDP mode with chunk tracking (linear_dataset), ranks work on same data
                        # In DDP mode without chunk tracking (random), ranks sample different ranges
                        linear_mode = getattr(config, 'linear_dataset', False) if config else False
                        if ddp_actually_working and not linear_mode:
                            # Calculate per-rank sample range to avoid overlap (random mode)
                            samples_per_rank = N // world_sz
                            rank_start = current_rank * samples_per_rank
                            rank_end = rank_start + samples_per_rank if current_rank < world_sz - 1 else N
                            # Sample from rank-specific range
                            idx = torch.randint(rank_start, rank_end, (int(batch_size),))
                        else:
                            # Linear mode or single GPU: all ranks sample from same dataset
                            idx = torch.randint(0, N, (int(batch_size),))
                        inp = input_ids.index_select(0, idx)
                        tgt = labels.index_select(0, idx)
                        puzzle_ids = torch.zeros((inp.shape[0],), dtype=torch.int64)
                    except RuntimeError as tensor_error:
                        print({"FATAL_TENSOR_ERROR": str(tensor_error), "step": steps_done}, flush=True)
                        stopped_early = True
                        break
                
                # Clear CUDA cache before moving tensors
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
                
                batch = {
                    "inputs": inp,
                    "targets": tgt,
                    "puzzle_identifiers": puzzle_ids.to(dml_device or device_obj),
                }
                
                if deepspeed_engine is None:
                    opt.zero_grad(set_to_none=True)
                
                # Forward pass with optional AMP
                if use_amp and dev == "cuda":
                    import torch
                    with torch.amp.autocast(device_type='cuda'):  # type: ignore[attr-defined]
                        loss, metrics = segment_rollout(model_student, batch, max_segments=int(halt_max_steps), epsilon=0.1)
                else:
                    loss, metrics = segment_rollout(model_student, batch, max_segments=int(halt_max_steps), epsilon=0.1)
                
                # MoE Load Balancing Loss
                if hasattr(model_student, 'inner') and hasattr(model_student.inner, 'config'):
                    model_config = model_student.inner.config
                    if getattr(model_config, 'use_moe', False):
                        from aios.core.hrm_models.moe_layer import load_balancing_loss
                        
                        lb_loss_total = 0.0
                        lb_count = 0
                        
                        for level_name in ['H_level', 'L_level']:
                            if hasattr(model_student.inner, level_name):
                                level_module = getattr(model_student.inner, level_name)
                                for layer in level_module.layers:
                                    if hasattr(layer, 'mlp'):
                                        if hasattr(layer.mlp, 'moe'):
                                            moe = layer.mlp.moe
                                            if hasattr(moe, 'last_router_logits') and moe.last_router_logits is not None:
                                                lb_loss_total += load_balancing_loss(
                                                    moe.last_router_logits,
                                                    num_experts=model_config.num_experts
                                                )
                                                lb_count += 1
                                        elif hasattr(layer.mlp, 'last_router_logits') and layer.mlp.last_router_logits is not None:
                                            lb_loss_total += load_balancing_loss(
                                                layer.mlp.last_router_logits,
                                                num_experts=model_config.num_experts
                                            )
                                            lb_count += 1
                        
                        if lb_count > 0:
                            lb_loss = lb_loss_total / lb_count
                            lb_coef = getattr(config, 'moe_load_balance_loss_coef', 0.05)
                            loss = loss + lb_coef * lb_loss
                            
                            if metrics is not None:
                                try:
                                    metrics['lb_loss'] = float(lb_loss.detach().cpu().item()) if torch.is_tensor(lb_loss) else lb_loss
                                    metrics['lb_coef'] = lb_coef
                                    metrics['moe_layers'] = lb_count
                                except Exception:
                                    pass
                
                # NaN detection
                if loss is None or torch.isnan(loss).any() if torch.is_tensor(loss) else False or torch.isinf(loss).any() if torch.is_tensor(loss) else False:
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
                    
                    if deepspeed_engine is None:
                        opt.zero_grad(set_to_none=True)
                    
                    continue

                # Distributed average
                log_metrics_extra: Dict[str, Any] = {}
                if ddp_actually_working:
                    try:
                        import torch.distributed as dist
                        if loss is not None and hasattr(loss, "detach"):
                            _dl = loss.detach()
                            dist.all_reduce(_dl, op=dist.ReduceOp.SUM)
                            world_safe = float(max(1, world_sz))
                            try:
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

                # Get gradient accumulation config
                gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1) if config is not None else 1
                accumulation_steps = max(1, int(gradient_accumulation_steps))
                
                # Scale loss for gradient accumulation
                # CRITICAL: This ensures gradients average instead of sum
                scaled_loss = loss / accumulation_steps

                # Track micro-batch loss for averaged signal.
                try:
                    micro_loss_buffer.append(float(loss.detach().cpu().item()))
                except Exception:
                    pass
                
                # Backward pass (gradients accumulate automatically)
                if deepspeed_engine is not None:
                    # DeepSpeed handles accumulation internally
                    deepspeed_engine.backward(scaled_loss)
                    
                    # Only step optimizer every N batches
                    if (steps_done + 1) % accumulation_steps == 0:
                        deepspeed_engine.step()

                        # Adaptive LR: observe once per optimizer step (after warmup).
                        if (
                            adaptive_lr_scheduler is not None
                            and base_lr > 0
                            and (warmup_steps <= 0 or steps_done >= warmup_steps)
                        ):
                            try:
                                obs_loss = (
                                    sum(micro_loss_buffer) / max(1, len(micro_loss_buffer))
                                    if micro_loss_buffer
                                    else float(loss.detach().cpu().item())
                                )
                                adaptive_lr_scheduler.observe(obs_loss)
                            except Exception:
                                pass
                            micro_loss_buffer.clear()
                        
                elif use_amp and scaler is not None and dev == "cuda":
                    # AMP with gradient accumulation
                    scaler.scale(scaled_loss).backward()
                    
                    # Only update weights every N batches
                    if (steps_done + 1) % accumulation_steps == 0:
                        scaler.unscale_(opt)
                        average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)

                        did_update = False
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            opt.zero_grad(set_to_none=True)
                            scaler.update()
                        else:
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad(set_to_none=True)
                            did_update = True

                        # Adaptive LR: observe once per optimizer step (after warmup).
                        if (
                            adaptive_lr_scheduler is not None
                            and base_lr > 0
                            and (warmup_steps <= 0 or steps_done >= warmup_steps)
                        ):
                            try:
                                if did_update:
                                    obs_loss = (
                                        sum(micro_loss_buffer) / max(1, len(micro_loss_buffer))
                                        if micro_loss_buffer
                                        else float(loss.detach().cpu().item())
                                    )
                                    adaptive_lr_scheduler.observe(obs_loss)
                            except Exception:
                                pass
                            micro_loss_buffer.clear()
                else:
                    # Standard mode with gradient accumulation
                    scaled_loss.backward()
                    
                    # Only update weights every N batches
                    if (steps_done + 1) % accumulation_steps == 0:
                        average_gradients_if_distributed(model_student, is_distributed=ddp_actually_working, world_sz=world_sz)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_student.parameters(), 0.5)

                        did_update = False
                        
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            opt.zero_grad(set_to_none=True)
                        else:
                            opt.step()
                            opt.zero_grad(set_to_none=True)
                            did_update = True

                        # Adaptive LR: observe once per optimizer step (after warmup).
                        if (
                            adaptive_lr_scheduler is not None
                            and base_lr > 0
                            and (warmup_steps <= 0 or steps_done >= warmup_steps)
                        ):
                            try:
                                if did_update:
                                    obs_loss = (
                                        sum(micro_loss_buffer) / max(1, len(micro_loss_buffer))
                                        if micro_loss_buffer
                                        else float(loss.detach().cpu().item())
                                    )
                                    adaptive_lr_scheduler.observe(obs_loss)
                            except Exception:
                                pass
                            micro_loss_buffer.clear()
                
                # Always increment step counter (counts batches processed)
                steps_done += 1
                
                # Hot-reload inference model
                if inference_manager is not None and hot_reload_steps > 0:
                    if steps_done % hot_reload_steps == 0:
                        try:
                            inference_manager.reload_from_checkpoint("actv1_student.safetensors")
                        except Exception as e:
                            print({"hot_reload": "failed", "step": steps_done, "error": str(e)})
                
                # Track dataset position
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
                
                if steps_done % 5 == 0 or steps_done == 1:
                    try:
                        import torch
                        if dev == "cuda" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            allocated_gb = torch.cuda.memory_allocated(device_obj) / 1024**3
                            reserved_gb = torch.cuda.memory_reserved(device_obj) / 1024**3
                            max_allocated_gb = torch.cuda.max_memory_allocated(device_obj) / 1024**3
                            total_gb = torch.cuda.get_device_properties(device_obj).total_memory / 1024**3
                            
                            # Get accumulation info
                            gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
                            accumulation_steps = max(1, int(gradient_accumulation_steps))
                            weight_updates = steps_done // accumulation_steps
                            
                            write_jsonl({
                                "step": steps_done,
                                "weight_updates": weight_updates,
                                "loss": float(loss.item()),  # Unscaled loss for logging
                                "lr": _get_optimizer_lr(),
                                "gradient_accumulation_steps": accumulation_steps,
                                "effective_batch_size": batch_size * accumulation_steps,
                                "physical_batch_size": batch_size,
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
                        pass
                
                # Log expert usage statistics
                if steps_done % 100 == 0:
                    try:
                        if hasattr(model_student, 'inner') and hasattr(model_student.inner, 'config'):
                            model_config = model_student.inner.config
                            if getattr(model_config, 'use_moe', False):
                                from aios.core.hrm_models.moe_layer import get_expert_usage_stats
                                
                                if hasattr(model_student.inner, 'H_level'):
                                    level_module = model_student.inner.H_level
                                    if len(level_module.layers) > 0:
                                        layer = level_module.layers[0]
                                        if hasattr(layer, 'mlp'):
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
                        pass
                
                # GPU utilization throttle
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
                # CPU utilization throttle
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
                    logger.warning(f"OOM detected on attempt {attempt + 1}/{max_attempts}, batch_size={batch_size}")
                    
                    import gc
                    gc.collect()
                    
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            logger.debug("Cleared CUDA cache after OOM")
                    except Exception:
                        pass
                    
                    try:
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                            total = torch.cuda.get_device_properties(0).total_memory / 1e9
                            logger.info(
                                f"OOM memory state: allocated={allocated:.2f}GB, "
                                f"reserved={reserved:.2f}GB, total={total:.2f}GB"
                            )
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
                    
                    if attempt >= max_attempts:
                        logger.error(
                            f"OOM recovery failed after {attempt} attempts, batch_size={batch_size}"
                        )
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
                        logger.info(f"OOM recovery: reduced batch_size to {batch_size} (attempt {attempt})")
                        try:
                            print({"event": "oom_backoff", "batch_size": int(batch_size), "attempt": int(attempt)})
                        except Exception:
                            pass
                        continue
                    else:
                        logger.error("OOM recovery failed: already at minimum batch_size=1")
                        print({
                            "event": "oom_at_min_batch",
                            "batch_size": 1,
                            "suggestion": "Reduce max_seq_len, enable gradient checkpointing, or use smaller model"
                        })
                        raise RuntimeError(
                            f"Out of memory even at batch_size=1. "
                            f"Try reducing max_seq_len or using gradient checkpointing."
                        ) from e
                raise

        if stopped_early:
            break

        # Logging after successful step
        # In distributed mode, only rank 0 should log to avoid duplicate output
        try:
            should_log = True
            
            if ddp_actually_working and current_rank != 0:
                should_log = False
            
            if should_log:
                cur_loss = None
                try:
                    if loss is not None:
                        cur_loss = float(loss.detach().cpu().item())
                except Exception:
                    cur_loss = None
                payload: Dict[str, Any] = {"event": "train", "step": int(step_offset + steps_done)}
                lr_now = _get_optimizer_lr()
                if lr_now is not None:
                    payload["lr"] = lr_now
                try:
                    payload["rank"] = current_rank
                    payload["ddp"] = ddp_actually_working
                except Exception:
                    pass
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
                if ddp_actually_working:
                    try:
                        import torch.distributed as dist
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
                
                # Print clean step progress like parallel training
                if cur_loss is not None:
                    rank_label = f"Rank {current_rank}" if ddp_actually_working else "GPU 0"
                    print(f"[{rank_label}] Step {steps_done}/{steps}: Loss={cur_loss:.4f}")
                
                write_jsonl(payload)
                
                # Ensure stdout is flushed for GUI progress monitoring
                try:
                    __sys.stdout.flush()
                except Exception:
                    pass
                
                # Write progress to shared file for parent process monitoring (DDP)
                if progress_file and current_rank == 0:
                    try:
                        import json
                        with open(progress_file, 'w') as pf:
                            json.dump({
                                "step": int(step_offset + steps_done),
                                "total_steps": int(steps),
                                "loss": float(cur_loss) if cur_loss is not None else None,
                                "timestamp": __time.time(),
                            }, pf)
                    except Exception:
                        pass
        except Exception:
            pass

        if eval_maybe is not None:
            try:
                eval_maybe()
            except Exception:
                pass
        
        # Check if graceful stop was requested - exit after current step completes
        if graceful_stop_pending:
            stopped_early = True
            stop_reason = "graceful_stop"
            print(f"[train_epoch] Graceful stop - exiting after completing step {display_step}")
            break
    
    # Check if graceful stop was requested - exit after chunk completion
    if graceful_stop_pending:
        stopped_early = True
        stop_reason = "graceful_stop"
        try:
            write_jsonl({"event": "graceful_stopped", "phase": "train", "step": int(steps_done), "stop_type": "graceful", "chunk_finished": True})
        except Exception:
            pass
        print({"GRACEFUL_STOP_HONORED": True, "chunk_steps_completed": steps_done})
    
    # NOTE: stop_after_epoch logic moved to run_iterate_mode() where sample
    # accumulation across cycles can be properly tracked for block/epoch completion
    
    # Clean up GRACEFUL_STOP file if it was triggered
    if stop_reason == "graceful_stop" and stop_file:
        try:
            from pathlib import Path as _Path
            graceful_stop_file = _Path(stop_file).parent / "GRACEFUL_STOP"
            if graceful_stop_file.exists():
                graceful_stop_file.unlink()
                print({"GRACEFUL_STOP_FILE_CLEANED": True})
        except Exception as e:
            print({"GRACEFUL_STOP_CLEANUP_ERROR": str(e)})

    return steps_done, stopped_early, batch_size, stop_reason
