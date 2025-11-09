from __future__ import annotations

# CRITICAL: Set CUDA allocator config BEFORE any torch imports!
# This must be done before CUDA is initialized to take effect.
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'

from typing import TYPE_CHECKING, Optional

import typer
from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

from aios.data.datasets import read_text_lines_sample_any
from aios.core.hrm_models.hf_adapter import build_hf_adapter  # re-export safety if imported elsewhere
from ..hrm_hf_utils import load_tokenizer as _load_tokenizer_helper
from .training_helpers import (
    write_last_safe_batches as _write_last_safe_batches_helper,
    write_jsonl as _write_jsonl_helper,
    should_stop as _should_stop_helper,
    eval_once as _eval_once_helper,
)
from .ddp_utils import maybe_spawn_and_exit_if_parent as _maybe_spawn
from .helpers_ddpworker import ddp_worker as _ddp_worker, serialize_config_for_spawn
from .training_logic import train_epoch as _train_epoch_helper
from .helpers_device import (
    resolve_device as _resolve_device_helper,
    init_cuda_distributed_if_needed as _init_cuda_dist_helper,
    get_dist_context as _get_dist_context,
)
from .helpers_data import (
    get_training_lines as _get_training_lines_helper,
)
from .helpers_finalize import (
    finalize_training as _finalize_training_helper,
    broadcast_final_payload as _broadcast_final_payload_helper,
)
from .helpers_encode import (
    encode_lines as _encode_lines_helper,
    # Teacher loading removed - training from datasets only
    adjust_tokenizer_padding as _adjust_tok_padding,
    encode_eval_lines as _encode_eval_lines,
)
from .helpers_model import build_student as _build_student, build_actv1_config as _build_cfg


def train_actv1_impl(config: "TrainingConfig") -> None:
    """Train the ACT V1 HRM model.
    
    Args:
        config: Training configuration object containing all parameters
    """
    """Train the ACT V1 HRM model.
    
    Args:
        config: Training configuration object containing all parameters
    """
    
    # Extract configuration values for use throughout
    model = config.model
    dataset_file = config.dataset_file
    max_seq_len = config.max_seq_len
    batch_size = config.batch_size
    steps = config.steps
    lr = config.lr
    device = config.device
    halt_max_steps = config.halt_max_steps
    save_dir = config.save_dir
    teacher = config.teacher
    teacher_device = config.teacher_device
    kl = config.kl
    kl_temp = config.kl_temp
    ascii_only = config.ascii_only
    eval_file = config.eval_file
    eval_batches = config.eval_batches
    sys_mem_cap_pct = config.sys_mem_cap_pct
    stop_file = config.stop_file
    log_file = config.log_file
    student_init = config.student_init
    brain_name = config.brain_name
    bundle_dir = config.bundle_dir
    h_layers = config.h_layers
    l_layers = config.l_layers
    hidden_size = config.hidden_size
    expansion = config.expansion
    num_heads = config.num_heads
    h_cycles = config.h_cycles
    l_cycles = config.l_cycles
    pos_encodings = config.pos_encodings
    cuda_ids = config.cuda_ids
    iterate = config.iterate
    optimize = config.optimize
    gradient_checkpointing = config.gradient_checkpointing
    use_amp = config.use_amp
    use_cpu_offload = config.use_cpu_offload
    zero_stage = config.zero_stage
    ddp = config.ddp
    world_size = config.world_size
    strict = config.strict
    
    # Import heavy deps lazily inside implementation
    try:
        import os
        from pathlib import Path
        import torch
        
        # Suppress deprecation warnings BEFORE importing transformers
        import warnings as _warnings
        _warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)
        _warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Suppress noisy deprecation warnings from Transformers
        try:
            from transformers.utils import logging as _hf_logging  # type: ignore
            _hf_logging.set_verbosity_error()
        except Exception:
            pass
        from aios.core.hrm_models import build_act_v1
        from aios.core.hrm_models.auto_chunking import auto_chunked_segment_rollout, get_recommended_chunk_size
    except Exception as e:
        print({"started": False, "error": f"Missing deps: {e}", "hint": "pip install -e .[hf]"})
        raise typer.Exit(code=1)

    # Normalize strict to bool for safety
    strict = bool(strict)


    # The following is a near verbatim move of the body from hrm_hf_cli.train_actv1
    # to preserve behavior. Only minimal adjustments for imports/structure were made.

    # Respect explicit CUDA device list if provided
    if cuda_ids:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(int(x)) for x in str(cuda_ids).split(",") if str(x).strip() != ""])  # noqa: E501
        except Exception:
            pass

    # Optional: Launch multi-process DDP; on Windows fallback uses gloo; return early in parent after spawn
    should_exit = _maybe_spawn(
        ddp=ddp,
        device=device,
        torch=torch,
        os=os,
        platform=__import__('platform'),
        world_size=world_size,
        strict=strict,
        worker_target=_ddp_worker,
        spawn_kwargs=serialize_config_for_spawn(config),
    )
    if should_exit:
        return

    # Resolve device (with validation; in strict mode disallow silent fallbacks)
    dev, device_obj, dml_device = _resolve_device_helper(device, strict, torch)

    # Resolve output/bundle directory based on brain_name
    out_dir_path: Optional[Path] = None
    if brain_name:
        try:
            out_dir_path = Path(bundle_dir) / str(brain_name)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            # Default log file to bundle if not explicitly provided
            if not log_file:
                log_file = str(out_dir_path / "metrics.jsonl")
            # Always prefer saving artifacts into the bundle dir
            save_dir = str(out_dir_path)
        except Exception:
            out_dir_path = None

    # Keep the original user-provided training batch size as a cap (not auto-tuned here)

    # Auto-load brain metadata if resuming training from existing brain
    brain_metadata = None
    if student_init:
        try:
            import json
            student_path = Path(student_init)
            # Check if student_init points to a brain bundle directory or the .pt file itself
            if student_path.is_dir():
                brain_json_path = student_path / "brain.json"
            else:
                # Try parent directory (common pattern: artifacts/brains/actv1/BrainName/actv1_student.pt)
                brain_json_path = student_path.parent / "brain.json"
            
            if brain_json_path.exists():
                with open(brain_json_path, 'r') as f:
                    brain_metadata = json.load(f)
                    # If model wasn't explicitly provided or is default, use brain's tokenizer
                    if model == "base_model" and "tokenizer_model" in brain_metadata:
                        model = brain_metadata["tokenizer_model"]
                        print({"brain_loaded": brain_metadata.get("name"), "tokenizer_from_brain": model})
        except Exception as e:
            # Non-fatal: just continue with user-provided model parameter
            print({"brain_metadata_load": "failed", "error": str(e), "hint": "Continuing with --model parameter"})

    # Tokenizer from model path
    tok = _load_tokenizer_helper(model)
    _adjust_tok_padding(tok)

    # Distributed context detection (intra-process spawn or torchrun)
    is_distributed, rank_id, world_sz, init_file_env = _get_dist_context(os)

    # Early JSONL logger (rank0 only when distributed)
    def _write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(log_file=log_file, payload=payload, is_distributed=is_distributed, rank_id=rank_id)

    # Shared STOP checker
    def _should_stop() -> bool:
        return _should_stop_helper(stop_file)

    lines: list[str] = []

    # Drop util target maps and GPU utilization-based tuning

    def _load_or_generate_lines() -> list[str]:
        nonlocal log_file
        return _get_training_lines_helper(
            dataset_file=dataset_file,
            ascii_only=ascii_only,
            read_text_lines_sample_any=read_text_lines_sample_any,
        )

    stopped_early = False
    # Use the unified helper to load/generate lines to avoid duplication and indentation issues
    # In distributed mode, avoid generating teacher lines on every rank to reduce memory usage and OOMs.
    # Non-distributed or helper-managed DDP sync
    lines = _load_or_generate_lines()
    if not lines:
        print({"started": False, "error": "no lines"})
        raise typer.Exit(code=1)

    def _train_with_lines(lines_in: list[str]) -> None:
        nonlocal input_ids, labels
        input_ids, labels = _encode_lines_helper(tok, lines_in, max_seq_len)

    # Estimate dataset memory usage and decide on loading strategy
    num_lines = len(lines)
    estimated_dataset_gb = (num_lines * max_seq_len * 4 * 2) / (1024 ** 3)  # input_ids + labels, int32
    
    # Force streaming for extreme context lengths to avoid keeping full sequences in GPU memory
    # Even if dataset is small, long sequences cause OOM when sliced (views keep full tensor)
    force_streaming_long_context = max_seq_len >= 8192
    use_streaming = estimated_dataset_gb > 4.0 or force_streaming_long_context
    
    if use_streaming:
        # Use streaming mode to avoid loading entire dataset into memory
        try:
            reason = f"Dataset would use {estimated_dataset_gb:.2f} GB (>{4.0} GB threshold)" if estimated_dataset_gb > 4.0 else f"Long context ({max_seq_len} tokens >= 8192)"
            print({
                "dataset_loading": "streaming",
                "reason": reason,
                "num_samples": num_lines,
                "max_seq_len": max_seq_len,
            })
        except Exception:
            pass
        
        # Don't tokenize all lines at once - will be done lazily during training
        input_ids = None
        labels = None
    else:
        # Small dataset - use eager loading (current behavior)
        try:
            print({
                "dataset_loading": "eager (all in memory)",
                "estimated_gb": f"{estimated_dataset_gb:.2f}",
                "num_samples": num_lines,
            })
        except Exception:
            pass
        input_ids, labels = _encode_lines_helper(tok, lines, max_seq_len)

    vocab_size = int(getattr(tok, "vocab_size", 50257) or 50257)
    cfg = _build_cfg(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        h_layers=h_layers,
        l_layers=l_layers,
        hidden_size=hidden_size,
        expansion=expansion,
        num_heads=num_heads,
        pos_encodings=pos_encodings,
        halt_max_steps=halt_max_steps,
    )
    model_student = _build_student(cfg, student_init=student_init, print_fn=print)
    
    # Count model parameters for memory-aware chunk sizing
    total_params = sum(p.numel() for p in model_student.parameters())
    
    # INTELLIGENT OPTIMIZATION: Auto-find best settings if requested
    if optimize:
        from aios.core.hrm_models.training_optimizer import optimize_training_config
        
        print({
            "optimization": "auto_optimization_started",
            "note": "Finding optimal context length (4K-100K) and batch size for available VRAM..."
        })
        
        opt_config = optimize_training_config(
            model_params=total_params,
            hidden_size=hidden_size,
            num_layers=h_layers + l_layers,
            min_context=4000,
            max_context=100000,
        )
        
        # Override settings with optimized values
        max_seq_len = opt_config.context_length
        batch_size = opt_config.batch_size
        # Note: gradient_checkpointing and use_amp already enabled by default
        
        # Update zero_stage if DeepSpeed recommended
        if opt_config.use_deepspeed and zero_stage == "none":
            if opt_config.deepspeed_stage == 1:
                zero_stage = "zero1"
            elif opt_config.deepspeed_stage == 2:
                zero_stage = "zero2"
            elif opt_config.deepspeed_stage == 3:
                zero_stage = "zero3"
        
        print({
            "optimization": "complete",
            "optimized_context": max_seq_len,
            "optimized_batch": batch_size,
            "chunk_size": opt_config.chunk_size,
            "deepspeed_stage": zero_stage,
            "estimated_vram_gb": round(opt_config.estimated_vram_gb, 2),
            "available_vram_gb": round(opt_config.available_vram_gb, 2),
            "optimization_score": round(opt_config.optimization_score, 1),
            "warnings": opt_config.warnings,
            "recommendations": opt_config.recommendations,
        })
    
    # EXTREME-SCALE OPTIMIZATION: Enable aggressive memory management for large contexts/models
    if max_seq_len >= 100_000 or total_params >= 500_000_000:
        from aios.core.hrm_models.extreme_scale_optimizations import enable_extreme_memory_mode
        enable_extreme_memory_mode()
        print({
            "optimization": "extreme_scale_mode_enabled",
            "context_length": max_seq_len,
            "model_params": total_params,
            "note": "Ultra-aggressive memory management for extreme scale"
        })
    
    try:
        print({
            "arch": {
                "H_layers": int(cfg["H_layers"]), "L_layers": int(cfg["L_layers"]), "hidden_size": int(cfg["hidden_size"]),
                "num_heads": int(cfg["num_heads"]), "expansion": float(cfg["expansion"]),
            }
        })
    except Exception:
        pass
    
    # Create auto-chunking segment_rollout wrapper
    # This automatically uses chunking for long sequences (>8K tokens)
    # Pass model_params for memory-aware chunk size selection
    
    # Detect available VRAM for optimal chunk sizing
    available_vram_gb = 20.0  # Conservative default
    try:
        if torch.cuda.is_available():
            # Get total VRAM and subtract 2GB for overhead and optimizer
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Reserve 20% for overhead, optimizer states, and fragmentation
            available_vram_gb = total_vram_gb * 0.70  # Use 70% of total
            print({
                "vram_detection": {
                    "total_gb": round(total_vram_gb, 2),
                    "available_for_model_gb": round(available_vram_gb, 2),
                    "note": "70% of total VRAM allocated for model, 30% for overhead"
                }
            })
    except Exception as e:
        print({"vram_detection": "failed", "using_default": 20.0, "error": str(e)})
    
    chunk_size_recommended = get_recommended_chunk_size(
        max_seq_len, 
        available_vram_gb=available_vram_gb,
        model_params=total_params
    )
    
    # Print extreme-scale recommendations if needed
    if max_seq_len >= 50_000 or total_params >= 200_000_000:
        try:
            from aios.core.hrm_models.extreme_scale_optimizations import print_extreme_scale_recommendations
            print_extreme_scale_recommendations(
                model_params=total_params,
                seq_len=max_seq_len,
                available_vram_gb=available_vram_gb,
                h_layers=h_layers,
                l_layers=l_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                expansion=expansion,
                vocab_size=vocab_size
            )
        except Exception as e:
            print({"extreme_scale_recommendations": "failed", "error": str(e)})
    
    segment_rollout = auto_chunked_segment_rollout(
        max_seq_len=max_seq_len,
        chunk_threshold=8192,  # Use chunking for sequences > 8K tokens
        chunk_size=chunk_size_recommended,
        gradient_checkpointing=gradient_checkpointing,
    )
    use_chunking = max_seq_len > 8192
    try:
        print({
            "training_mode": "chunked" if use_chunking else "standard",
            "max_seq_len": max_seq_len,
            "chunk_size": chunk_size_recommended if use_chunking else "N/A",
            "gradient_checkpointing": gradient_checkpointing,
        })
    except Exception:
        pass

    # Device and (optional) distributed initialization and placement
    dml_device = None
    if dev == "dml":
        try:
            import torch_directml as _dml  # type: ignore
            dml_device = _dml.device()
        except Exception:
            dev = "cpu"
    device_obj = (dml_device if dml_device is not None else torch.device(dev))
    # Apply GPU memory fraction cap if provided
    try:
        if str(device_obj) == "cuda" and torch.cuda.is_available():
            frac_env = __import__('os').environ.get("AIOS_GPU_MEM_FRACTION")
            if frac_env:
                try:
                    frac = float(frac_env)
                    # Best-effort; applies to current process on this device
                    torch.cuda.set_per_process_memory_fraction(float(max(0.05, min(0.99, frac))), device=device_obj)
                except Exception:
                    pass
    except Exception:
        pass
    device_obj, ddp_initialized = _init_cuda_dist_helper(
        dev=str(device_obj),
        is_distributed=is_distributed,
        torch=torch,
        os=os,
        rank_id=rank_id,
        world_sz=world_sz,
        init_file_env=init_file_env,
    )
    model_student.to(device_obj)
    
    # Wrap model in DDP if distributed training is initialized
    if ddp_initialized and is_distributed:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # Wrap with DDP to enable multi-GPU training
            model_student = DDP(
                model_student,
                device_ids=[rank_id],
                output_device=rank_id,
                find_unused_parameters=False  # Set to True if you have unused parameters
            )
            print({
                "ddp_model_wrapped": True,
                "rank": rank_id,
                "world_size": world_sz,
                "device": str(device_obj)
            })
        except Exception as e:
            print({
                "ddp_model_wrapped": False,
                "error": str(e),
                "rank": rank_id
            })
    
    try:
        if (dml_device is None) and (str(device_obj) == "cuda") and torch.cuda.is_available() and (not is_distributed) and torch.cuda.device_count() > 1:
            try:
                print({"multi_gpu": False, "note": "Multiple CUDA devices detected; enable --ddp with --cuda-ids to use multi-GPU."})
            except Exception:
                pass
    except Exception:
        pass

    # Optional eval set
    eval_ids, eval_labels = _encode_eval_lines(
        tok=tok,
        eval_file=eval_file,
        ascii_only=ascii_only,
        max_seq_len=max_seq_len,
        read_text_lines_sample_any=read_text_lines_sample_any,
    )
    model_student.train(True)

    params = [p for p in model_student.parameters() if p.requires_grad]
    OptClass = getattr(torch.optim, "AdamW", None) or getattr(torch.optim, "Adam")
    opt = OptClass(params, lr=float(lr))
    
    # Initialize AMP GradScaler for mixed precision training on CUDA
    scaler = None
    use_amp = False
    if dev == "cuda" and torch.cuda.is_available():
        try:
            # Enable mixed precision training (saves ~40-50% memory)
            use_amp = True
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                scaler = torch.cuda.amp.GradScaler()
            print({"amp_enabled": True, "note": "Mixed precision training enabled for memory efficiency"})
        except Exception as e:
            print({"amp_enabled": False, "error": str(e), "note": "Falling back to FP32"})
            use_amp = False
    
    try:
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    # REMOVED: Teacher model loading - training from datasets only (no KL distillation)
    # This saves 3-4GB VRAM by not loading teacher models
    print({
        "training_mode": "dataset_only",
        "note": "Training student model from scratch using dataset - no teacher model",
        "vram_saved": "~3-4 GB (teacher not loaded)"
    })

    # Handle streaming vs eager dataset loading
    if use_streaming:
        # Create streaming dataset
        from .streaming_dataset import create_streaming_dataset
        streaming_dataset = create_streaming_dataset(
            lines=lines,
            tokenizer=tok,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            shuffle=True,
        )
        N = len(lines)  # Number of samples (for logging)
    else:
        if input_ids is not None:
            N = input_ids.shape[0]
        else:
            N = len(lines)
        streaming_dataset = None
    
    last = 0.0
    import time as _time

    # Evaluation now runs only at the end of training
    # No upfront auto-tuning; use provided batch_size as-is

    def _eval_once() -> None:
        _eval_once_helper(
            model_student=model_student,
            eval_ids=eval_ids,
            eval_labels=eval_labels,
            batch_size=batch_size,
            device_obj=device_obj,
            dml_device=dml_device,
            halt_max_steps=halt_max_steps,
            eval_batches=eval_batches,
            segment_rollout=segment_rollout,
            write_jsonl=_write_jsonl,
            tokenizer=tok,
            enable_english_logic_eval=True,
        )

    # No runtime auto-tune on CUDA

    # Drop utilization-based CPU/GPU auto-tune and CPU mem-cap adjustment

    steps_done = 0

    # Parse optional GPU/CPU utilization targets and throttle settings (0 or missing disables)
    try:
        _gpu_util_target = __import__('os').environ.get("AIOS_GPU_UTIL_TARGET")
        gpu_util_target = int(_gpu_util_target) if _gpu_util_target is not None else 0
    except Exception:
        gpu_util_target = 0
    try:
        _cpu_util_target = __import__('os').environ.get("AIOS_CPU_UTIL_TARGET")
        cpu_util_target = int(_cpu_util_target) if _cpu_util_target is not None else 0
    except Exception:
        cpu_util_target = 0
    try:
        import os as __os
        gpu_util_mode = str(__os.environ.get("AIOS_GPU_UTIL_MODE") or "duty").lower()
        gpu_util_poll_ms = int(__os.environ.get("AIOS_GPU_UTIL_POLL_MS") or "50")
    except Exception:
        gpu_util_mode = "duty"
        gpu_util_poll_ms = 50

    def _do_train_epoch():
        nonlocal steps_done, batch_size, stopped_early
        # Periodic evaluation removed - only runs at end now
        def _maybe_eval():
            pass  # No-op during training
        s_done, early, new_bs = _train_epoch_helper(
            model_student=model_student,
            segment_rollout=segment_rollout,
            opt=opt,
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
            write_jsonl=_write_jsonl,
            should_stop=_should_stop,
            write_last_safe_batches=_write_last_safe_batches_helper,
            eval_maybe=_maybe_eval,
            gpu_util_target=gpu_util_target,
            cpu_util_target=cpu_util_target,
            gpu_util_mode=gpu_util_mode,
            gpu_util_poll_ms=gpu_util_poll_ms,
            streaming_dataset=streaming_dataset,  # NEW: Add streaming support
            tokenizer=tok,  # NEW: Pass tokenizer for streaming
            lines=lines if use_streaming else None,  # NEW: Pass lines for streaming
            use_amp=use_amp,  # NEW: Enable AMP if available
            scaler=scaler,  # NEW: Pass GradScaler for AMP
        )
        steps_done += s_done
        stopped_early = early
        batch_size = new_bs

    if not iterate:
        for _ in [0]:
            _do_train_epoch()
    else:
        cycle = 0
        while True:
            if stop_file and isinstance(stop_file, str):
                try:
                    from pathlib import Path as _Path
                    if _Path(stop_file).exists():
                        print({"stopped": True, "phase": "iterate", "cycle": int(cycle)})
                        stopped_early = True
                        break
                except Exception:
                    pass
            new_lines = _load_or_generate_lines()
            if not new_lines:
                print({"started": False, "error": "no lines (iterate)"})
                break
            _train_with_lines(new_lines)
            try:
                _write_jsonl({"event": "iterate_cycle", "cycle": int(cycle)})
            except Exception:
                pass
            _do_train_epoch()
            cycle += 1

    # Run final evaluation after training completes
    if eval_file:
        try:
            _write_jsonl({"event": "final_evaluation_start"})
            _eval_once()
            _write_jsonl({"event": "final_evaluation_complete"})
        except Exception as e:
            _write_jsonl({"event": "final_evaluation_error", "error": str(e)})

    final_payload = _finalize_training_helper(
        model_student=model_student,
        save_dir=save_dir,
        stopped_early=stopped_early,
        steps_done=steps_done,
        is_distributed=is_distributed,
        rank_id=rank_id,
        tok=tok,
        h_layers=h_layers,
        l_layers=l_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        expansion=expansion,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        pos_encodings=pos_encodings,
        log_file=log_file,
        write_jsonl=_write_jsonl,
        brain_name=brain_name,
        teacher=teacher,
        model=model,
        max_seq_len=max_seq_len,
        halt_max_steps=halt_max_steps,
    )

    final_payload = _broadcast_final_payload_helper(
        final_payload=final_payload,
        is_distributed=is_distributed,
        rank_id=rank_id,
        torch=torch,
    )
    if (not is_distributed) or (rank_id == 0):
        print(final_payload)
        _write_jsonl({"event": "final", **final_payload})
    try:
        _write_last_safe_batches_helper(train_bs=int(batch_size))
    except Exception:
        pass
    # Clean up process group if initialized
    try:
        if is_distributed and (str(device_obj) == "cuda"):
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
    except Exception:
        pass
