"""Memory optimization and tracking utilities."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def detect_available_vram(log_fn) -> float:
    """Detect available VRAM for optimal chunk sizing.
    
    Supports both CUDA (NVIDIA/AMD ROCm) and Intel XPU devices.
    
    Args:
        log_fn: Logging function
        
    Returns:
        Available VRAM in GB
    """
    available_vram_gb = 20.0  # Conservative default
    backend_used = "default"
    
    try:
        # Check CUDA devices first (NVIDIA + AMD ROCm)
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Reserve 30% for overhead, optimizer states, and fragmentation
            available_vram_gb = total_vram_gb * 0.70
            backend_used = "cuda"
            
            # Check for ROCm build (AMD)
            rocm = bool(getattr(torch.version, "hip", None))
            
            log_fn({
                "vram_detection": {
                    "backend": "cuda" if not rocm else "rocm",
                    "total_gb": round(total_vram_gb, 2),
                    "available_for_model_gb": round(available_vram_gb, 2),
                    "note": "70% of total VRAM allocated for model, 30% for overhead"
                }
            })
        # Check Intel XPU devices
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                props = torch.xpu.get_device_properties(0)
                total_vram_gb = getattr(props, "total_memory", 0) / (1024 ** 3)
                
                # Try to get actual available memory
                try:
                    free_mem, _ = torch.xpu.mem_get_info(0)
                    available_vram_gb = (free_mem / (1024 ** 3)) * 0.85  # 85% of free
                except Exception:
                    available_vram_gb = total_vram_gb * 0.70
                
                backend_used = "xpu"
                log_fn({
                    "vram_detection": {
                        "backend": "xpu",
                        "total_gb": round(total_vram_gb, 2),
                        "available_for_model_gb": round(available_vram_gb, 2),
                        "note": "Intel XPU device detected"
                    }
                })
            except Exception as e:
                log_fn({
                    "vram_detection": "xpu_failed",
                    "using_default": 20.0,
                    "error": str(e)
                })
        else:
            log_fn({
                "vram_detection": "no_gpu",
                "using_default": available_vram_gb,
                "note": "No CUDA or XPU devices available"
            })
    except Exception as e:
        log_fn({
            "vram_detection": "failed",
            "using_default": 20.0,
            "error": str(e)
        })
    
    return available_vram_gb


def setup_memory_tracking(
    model: Any,
    config: "TrainingConfig",
    device_obj: torch.device,
    log_fn
) -> Any:
    """Initialize memory tracking.
    
    Args:
        model: The model
        config: Training configuration
        device_obj: PyTorch device
        log_fn: Logging function
        
    Returns:
        MemoryTracker instance
    """
    from aios.core.hrm_training.memory_utils import (
        MemoryTracker,
        estimate_model_memory,
        estimate_activation_memory,
    )
    
    memory_tracker = MemoryTracker(device=str(device_obj))
    
    # Snapshot: After model creation
    num_params = sum(p.numel() for p in model.parameters())
    memory_tracker.snapshot('model_created', metadata={
        'parameters': num_params,
        'h_layers': config.h_layers,
        'l_layers': config.l_layers,
        'hidden_size': config.hidden_size,
    })
    
    # Log theoretical memory requirements
    theoretical_memory = estimate_model_memory(
        num_parameters=num_params,
        precision='fp16' if config.use_amp else 'fp32',
        include_optimizer=True,
        optimizer_type='adamw8bit' if config.use_8bit_optimizer else 'adamw'
    )
    log_fn({"theoretical_memory_requirements": theoretical_memory})
    
    # Estimate activation memory
    activation_estimate = estimate_activation_memory(
        batch_size=config.batch_size,
        sequence_length=config.max_seq_len,
        hidden_size=config.hidden_size,
        num_layers=config.h_layers + config.l_layers,
        num_heads=config.num_heads,
        gradient_checkpointing=config.gradient_checkpointing,
        precision='fp16' if config.use_amp else 'fp32'
    )
    log_fn({"estimated_activation_memory": activation_estimate})
    
    return memory_tracker


def enable_extreme_scale_mode(
    max_seq_len: int,
    total_params: int,
    log_fn
) -> None:
    """Enable extreme-scale optimizations if needed.
    
    Args:
        max_seq_len: Maximum sequence length
        total_params: Total model parameters
        log_fn: Logging function
    """
    if max_seq_len < 100_000 and total_params < 500_000_000:
        return
    
    try:
        from aios.core.hrm_models.extreme_scale_optimizations import enable_extreme_memory_mode
        enable_extreme_memory_mode()
        log_fn({
            "optimization": "extreme_scale_mode_enabled",
            "context_length": max_seq_len,
            "model_params": total_params,
            "note": "Ultra-aggressive memory management for extreme scale"
        })
    except Exception as e:
        log_fn({
            "extreme_scale_mode": "unavailable",
            "error": str(e)
        })


def run_training_optimizer(
    config: "TrainingConfig",
    total_params: int,
    log_fn
) -> Tuple[int, int, Optional[str]]:
    """Run intelligent training optimization to find best settings.
    
    Args:
        config: Training configuration
        total_params: Total model parameters
        log_fn: Logging function
        
    Returns:
        Tuple of (optimized_max_seq_len, optimized_batch_size, optimized_zero_stage)
    """
    from aios.core.hrm_models.training_optimizer import optimize_training_config
    
    log_fn({
        "optimization": "auto_optimization_started",
        "note": "Finding optimal context length (4K-100K) and batch size for available VRAM..."
    })
    
    opt_config = optimize_training_config(
        model_params=total_params,
        hidden_size=config.hidden_size,
        num_layers=config.h_layers + config.l_layers,
        min_context=4000,
        max_context=100000,
    )
    
    # Override settings with optimized values
    max_seq_len = opt_config.context_length
    batch_size = opt_config.batch_size
    
    # Determine zero_stage
    zero_stage = config.zero_stage
    if opt_config.use_deepspeed and zero_stage == "none":
        if opt_config.deepspeed_stage == 1:
            zero_stage = "zero1"
        elif opt_config.deepspeed_stage == 2:
            zero_stage = "zero2"
        elif opt_config.deepspeed_stage == 3:
            zero_stage = "zero3"
    
    log_fn({
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
    
    return max_seq_len, batch_size, zero_stage


def configure_chunking(
    max_seq_len: int,
    chunk_size: int,
    use_chunked_training: bool,
    gradient_checkpointing: bool,
    use_cpu_offload: bool,
    log_fn
) -> Tuple[Any, bool, int]:
    """Configure auto-chunking for training.
    
    Args:
        max_seq_len: Maximum sequence length
        chunk_size: Chunk size
        use_chunked_training: Whether to use chunked training
        gradient_checkpointing: Whether to use gradient checkpointing
        use_cpu_offload: Whether to use CPU offload
        log_fn: Logging function
        
    Returns:
        Tuple of (segment_rollout function, use_chunking bool, final_chunk_size)
    """
    from aios.core.hrm_models.auto_chunking import auto_chunked_segment_rollout
    
    # Respect user's explicit chunk_size choice
    final_chunk_size = chunk_size
    log_fn({
        "chunk_size_source": "user_specified",
        "chunk_size": chunk_size,
        "note": "Respecting user's explicit choice (GUI dropdown or CLI --chunk-size flag)"
    })
    
    # Only enable chunking if use_chunked_training is True
    segment_rollout = auto_chunked_segment_rollout(
        max_seq_len=max_seq_len,
        chunk_threshold=0 if use_chunked_training else 999999,
        chunk_size=final_chunk_size,
        gradient_checkpointing=gradient_checkpointing,
        use_cpu_offload=use_cpu_offload,
    )
    
    use_chunking = use_chunked_training
    log_fn({
        "training_mode": "chunked" if use_chunking else "standard",
        "max_seq_len": max_seq_len,
        "chunk_size": final_chunk_size if use_chunking else "N/A",
        "chunked_training_forced": use_chunked_training,
        "gradient_checkpointing": gradient_checkpointing,
        "cpu_offload": use_cpu_offload if use_chunking else "N/A",
    })
    
    return segment_rollout, use_chunking, final_chunk_size


def log_optimization_summary(
    model_memory_gb: float,
    config: "TrainingConfig",
    world_sz: int,
    is_distributed: bool,
    log_fn
) -> dict:
    """Log comprehensive optimization summary.
    
    Args:
        model_memory_gb: Estimated model memory in GB
        config: Training configuration
        world_sz: World size for distributed training
        is_distributed: Whether distributed training is active
        log_fn: Logging function
        
    Returns:
        Optimization summary dictionary
    """
    from aios.core.hrm_training.memory_utils import log_optimization_summary as _log_opt
    
    opt_summary = _log_opt(
        model_memory_gb=model_memory_gb,
        use_8bit_optimizer=config.use_8bit_optimizer,
        gradient_checkpointing=config.gradient_checkpointing,
        use_amp=config.use_amp,
        use_chunked_training=config.use_chunked_training,
        chunk_size=config.chunk_size if config.use_chunked_training else None,
        zero_stage=config.zero_stage,
        num_gpus=world_sz if is_distributed else 1
    )
    
    log_fn({"optimization_summary": opt_summary})
    return opt_summary


def finalize_memory_report(
    memory_tracker: Any,
    steps_done: int,
    stopped_early: bool,
    batch_size: int,
    write_jsonl,
    log_fn
) -> None:
    """Generate and log final memory report.
    
    Args:
        memory_tracker: MemoryTracker instance
        steps_done: Number of steps completed
        stopped_early: Whether training stopped early
        batch_size: Final batch size
        write_jsonl: JSONL logging function
        log_fn: Logging function
    """
    try:
        memory_tracker.snapshot('training_complete', metadata={
            'steps_completed': steps_done,
            'stopped_early': stopped_early,
        })
        memory_report = memory_tracker.get_report()
        log_fn({"memory_profile_report": memory_report})
        write_jsonl({"event": "memory_profile", "report": memory_report})
        
        # Log current memory state
        final_memory = memory_tracker.log_current('final_state', {
            'total_steps': steps_done,
            'batch_size': batch_size,
        })
        log_fn({"final_memory_state": final_memory})
    except Exception as e:
        log_fn({"memory_report_error": str(e)})
