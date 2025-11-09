"""
Model Builder Module for ACT-V1 Training

Handles the complete model and data setup pipeline:
- Device configuration and distributed setup
- Tokenizer loading and configuration  
- Dataset preparation (streaming or eager loading)
- Model architecture instantiation
- Brain metadata loading for resume training

This module orchestrates all the "building blocks" before optimization begins.
"""

import os
from pathlib import Path
from typing import Any, Optional
import warnings

import torch

# Suppress deprecation warnings before importing transformers
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

from transformers import AutoTokenizer

# Import HRM models and helpers
from aios.core.hrm_models import build_act_v1
from aios.core.hrm_models.auto_chunking import auto_chunked_segment_rollout

# Import helper functions from parent module
from ..helpers import (
    _resolve_device_helper,
    _maybe_spawn,
    _load_tokenizer_helper,
    _adjust_tok_padding,
    _get_dist_context,
    _write_jsonl_helper,
    _should_stop_helper,
    _get_training_lines_helper,
    _encode_lines_helper,
    _build_cfg,
    _build_student,
    _encode_eval_lines,
)

from .checkpoint_utils import _recover_checkpoint_artifacts, _resolve_student_init_path
from .config_processing import serialize_config_for_spawn
from ..ddp import _ddp_worker


def setup_model_and_data(
    config: Any,  # TrainingConfig
    device: str,
    strict: bool,
    bundle_dir: str,
    brain_name: Optional[str],
    model: str,
    student_init: Optional[str],
    save_dir: str,
    log_file: Optional[str],
    dataset_file: str,
    ascii_only: bool,
    read_text_lines_sample_any: Any,
    max_seq_len: int,
    batch_size: int,
    steps: int,
    eval_file: Optional[str],
    stop_file: Optional[str],
) -> dict[str, Any]:
    """
    Setup model, tokenizer, datasets, and device configuration.
    
    This is the first major phase of training setup. It:
    1. Configures CUDA devices and handles distributed training setup
    2. Loads and configures the tokenizer
    3. Prepares training and evaluation datasets
    4. Instantiates the model architecture
    5. Loads brain metadata if resuming training
    
    Args:
        config: Training configuration object with all hyperparameters
        device: Target device (cuda/cpu/dml)
        strict: Whether to enforce strict device requirements
        bundle_dir: Directory for brain bundles
        brain_name: Name of the brain being trained
        model: Model/tokenizer identifier
        student_init: Path to checkpoint for initialization
        save_dir: Directory for saving checkpoints
        log_file: Path to JSONL log file
        dataset_file: Path to training data
        ascii_only: Whether to filter non-ASCII characters
        read_text_lines_sample_any: Function for reading training lines
        max_seq_len: Maximum sequence length
        batch_size: Training batch size
        steps: Number of training steps
        eval_file: Path to evaluation data (optional)
        stop_file: Path to stop file for early termination
        
    Returns:
        Dictionary containing all setup components:
        - model_student: The instantiated model
        - tokenizer: Configured tokenizer
        - device_obj: PyTorch device object
        - input_ids, labels: Training data tensors (or None for streaming)
        - eval_ids, eval_labels: Evaluation data tensors
        - streaming_dataset: Dataset object for streaming mode
        - cfg: Model configuration dictionary
        - And many other configuration values needed downstream
    """
    
    # ============================================================================
    # Import Heavy Dependencies
    # ============================================================================
    # Import here to keep startup fast if not using this module
    try:
        import json
        # Suppress transformers logging
        try:
            from transformers.utils import logging as _hf_logging  # type: ignore
            _hf_logging.set_verbosity_error()
        except Exception:
            pass
    except Exception as e:
        print({"started": False, "error": f"Missing deps: {e}", "hint": "pip install -e .[hf]"})
        import typer
        raise typer.Exit(code=1)
    
    # Extract config values
    h_layers = config.h_layers
    l_layers = config.l_layers
    hidden_size = config.hidden_size
    expansion = config.expansion
    num_heads = config.num_heads
    h_cycles = config.h_cycles
    l_cycles = config.l_cycles
    pos_encodings = config.pos_encodings
    window_size = config.window_size
    cuda_ids = config.cuda_ids
    iterate = config.iterate
    gradient_checkpointing = config.gradient_checkpointing
    use_cpu_offload = config.use_cpu_offload
    use_chunked_training = config.use_chunked_training
    chunk_size = config.chunk_size
    zero_stage = config.zero_stage
    ddp = config.ddp
    world_size = config.world_size
    halt_max_steps = config.halt_max_steps
    
    # Normalize strict to bool
    strict = bool(strict)
    
    # ============================================================================
    # CUDA Device Configuration
    # ============================================================================
    # Respect explicit CUDA device list if provided
    if cuda_ids:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([
                str(int(x)) for x in str(cuda_ids).split(",") if str(x).strip() != ""
            ])
        except Exception:
            pass
    
    # ============================================================================
    # Multi-Process DDP Spawn (if applicable)
    # ============================================================================
    # Launch multi-process DDP; return early in parent after spawn
    import platform
    should_exit = _maybe_spawn(
        ddp=ddp,
        device=device,
        torch=torch,
        os=os,
        platform=platform,
        world_size=world_size,
        strict=strict,
        worker_target=_ddp_worker,
        spawn_kwargs=serialize_config_for_spawn(config),
    )
    if should_exit:
        return {"should_exit": True}
    
    # ============================================================================
    # Device Resolution
    # ============================================================================
    # Resolve device with validation (strict mode disallows silent fallbacks)
    dev, device_obj, dml_device = _resolve_device_helper(device, strict, torch)
    
    # ============================================================================
    # Windows ZeRO Multi-GPU Handling
    # ============================================================================
    # Windows-specific: Auto-convert multi-GPU to single GPU when ZeRO is enabled
    # (ZeRO requires proper DDP which has issues on Windows with gloo backend)
    if os.name == "nt" and zero_stage and zero_stage != "none":
        num_gpus = 0
        if cuda_ids:
            try:
                num_gpus = len([x for x in str(cuda_ids).split(",") if x.strip()])
            except Exception:
                num_gpus = 0
        
        if num_gpus > 1 or (ddp and world_size is not None and world_size > 1):
            # Auto-convert to single GPU
            print({
                "windows_zero_multi_gpu_detected": True,
                "action": "auto_convert_to_single_gpu",
                "reason": "Windows does not support ZeRO with multi-GPU (gloo backend limitations)",
                "previous_config": {"cuda_ids": cuda_ids, "world_size": world_size, "ddp": ddp},
                "new_config": "single GPU (first device)"
            })
            # Keep only first GPU
            if cuda_ids:
                first_gpu = str(cuda_ids).split(",")[0].strip()
                cuda_ids = first_gpu
                os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
            # Disable DDP
            ddp = False
            world_size = 1
            print({"note": "To use multi-GPU on Windows, disable ZeRO and use standard DDP"})
    
    # ============================================================================
    # Output Directory Setup
    # ============================================================================
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
    
    # ============================================================================
    # Checkpoint Recovery
    # ============================================================================
    # Attempt to recover any partial checkpoints before loading
    candidate_dirs: list[Path] = []
    if out_dir_path is not None:
        candidate_dirs.append(out_dir_path)
    try:
        candidate_dirs.append(Path(save_dir))
    except Exception:
        pass
    if student_init:
        try:
            init_candidate = Path(student_init)
            candidate_dirs.append(
                init_candidate if init_candidate.is_dir() else init_candidate.parent
            )
        except Exception:
            pass
    
    recovered_checkpoint = _recover_checkpoint_artifacts(candidate_dirs, print)
    student_init = _resolve_student_init_path(student_init, recovered_checkpoint, candidate_dirs, print)
    config.student_init = student_init
    
    # ============================================================================
    # Brain Metadata Loading
    # ============================================================================
    # Auto-load brain metadata if resuming training from existing brain
    brain_metadata = None
    if student_init:
        try:
            student_path = Path(student_init)
            # Check if student_init points to a brain bundle directory or the safetensors file itself
            if student_path.is_dir():
                brain_json_path = student_path / "brain.json"
            else:
                # Try parent directory
                brain_json_path = student_path.parent / "brain.json"
            
            if brain_json_path.exists():
                with open(brain_json_path, 'r') as f:
                    brain_metadata = json.load(f)
                    # If model wasn't explicitly provided or is default, use brain's tokenizer
                    if model == "base_model" and "tokenizer_model" in brain_metadata:
                        model = brain_metadata["tokenizer_model"]
                        print({"brain_loaded": brain_metadata.get("name"), "tokenizer_from_brain": model})
                    
                    # Load MoE configuration from brain metadata
                    if "use_moe" in brain_metadata:
                        brain_use_moe = brain_metadata.get("use_moe", False)
                        brain_num_experts = brain_metadata.get("num_experts", 8)
                        brain_num_experts_per_tok = brain_metadata.get("num_experts_per_tok", 2)
                        
                        # Override config values with brain metadata
                        config.use_moe = brain_use_moe
                        config.num_experts = brain_num_experts
                        config.num_experts_per_tok = brain_num_experts_per_tok
                        
                        print({
                            "moe_config_loaded": True,
                            "use_moe": brain_use_moe,
                            "num_experts": brain_num_experts,
                            "num_experts_per_tok": brain_num_experts_per_tok,
                            "source": "brain.json"
                        })
        except Exception as e:
            # Non-fatal: just continue with user-provided model parameter
            print({
                "brain_metadata_load": "failed",
                "error": str(e),
                "hint": "Continuing with --model parameter"
            })
    
    # ============================================================================
    # Tokenizer Loading
    # ============================================================================
    tok = _load_tokenizer_helper(model)
    _adjust_tok_padding(tok)
    
    # ============================================================================
    # Distributed Context Detection
    # ============================================================================
    # Detect distributed context (intra-process spawn or torchrun)
    is_distributed, rank_id, world_sz, init_file_env = _get_dist_context(os)
    
    # ============================================================================
    # JSONL Logger Setup
    # ============================================================================
    # Early JSONL logger (rank0 only when distributed)
    def _write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(
            log_file=log_file,
            payload=payload,
            is_distributed=is_distributed,
            rank_id=rank_id
        )
    
    # ============================================================================
    # Stop File Checker
    # ============================================================================
    def _should_stop() -> bool:
        return _should_stop_helper(stop_file)
    
    # ============================================================================
    # Dataset Loading
    # ============================================================================
    def _load_or_generate_lines(cycle: int = 0) -> list[str]:
        return _get_training_lines_helper(
            dataset_file=dataset_file,
            ascii_only=ascii_only,
            read_text_lines_sample_any=read_text_lines_sample_any,
            cycle=cycle,
        )
    
    # Load training lines
    lines = _load_or_generate_lines()
    if not lines:
        print({"started": False, "error": "no lines"})
        import typer
        raise typer.Exit(code=1)
    
    # Track cycle count for epoch-based shuffling
    cycle_count = 0
    
    # ============================================================================
    # Training Parameters Calculation
    # ============================================================================
    # Calculate warmup steps: 10% of total steps or 200, whichever is smaller
    warmup_steps = min(200, max(10, steps // 10))
    
    # Validate dataset coverage
    num_lines = len(lines)
    expected_samples_per_epoch = steps * batch_size
    coverage_pct = (expected_samples_per_epoch / num_lines) * 100 if num_lines > 0 else 0
    
    if coverage_pct < 50 and num_lines > 100:
        print({
            "warning": "low_dataset_coverage",
            "dataset_samples": num_lines,
            "steps": steps,
            "batch_size": batch_size,
            "samples_per_epoch": expected_samples_per_epoch,
            "coverage_percent": round(coverage_pct, 1),
            "recommendation": f"Increase --steps to {num_lines // batch_size} for full dataset coverage",
            "impact": "Low coverage means the model sees only a small fraction of your data each epoch",
        })
    
    # ============================================================================
    # Streaming vs Eager Dataset Decision
    # ============================================================================
    # Estimate dataset memory usage and decide on loading strategy
    estimated_dataset_gb = (num_lines * max_seq_len * 4 * 2) / (1024 ** 3)  # input_ids + labels, int32
    
    # Force streaming for extreme context lengths
    force_streaming_long_context = max_seq_len >= 8192
    use_streaming = estimated_dataset_gb > 4.0 or force_streaming_long_context
    
    streaming_dataset = None
    input_ids = None
    labels = None
    
    if use_streaming:
        reason = (f"Dataset would use {estimated_dataset_gb:.2f} GB (>{4.0} GB threshold)"
                  if estimated_dataset_gb > 4.0
                  else f"Long context ({max_seq_len} tokens >= 8192)")
        print({
            "dataset_loading": "streaming",
            "reason": reason,
            "num_samples": num_lines,
            "max_seq_len": max_seq_len,
        })
        # Don't tokenize all lines at once - will be done lazily during training
    else:
        # Small dataset - use eager loading
        print({
            "dataset_loading": "eager (all in memory)",
            "estimated_gb": f"{estimated_dataset_gb:.2f}",
            "num_samples": num_lines,
        })
        input_ids, labels = _encode_lines_helper(tok, lines, max_seq_len)
    
    # ============================================================================
    # Vocabulary Size Calculation
    # ============================================================================
    # Calculate actual vocab size needed (base vocab + added special tokens)
    vocab_size = int(getattr(tok, "vocab_size", 50257) or 50257)
    
    # Check if tokenizer has special tokens beyond vocab_size
    if hasattr(tok, 'all_special_ids') and tok.all_special_ids:
        max_special_id = max(tok.all_special_ids)
        if max_special_id >= vocab_size:
            actual_vocab_size = max_special_id + 1
            print({
                "vocab_size_adjustment": {
                    "original_vocab_size": vocab_size,
                    "max_special_token_id": max_special_id,
                    "adjusted_vocab_size": actual_vocab_size,
                    "reason": "Special tokens extend beyond base vocabulary"
                }
            })
            vocab_size = actual_vocab_size
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
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
        window_size=window_size,
        use_moe=config.use_moe,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_capacity_factor=config.moe_capacity_factor,
    )
    
    # ============================================================================
    # Model Instantiation
    # ============================================================================
    model_student = _build_student(cfg, student_init=student_init, print_fn=print)
    
    # Log architecture
    try:
        print({
            "arch": {
                "H_layers": int(cfg["H_layers"]),
                "L_layers": int(cfg["L_layers"]),
                "hidden_size": int(cfg["hidden_size"]),
                "num_heads": int(cfg["num_heads"]),
                "expansion": float(cfg["expansion"]),
            }
        })
    except Exception:
        pass
    
    # ============================================================================
    # Evaluation Data Preparation
    # ============================================================================
    eval_ids, eval_labels = _encode_eval_lines(
        tok=tok,
        eval_file=eval_file,
        ascii_only=ascii_only,
        max_seq_len=max_seq_len,
        read_text_lines_sample_any=read_text_lines_sample_any,
    )
    
    # Set model to training mode
    model_student.train(True)
    
    # ============================================================================
    # Return All Setup Components
    # ============================================================================
    return {
        # Model and tokenizer
        "model_student": model_student,
        "tokenizer": tok,
        "cfg": cfg,
        "vocab_size": vocab_size,
        
        # Device configuration
        "device_obj": device_obj,
        "dev": dev,
        "dml_device": dml_device,
        
        # Distributed training
        "is_distributed": is_distributed,
        "rank_id": rank_id,
        "world_sz": world_sz,
        "init_file_env": init_file_env,
        
        # Dataset
        "input_ids": input_ids,
        "labels": labels,
        "lines": lines,
        "streaming_dataset": streaming_dataset,
        "use_streaming": use_streaming,
        "cycle_count": cycle_count,
        "num_lines": num_lines,
        
        # Evaluation data
        "eval_ids": eval_ids,
        "eval_labels": eval_labels,
        
        # Training parameters
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        
        # Helper functions
        "write_jsonl": _write_jsonl,
        "should_stop": _should_stop,
        "load_or_generate_lines": _load_or_generate_lines,
        
        # Paths and metadata
        "out_dir_path": out_dir_path,
        "save_dir": save_dir,
        "log_file": log_file,
        "brain_metadata": brain_metadata,
        
        # Updated config values
        "cuda_ids": cuda_ids,
        "ddp": ddp,
        "world_size": world_size,
    }
