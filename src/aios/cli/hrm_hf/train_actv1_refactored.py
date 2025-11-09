"""ACT V1 HRM Training - Main Orchestrator.

This module coordinates all training components through a clean, modular interface.
Each major aspect of training (config, data, model, optimization, execution) is
handled by specialized modules.
"""
from __future__ import annotations

# CRITICAL: Set CUDA allocator config BEFORE any torch imports!
import os
import warnings

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*", category=UserWarning)

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import typer
from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig

# Import modular components
from aios.data.datasets import read_text_lines_sample_any
from ..hrm_hf_utils import load_tokenizer as _load_tokenizer
from .checkpoint_recovery import recover_checkpoint_artifacts, resolve_student_init_path
from .expert_training import train_expert_only
from .config_validation import (
    extract_and_log_config,
    auto_adjust_moe_learning_rate,
    setup_cuda_devices,
    validate_dependencies,
    setup_output_directory,
)
from .resume_detection import detect_resume_state
from .brain_metadata import load_brain_metadata, extract_model_from_metadata
from .data import get_training_lines as _get_training_lines
from .dataset_setup import (
    setup_dataset_mode,
    calculate_warmup_and_coverage,
    setup_epoch_tracking,
)
from .model_building import (
    calculate_vocab_size,
    build_model_config,
    build_model,
    count_model_parameters,
    print_extreme_scale_recommendations,
)
from .model_precision import apply_quantization, apply_dtype_conversion, apply_peft
from .memory_optimization import (
    detect_available_vram,
    setup_memory_tracking,
    enable_extreme_scale_mode,
    run_training_optimizer,
    configure_chunking,
    log_optimization_summary,
    finalize_memory_report,
)
from .device import (
    resolve_device as _resolve_device,
    init_cuda_distributed_if_needed as _init_cuda_dist,
    get_dist_context as _get_dist_context,
)
from .distributed_setup import (
    initialize_deepspeed,
    wrap_with_ddp,
    handle_windows_zero_multi_gpu,
    ensure_model_on_device,
)
from .optimizer_setup import (
    create_optimizer,
    setup_amp_scaler,
    setup_inference_manager,
    log_optimizer_memory,
)
from .training_loop import (
    execute_training_epoch,
    run_iterate_mode,
    run_single_epoch_mode,
)
from .training_helpers import (
    write_jsonl as _write_jsonl_helper,
    should_stop as _should_stop_helper,
    eval_once as _eval_once_helper,
)
from .encoding import (
    adjust_tokenizer_padding as _adjust_tok_padding,
    encode_eval_lines as _encode_eval_lines,
)
from .ddp import (
    maybe_spawn_and_exit_if_parent as _maybe_spawn,
    ddp_worker as _ddp_worker,
    serialize_config_for_spawn,
)
from .finalization import (
    finalize_training as _finalize_training,
    broadcast_final_payload as _broadcast_final_payload,
)


def train_actv1_impl(config: "TrainingConfig") -> None:
    """Train the ACT V1 HRM model or expert module.
    
    This is the main orchestrator that coordinates all training components.
    It delegates specialized tasks to modular components for maintainability.
    
    Args:
        config: Training configuration object containing all parameters
    """
    # ========================================================================
    # EXPERT-ONLY TRAINING MODE
    # ========================================================================
    if config.expert_id:
        return train_expert_only(config)
    
    # ========================================================================
    # INITIALIZATION & VALIDATION
    # ========================================================================
    if not validate_dependencies(print):
        raise typer.Exit(code=1)
    
    # Extract and log configuration
    extract_and_log_config(config, print)
    
    # Auto-adjust learning rate for MoE if enabled
    lr = auto_adjust_moe_learning_rate(config, print)
    
    # Setup CUDA devices
    setup_cuda_devices(config.cuda_ids, print)
    
    # ========================================================================
    # RESUME DETECTION
    # ========================================================================
    step_offset, resume_cycle, resume_session = detect_resume_state(
        config, config.bundle_dir, config.brain_name, print
    )
    
    # ========================================================================
    # DISTRIBUTED TRAINING SETUP
    # ========================================================================
    # Optional: Launch multi-process DDP (Windows uses gloo backend)
    should_exit = _maybe_spawn(
        ddp=config.ddp,
        device=config.device,
        torch=torch,
        os=os,
        platform=__import__('platform'),
        world_size=config.world_size,
        strict=config.strict,
        worker_target=_ddp_worker,
        spawn_kwargs=serialize_config_for_spawn(config),
    )
    if should_exit:
        return
    
    # Windows-specific: Auto-convert multi-GPU to single GPU when ZeRO enabled
    ddp, world_size = handle_windows_zero_multi_gpu(config, print)
    config.ddp = ddp
    config.world_size = world_size
    
    # Resolve device with validation
    dev, device_obj, dml_device = _resolve_device(config.device, config.strict, torch)
    
    # Detect distributed context
    is_distributed, rank_id, world_sz, init_file_env = _get_dist_context(os)
    
    # ========================================================================
    # OUTPUT DIRECTORY & CHECKPOINTS
    # ========================================================================
    out_dir_path, save_dir, log_file = setup_output_directory(
        config.brain_name, config.bundle_dir, config.log_file, config.save_dir, print
    )
    config.save_dir = save_dir
    config.log_file = log_file
    
    # Recover partial checkpoints
    candidate_dirs: list[Path] = []
    if out_dir_path:
        candidate_dirs.append(out_dir_path)
    if config.save_dir:
        candidate_dirs.append(Path(config.save_dir))
    if config.student_init:
        init_path = Path(config.student_init)
        candidate_dirs.append(init_path if init_path.is_dir() else init_path.parent)
    
    recovered_checkpoint = recover_checkpoint_artifacts(candidate_dirs, print)
    student_init = resolve_student_init_path(
        config.student_init, recovered_checkpoint, candidate_dirs, print
    )
    config.student_init = student_init
    
    # Load brain metadata if resuming
    brain_metadata = load_brain_metadata(student_init, print)
    model_path = extract_model_from_metadata(brain_metadata, config.model)
    
    # ========================================================================
    # TOKENIZER & DATASET
    # ========================================================================
    tok = _load_tokenizer(model_path)
    _adjust_tok_padding(tok)
    
    # Setup JSONL logging (rank0 only when distributed)
    def write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(log_file, payload, is_distributed, rank_id)
    
    def should_stop() -> bool:
        return _should_stop_helper(config.stop_file)
    
    # Load training data
    lines = _get_training_lines(
        dataset_file=config.dataset_file,
        ascii_only=config.ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
    )
    
    if not lines:
        print({"started": False, "error": "no lines"})
        raise typer.Exit(code=1)
    
    # Setup epoch tracking
    setup_epoch_tracking(config, print)
    
    # Calculate warmup steps and validate coverage
    warmup_steps = calculate_warmup_and_coverage(config, len(lines), print)
    
    # Setup dataset mode (streaming vs eager)
    input_ids, labels, streaming_dataset, N = setup_dataset_mode(config, lines, print)
    
    # ========================================================================
    # MODEL BUILDING
    # ========================================================================
    vocab_size = calculate_vocab_size(tok, print)
    cfg = build_model_config(config, vocab_size, print)
    model_student = build_model(cfg, student_init, print)
    
    # Count parameters
    total_params = count_model_parameters(model_student)
    
    # Apply quantization and precision
    initial_memory_mb = sum(p.numel() * p.element_size() for p in model_student.parameters()) / (1024**2)
    apply_quantization(model_student, config, print)
    apply_dtype_conversion(model_student, config, initial_memory_mb, print)
    
    # Apply PEFT (LoRA/AdaLoRA/IA3)
    model_student = apply_peft(model_student, config, print)
    
    # ========================================================================
    # OPTIMIZATION & MEMORY MANAGEMENT
    # ========================================================================
    # Run intelligent optimization if requested
    if config.optimize:
        max_seq_len, batch_size, zero_stage = run_training_optimizer(config, total_params, print)
        config.max_seq_len = max_seq_len
        config.batch_size = batch_size
        config.zero_stage = zero_stage
    
    # Enable extreme-scale mode if needed
    enable_extreme_scale_mode(config.max_seq_len, total_params, print)
    
    # Print recommendations for extreme scale
    print_extreme_scale_recommendations(config.max_seq_len, total_params, print)
    
    # Detect VRAM
    available_vram_gb = detect_available_vram(print)
    
    # Configure chunking
    segment_rollout, use_chunking, final_chunk_size = configure_chunking(
        config.max_seq_len, config.chunk_size, config.use_chunked_training,
        config.gradient_checkpointing, config.use_cpu_offload, print
    )
    
    # ========================================================================
    # DEVICE PLACEMENT & DISTRIBUTED SETUP
    # ========================================================================
    # Initialize memory tracking
    memory_tracker = setup_memory_tracking(model_student, config, device_obj, print)
    
    # Ensure model on correct device (ZeRO-3 needs CPU initially)
    model_student = ensure_model_on_device(model_student, device_obj, config.zero_stage, print)
    
    # Initialize CUDA distributed if needed
    device_obj, ddp_initialized = _init_cuda_dist(
        dev=str(device_obj),
        is_distributed=is_distributed,
        torch=torch,
        os=os,
        rank_id=rank_id,
        world_sz=world_sz,
        init_file_env=init_file_env,
    )
    
    # Initialize DeepSpeed
    deepspeed_engine, use_deepspeed_optimizer = initialize_deepspeed(
        model_student, config, device_obj, print
    )
    
    # Wrap with DDP if multi-GPU without DeepSpeed
    if deepspeed_engine is None:
        model_student = wrap_with_ddp(model_student, device_obj, ddp_initialized and is_distributed, rank_id, print)
    
    # ========================================================================
    # OPTIMIZER & AMP
    # ========================================================================
    opt = create_optimizer(model_student, config, use_deepspeed_optimizer, print)
    scaler = setup_amp_scaler(config.use_amp, dev, print)
    
    # Log optimizer memory
    log_optimizer_memory(memory_tracker, config.use_8bit_optimizer, use_deepspeed_optimizer, print)
    
    # Log optimization summary
    theoretical_memory_gb = sum(p.numel() * p.element_size() for p in model_student.parameters()) / (1024**3)
    log_optimization_summary(theoretical_memory_gb, config, world_sz, is_distributed, print)
    
    # ========================================================================
    # INFERENCE MANAGER (OPTIONAL)
    # ========================================================================
    inference_manager = setup_inference_manager(config, model_student, tok, print)
    
    # ========================================================================
    # EVALUATION SETUP
    # ========================================================================
    eval_ids, eval_labels = _encode_eval_lines(
        tok=tok,
        eval_file=config.eval_file,
        ascii_only=config.ascii_only,
        max_seq_len=config.max_seq_len,
        read_text_lines_sample_any=read_text_lines_sample_any,
    )
    model_student.train(True)
    
    def eval_once() -> None:
        _eval_once_helper(
            model_student=model_student,
            eval_ids=eval_ids,
            eval_labels=eval_labels,
            batch_size=config.batch_size,
            device_obj=device_obj,
            dml_device=dml_device,
            halt_max_steps=config.halt_max_steps,
            eval_batches=config.eval_batches,
            segment_rollout=segment_rollout,
            write_jsonl=write_jsonl,
            tokenizer=tok,
            enable_english_logic_eval=True,
        )
    
    # ========================================================================
    # TRAINING EXECUTION
    # ========================================================================
    def load_new_lines(cycle: int = 0) -> list[str]:
        return _get_training_lines(
            dataset_file=config.dataset_file,
            ascii_only=config.ascii_only,
            read_text_lines_sample_any=read_text_lines_sample_any,
            cycle=cycle,
        )
    
    def train_with_lines(new_lines: list[str]) -> None:
        nonlocal input_ids, labels, streaming_dataset
        if streaming_dataset is not None:
            from .streaming_dataset import create_streaming_dataset
            shuffle_mode = not config.linear_dataset
            start_offset = config.dataset_start_offset if config.linear_dataset else 0
            streaming_dataset = create_streaming_dataset(
                lines=new_lines,
                tokenizer=tok,
                max_seq_len=config.max_seq_len,
                batch_size=config.batch_size,
                shuffle=shuffle_mode,
                epoch=0,
                start_offset=start_offset,
            )
    
    # Create epoch executor
    def execute_epoch():
        nonlocal batch_size, stopped_early, last_stop_reason, steps_done
        s_done, early, new_bs, stop_reason = execute_training_epoch(
            model=model_student,
            segment_rollout=segment_rollout,
            opt=opt,
            device_obj=device_obj,
            dml_device=dml_device,
            input_ids=input_ids,
            labels=labels,
            config=config,
            streaming_dataset=streaming_dataset,
            tokenizer=tok,
            lines=lines if streaming_dataset else None,
            scaler=scaler,
            deepspeed_engine=deepspeed_engine,
            inference_manager=inference_manager,
            warmup_steps=warmup_steps,
            base_lr=lr,
            step_offset=step_offset,
            is_distributed=is_distributed,
            world_sz=world_sz,
            write_jsonl=write_jsonl,
            should_stop=should_stop,
        )
        steps_done += s_done
        stopped_early = early
        batch_size = new_bs
        last_stop_reason = stop_reason
        return s_done, early, new_bs, stop_reason
    
    # Initialize tracking variables
    steps_done = 0
    stopped_early = False
    last_stop_reason = None
    batch_size = config.batch_size
    cycle = resume_cycle if config.iterate else 0
    
    # Run training
    if config.iterate:
        steps_done, stopped_early, last_stop_reason, cycle = run_iterate_mode(
            config, lines, resume_cycle, execute_epoch,
            load_new_lines, train_with_lines, print, write_jsonl,
        )
    else:
        steps_done, stopped_early, last_stop_reason = run_single_epoch_mode(execute_epoch, print)
    
    print({
        "EXITED_TRAINING": True,
        "stopped_early": stopped_early,
        "steps_done": steps_done,
        "stop_reason": last_stop_reason
    })
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    # Final evaluation
    if config.eval_file:
        try:
            print({"final_evaluation": "running"})
            eval_once()
        except Exception as e:
            print({"final_evaluation": "failed", "error": str(e)})
    
    # Generate memory report
    finalize_memory_report(memory_tracker, steps_done, stopped_early, batch_size, write_jsonl, print)
    
    print({"ABOUT_TO_CALL_FINALIZATION": True, "steps_done": steps_done})
    
    # Finalize training and save artifacts
    try:
        final_payload = _finalize_training(
            model_student=model_student,
            save_dir=save_dir,
            stopped_early=stopped_early,
            steps_done=steps_done,
            is_distributed=is_distributed,
            rank_id=rank_id,
            tok=tok,
            h_layers=config.h_layers,
            l_layers=config.l_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expansion=config.expansion,
            h_cycles=config.h_cycles,
            l_cycles=config.l_cycles,
            pos_encodings=config.pos_encodings,
            log_file=log_file,
            write_jsonl=write_jsonl,
            brain_name=config.brain_name,
            model=model_path,
            max_seq_len=config.max_seq_len,
            halt_max_steps=config.halt_max_steps,
            default_goal=config.default_goal,
            dataset_file=config.dataset_file,
            use_moe=config.use_moe,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_capacity_factor=config.moe_capacity_factor,
            stop_reason=last_stop_reason,
            iterate_cycle=cycle if config.iterate else None,
            training_config=config,
        )
    except Exception as finalize_error:
        print({
            "finalize_training": "ERROR",
            "error": str(finalize_error),
            "traceback": str(__import__('traceback').format_exc()),
        })
        final_payload = {"trained": False, "error": str(finalize_error)}
    
    final_payload = _broadcast_final_payload(final_payload, is_distributed, rank_id, torch)
    
    if (not is_distributed) or (rank_id == 0):
        print(final_payload)
        write_jsonl({"event": "final", **final_payload})
    
    # Cleanup
    if inference_manager is not None:
        try:
            inference_manager.cleanup()
        except Exception as e:
            print({"inference_manager_cleanup": "failed", "error": str(e)})
    
    if is_distributed and str(device_obj).startswith("cuda"):
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass
    
    print({
        "training_complete": True,
        "exit_code": 0,
        "stopped_early": stopped_early,
        "steps": steps_done
    })
