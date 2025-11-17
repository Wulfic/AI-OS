"""ACT V1 HRM Training - Main Orchestrator.

This module coordinates all training components through a clean, modular interface.
Each major aspect of training (config, data, model, optimization, execution) is
handled by specialized modules.
"""
from __future__ import annotations

# CRITICAL: Setup CUDA environment variables BEFORE any torch imports!
import os
import warnings

# Set CUDA allocator config BEFORE torch import (PYTORCH_CUDA_ALLOC_CONF deprecated)
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*", category=UserWarning)

# CRITICAL: Set CUDA_VISIBLE_DEVICES before importing torch if cuda_ids provided
# This must happen BEFORE torch.cuda.is_available() is called to be effective
# We extract cuda_ids from sys.argv to set it early (for CLI) or from environment (for GUI)
def _setup_cuda_visible_devices_early():
    """Setup CUDA_VISIBLE_DEVICES from command line args or environment before torch import.
    
    This is critical because once torch.cuda is initialized, changing 
    CUDA_VISIBLE_DEVICES has no effect on the current process.
    
    Handles two cases:
    1. CLI invocation: Extracts --cuda-ids from sys.argv
    2. GUI/multiprocessing: Checks AIOS_CUDA_IDS environment variable set by parent process
    """
    import sys
    
    cuda_ids_str = None
    source = "none"
    
    # Case 1: Check environment variable (set by GUI before spawning process)
    if "AIOS_CUDA_IDS" in os.environ:
        cuda_ids_str = os.environ["AIOS_CUDA_IDS"]
        source = "AIOS_CUDA_IDS env var"
    
    # Case 2: Extract from command line arguments
    if not cuda_ids_str:
        try:
            # Look for --cuda-ids argument in command line
            for i, arg in enumerate(sys.argv):
                if arg == '--cuda-ids' and i + 1 < len(sys.argv):
                    cuda_ids_str = sys.argv[i + 1]
                    source = "CLI --cuda-ids"
                    break
        except Exception:
            pass
    
    # Apply CUDA_VISIBLE_DEVICES if we found cuda_ids
    if cuda_ids_str:
        try:
            # Parse and validate
            device_list = [
                str(int(x.strip())) for x in cuda_ids_str.split(",") 
                if x.strip() and x.strip().isdigit()
            ]
            if device_list:
                existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)
                # Note: Can't use rich.print here as it may not be imported yet
                print(f"[EARLY INIT] CUDA_VISIBLE_DEVICES set to: {','.join(device_list)} (from {source})")
                if existing and existing != ",".join(device_list):
                    print(f"[EARLY INIT] Previous value was: {existing}")
        except Exception as e:
            print(f"[EARLY INIT] Warning: Failed to set CUDA_VISIBLE_DEVICES early: {e}")

_setup_cuda_visible_devices_early()

# Now safe to import torch
try:
    import torch
    if torch.cuda.is_available():
        # Prefer mem-efficient kernels; disable math fallback which may inflate memory
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash(False)  # keep False unless FlashAttention-compatible
        sdp_kernel.enable_math(False)
        sdp_kernel.enable_mem_efficient(True)
except Exception:
    pass

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
from .checkpoint_saver import CheckpointSaver
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


def run_training_multiprocessing_entry(
    config: "TrainingConfig",
    stop_event=None,
    graceful_stop_event=None,
    stop_ack_event=None,
    graceful_stop_ack_event=None,
) -> None:
    """Entry point for multiprocessing.Process.
    
    This function is called when training is launched via multiprocessing.Process
    instead of subprocess. It allows sharing of Events for stop signaling.
    
    Args:
        config: Training configuration
        stop_event: Multiprocessing Event for immediate stop
        graceful_stop_event: Multiprocessing Event for graceful stop
        stop_ack_event: Event set by worker when immediate stop is observed
        graceful_stop_ack_event: Event set by worker when graceful stop is observed
    """
    import sys
    import traceback
    import faulthandler
    import signal

    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    try:
        sigusr2 = getattr(signal, "SIGUSR2", None)
        if sigusr2 is not None:
            faulthandler.register(sigusr2, file=sys.__stderr__, all_threads=True, chain=False)
    except Exception:
        pass
    
    try:
        # Call the main training implementation with Events
        train_actv1_impl(
            config,
            stop_event=stop_event,
            graceful_stop_event=graceful_stop_event,
            stop_ack_event=stop_ack_event,
            graceful_stop_ack_event=graceful_stop_ack_event,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


def train_actv1_impl(
    config: "TrainingConfig",
    stop_event=None,
    graceful_stop_event=None,
    stop_ack_event=None,
    graceful_stop_ack_event=None,
) -> None:
    """Train the ACT V1 HRM model or expert module.
    
    This is the main orchestrator that coordinates all training components.
    It delegates specialized tasks to modular components for maintainability.
    
    Args:
        config: Training configuration object containing all parameters
        stop_event: Optional multiprocessing Event for immediate stop
        graceful_stop_event: Optional multiprocessing Event for graceful stop
        stop_ack_event: Optional Event set once stop_event is observed by workers
        graceful_stop_ack_event: Optional Event set once graceful stop is observed
    """
    # ========================================================================
    # OUTPUT DIRECTORY SETUP (must happen before any training mode)
    # ========================================================================
    # Setup output directory based on brain_name. This MUST happen before
    # parallel independent or other training modes are dispatched.
    out_dir_path, save_dir, log_file = setup_output_directory(
        config.brain_name, config.bundle_dir, config.log_file, config.save_dir, print
    )
    config.save_dir = save_dir
    config.log_file = log_file
    
    # ========================================================================
    # AUTO-ENABLE PARALLEL TRAINING FOR ALL GPU CONFIGURATIONS
    # ========================================================================
    # Parse cuda_ids to determine GPU configuration
    import platform
    cuda_ids = config.cuda_ids
    if isinstance(cuda_ids, str):
        cuda_id_list = [int(x.strip()) for x in cuda_ids.split(',') if x.strip()]
    elif cuda_ids is not None:
        cuda_id_list = list(cuda_ids) if hasattr(cuda_ids, '__iter__') else [cuda_ids]
    else:
        cuda_id_list = []
    
    # On Windows, DDP is broken (PyTorch limitation). Auto-enable parallel independent
    # training when multiple GPUs are detected unless explicitly disabled.
    if platform.system() == "Windows":
        # If multi-GPU on Windows and not explicitly disabled, enable parallel independent
        if len(cuda_id_list) > 1 and not config.parallel_independent and not config.ddp:
            print({
                "info": "Windows multi-GPU detected",
                "note": "Automatically enabling parallel independent training (DDP is broken on Windows)",
                "gpus": cuda_id_list,
                "tip": "Use --no-parallel-independent to disable this behavior"
            })
            config.parallel_independent = True
    
    # ========================================================================
    # UNIFIED PARALLEL TRAINING MODE (Single-GPU and Multi-GPU)
    # ========================================================================
    # CRITICAL: Route ALL GPU training (single or multi) through parallel_training_v3
    # This ensures consistent batch handling, gradient accumulation, and step counting.
    # 
    # Benefits:
    # - Proper gradient accumulation (effective_batch_size = batch_size * grad_accum_steps)
    # - Correct step counting (optimizer steps, not micro-batches)
    # - Unified codebase reduces maintenance burden
    # - Single GPU gets same robust chunk tracking as parallel mode
    #
    # The parallel_training_v3 code handles single-GPU case by just using 1 GPU worker.
    if config.parallel_independent or (len(cuda_id_list) == 1 and not config.ddp):
        from .parallel_training_v3 import run_parallel_training_v3
        
        # For single GPU, ensure parallel_independent is set so parallel_training_v3 activates
        if len(cuda_id_list) == 1 and not config.parallel_independent:
            print({
                "info": "Single GPU training routing through unified parallel training path",
                "note": "This ensures consistent gradient accumulation and step counting",
                "gpu": cuda_id_list[0],
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": getattr(config, 'gradient_accumulation_steps', 1),
                "effective_batch_size": config.batch_size * getattr(config, 'gradient_accumulation_steps', 1),
            })
            config.parallel_independent = True
        
        return run_parallel_training_v3(
            config,
            stop_event=stop_event,
            graceful_stop_event=graceful_stop_event,
            stop_ack_event=stop_ack_event,
            graceful_stop_ack_event=graceful_stop_ack_event,
        )
    
    # ========================================================================
    # EXPERT-ONLY TRAINING MODE
    # ========================================================================
    if config.expert_id:
        return train_expert_only(config)
    
    # ========================================================================
    # ALTERNATIVE TRAINING PATH (DDP ONLY)
    # ========================================================================
    # NOTE: This is an alternative training implementation for DDP (torch.distributed).
    # 
    # Current routing:
    # - Single GPU training: Routes through parallel_training_v3 (above)
    # - Multi-GPU training: Routes through parallel_training_v3 (above)
    # - DDP training: Uses this path (Linux/multi-GPU with torch.distributed)
    #
    # Use case: DDP-based distributed training (--ddp flag)
    #
    # Known limitations:
    # - Different step counting behavior (counts micro-batches instead of optimizer steps)
    # - Separate gradient accumulation implementation from parallel_training_v3
    # - Requires torch.distributed setup
    #
    # Future: DDP functionality may be unified with parallel_training_v3 in later releases.
    # ========================================================================
    
    # INITIALIZATION & VALIDATION
    # ========================================================================
    if not validate_dependencies(print):
        raise typer.Exit(code=1)
    
    # Extract and log configuration
    extract_and_log_config(config, print)
    
    # Auto-adjust learning rate for MoE if enabled
    lr = auto_adjust_moe_learning_rate(config, print)
    # CRITICAL: Update config with adjusted LR so optimizer uses it
    config.lr = lr
    
    # Setup CUDA devices (already set early before torch import, this is for logging/validation)
    setup_cuda_devices(config.cuda_ids, print)
    
    # ========================================================================
    # RESUME DETECTION (must happen before DDP spawn)
    # ========================================================================
    # Check if step_offset/resume_cycle already in config (from DDP parent process)
    if hasattr(config, 'step_offset') and hasattr(config, 'resume_cycle'):
        step_offset = config.step_offset  # type: ignore[attr-defined]
        resume_cycle = config.resume_cycle  # type: ignore[attr-defined]
        resume_session = None
        print({
            "resume": "using_parent_values",
            "step_offset": step_offset,
            "resume_cycle": resume_cycle,
            "note": "DDP worker inheriting resume state from parent"
        })
    else:
        # Main process: detect resume state
        step_offset, resume_cycle, resume_session = detect_resume_state(
            config, config.bundle_dir, config.brain_name, print
        )
        
        # Store resume state in config so it can be passed to DDP workers
        config.step_offset = step_offset  # type: ignore[attr-defined]
        config.resume_cycle = resume_cycle  # type: ignore[attr-defined]
    
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
    # CHECKPOINTS
    # ========================================================================
    # Note: output directory setup already happened at the start of train_actv1_impl()
    # Now we just recover checkpoints from the configured directories.
    
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
    
    # Load brain metadata if resuming (or check save_dir for fresh starts)
    brain_metadata = load_brain_metadata(
        student_init=student_init,
        log_fn=print,
        save_dir=config.save_dir  # Fallback to brain directory for fresh starts
    )
    model_path = extract_model_from_metadata(brain_metadata, config.model)
    
    # ========================================================================
    # TOKENIZER & DATASET
    # ========================================================================
    # Import tokenizer extraction function
    from .brain_metadata import extract_tokenizer_from_metadata
    tokenizer_path = extract_tokenizer_from_metadata(brain_metadata, model_path)
    tok = _load_tokenizer(tokenizer_path)
    _adjust_tok_padding(tok)
    
    # Setup JSONL logging (rank0 only when distributed)
    def write_jsonl(payload: dict) -> None:
        _write_jsonl_helper(
            log_file=log_file,
            payload=payload,
            is_distributed=is_distributed,
            rank_id=rank_id
        )
    
    def should_stop() -> bool:
        return _should_stop_helper(config.stop_file)
    
    # Load training data (use resume_cycle to load correct chunk when resuming)
    lines = _get_training_lines(
        dataset_file=config.dataset_file,
        ascii_only=config.ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
        cycle=resume_cycle,  # Use resume_cycle to continue from correct chunk
        dataset_chunk_size=config.dataset_chunk_size,
    )
    
    if not lines:
        write_jsonl({"started": False, "error": "no lines"})
        raise typer.Exit(code=1)
    
    # Setup epoch tracking
    setup_epoch_tracking(config, write_jsonl)
    
    # Calculate warmup steps and validate coverage
    warmup_steps = calculate_warmup_and_coverage(config, len(lines), write_jsonl)
    
    # Setup dataset mode (streaming vs eager)
    input_ids, labels, streaming_dataset, N = setup_dataset_mode(
        config,
        lines,
        write_jsonl,
        rank=rank_id,
        world_size=world_sz,
        tok=tok,
    )
    
    # ========================================================================
    # MODEL BUILDING
    # ========================================================================
    vocab_size = calculate_vocab_size(tok, write_jsonl)
    cfg = build_model_config(config, vocab_size, write_jsonl)
    model_student = build_model(cfg, student_init, write_jsonl)
    
    # Count parameters
    total_params = count_model_parameters(model_student)
    
    # Apply quantization and precision
    initial_memory_mb = sum(p.numel() * p.element_size() for p in model_student.parameters()) / (1024**2)
    apply_quantization(model_student, config, write_jsonl)
    apply_dtype_conversion(model_student, config, initial_memory_mb, write_jsonl)
    
    # Apply PEFT (LoRA/AdaLoRA/IA3)
    model_student = apply_peft(model_student, config, write_jsonl)
    
    # ========================================================================
    # OPTIMIZATION & MEMORY MANAGEMENT
    # ========================================================================
    # Run intelligent optimization if requested
    if config.optimize:
        max_seq_len, batch_size, zero_stage = run_training_optimizer(config, total_params, write_jsonl)
        config.max_seq_len = max_seq_len
        config.batch_size = batch_size
        config.zero_stage = zero_stage  # type: ignore[assignment]
    
    # Enable extreme-scale mode if needed
    enable_extreme_scale_mode(config.max_seq_len, total_params, write_jsonl)
    
    # Print recommendations for extreme scale
    print_extreme_scale_recommendations(config.max_seq_len, total_params, write_jsonl)
    
    # Detect VRAM
    available_vram_gb = detect_available_vram(write_jsonl)
    
    # Configure chunking
    segment_rollout, use_chunking, final_chunk_size = configure_chunking(
        config.max_seq_len, config.chunk_size, config.use_chunked_training,
        config.gradient_checkpointing, config.use_cpu_offload, write_jsonl
    )
    
    # ========================================================================
    # DEVICE PLACEMENT & DISTRIBUTED SETUP
    # ========================================================================
    # Initialize memory tracking
    memory_tracker = setup_memory_tracking(model_student, config, device_obj, write_jsonl)
    
    # Ensure model on correct device (ZeRO-3 needs CPU initially)
    model_student = ensure_model_on_device(model_student, device_obj, config.zero_stage, write_jsonl)
    
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
        model_student, config, device_obj, write_jsonl
    )
    
    # Wrap with DDP if multi-GPU without DeepSpeed
    if deepspeed_engine is None:
        model_student = wrap_with_ddp(model_student, device_obj, ddp_initialized and is_distributed, rank_id, write_jsonl)
    else:
        # When using DeepSpeed, the engine IS the model for training
        # Replace model_student with engine so all subsequent operations use it
        write_jsonl({
            "model": "using_deepspeed_engine",
            "note": "DeepSpeed engine replaces model for training"
        })
        model_student = deepspeed_engine
    
    # ========================================================================
    # CHECKPOINT SAVER & SIGNAL HANDLING
    # ========================================================================
    checkpoint_saver = CheckpointSaver(
        model=model_student,
        save_dir=config.save_dir,
        config=config,
        print_fn=write_jsonl,
    )
    checkpoint_saver.setup_signal_handlers()
    
    # ========================================================================
    # OPTIMIZER & AMP
    # ========================================================================
    opt = create_optimizer(model_student, config, use_deepspeed_optimizer, write_jsonl)
    scaler = setup_amp_scaler(config.use_amp, dev, write_jsonl)
    
    # Log optimizer memory
    log_optimizer_memory(memory_tracker, config.use_8bit_optimizer, use_deepspeed_optimizer, write_jsonl)
    
    # Log optimization summary
    theoretical_memory_gb = sum(p.numel() * p.element_size() for p in model_student.parameters()) / (1024**3)
    log_optimization_summary(theoretical_memory_gb, config, world_sz, is_distributed, write_jsonl)
    
    # ========================================================================
    # INFERENCE MANAGER (OPTIONAL)
    # ========================================================================
    inference_manager = setup_inference_manager(config, model_student, tok, write_jsonl)
    
    # ========================================================================
    # EVALUATION SETUP
    # ========================================================================
    eval_ids, eval_labels = _encode_eval_lines(
        tok=tok,
        eval_file=config.eval_file,
        ascii_only=config.ascii_only,
        max_seq_len=config.max_seq_len,
        read_text_lines_sample_any=read_text_lines_sample_any,
        dataset_chunk_size=config.dataset_chunk_size,
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
            dataset_chunk_size=config.dataset_chunk_size,
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
                rank=rank_id,
                world_size=world_sz,
            )
    
    # Create epoch executor
    def execute_epoch():
        nonlocal batch_size, stopped_early, last_stop_reason, steps_done
        # Use cumulative steps_done as offset for display in iterate mode
        # This ensures step counters continue from previous cycles (e.g., 1001, 1002...)
        # instead of resetting to 1 each cycle
        current_offset = step_offset + steps_done
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
            lines=lines if streaming_dataset else None,  # pyright: ignore[reportArgumentType]
            scaler=scaler,
            deepspeed_engine=deepspeed_engine,
            inference_manager=inference_manager,
            warmup_steps=warmup_steps,
            base_lr=lr,
            step_offset=current_offset,
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
        # Enable chunk/block tracking for iterate mode in single-GPU/DDP/standard modes when linear dataset is enabled
        tracker_obj = None
        gpu_rank = rank_id if is_distributed else 0
        # Store rank info in config for training loop to access
        config._tracker_rank_id = gpu_rank  # type: ignore
        config._tracker_is_rank_0 = (not is_distributed) or (rank_id == 0)  # type: ignore
        try:
            if getattr(config, 'linear_dataset', False):
                from .chunk_tracker import ChunkTracker as _ChunkTracker
                # Save tracker state alongside training artifacts
                from pathlib import Path as _P
                state_dir = _P(config.save_dir) if config.save_dir else _P("artifacts/brains/actv1")
                tracker_state_file = state_dir / "chunk_tracker_state.json"
                tracker_obj = _ChunkTracker(state_file=tracker_state_file)
                if (not is_distributed) or (rank_id == 0):
                    write_jsonl({
                        "chunk_tracker": "enabled",
                        "state_file": str(tracker_state_file),
                        "mode": "DDP" if is_distributed else "single-GPU",
                        "world_size": world_sz if is_distributed else 1,
                    })
            else:
                if (not is_distributed) or (rank_id == 0):
                    write_jsonl({
                        "chunk_tracker": "disabled",
                        "reason": "linear_dataset is False (randomized sampling)",
                    })
        except Exception as _e:
            if (not is_distributed) or (rank_id == 0):
                write_jsonl({"chunk_tracker": "init_failed", "error": str(_e)})

        steps_done, stopped_early, last_stop_reason, cycle = run_iterate_mode(
            config, lines, resume_cycle, execute_epoch,
            load_new_lines, train_with_lines, write_jsonl, write_jsonl,
            checkpoint_saver,
            chunk_tracker=tracker_obj,
        )
    else:
        # Enable chunk/block tracking for single-epoch mode as well (one chunk)
        tracker_obj = None
        gpu_rank = rank_id if is_distributed else 0
        config._tracker_rank_id = gpu_rank  # type: ignore
        config._tracker_is_rank_0 = (not is_distributed) or (rank_id == 0)  # type: ignore
        if getattr(config, 'linear_dataset', False):
            try:
                from .chunk_tracker import ChunkTracker as _ChunkTracker
                from pathlib import Path as _P
                state_dir = _P(config.save_dir) if config.save_dir else _P("artifacts/brains/actv1")
                tracker_state_file = state_dir / "chunk_tracker_state.json"
                tracker_obj = _ChunkTracker(state_file=tracker_state_file)
                if (not is_distributed) or (rank_id == 0):
                    write_jsonl({
                        "chunk_tracker": "enabled",
                        "state_file": str(tracker_state_file),
                    })
            except Exception as _e:
                if (not is_distributed) or (rank_id == 0):
                    write_jsonl({"chunk_tracker": "init_failed", "error": str(_e)})

        # Pre-claim the logical chunk before running the epoch
        samples_per_block = getattr(config, 'samples_per_block', 100000) or 100000
        dataset_chunk_size = getattr(config, 'dataset_chunk_size', 4000)
        chunks_per_block = max(1, (int(samples_per_block) + int(dataset_chunk_size) - 1) // int(dataset_chunk_size))
        if getattr(config, 'current_block_samples', None) is None:
            try:
                config.current_block_samples = 0
            except Exception:
                pass
        current_block_id = getattr(config, 'current_block_id', 0)
        chunk_already_trained = False
        try:
            if tracker_obj is not None:
                chunk_id_in_block = (int(getattr(config, 'current_block_samples', 0)) // int(dataset_chunk_size)) % chunks_per_block
                gpu_rank = getattr(config, '_tracker_rank_id', 0)
                # CRITICAL: Check if chunk was already trained (resume detection)
                can_claim = tracker_obj.claim_chunk(int(current_block_id), int(chunk_id_in_block), gpu_id=gpu_rank)
                if not can_claim:
                    chunk_already_trained = True
                    write_jsonl({
                        "event": "chunk_skipped",
                        "reason": "already_trained",
                        "block_id": int(current_block_id),
                        "chunk_id": int(chunk_id_in_block),
                        "note": "Chunk already trained in previous session - skipping"
                    })
        except Exception:
            pass

        # Train only if chunk wasn't already trained
        if not chunk_already_trained:
            steps_done, stopped_early, last_stop_reason = run_single_epoch_mode(
                execute_epoch, write_jsonl, checkpoint_saver, steps_done
            )
        else:
            # Skip training - keep steps_done at 0 (no new steps this session)
            # Finalization expects steps_done to be session-relative, not cumulative
            try:
                if tracker_obj is not None:
                    stats = tracker_obj.get_progress_stats()
                    cumulative_steps = stats.get("total_gpu_steps", 0)
                    write_jsonl({
                        "event": "resume_skip_complete",
                        "cumulative_steps": cumulative_steps,
                        "new_steps_this_session": 0,
                        "note": "Skipped already-trained chunk - no new steps in this session"
                    })
            except Exception:
                pass

        # Mark chunk complete after training (ONLY if we actually trained)
        if not chunk_already_trained:
            try:
                if tracker_obj is not None:
                    trained_samples = len(lines)
                    chunk_id_in_block = (int(getattr(config, 'current_block_samples', 0)) // int(dataset_chunk_size)) % chunks_per_block
                    gpu_rank = getattr(config, '_tracker_rank_id', 0)
                    is_rank_0 = getattr(config, '_tracker_is_rank_0', True)
                    tracker_obj.mark_chunk_complete(
                        block_id=int(current_block_id),
                        chunk_id=int(chunk_id_in_block),
                        gpu_id=gpu_rank,
                        step=int(steps_done),
                        samples_trained=int(trained_samples),
                    )
                    
                    # CRITICAL: Save checkpoint immediately after chunk completes (matching parallel behavior)
                    # steps_done here is session-relative (new steps trained this session)
                    if checkpoint_saver and is_rank_0:
                        checkpoint_saver.update_progress(steps_done, 0)
                        checkpoint_saver.save_checkpoint(
                            reason="chunk_complete",
                            step=steps_done,
                            cycle=0,
                        )
                    # Emit parallel-compatible stats payload (rank 0 only)
                    if is_rank_0:
                        try:
                            stats = tracker_obj.get_progress_stats()
                        except Exception:
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
                            "total_gpu_steps": stats.get("total_gpu_steps", int(steps_done)),
                            "total_chunks_trained": stats.get("total_chunks_trained", 0),
                            "blocks_completed": stats.get("blocks_completed", 0),
                            "current_epoch": stats.get("current_epoch", 0),
                        })
                    # Update samples within block and possibly mark block complete
                    try:
                        config.current_block_samples = int(getattr(config, 'current_block_samples', 0)) + int(trained_samples)
                    except Exception:
                        pass
                    if int(getattr(config, 'current_block_samples', 0)) >= int(samples_per_block):
                        try:
                            tracker_obj.mark_block_complete(int(current_block_id))
                        except Exception:
                            pass
                        # Advance logical block id and reset counters
                        try:
                            current_block_id = int(current_block_id) + 1
                            setattr(config, 'current_block_id', int(current_block_id))
                            config.current_block_samples = 0
                        except Exception:
                            pass
            except Exception:
                pass
    
    write_jsonl({
        "EXITED_TRAINING": True,
        "stopped_early": stopped_early,
        "steps_done": steps_done,
        "stop_reason": last_stop_reason
    })
    
    # Cleanup signal handlers
    checkpoint_saver.cleanup()
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    # Final evaluation
    if config.eval_file:
        try:
            write_jsonl({"final_evaluation": "running"})
            eval_once()
        except Exception as e:
            write_jsonl({"final_evaluation": "failed", "error": str(e)})
    
    # Generate memory report
    finalize_memory_report(memory_tracker, steps_done, stopped_early, batch_size, write_jsonl, write_jsonl)
    
    write_jsonl({"ABOUT_TO_CALL_FINALIZATION": True, "steps_done": steps_done})
    
    # Finalize training and save artifacts
    # CRITICAL: This must run even if training stopped early
    # Use try/except to ensure we attempt finalization no matter what
    final_payload = None
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
        write_jsonl({
            "finalize_training": "ERROR",
            "error": str(finalize_error),
            "traceback": str(__import__('traceback').format_exc()),
        })
        # Even if finalization fails, create a minimal payload
        # This ensures we don't lose all progress
        final_payload = {
            "trained": False,
            "error": str(finalize_error),
            "steps": steps_done,
            "stopped_early": stopped_early,
        }
        # Try to at least save the checkpoint even if brain.json fails
        try:
            from pathlib import Path as _FinalPath
            out_dir = _FinalPath(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = out_dir / "actv1_student.safetensors"
            tmp_path = out_dir / "actv1_student.safetensors.tmp"
            
            write_jsonl({"emergency_checkpoint_save": "attempting", "path": str(checkpoint_path)})
            
            # Get state dict (handle DDP)
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
                if isinstance(model_student, DDP):
                    state_dict = model_student.module.state_dict()
                else:
                    state_dict = model_student.state_dict()
            except Exception:
                state_dict = model_student.state_dict()
            
            # Save checkpoint
            from safetensors.torch import save_file as _save_safetensors
            _save_safetensors(state_dict, str(tmp_path))
            
            # Move to final location
            import os as _os
            if checkpoint_path.exists():
                backup_path = out_dir / "actv1_student.safetensors.emergency_backup"
                _os.replace(str(checkpoint_path), str(backup_path))
            _os.replace(str(tmp_path), str(checkpoint_path))
            
            write_jsonl({
                "emergency_checkpoint_save": "SUCCESS",
                "path": str(checkpoint_path),
                "size_mb": round(checkpoint_path.stat().st_size / (1024**2), 2),
            })
        except Exception as emergency_error:
            write_jsonl({
                "emergency_checkpoint_save": "FAILED",
                "error": str(emergency_error),
            })
    
    # Ensure we have a final_payload
    if final_payload is None:
        final_payload = {"trained": False, "error": "finalization_returned_none"}
    
    final_payload = _broadcast_final_payload(
        final_payload=final_payload,
        is_distributed=is_distributed,
        rank_id=rank_id,
        torch=torch
    )
    
    if (not is_distributed) or (rank_id == 0):
        write_jsonl(final_payload)
        write_jsonl({"event": "final", **final_payload})
    
    # Cleanup
    if inference_manager is not None:
        try:
            inference_manager.cleanup()
        except Exception as e:
            write_jsonl({"inference_manager_cleanup": "failed", "error": str(e)})
    
    if is_distributed and str(device_obj).startswith("cuda"):
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass
    
    # If chunk tracker was used, emit a parallel-compatible final stats event (rank 0 only)
    try:
        if 'tracker_obj' in locals() and tracker_obj is not None:
            # Only rank 0 saves tracker state and emits final stats
            if (not is_distributed) or (rank_id == 0):
                # Persist tracker state to allow seamless switching between modes
                try:
                    tracker_obj.save()
                except Exception:
                    pass
                final_stats = tracker_obj.get_progress_stats()
                write_jsonl({
                    "event": "training_complete",
                    "total_gpu_steps": final_stats.get("total_gpu_steps", steps_done),
                    "total_samples_trained": final_stats.get("total_samples_trained", 0),
                    "total_chunks_trained": final_stats.get("total_chunks_trained", 0),
                    "blocks_completed": final_stats.get("blocks_completed", 0),
                    "current_epoch": final_stats.get("current_epoch", 0),
                })
    except Exception:
        pass

    write_jsonl({
        "training_complete": True,
        "exit_code": 0,
        "stopped_early": stopped_early,
        "steps": steps_done
    })
