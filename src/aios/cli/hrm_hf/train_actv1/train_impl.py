"""Main training implementation orchestrator.

Thin orchestrator that coordinates all training modules.
This module delegates to specialized modules for each phase of training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import print

if TYPE_CHECKING:
    from aios.core.hrm_training.training_config import TrainingConfig


def train_actv1_impl(config: "TrainingConfig") -> None:
    """Train the ACT V1 HRM model or expert module.
    
    This is the main entry point that orchestrates the entire training pipeline.
    It delegates to specialized modules for each phase:
    - Expert training (if expert_id specified)
    - Config processing and LR adjustment
    - Resume detection
    - Model building and setup
    - Training execution
    - Finalization and checkpointing
    
    Args:
        config: Training configuration object containing all parameters
    """
    # Import sub-modules
    from .expert_training import train_expert_only
    from .config_processing import extract_and_process_config
    from .resume_handler import detect_resume_checkpoint
    
    # Expert-specific training mode: train standalone FeedForward expert
    if config.expert_id:
        return train_expert_only(config)
    
    # Standard full HRM model training
    print({"training_mode": "full_hrm_actv1", "model": config.model})
    
    # Phase 1: Process configuration and apply MoE LR adjustment
    config_values, adjusted_lr = extract_and_process_config(config)
    
    # Phase 2: Detect resume checkpoint
    step_offset, resume_cycle, resume_session = detect_resume_checkpoint(
        resume=config.resume,
        brain_name=config.brain_name,
        bundle_dir=config.bundle_dir,
        dataset_file=config.dataset_file
    )
    
    # Phase 3-6: Main training pipeline (fully modularized)
    # The remaining functionality has been extracted into 4 modules:
    # - model_builder.py: Device setup, tokenizer, dataset, model instantiation
    # - model_optimizer.py: Quantization, PEFT, DeepSpeed, optimizer setup
    # - training_executor.py: Training loop, iterate mode, monitoring
    # - training_finalizer.py: Evaluation, checkpointing, metadata
    
    # Extract all config values needed for module calls
    device = config.device
    strict = config.strict
    bundle_dir = config.bundle_dir
    brain_name = config.brain_name
    model = config.model
    student_init = config.student_init
    save_dir = config.save_dir
    log_file = config.log_file
    dataset_file = config.dataset_file
    ascii_only = config.ascii_only
    read_text_lines_sample_any = config.read_text_lines_sample_any
    max_seq_len = config.max_seq_len
    batch_size = config.batch_size
    steps = config.steps
    eval_file = config.eval_file
    stop_file = config.stop_file
    sys_mem_cap_pct = config.sys_mem_cap_pct
    eval_batches = config.eval_batches
    
    # Phase 3: Model and Data Setup
    from .model_builder import setup_model_and_data
    
    setup_result = setup_model_and_data(
        config=config,
        device=device,
        strict=strict,
        bundle_dir=bundle_dir,
        brain_name=brain_name,
        model=model,
        student_init=student_init,
        save_dir=save_dir,
        log_file=log_file,
        dataset_file=dataset_file,
        ascii_only=ascii_only,
        read_text_lines_sample_any=read_text_lines_sample_any,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        steps=steps,
        eval_file=eval_file,
        stop_file=stop_file,
    )
    
    # Check if we should exit early (DDP parent process)
    if setup_result.get("should_exit"):
        return
    
    # Phase 4: Model Optimization
    from .model_optimizer import setup_optimization
    
    optimization_result = setup_optimization(
        model_student=setup_result["model_student"],
        config=config,
        device_obj=setup_result["device_obj"],
        dev=setup_result["dev"],
        dml_device=setup_result["dml_device"],
        is_distributed=setup_result["is_distributed"],
        rank_id=setup_result["rank_id"],
        world_sz=setup_result["world_sz"],
        init_file_env=setup_result["init_file_env"],
        batch_size=setup_result["batch_size"],
        max_seq_len=max_seq_len,
        steps=steps,
        hidden_size=config.hidden_size,
        h_layers=config.h_layers,
        l_layers=config.l_layers,
        lr=adjusted_lr,  # Use adjusted LR from config processing
        out_dir_path=setup_result["out_dir_path"],
        save_dir=setup_result["save_dir"],
        tokenizer=setup_result["tokenizer"],
        h_cycles=config.h_cycles,
        l_cycles=config.l_cycles,
        expansion=config.expansion,
        num_heads=config.num_heads,
        pos_encodings=config.pos_encodings,
        halt_max_steps=config.halt_max_steps,
        window_size=config.window_size,
        vocab_size=setup_result["vocab_size"],
    )
    
    # Phase 5: Training Execution
    from .training_executor import execute_training
    
    training_result = execute_training(
        model_student=optimization_result["model_student"],
        optimizer=optimization_result["optimizer"],
        segment_rollout=optimization_result["segment_rollout"],
        device_obj=optimization_result["device_obj"],
        dml_device=setup_result["dml_device"],
        input_ids=setup_result["input_ids"],
        labels=setup_result["labels"],
        batch_size=optimization_result["batch_size"],
        steps=steps,
        halt_max_steps=config.halt_max_steps,
        sys_mem_cap_pct=sys_mem_cap_pct,
        dev=setup_result["dev"],
        is_distributed=setup_result["is_distributed"],
        world_sz=setup_result["world_sz"],
        stop_file=stop_file,
        write_jsonl=setup_result["write_jsonl"],
        should_stop=setup_result["should_stop"],
        load_or_generate_lines=setup_result["load_or_generate_lines"],
        lines=setup_result["lines"],
        streaming_dataset=setup_result["streaming_dataset"],
        use_streaming=setup_result["use_streaming"],
        cycle_count=setup_result["cycle_count"],
        tokenizer=setup_result["tokenizer"],
        max_seq_len=max_seq_len,
        use_amp=optimization_result["use_amp"],
        scaler=optimization_result["scaler"],
        deepspeed_engine=optimization_result["deepspeed_engine"],
        inference_manager=optimization_result["inference_manager"],
        hot_reload_steps=config.hot_reload_steps,
        warmup_steps=setup_result["warmup_steps"],
        base_lr=adjusted_lr,
        config=config,
        step_offset=step_offset,
        iterate=config.iterate,
        resume_cycle=resume_cycle,
    )
    
    # Phase 6: Training Finalization
    from .training_finalizer import finalize_training
    
    final_payload = finalize_training(
        model_student=optimization_result["model_student"],
        optimizer=optimization_result["optimizer"],
        memory_tracker=optimization_result["memory_tracker"],
        eval_ids=setup_result["eval_ids"],
        eval_labels=setup_result["eval_labels"],
        eval_file=eval_file,
        batch_size=training_result["batch_size"],
        device_obj=optimization_result["device_obj"],
        dml_device=setup_result["dml_device"],
        halt_max_steps=config.halt_max_steps,
        eval_batches=eval_batches,
        segment_rollout=optimization_result["segment_rollout"],
        write_jsonl=setup_result["write_jsonl"],
        tokenizer=setup_result["tokenizer"],
        steps_done=training_result["steps_done"],
        stopped_early=training_result["stopped_early"],
        last_stop_reason=training_result["last_stop_reason"],
        is_distributed=setup_result["is_distributed"],
        rank_id=setup_result["rank_id"],
        save_dir=setup_result["save_dir"],
        h_layers=config.h_layers,
        l_layers=config.l_layers,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        expansion=config.expansion,
        h_cycles=config.h_cycles,
        l_cycles=config.l_cycles,
        pos_encodings=config.pos_encodings,
        log_file=setup_result["log_file"],
        brain_name=brain_name,
        model=model,
        max_seq_len=max_seq_len,
        default_goal=config.default_goal,
        dataset_file=dataset_file,
        config=config,
        cycle=training_result["cycle"],
        iterate=config.iterate,
        inference_manager=optimization_result["inference_manager"],
    )
    
    return final_payload
