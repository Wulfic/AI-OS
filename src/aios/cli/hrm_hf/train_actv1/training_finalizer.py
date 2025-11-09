"""
Training Finalizer Module for ACT-V1 Training

Handles post-training cleanup and finalization:
- Final evaluation execution
- Memory profiling and reporting
- Model checkpoint saving
- Brain metadata generation
- Distributed payload broadcasting
- Process group cleanup
- Inference manager cleanup

This module ensures proper cleanup and generates final training outputs.
"""

from typing import Any, Optional, Callable
import torch


def finalize_training(
    model_student: Any,
    optimizer: Any,
    memory_tracker: Any,
    eval_ids: Optional[Any],
    eval_labels: Optional[Any],
    eval_file: Optional[str],
    batch_size: int,
    device_obj: Any,
    dml_device: Any,
    halt_max_steps: int,
    eval_batches: int,
    segment_rollout: Callable,
    write_jsonl: Callable,
    tokenizer: Any,
    steps_done: int,
    stopped_early: bool,
    last_stop_reason: Optional[str],
    is_distributed: bool,
    rank_id: int,
    save_dir: str,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    num_heads: int,
    expansion: float,
    h_cycles: int,
    l_cycles: int,
    pos_encodings: str,
    log_file: Optional[str],
    brain_name: Optional[str],
    model: str,
    max_seq_len: int,
    default_goal: Optional[str],
    dataset_file: str,
    config: Any,  # TrainingConfig
    cycle: int,
    iterate: bool,
    inference_manager: Optional[Any],
) -> dict[str, Any]:
    """
    Finalize training and generate outputs.
    
    This function handles:
    1. Final evaluation on eval set
    2. Memory profiling report generation
    3. Model checkpoint saving
    4. Brain metadata creation
    5. Distributed payload broadcasting
    6. Cleanup of distributed resources
    7. Inference manager cleanup
    
    Args:
        model_student: Trained model
        optimizer: Training optimizer
        memory_tracker: Memory tracking object
        eval_ids, eval_labels: Evaluation data tensors
        eval_file: Path to evaluation file
        batch_size: Final batch size
        device_obj: PyTorch device
        dml_device: DirectML device
        halt_max_steps: Maximum adaptive computation steps
        eval_batches: Number of evaluation batches
        segment_rollout: Chunked training function
        write_jsonl: JSONL logging function
        tokenizer: Tokenizer object
        steps_done: Total training steps completed
        stopped_early: Whether training stopped early
        last_stop_reason: Reason for stopping
        is_distributed: Whether using distributed training
        rank_id: Process rank
        save_dir: Save directory
        h_layers, l_layers, hidden_size, num_heads, expansion, h_cycles, l_cycles, pos_encodings:
            Model architecture parameters
        log_file: JSONL log file path
        brain_name: Brain name
        model: Model/tokenizer identifier
        max_seq_len: Maximum sequence length
        default_goal: Default training goal
        dataset_file: Training dataset path
        config: Training configuration
        cycle: Final training cycle
        iterate: Whether iterate mode was used
        inference_manager: Multi-GPU inference manager
        
    Returns:
        Dictionary containing final training results:
        - trained: Whether training succeeded
        - steps: Total steps completed
        - saved_to: Save directory
        - Additional metadata and configuration
    """
    
    # Import helper functions
    from ..helpers import (
        _eval_once_helper,
        _finalize_training_helper,
        _broadcast_final_payload_helper,
        _write_last_safe_batches_helper,
    )
    
    # ============================================================================
    # Final Evaluation
    # ============================================================================
    if eval_file:
        try:
            write_jsonl({"event": "final_evaluation_start"})
            
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
                write_jsonl=write_jsonl,
                tokenizer=tokenizer,
                enable_english_logic_eval=True,
            )
            
            write_jsonl({"event": "final_evaluation_complete"})
        except Exception as e:
            write_jsonl({"event": "final_evaluation_error", "error": str(e)})
    
    # ============================================================================
    # Memory Profiling Report
    # ============================================================================
    try:
        memory_tracker.snapshot('training_complete', metadata={
            'steps_completed': steps_done,
            'stopped_early': stopped_early,
        })
        memory_report = memory_tracker.get_report()
        print({"memory_profile_report": memory_report})
        write_jsonl({"event": "memory_profile", "report": memory_report})
        
        # Log current memory state
        final_memory = memory_tracker.log_current('final_state', {
            'total_steps': steps_done,
            'batch_size': batch_size,
        })
        print({"final_memory_state": final_memory})
    except Exception as e:
        print({"memory_report_error": str(e)})
    
    print({
        "ABOUT_TO_CALL_FINALIZATION": True,
        "steps_done": steps_done,
        "stopped_early": stopped_early
    })
    
    # ============================================================================
    # Model Checkpoint Saving
    # ============================================================================
    # CRITICAL: Always call finalization, even if training crashed
    try:
        final_payload = _finalize_training_helper(
            model_student=model_student,
            save_dir=save_dir,
            stopped_early=stopped_early,
            steps_done=steps_done,
            is_distributed=is_distributed,
            rank_id=rank_id,
            tok=tokenizer,
            h_layers=h_layers,
            l_layers=l_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
            pos_encodings=pos_encodings,
            log_file=log_file,
            write_jsonl=write_jsonl,
            brain_name=brain_name,
            model=model,
            max_seq_len=max_seq_len,
            halt_max_steps=halt_max_steps,
            default_goal=default_goal,
            dataset_file=dataset_file,
            # MoE configuration
            use_moe=config.use_moe,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_capacity_factor=config.moe_capacity_factor,
            stop_reason=last_stop_reason,
            iterate_cycle=cycle if iterate else None,
        )
    except Exception as finalize_error:
        import traceback
        print({
            "finalize_training": "ERROR",
            "error": str(finalize_error),
            "traceback": str(traceback.format_exc()),
        })
        # Set error flag but continue to cleanup
        final_payload = {"trained": False, "error": str(finalize_error)}
    
    # ============================================================================
    # Distributed Payload Broadcasting
    # ============================================================================
    final_payload = _broadcast_final_payload_helper(
        final_payload=final_payload,
        is_distributed=is_distributed,
        rank_id=rank_id,
        torch=torch,
    )
    
    if (not is_distributed) or (rank_id == 0):
        print(final_payload)
        write_jsonl({"event": "final", **final_payload})
    
    try:
        _write_last_safe_batches_helper(train_bs=int(batch_size))
    except Exception:
        pass
    
    # ============================================================================
    # Cleanup Resources
    # ============================================================================
    # Clean up inference manager
    if inference_manager is not None:
        try:
            inference_manager.cleanup()
        except Exception as e:
            print({"inference_manager_cleanup": "error", "message": str(e)})
    
    # Clean up process group if initialized
    try:
        if is_distributed and (str(device_obj) == "cuda"):
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
    except Exception:
        pass
    
    # ============================================================================
    # Return Final Payload
    # ============================================================================
    return final_payload
