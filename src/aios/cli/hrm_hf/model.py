from __future__ import annotations

from typing import Dict, Any, Optional, Callable


def build_student(
    cfg: Dict[str, Any],
    *,
    student_init: Optional[str] = None,
    build_act_v1=None,
    print_fn: Optional[Callable[[dict], None]] = None,
):
    """Build ACT V1 student model, optionally load weights, and print param stats.

    - cfg: dict passed to build_act_v1
    - student_init: optional .safetensors path or brain bundle dir to load from
    - build_act_v1: injectable constructor to avoid import-time cycles
    - print_fn: optional printer (defaults to rich.print)
    
    Loading behavior:
    - Loads from safetensors format only (secure, fast, memory-mapped)
    - If student_init is a directory (brain bundle), uses checkpoint_file from brain.json
    - If student_init is a file, loads directly (must be .safetensors)
    
    Note: All checkpoints are now saved in safetensors format for security and performance.
    """
    if build_act_v1 is None:
        from aios.core.hrm_models import build_act_v1 as _build
        build_act_v1 = _build
    model_student = build_act_v1(cfg)
    
    if student_init:
        try:
            from pathlib import Path
            from safetensors.torch import load_file as _load_safetensors
            
            # Determine checkpoint path
            student_path = Path(student_init)
            checkpoint_path = None
            
            # If it's a directory (brain bundle), load from brain.json metadata
            if student_path.is_dir():
                try:
                    import json
                    brain_json = student_path / "brain.json"
                    if brain_json.exists():
                        with open(brain_json, 'r') as f:
                            metadata = json.load(f)
                        
                        # Get checkpoint file from metadata
                        checkpoint_file = metadata.get("checkpoint_file", "actv1_student.safetensors")
                        checkpoint_path = student_path / checkpoint_file
                        
                        if not checkpoint_path.exists():
                            # Fallback to default name if specified file doesn't exist
                            checkpoint_path = student_path / "actv1_student.safetensors"
                    else:
                        # No brain.json, use default name
                        checkpoint_path = student_path / "actv1_student.safetensors"
                        
                except Exception as meta_error:
                    if print_fn:
                        try:
                            print_fn({"checkpoint_metadata_warning": str(meta_error)})
                        except Exception:
                            pass
                    checkpoint_path = student_path / "actv1_student.safetensors"
            
            # If it's a file, use it directly
            elif student_path.is_file():
                checkpoint_path = student_path
            
            # Load the checkpoint
            if checkpoint_path and checkpoint_path.exists():
                try:
                    sd = _load_safetensors(str(checkpoint_path), device="cpu")
                    missing, unexpected = model_student.load_state_dict(sd, strict=False)
                    if print_fn:
                        try:
                            print_fn({
                                "loaded_student": True,
                                "format": "safetensors",
                                "path": str(checkpoint_path),
                                "missing": list(missing),
                                "unexpected": list(unexpected),
                            })
                        except Exception:
                            pass
                except Exception as load_error:
                    error_str = str(load_error)
                    if print_fn:
                        try:
                            print_fn({
                                "checkpoint_load": "FAILED",
                                "format": "safetensors",
                                "path": str(checkpoint_path),
                                "error": error_str,
                            })
                        except Exception:
                            pass
                    
                    # If checkpoint is corrupted, delete it and start fresh
                    if "incomplete metadata" in error_str or "not fully covered" in error_str:
                        if print_fn:
                            try:
                                print_fn({
                                    "checkpoint_recovery": "corrupted_checkpoint_detected",
                                    "path": str(checkpoint_path),
                                    "action": "removing and starting fresh",
                                    "message": "Checkpoint file was corrupted (incomplete write). Starting with fresh model.",
                                })
                            except Exception:
                                pass
                        try:
                            checkpoint_path.unlink()
                            if print_fn:
                                print_fn({
                                    "checkpoint_recovery": "corrupted_file_removed",
                                    "path": str(checkpoint_path),
                                })
                        except Exception as unlink_error:
                            if print_fn:
                                print_fn({
                                    "checkpoint_recovery": "failed_to_remove",
                                    "path": str(checkpoint_path),
                                    "error": str(unlink_error),
                                })
                        # Don't raise - just start with fresh model
                        return model_student
                    
                    # For other errors, raise them
                    raise
            else:
                if print_fn:
                    try:
                        print_fn({
                            "loaded_student": False,
                            "error": "Checkpoint file not found",
                            "path": str(checkpoint_path) if checkpoint_path else str(student_init),
                            "note": "Ensure checkpoint exists and is in safetensors format"
                        })
                    except Exception:
                        pass
                        
        except ImportError as import_error:
            error_msg = "safetensors package is required. Install with: pip install safetensors"
            if print_fn:
                try:
                    print_fn({
                        "loaded_student": False,
                        "error": error_msg,
                        "path": str(student_init)
                    })
                except Exception:
                    pass
            raise RuntimeError(error_msg) from import_error
            
        except Exception as e:
            if print_fn:
                try:
                    print_fn({"loaded_student": False, "error": str(e), "path": str(student_init)})
                except Exception:
                    pass
            raise
            
    try:
        total_params = sum(p.numel() for p in model_student.parameters())
        trainable_params = sum(p.numel() for p in model_student.parameters() if p.requires_grad)
        if print_fn:
            try:
                print_fn({
                    "params": {"total": int(total_params), "trainable": int(trainable_params)}
                })
            except Exception:
                pass
    except Exception:
        pass
    return model_student


def build_actv1_config(
    *,
    batch_size: int,
    max_seq_len: int,
    vocab_size: int,
    h_cycles: int,
    l_cycles: int,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    expansion: float,
    num_heads: int,
    pos_encodings: str,
    halt_max_steps: int,
    use_flash_attn: bool = False,
    use_gradient_checkpointing: bool = False,
    window_size: int | None = None,
    use_moe: bool = True,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    moe_capacity_factor: float = 1.25,
) -> Dict[str, Any]:
    """Return the config dict expected by build_act_v1."""
    config = dict(
        batch_size=int(batch_size),
        seq_len=int(max_seq_len),
        num_puzzle_identifiers=4,
        puzzle_emb_ndim=0,
        vocab_size=int(vocab_size),
        H_cycles=int(h_cycles),
        L_cycles=int(l_cycles),
        H_layers=int(h_layers),
        L_layers=int(l_layers),
        hidden_size=int(hidden_size),
        expansion=float(expansion),
        num_heads=int(num_heads),
        pos_encodings=str(pos_encodings),
        halt_max_steps=int(halt_max_steps),
        halt_exploration_prob=0.1,
        forward_dtype="float32",
        use_flash_attn=bool(use_flash_attn),
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
        use_moe=bool(use_moe),
        num_experts=int(num_experts),
        num_experts_per_tok=int(num_experts_per_tok),
        moe_capacity_factor=float(moe_capacity_factor),
    )
    if window_size is not None:
        config["window_size"] = int(window_size)
    return config
