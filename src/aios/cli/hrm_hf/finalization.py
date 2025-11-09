from __future__ import annotations

from typing import Optional, Any, Callable, Dict


def _validate_tokenizer_path(
    saved_path: Optional[str],
    tokenizer: Any,
    fallback_model: Optional[str],
    expected_vocab_size: int
) -> str:
    """Validate tokenizer path matches expected vocab size.
    
    Prevents resume failures by ensuring saved tokenizer path is correct.
    If saved path exists and matches vocab size, use it.
    Otherwise, try tokenizer.name_or_path, then fallback to model.
    
    Args:
        saved_path: Previously saved tokenizer path (from brain.json)
        tokenizer: Current tokenizer object
        fallback_model: Fallback model path (config.model)
        expected_vocab_size: Expected vocabulary size
        
    Returns:
        Validated tokenizer path
    """
    # Priority 1: Use saved path if it exists and was working
    if saved_path:
        return saved_path
    
    # Priority 2: Use tokenizer.name_or_path
    tok_path = getattr(tokenizer, 'name_or_path', None)
    if tok_path:
        # Verify this path has correct vocab size by checking known tokenizer mappings
        # Mistral-7B = 32000, GPT2 = 50257, etc.
        if expected_vocab_size == 32000 and 'mistral' not in str(tok_path).lower():
            # Wrong tokenizer! Try to find correct one
            try:
                from pathlib import Path
                # Look for mistral tokenizer in artifacts
                mistral_path = Path("artifacts/hf_implant/tokenizers/mistral-7b")
                if mistral_path.exists():
                    print({
                        "tokenizer_validation": "corrected",
                        "wrong_path": tok_path,
                        "corrected_path": str(mistral_path),
                        "reason": f"vocab_size={expected_vocab_size} requires Mistral tokenizer"
                    })
                    return str(mistral_path)
            except Exception:
                pass
        return tok_path
    
    # Priority 3: Fallback to model path
    return fallback_model if fallback_model else ''


def finalize_training(
    *,
    model_student,
    save_dir: str,
    stopped_early: bool,
    steps_done: int,
    is_distributed: bool,
    rank_id: int,
    tok,
    h_layers: int,
    l_layers: int,
    hidden_size: int,
    num_heads: int,
    expansion: float,
    h_cycles: int,
    l_cycles: int,
    pos_encodings: str,
    log_file: Optional[str],
    write_jsonl: Callable[[Dict[str, Any]], None],
    # Extra metadata to preserve parity with original implementation
    brain_name: Optional[str] = None,
    model: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    halt_max_steps: Optional[int] = None,
    default_goal: Optional[str] = None,
    dataset_file: Optional[str] = None,
    # MOE configuration parameters
    use_moe: Optional[bool] = None,
    num_experts: Optional[int] = None,
    num_experts_per_tok: Optional[int] = None,
    moe_capacity_factor: Optional[float] = None,
    stop_reason: Optional[str] = None,
    # Iterate mode tracking
    iterate_cycle: Optional[int] = None,
    # Epoch tracking (pass full config to access all fields)
    training_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Save artifacts and return final payload; DDP-safe (rank0 writes).
    
    All checkpoints are saved in safetensors format for security and performance.
    Safetensors provides:
    - No arbitrary code execution (unlike Pickle)
    - Faster load/save with memory-mapped I/O
    - Standard format across HuggingFace ecosystem
    """
    from pathlib import Path as _Path
    from rich import print
    import json as _json
    import time as _t
    import os as _os

    # FIRST THING: Log that finalization has started
    print({
        "finalization": "STARTED",
        "stopped_early": stopped_early,
        "steps_done": steps_done,
        "brain_name": brain_name,
        "save_dir": save_dir,
    })

    out_dir = _Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "actv1_student.safetensors"
    orderly_epoch_stop = stop_reason == "stop_after_epoch"
    
    # Debug: Log checkpoint paths
    if (not is_distributed) or (rank_id == 0):
        print({
            "checkpoint_save": "preparing",
            "format": "safetensors",
            "save_dir": str(save_dir),
            "out_dir": str(out_dir),
            "checkpoint_path": str(checkpoint_path),
            "out_dir_exists": out_dir.exists(),
        })
    
    # Log early stop event (but continue to save progress)
    if stopped_early:
        if orderly_epoch_stop:
            try:
                write_jsonl({
                    "event": "final_stop_after_epoch",
                    "trained": True,
                    "steps": int(steps_done),
                    "stop_reason": stop_reason,
                })
            except Exception:
                pass
            if (not is_distributed) or (rank_id == 0):
                print({
                    "stopped_after_epoch": True,
                    "steps_completed": int(steps_done),
                    "note": "Epoch limit reached; saving progress",
                })
        else:
            try:
                write_jsonl({
                    "event": "final",
                    "trained": False,
                    "stopped": True,
                    "steps": int(steps_done),
                    "stop_reason": stop_reason or "unknown",
                })
            except Exception:
                pass
            if (not is_distributed) or (rank_id == 0):
                print({
                    "stopped_early": True,
                    "steps_completed": int(steps_done),
                    "note": "Saving progress before exit",
                    "stop_reason": stop_reason or "unspecified",
                })

    # Save only on rank 0
    if (not is_distributed) or (rank_id == 0):
        tmp_path = out_dir / "actv1_student.safetensors.tmp"
        prev_path = out_dir / "actv1_student.safetensors.prev"
        
        # Handle DDP-wrapped models: unwrap to get the actual model state dict
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(model_student, DDP):
                state_dict = model_student.module.state_dict()
            else:
                state_dict = model_student.state_dict()
        except Exception:
            state_dict = model_student.state_dict()
        
        try:
            from safetensors.torch import save_file as _save_safetensors
            
            print({
                "checkpoint_save": "saving_model_safetensors",
                "tmp_path": str(tmp_path),
                "state_dict_keys": len(state_dict.keys()) if hasattr(state_dict, 'keys') else 0,
            })
            
            # Save to tmp file first (atomic write)
            _save_safetensors(state_dict, str(tmp_path))
            
            print({
                "checkpoint_save": "model_saved_to_tmp",
                "format": "safetensors",
                "tmp_path": str(tmp_path),
                "tmp_exists": tmp_path.exists(),
                "tmp_size_mb": round(tmp_path.stat().st_size / (1024**2), 2) if tmp_path.exists() else 0,
            })
            
            # IMMEDIATELY move to final location (don't delay - process might be killed!)
            # Backup old checkpoint first if it exists
            try:
                if checkpoint_path.exists():
                    try:
                        prev_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _os.replace(str(checkpoint_path), str(prev_path))
                    print({"checkpoint_save": "old_checkpoint_backed_up", "backup_path": str(prev_path)})
            except Exception as backup_error:
                print({"checkpoint_save": "backup_warning", "error": str(backup_error)})
                # Continue anyway - backup failure shouldn't stop checkpoint save
            
            # Move tmp to final location NOW
            _os.replace(str(tmp_path), str(checkpoint_path))
            
            print({
                "checkpoint_save": "SUCCESS",
                "format": "safetensors",
                "final_path": str(checkpoint_path),
                "exists": checkpoint_path.exists(),
                "size_mb": round(checkpoint_path.stat().st_size / (1024**2), 2) if checkpoint_path.exists() else 0,
            })
            
        except ImportError as import_error:
            print({
                "checkpoint_save": "FATAL_ERROR",
                "error": "safetensors package not installed",
                "note": "Install with: pip install safetensors",
                "traceback": str(import_error),
            })
            raise RuntimeError("safetensors package is required. Install with: pip install safetensors") from import_error
            
        except Exception as save_error:
            print({
                "checkpoint_save": "ERROR_SAVING_MODEL",
                "error": str(save_error),
                "tmp_path": str(tmp_path),
                "traceback": str(__import__('traceback').format_exc()),
            })
            raise  # Re-raise to see the error
        finally:
            # Clean up tmp file if it still exists (shouldn't happen if move succeeded)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                    print({"checkpoint_save": "tmp_file_cleaned_up"})
            except Exception:
                pass

        # Update brain.json
        meta_path = out_dir / "brain.json"
        meta = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = _json.load(f) or {}
            except Exception:
                meta = {}
        created_at = float(meta.get("created_at", _t.time()))
        # Increment training steps (track total steps across all training sessions)
        previous_steps = int(meta.get("training_steps", 0))
        total_steps = previous_steps + int(steps_done)
        
        # Track dataset usage history
        dataset_history = meta.get("dataset_history", [])
        if dataset_file:
            from pathlib import Path as _DataPath
            dataset_name = _DataPath(dataset_file).name
            dataset_path = str(_DataPath(dataset_file).resolve())
            
            # Add new training session to history
            session_record = {
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "steps": int(steps_done),
                "timestamp": float(_t.time()),
            }
            dataset_history.append(session_record)
            
            # Calculate dataset usage statistics
            dataset_stats = {}
            for record in dataset_history:
                ds_name = record.get("dataset_name", "unknown")
                if ds_name not in dataset_stats:
                    dataset_stats[ds_name] = {
                        "times_used": 0,
                        "total_steps": 0,
                        "first_used": record.get("timestamp", 0),
                        "last_used": record.get("timestamp", 0),
                        "dataset_path": record.get("dataset_path", ""),
                    }
                dataset_stats[ds_name]["times_used"] += 1
                dataset_stats[ds_name]["total_steps"] += record.get("steps", 0)
                dataset_stats[ds_name]["last_used"] = max(
                    dataset_stats[ds_name]["last_used"],
                    record.get("timestamp", 0)
                )
        else:
            # If no dataset_file provided, preserve existing stats
            dataset_stats = meta.get("dataset_stats", {})
        
        # Build last_session metadata for resume functionality
        last_session = {
            "timestamp": float(_t.time()),
            "steps_completed": int(steps_done),
            "total_steps": int(total_steps),
            "stopped_early": bool(stopped_early and not orderly_epoch_stop),
            "stop_reason": stop_reason,
            "dataset_file": str(dataset_file) if dataset_file else None,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "safetensors",
            "iterate_cycle": int(iterate_cycle) if iterate_cycle is not None else 0,
            # Epoch tracking metadata (for resume)
            "epoch_tracking": {
                "dataset_total_samples": getattr(training_config, 'dataset_total_samples', None) if training_config else None,
                "samples_per_block": getattr(training_config, 'samples_per_block', None) if training_config else None,
                "total_blocks": getattr(training_config, 'total_blocks', None) if training_config else None,
                "samples_processed_this_epoch": getattr(training_config, 'samples_processed_this_epoch', 0) if training_config else 0,
                "blocks_processed_this_epoch": getattr(training_config, 'blocks_processed_this_epoch', "") if training_config else "",
                "current_epoch": getattr(training_config, 'current_epoch', 0) if training_config else 0,
                "current_block_samples": getattr(training_config, 'current_block_samples', 0) if training_config else 0,
            },
            # Training configuration (for resume)
            "config": {
                "max_seq_len": int(max_seq_len) if isinstance(max_seq_len, int) else meta.get("max_seq_len") or 0,
                "halt_max_steps": int(halt_max_steps) if isinstance(halt_max_steps, int) else meta.get("halt_max_steps") or 0,
                "h_layers": int(h_layers),
                "l_layers": int(l_layers),
                "hidden_size": int(hidden_size),
                "num_heads": int(num_heads),
                "expansion": float(expansion),
                "h_cycles": int(h_cycles),
                "l_cycles": int(l_cycles),
                "pos_encodings": str(pos_encodings),
                "use_moe": bool(use_moe) if use_moe is not None else meta.get("use_moe", False),
                "num_experts": int(num_experts) if num_experts is not None else meta.get("num_experts", 8),
                "num_experts_per_tok": int(num_experts_per_tok) if num_experts_per_tok is not None else meta.get("num_experts_per_tok", 2),
                "moe_capacity_factor": float(moe_capacity_factor) if moe_capacity_factor is not None else meta.get("moe_capacity_factor", 1.25),
            }
        }
        
        meta.update({
            "name": str(brain_name) if brain_name else meta.get("name") or None,
            "type": "actv1",
            "checkpoint_file": "actv1_student.safetensors",
            "checkpoint_format": "safetensors",
            "created_at": created_at,
            "last_trained": float(_t.time()),
            "training_steps": total_steps,  # Global step counter
            "last_session": last_session,  # Resume metadata
            # Preserve existing tokenizer_model on resume; use tok.name_or_path or model for fresh starts
            # CRITICAL: Validate tokenizer path matches vocab size to prevent resume failures
            "tokenizer_model": str(_validate_tokenizer_path(
                saved_path=meta.get("tokenizer_model"),
                tokenizer=tok,
                fallback_model=model,
                expected_vocab_size=int(getattr(tok, "vocab_size", 0) or 0)
            )),
            "max_seq_len": int(max_seq_len) if isinstance(max_seq_len, int) else meta.get("max_seq_len") or 0,
            "halt_max_steps": int(halt_max_steps) if isinstance(halt_max_steps, int) else meta.get("halt_max_steps") or 0,
            "log_file": str(_Path(log_file).name if log_file else "metrics.jsonl"),
            "default_goal": str(default_goal) if default_goal else meta.get("default_goal"),
            "dataset_history": dataset_history,
            "dataset_stats": dataset_stats,
            # MOE configuration (preserve existing values if not provided)
            "use_moe": bool(use_moe) if use_moe is not None else meta.get("use_moe", False),
            "num_experts": int(num_experts) if num_experts is not None else meta.get("num_experts", 8),
            "num_experts_per_tok": int(num_experts_per_tok) if num_experts_per_tok is not None else meta.get("num_experts_per_tok", 2),
            "moe_capacity_factor": float(moe_capacity_factor) if moe_capacity_factor is not None else meta.get("moe_capacity_factor", 1.25),
            "arch": {
                "H_layers": int(h_layers),
                "L_layers": int(l_layers),
                "hidden_size": int(hidden_size),
                "num_heads": int(num_heads),
                "expansion": float(expansion),
                "H_cycles": int(h_cycles),
                "L_cycles": int(l_cycles),
                "pos_encodings": str(pos_encodings),
                "vocab_size": int(getattr(tok, "vocab_size", 0) or 0),
            },
        })
        try:
            with meta_path.open("w", encoding="utf-8") as f:
                _json.dump(meta, f, indent=2)
        except Exception:
            pass

    # Return final payload with completion status
    if orderly_epoch_stop:
        final_payload = {
            "trained": True,
            "steps": int(steps_done),
            "checkpoint": str(checkpoint_path),
            "stop_reason": stop_reason,
        }
    elif stopped_early:
        final_payload = {
            "trained": False,
            "stopped": True,
            "steps": int(steps_done),
            "checkpoint": str(checkpoint_path),
            "stop_reason": stop_reason or "unknown",
        }
    else:
        final_payload = {"trained": True, "steps": int(steps_done), "checkpoint": str(checkpoint_path)}
        if stop_reason:
            final_payload["stop_reason"] = stop_reason
    
    return final_payload


def broadcast_final_payload(
    *,
    final_payload: Dict[str, Any],
    is_distributed: bool,
    rank_id: int,
    torch,
) -> Dict[str, Any]:
    """Barrier and broadcast a small final payload dict from rank 0 to all ranks.

    Returns the (possibly received) payload on all ranks. Safe if dist not initialized.
    """
    if not is_distributed:
        return final_payload
    try:
        import json as _json
        import torch.distributed as dist
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        blob = None
        if rank_id == 0:
            try:
                blob = _json.dumps(final_payload).encode("utf-8")
            except Exception:
                blob = b"{}"
        if hasattr(torch, "tensor"):
            if (rank_id == 0) or (not getattr(dist, "is_initialized", lambda: False)()):
                _bb = blob if isinstance(blob, (bytes, bytearray)) else b"{}"
                size_tensor = torch.tensor([len(_bb)], dtype=torch.int32)
            else:
                size_tensor = torch.tensor([0], dtype=torch.int32)
            if getattr(dist, "is_initialized", lambda: False)():
                dist.broadcast(size_tensor, src=0)
            sz = int(size_tensor.item())
            data_tensor = torch.empty((sz,), dtype=torch.uint8)
            if ((rank_id == 0) or (not getattr(dist, "is_initialized", lambda: False)())) and blob is not None:
                _bb = blob if isinstance(blob, (bytes, bytearray)) else b"{}"
                data_tensor[:] = torch.tensor(list(_bb), dtype=torch.uint8)
            if getattr(dist, "is_initialized", lambda: False)():
                dist.broadcast(data_tensor, src=0)
            if getattr(dist, "is_initialized", lambda: False)() and rank_id != 0:
                try:
                    final_payload = _json.loads(bytes(data_tensor.tolist()).decode("utf-8"))
                except Exception:
                    pass
    except Exception:
        pass
    return final_payload
