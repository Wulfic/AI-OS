from __future__ import annotations

from typing import Optional, Any, Callable, Dict


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
    teacher: Optional[str] = None,
    model: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    halt_max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Save artifacts and return final payload; DDP-safe (rank0 writes)."""
    from pathlib import Path as _Path
    from rich import print
    import json as _json
    import time as _t
    import os as _os

    out_dir = _Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_path = out_dir / "actv1_student.pt"
    # Always proceed with normal save operations
    if stopped_early:
        try:
            write_jsonl({"event": "final", "trained": False, "stopped": True, "steps": int(steps_done)})
        except Exception:
            pass
        if (not is_distributed) or (rank_id == 0):
            print({"trained": False, "stopped": True, "steps": int(steps_done)})
        return {"trained": False, "steps": int(steps_done)}

    # Save only on rank 0
    if (not is_distributed) or (rank_id == 0):
        tmp_path = out_dir / "actv1_student.pt.tmp"
        prev_path = out_dir / "actv1_student.pt.prev"
        import torch as _tch
        
        # Handle DDP-wrapped models: unwrap to get the actual model state dict
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(model_student, DDP):
                state_dict = model_student.module.state_dict()
            else:
                state_dict = model_student.state_dict()
        except Exception:
            state_dict = model_student.state_dict()
        
        _tch.save(state_dict, str(tmp_path))
        try:
            try:
                prev_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                pass
            if student_path.exists():
                import os as _os
                _os.replace(str(student_path), str(prev_path))
            import os as _os
            _os.replace(str(tmp_path), str(student_path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
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
            meta.update({
                "name": str(brain_name) if brain_name else meta.get("name") or None,
                "type": "actv1",
                "student_pt": "actv1_student.pt",
                "created_at": created_at,
                "last_trained": float(_t.time()),
                "tokenizer_model": str(model or getattr(tok, 'name_or_path', None) or ''),
                "teacher_model": str(teacher) if teacher else meta.get("teacher_model") or None,
                "max_seq_len": int(max_seq_len) if isinstance(max_seq_len, int) else meta.get("max_seq_len") or 0,
                "halt_max_steps": int(halt_max_steps) if isinstance(halt_max_steps, int) else meta.get("halt_max_steps") or 0,
                "log_file": str(_Path(log_file).name if log_file else "metrics.jsonl"),
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

    final_payload = {"trained": True, "steps": int(steps_done), "student": str(student_path)}
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
