from __future__ import annotations

from typing import Optional, Any, Callable, Dict, Tuple, List


def resolve_device(device: str, strict: bool, torch) -> Tuple[str, Any, Any]:
    """Resolve device string and return (dev_str, device_obj, dml_device).

    Handles auto, cuda, dml with strict mode constraints.
    """
    dev = device
    dml_device = None
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            try:
                import torch_directml as _dml  # type: ignore
                _ = _dml.device()
                dev = "dml"
            except Exception:
                dev = "cpu"
    else:
        if str(dev).lower() == "cuda" and not torch.cuda.is_available():
            if strict:
                from rich import print
                import typer
                print({"error": "CUDA requested but not available", "hint": "Install CUDA PyTorch or choose --device cpu/dml", "device_request": "cuda"})
                raise typer.Exit(code=2)
            else:
                try:
                    from rich import print
                    print({"device_request": "cuda", "using": "cpu", "reason": "cuda_unavailable_or_not_compiled"})
                except Exception:
                    pass
                dev = "cpu"

    device_obj = None
    if dev == "dml":
        try:
            import torch_directml as _dml  # type: ignore
            dml_device = _dml.device()
            device_obj = dml_device
        except Exception:
            dev = "cpu"
            device_obj = torch.device(dev)
    else:
        device_obj = torch.device(dev)
    return dev, device_obj, dml_device


def get_training_lines(
    *,
    teacher_dataset: bool,
    dataset_file: Optional[str],
    ascii_only: bool,
    tok,
    teacher: Optional[str],
    teacher_device: str,
    strict: bool,
    td_num_samples: int,
    td_max_new_tokens: int,
    td_batch: int,
    td_temperature: float,
    td_top_p: float,
    td_top_k: int,
    td_prompt: Optional[str],
    td_seed: Optional[int],
    sys_mem_cap_pct: Optional[int],
    stop_file: Optional[str],
    is_distributed: bool,
    rank_id: int,
    torch,
    AutoModelForCausalLM,
    read_text_lines_sample_any: Callable[..., list[str]],
    load_teacher_for_gen: Callable[..., Tuple[Any, Any]],
    write_jsonl: Callable[[Dict[str, Any]], None],
    write_last_safe_batches: Callable[..., None],
) -> list[str]:
    """Load training lines from dataset or generate via teacher model.

    Encapsulates the large logic including DDP rank0 generation + file sync for other ranks.
    """
    from pathlib import Path as _Path
    from rich import print

    def _is_ascii(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except Exception:
            return False

    def _generate_lines() -> list[str]:
        lines_local: list[str] = []
        teacher_name = teacher or (tok.name_or_path if hasattr(tok, "name_or_path") else None)
        if not teacher_name:
            raise RuntimeError("teacher name not resolved")
        try:
            t_model_gen, device_for_gen = load_teacher_for_gen(
                teacher_name=teacher_name,
                teacher_device=teacher_device,
                strict=strict,
                torch=torch,
                AutoModelForCausalLM=AutoModelForCausalLM,
            )
        except Exception as e:
            print({"started": False, "error": f"cannot load teacher for dataset: {e}"})
            import typer
            raise typer.Exit(code=1)
        # Rank-aware device mapping for CUDA
        try:
            if getattr(device_for_gen, 'type', 'cpu') == 'cuda' and torch.cuda.is_available():
                pick_idx = int(rank_id) if is_distributed else getattr(device_for_gen, 'index', 0) or 0
                device_for_gen = torch.device("cuda", int(pick_idx))
                t_model_gen = t_model_gen.to(device_for_gen)  # type: ignore[misc]
        except Exception:
            pass
        try:
            if (not is_distributed) or (rank_id == 0):
                write_jsonl({"event": "gen_device", "device": str(device_for_gen), "rank": int(rank_id)})
        except Exception:
            pass
        max_new_cap = int(max(1, int(td_max_new_tokens)))

        def _gen_call(_inp, _mask, _max_new):
            max_new_eff = int(min(int(_max_new), int(max_new_cap)))
            return t_model_gen.generate(
                input_ids=_inp,
                attention_mask=_mask,
                max_new_tokens=max_new_eff,
                do_sample=True,
                temperature=float(td_temperature),
                top_p=float(td_top_p) if float(td_top_p) > 0 else 1.0,
                top_k=int(td_top_k) if int(td_top_k) > 0 else 0,
                pad_token_id=None,
                eos_token_id=tok.eos_token_id,
                use_cache=False,
            )

        if td_seed is not None:
            try:
                torch.manual_seed(int(td_seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(td_seed))
            except Exception:
                pass
        if td_prompt is not None and len(td_prompt) > 0:
            base_enc = tok(td_prompt, return_tensors="pt")
            base_ids = base_enc["input_ids"]
        else:
            pid = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id if tok.pad_token_id is not None else 0)
            import torch as _t
            base_ids = _t.tensor([[pid]], dtype=_t.long)
        base_ids = base_ids.to(device_for_gen)

        # Warmup probe
        try:
            if getattr(device_for_gen, 'type', 'cpu') == 'cuda' and torch.cuda.is_available():
                with torch.no_grad():
                    _inp = base_ids.expand(1, -1)
                    _mask = None
                    try:
                        import torch as _t
                        _mask = _t.ones_like(_inp, dtype=_t.bool)
                    except Exception:
                        _mask = None
                    _ = _gen_call(_inp, _mask, int(max(4, min(8, int(td_max_new_tokens)))))
                    torch.cuda.synchronize()
        except Exception as __e_warm:
            _msg = str(__e_warm).lower()
            if ("out of memory" in _msg or "cuda error" in _msg) and torch.cuda.is_available():
                if strict and str(teacher_device).lower() == "cuda":
                    print({"error": "CUDA OOM during teacher warmup in strict mode", "hint": "Lower --td-batch or --td-max-new-tokens"})
                    import typer
                    raise typer.Exit(code=4)
                else:
                    try:
                        device_for_gen = torch.device("cpu")
                        t_model_gen = t_model_gen.to(device_for_gen)  # type: ignore[misc]
                        base_ids = base_ids.to(device_for_gen)
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        remain = int(max(1, td_num_samples))
        gen_batch = int(max(1, td_batch))
        try:
            write_jsonl({"event": "gen_start", "total": int(td_num_samples)})
        except Exception:
            pass
        from math import ceil
        mem_cap_pct = None
        try:
            mem_cap_pct = int(sys_mem_cap_pct) if sys_mem_cap_pct is not None else None
        except Exception:
            mem_cap_pct = None

        def _sys_mem_used_pct() -> Optional[float]:
            try:
                import psutil  # type: ignore
                v = psutil.virtual_memory()
                return float(v.percent)
            except Exception:
                return None

        try:
            gen_batch = int(max(1, min(int(gen_batch), 64)))
        except Exception:
            gen_batch = int(max(1, gen_batch))
        iters = ceil(remain / gen_batch)
        generated = 0
        for _ in range(iters):
            if stop_file and isinstance(stop_file, str):
                try:
                    sp = _Path(stop_file)
                    if sp.exists():
                        try:
                            write_jsonl({"event": "stopped", "phase": "teacher_dataset", "generated": int(generated), "total": int(td_num_samples)})
                        except Exception:
                            pass
                        print({"stopped": True, "phase": "teacher_dataset", "generated": int(generated), "total": int(td_num_samples)})
                        lines_local = lines_local[:generated]
                        break
                except Exception:
                    pass
            if (getattr(device_for_gen, 'type', 'cpu') == 'cpu') and mem_cap_pct is not None and gen_batch > 1:
                used = _sys_mem_used_pct()
                if used is not None and used > float(mem_cap_pct):
                    gen_batch = max(1, int(gen_batch * 0.8))
                    write_last_safe_batches(td_bs=int(gen_batch))
            bs = min(gen_batch, remain)
            remain -= bs
            max_new_try = int(td_max_new_tokens)
            while True:
                try:
                    inp = base_ids.expand(bs, -1)
                    attn_mask = None
                    try:
                        import torch as _t
                        attn_mask = _t.ones_like(inp, dtype=_t.bool)
                    except Exception:
                        attn_mask = None
                    with torch.no_grad():
                        out = _gen_call(inp, attn_mask, int(max_new_try))
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or ("cuda error" in msg and "out of memory" in msg)) and (getattr(device_for_gen, 'type', 'cpu') == 'cuda') and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if bs > 1:
                            bs = max(1, bs // 2)
                            try:
                                write_jsonl({"event": "gen_oom_backoff", "new_bs": int(bs)})
                            except Exception:
                                pass
                            continue
                        if max_new_try > 16:
                            max_new_try = max(8, int(max_new_try * 0.7))
                            try:
                                write_jsonl({"event": "gen_oom_backoff_tokens", "max_new_tokens": int(max_new_try)})
                            except Exception:
                                pass
                            continue
                        if strict and str(teacher_device).lower() == "cuda":
                            print({"error": "Repeated CUDA OOM during teacher generation in strict mode", "hint": "Reduce --td-batch or --td-max-new-tokens"})
                            import typer
                            raise typer.Exit(code=5)
                        try:
                            device_for_gen = torch.device("cpu")
                            t_model_gen = t_model_gen.to(device_for_gen)  # type: ignore[misc]
                            base_ids = base_ids.to(device_for_gen)
                        except Exception:
                            pass
                        continue
                    raise
            texts = tok.batch_decode(out, skip_special_tokens=True)
            lines_local.extend(texts)
            generated += bs
            try:
                write_jsonl({"event": "gen_progress", "generated": int(generated), "total": int(td_num_samples)})
            except Exception:
                pass
        lines_local = [ln for ln in lines_local if ln and str(ln).strip()]
        if ascii_only:
            lines_local = [ln for ln in lines_local if _is_ascii(str(ln))]
        try:
            write_jsonl({"event": "gen_done", "generated": int(generated), "total": int(td_num_samples)})
        except Exception:
            pass
        write_last_safe_batches(td_bs=int(gen_batch))
        try:
            if getattr(device_for_gen, 'type', 'cpu') == 'cuda' and torch.cuda.is_available():
                try:
                    del t_model_gen  # type: ignore[name-defined]
                except Exception:
                    pass
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            pass
        return lines_local

    if teacher_dataset:
        if is_distributed:
            sync_dir = _Path("artifacts/brains/actv1/_ddp")
            sync_dir.mkdir(parents=True, exist_ok=True)
            sync_file = sync_dir / "teacher_lines.rank0.txt"
            err_file = sync_dir / "teacher_lines.ERROR.json"
            if rank_id == 0:
                try:
                    lines = _generate_lines()
                    with sync_file.open("w", encoding="utf-8") as f:
                        for ln in lines:
                            f.write(str(ln).replace("\n", " ").strip() + "\n")
                except BaseException as __gen_e:
                    try:
                        import json as __json
                        payload = {"error": str(__gen_e)}
                        err_file.write_text(__json.dumps(payload), encoding="utf-8")
                    except Exception:
                        pass
                    raise
            else:
                import time as __wtime
                waited = 0.0
                last_emit = 0.0
                while not (sync_file.exists() or err_file.exists()):
                    if stop_file and isinstance(stop_file, str):
                        try:
                            sp = _Path(stop_file)
                            if sp.exists():
                                break
                        except Exception:
                            pass
                    __wtime.sleep(0.5)
                    waited += 0.5
                    if (waited - last_emit) >= 5.0:
                        last_emit = waited
                        try:
                            print({"event": "waiting_for_teacher_lines", "rank": int(rank_id), "waited_s": int(waited)})
                        except Exception:
                            pass
                    if waited > 900 and bool(strict):
                        print({"error": "Timeout waiting for rank0 teacher lines in strict mode"})
                        import typer
                        raise typer.Exit(code=8)
                if err_file.exists():
                    try:
                        import json as __json
                        payload = __json.loads(err_file.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {"error": "teacher generation failed on rank0"}
                    try:
                        print(payload)
                    except Exception:
                        pass
                    import typer
                    raise typer.Exit(code=int(payload.get("exit_code") or 9))
                try:
                    with sync_file.open("r", encoding="utf-8") as f:
                        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
                except Exception:
                    lines = []
        else:
            lines = _generate_lines()
    else:
        if not dataset_file:
            print({"started": False, "error": "no dataset provided; use --dataset-file"})
            import typer
            raise typer.Exit(code=1)
        lines = read_text_lines_sample_any(dataset_file, max_lines=4000)
        lines = [ln for ln in lines if ln and str(ln).strip()]
        if ascii_only:
            lines = [ln for ln in lines if _is_ascii(str(ln))]
    return lines


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

    out_dir = _Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_path = out_dir / "actv1_student.pt"
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
        _tch.save(model_student.state_dict(), str(tmp_path))
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


def encode_lines(tok, lines: List[str], max_seq_len: int):
    """Tokenize lines to (input_ids, labels) applying ignore_index for padding."""
    enc = tok(
        lines,
        padding="max_length",
        truncation=True,
        max_length=int(max_seq_len),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    try:
        if tok.pad_token_id is not None:
            labels[enc["attention_mask"] == 0] = -100
    except Exception:
        pass
    return input_ids, labels


def init_cuda_distributed_if_needed(
    *,
    dev: str,
    is_distributed: bool,
    torch,
    os,
    rank_id: int,
    world_sz: int,
    init_file_env: Optional[str],
):
    """Set per-rank CUDA device and initialize process group when needed.

    Returns (device_obj, ddp_initialized: bool).
    """
    device_obj = (torch.device(dev))
    ddp_initialized = False
    if is_distributed and str(device_obj) == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(rank_id))
            device_obj = torch.device("cuda", int(rank_id))
            import torch.distributed as dist
            import platform as __pf
            backend = os.environ.get("AIOS_DDP_BACKEND", "nccl")
            if __pf.system().lower() == "windows" and backend == "nccl":
                backend = "gloo"
            try:
                if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
                    dist.init_process_group(backend=backend, init_method="env://", world_size=int(world_sz), rank=int(rank_id))
                elif init_file_env:
                    dist.init_process_group(backend=backend, init_method=f"file://{init_file_env}", world_size=int(world_sz), rank=int(rank_id))
                else:
                    dist.init_process_group(backend=backend, world_size=int(world_sz), rank=int(rank_id))
                ddp_initialized = True
            except Exception:
                backend = "gloo"
                if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
                    dist.init_process_group(backend=backend, init_method="env://", world_size=int(world_sz), rank=int(rank_id))
                elif init_file_env:
                    dist.init_process_group(backend=backend, init_method=f"file://{init_file_env}", world_size=int(world_sz), rank=int(rank_id))
                else:
                    dist.init_process_group(backend=backend, world_size=int(world_sz), rank=int(rank_id))
                ddp_initialized = True
            try:
                from rich import print
                print({"ddp_worker": bool(ddp_initialized), "rank": int(rank_id), "world": int(world_sz), "backend": backend})
            except Exception:
                pass
        except Exception as _e_init:
            try:
                from rich import print
                print({"ddp_worker": False, "error": f"dist.init failed: {_e_init}"})
            except Exception:
                pass
    return device_obj, ddp_initialized


def load_teacher_kl_model(
    *,
    teacher_name: Optional[str],
    teacher_device: str,
    strict: bool,
    AutoModelForCausalLM,
    torch,
):
    """Load teacher model for KL regularization; returns model or None.

    Handles CUDA/DML/CPU placement with strict behavior.
    """
    if not teacher_name:
        return None
    try:
        try:
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name, use_safetensors=True)
        except Exception:
            try:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)
            except Exception:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name, from_tf=True)
        if str(teacher_device).lower() == "auto":
            t_dev_obj = torch.device("cpu")
        elif str(teacher_device).lower() == "cuda" and torch.cuda.is_available():
            t_dev_obj = torch.device("cuda")
        elif str(teacher_device).lower() == "dml":
            try:
                import torch_directml as _dml  # type: ignore
                t_dev_obj = _dml.device()
            except Exception:
                t_dev_obj = torch.device("cpu")
        else:
            t_dev_obj = torch.device("cpu")
        try:
            teacher_model.to(t_dev_obj)  # type: ignore[misc]
        except Exception as _t_move_e:
            if strict and str(teacher_device).lower() == "cuda":
                from rich import print
                import typer
                print({"error": f"Failed moving teacher KL model to CUDA in strict mode: {_t_move_e}"})
                raise typer.Exit(code=7)
            else:
                raise
        teacher_model.eval()
        return teacher_model
    except Exception:
        return None


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
