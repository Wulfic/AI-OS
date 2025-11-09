from __future__ import annotations

from typing import Any, Tuple


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


def init_cuda_distributed_if_needed(
    *,
    dev: str,
    is_distributed: bool,
    torch,
    os,
    rank_id: int,
    world_sz: int,
    init_file_env: str | None,
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
            from datetime import timedelta
            
            backend = os.environ.get("AIOS_DDP_BACKEND", "nccl")
            if __pf.system().lower() == "windows" and backend == "nccl":
                backend = "gloo"
                
            # Configure timeout based on optimization mode
            timeout_sec = 1800  # default 30 minutes
            try:
                if os.environ.get("AIOS_DDP_OPTIMIZE_MODE") == "1":
                    timeout_sec = int(os.environ.get("AIOS_DDP_TIMEOUT_SEC", "30"))
                else:
                    timeout_sec = int(os.environ.get("AIOS_DDP_TIMEOUT_SEC", "1800"))
            except Exception:
                pass
            timeout = timedelta(seconds=timeout_sec)
            
            # Pre-print to show we're attempting init
            try:
                from rich import print as _p
                _p({"ddp_init_starting": True, "rank": int(rank_id), "world": int(world_sz), "backend": backend})
            except Exception:
                pass
            
            try:
                # Prefer TCP-based init for better Windows compatibility
                # Disable libuv for Windows PyTorch builds that don't support it
                os.environ.setdefault("USE_LIBUV", "0")
                
                if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
                    init_method = "env://"
                elif init_file_env:
                    # Ensure init file directory exists and is writable
                    import pathlib
                    init_path = pathlib.Path(init_file_env)
                    init_path.parent.mkdir(parents=True, exist_ok=True)
                    # Clean up stale init file from previous runs
                    if init_path.exists() and rank_id == 0:
                        try:
                            init_path.unlink()
                        except Exception:
                            pass
                    init_method = f"file://{init_file_env}"
                else:
                    # Fallback: set up localhost TCP for Windows
                    if rank_id == 0:
                        os.environ["MASTER_ADDR"] = "127.0.0.1"
                        os.environ["MASTER_PORT"] = "29500"
                    init_method = "env://"
                
                dist.init_process_group(
                    backend=backend, 
                    init_method=init_method,
                    world_size=int(world_sz), 
                    rank=int(rank_id),
                    timeout=timeout
                )
                ddp_initialized = True
            except Exception as e1:
                # Fallback to gloo with TCP
                try:
                    from rich import print as _p
                    _p({"ddp_init_retry": True, "rank": int(rank_id), "error": str(e1)[:100]})
                except Exception:
                    pass
                backend = "gloo"
                # Set up localhost TCP as fallback
                if not os.environ.get("MASTER_ADDR"):
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = "29500"
                # Disable libuv for retry as well
                os.environ["USE_LIBUV"] = "0"
                dist.init_process_group(
                    backend=backend, 
                    init_method="env://",
                    world_size=int(world_sz), 
                    rank=int(rank_id),
                    timeout=timeout
                )
                ddp_initialized = True
            try:
                from rich import print
                optimize_mode = os.environ.get("AIOS_DDP_OPTIMIZE_MODE") == "1"
                print({
                    "ddp_worker": bool(ddp_initialized), 
                    "rank": int(rank_id), 
                    "world": int(world_sz), 
                    "backend": backend,
                    "timeout_sec": timeout_sec,
                    "optimize_mode": optimize_mode
                })
            except Exception:
                pass
        except Exception as _e_init:
            try:
                from rich import print
                print({"ddp_worker": False, "error": f"dist.init failed: {_e_init}"})
            except Exception:
                pass
    return device_obj, ddp_initialized


def get_dist_context(os):
    """Parse environment variables to derive distributed context.

    Returns (is_distributed: bool, rank_id: int, world_sz: int, init_file_env: Optional[str]).
    Supports torchrun env (LOCAL_RANK/RANK, WORLD_SIZE) and our spawn env (AIOS_DDP_*).
    """
    rank_env = os.environ.get("AIOS_DDP_RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    world_env = os.environ.get("AIOS_DDP_WORLD") or os.environ.get("WORLD_SIZE")
    init_file_env = os.environ.get("AIOS_DDP_INIT_FILE")
    is_distributed = False
    rank_id = 0
    world_sz = 1
    try:
        if rank_env is not None and world_env is not None:
            rank_id = int(rank_env)
            world_sz = max(1, int(world_env))
            is_distributed = world_sz > 1
    except Exception:
        is_distributed = False
        rank_id = 0
        world_sz = 1
    return is_distributed, rank_id, world_sz, init_file_env
