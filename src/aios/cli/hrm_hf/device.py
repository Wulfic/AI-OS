from __future__ import annotations

from typing import Any, Tuple


def resolve_device(device: str, strict: bool, torch) -> Tuple[str, Any, Any]:
    """Resolve device string and return (dev_str, device_obj, dml_device).

    Handles auto, cuda, xpu, dml with strict mode constraints.
    Device priority for auto: cuda > xpu > dml > cpu
    """
    dev = device
    dml_device = None
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = "xpu"
        else:
            try:
                import torch_directml as _dml  # type: ignore
                _ = _dml.device()
                dev = "dml"
            except Exception:
                dev = "cpu"
    else:
        # Validate requested device
        if str(dev).lower() == "cuda" and not torch.cuda.is_available():
            if strict:
                from rich import print
                import typer
                print({"error": "CUDA requested but not available", "hint": "Install CUDA PyTorch or choose --device cpu/xpu/dml", "device_request": "cuda"})
                raise typer.Exit(code=2)
            else:
                try:
                    from rich import print
                    print({"device_request": "cuda", "using": "cpu", "reason": "cuda_unavailable_or_not_compiled"})
                except Exception:
                    pass
                dev = "cpu"
        elif str(dev).lower() == "xpu":
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                if strict:
                    from rich import print
                    import typer
                    print({
                        "error": "XPU requested but not available",
                        "hint": "Install intel-extension-for-pytorch or choose --device cpu/cuda/dml",
                        "device_request": "xpu"
                    })
                    raise typer.Exit(code=2)
                else:
                    try:
                        from rich import print
                        print({"device_request": "xpu", "using": "cpu", "reason": "xpu_unavailable_or_ipex_not_installed"})
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
                # DEBUG: Print actual environment variables
                _p({"ddp_env_check": {
                    "MASTER_ADDR": os.environ.get("MASTER_ADDR", "NOT_SET"),
                    "MASTER_PORT": os.environ.get("MASTER_PORT", "NOT_SET"),
                    "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "NOT_SET"),
                    "init_file_env": init_file_env
                }})
            except Exception:
                pass
            
            try:
                # CRITICAL: Disable libuv for Windows PyTorch builds that don't support it
                # Must be set BEFORE any torch.distributed imports
                os.environ["USE_LIBUV"] = "0"  # Force set, not just default

                # CRITICAL FIXES for Windows/gloo:
                # 1) Force loopback interface to avoid invalid hostnames like 'kubernetes.docker.internal'
                # 2) Prefer file:// init when spawn provided AIOS_DDP_INIT_FILE to avoid poisoned env vars
                # 3) Clean any Docker/Kubernetes env vars that poison hostname resolution
                
                # Remove poisoned environment variables that cause hostname resolution issues
                poisoned_vars = [
                    "KUBERNETES_SERVICE_HOST",
                    "KUBERNETES_SERVICE_PORT",
                    "KUBERNETES_PORT", 
                    "DOCKER_HOST",
                ]
                for var in poisoned_vars:
                    if var in os.environ:
                        try:
                            del os.environ[var]
                        except Exception:
                            pass
                
                if os.name == 'nt':  # Windows
                    # For Windows + gloo: Don't set GLOO_SOCKET_IFNAME explicitly
                    # Let gloo auto-detect the interface when using 127.0.0.1
                    # Setting interface names like "Loopback Pseudo-Interface 1" causes errors
                    # Only override if not already set by user
                    if "GLOO_SOCKET_IFNAME" not in os.environ:
                        # Leave unset for auto-detection, or use empty string
                        pass  # Let gloo auto-detect based on MASTER_ADDR
                    # NOTE: Do NOT set MASTER_ADDR/MASTER_PORT here for Windows
                    # We will use file:// init method which doesn't need these env vars
                    # Setting them causes "makeDeviceForHostname(): unsupported gloo device" error

                # Detect Docker Desktop hostname pollution (for info only)
                # Docker Desktop adds kubernetes.docker.internal to hosts file
                docker_desktop_detected = False
                try:
                    import socket
                    try:
                        socket.gethostbyname('kubernetes.docker.internal')
                        docker_desktop_detected = True
                    except socket.gaierror:
                        pass
                except Exception:
                    pass
                
                # WINDOWS FILE-BASED INITIALIZATION:
                # As of PyTorch v1.7+, Windows supports gloo backend with FileStore instead of TCPStore.
                # TCPStore requires libuv which Windows PyTorch builds don't have.
                # Official docs: https://pytorch.org/docs/stable/distributed.html
                # File-based init_method is the ONLY officially supported method on Windows.
                
                import torch.distributed as dist
                use_file_init = (os.name == 'nt' and backend == 'gloo')
                
                if use_file_init:
                    # Windows + gloo: MUST use file:// init_method (FileStore)
                    # Create a temporary file for rendezvous
                    import tempfile
                    import pathlib
                    
                    # Use provided init file or create temporary one
                    if init_file_env:
                        init_path = pathlib.Path(init_file_env)
                    else:
                        # Create in user's temp directory with unique name for this session
                        temp_dir = pathlib.Path(tempfile.gettempdir()) / "aios_ddp"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        # Use port to make file unique per session
                        port = os.environ.get("MASTER_PORT", "29500")
                        init_path = temp_dir / f"ddp_init_{port}.tmp"
                    
                    # Ensure parent directory exists
                    init_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Clean up stale file from previous runs (rank 0 only)
                    if init_path.exists() and int(rank_id) == 0:
                        try:
                            init_path.unlink()
                        except Exception:
                            pass
                    
                    # Convert Windows path to file:// URL format
                    # Must use forward slashes and triple slash for local file
                    file_url = init_path.as_posix()
                    if file_url.startswith('/'):
                        init_method = f"file://{file_url}"
                    else:
                        # Windows absolute path (e.g., C:/...)
                        init_method = f"file:///{file_url}"
                    
                    # CRITICAL: Remove MASTER_ADDR/MASTER_PORT environment variables
                    # These cause gloo to attempt hostname resolution even with file:// init
                    # This is the root cause of "makeDeviceForHostname(): unsupported gloo device"
                    env_vars_removed = []
                    for env_var in ["MASTER_ADDR", "MASTER_PORT"]:
                        if env_var in os.environ:
                            try:
                                del os.environ[env_var]
                                env_vars_removed.append(env_var)
                            except Exception:
                                pass
                    
                    try:
                        from rich import print as _p
                        _p({
                            "ddp_init_method": "file_store (Windows)",
                            "rank": int(rank_id),
                            "init_file": str(init_path),
                            "file_url": init_method,
                            "note": "Using FileStore - official Windows gloo method",
                            "env_vars_removed": env_vars_removed
                        })
                    except Exception:
                        pass
                    
                    # Initialize with file:// method (uses FileStore internally)
                    dist.init_process_group(
                        backend=backend,
                        init_method=init_method,
                        world_size=int(world_sz),
                        rank=int(rank_id),
                        timeout=timeout
                    )
                    ddp_initialized = True
                    
                    try:
                        from rich import print as _p
                        _p({
                            "ddp": "init_successful",
                            "backend": backend,
                            "rank": int(rank_id),
                            "world_size": int(world_sz),
                            "method": "FileStore",
                            "init_file": str(init_path)
                        })
                    except Exception:
                        pass
                    
                elif docker_desktop_detected:
                    # Docker Desktop workaround: Use env:// with forced localhost
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                    
                    try:
                        from rich import print as _p
                        _p({
                            "ddp_init_method": "env (Docker workaround)",
                            "rank": int(rank_id),
                            "note": "Docker Desktop detected - using localhost"
                        })
                    except Exception:
                        pass
                    
                    dist.init_process_group(
                        backend=backend,
                        init_method="env://",
                        world_size=int(world_sz),
                        rank=int(rank_id),
                        timeout=timeout
                    )
                    ddp_initialized = True
                    
                else:
                    # Standard initialization for Linux/NCCL
                    # Choose init method: if we have an init file from our spawn, use it
                    if init_file_env:
                        # Ensure init file directory exists and is writable
                        import pathlib
                        init_path = pathlib.Path(init_file_env)
                        init_path.parent.mkdir(parents=True, exist_ok=True)
                        # Clean up stale init file from previous runs (rank 0 only)
                        if init_path.exists() and int(rank_id) == 0:
                            try:
                                init_path.unlink()
                            except Exception:
                                pass
                        init_method = f"file://{init_file_env}"
                    else:
                        # Fall back to env://
                        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                        os.environ.setdefault("MASTER_PORT", "29500")
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
                # Retry with different backend if available
                try:
                    from rich import print as _p
                    _p({"ddp_init_retry": True, "rank": int(rank_id), "error": str(e1)[:150]})
                except Exception:
                    pass
                
                # If gloo failed and NCCL is available, try NCCL
                if backend == "gloo" and torch.cuda.is_available():
                    try:
                        backend = "nccl"
                        os.environ["MASTER_ADDR"] = "127.0.0.1"
                        os.environ["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
                        
                        from rich import print as _p
                        _p({"ddp_backend_fallback": "nccl", "rank": int(rank_id), "note": "Trying NCCL backend"})
                        
                        dist.init_process_group(
                            backend=backend,
                            init_method="env://",
                            world_size=int(world_sz),
                            rank=int(rank_id),
                            timeout=timeout
                        )
                        ddp_initialized = True
                    except Exception as e2:
                        from rich import print as _p
                        _p({"ddp_nccl_failed": True, "error": str(e2)[:100]})
                        raise e1  # Raise original error
                else:
                    raise
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
                import traceback
                print({
                    "ddp_worker": False, 
                    "error": f"dist.init failed: {_e_init}",
                    "note": "DDP initialization failed",
                    "rank": int(rank_id),
                    "details": "All automatic workarounds attempted (TCPStore, NCCL fallback)",
                    "suggestion": "Check firewall settings or try single GPU: --cuda-ids 0",
                    "traceback": traceback.format_exc()[-500:]  # Last 500 chars of traceback
                })
            except Exception:
                pass
            
            # Check if we should abort (strict mode or explicit flag)
            strict_mode = str(os.environ.get("AIOS_DDP_STRICT", "0")) == "1"
            abort_on_fail = str(os.environ.get("AIOS_DDP_ABORT_ON_FAIL", "0")) == "1"
            
            if strict_mode or abort_on_fail:
                # In DDP worker mode, exit with error code instead of continuing
                # This prevents duplicate independent training
                if is_distributed and int(rank_id) >= 0:
                    try:
                        from rich import print
                        print({
                            "ddp_worker_exit": True,
                            "rank": int(rank_id),
                            "reason": "DDP init failed in strict mode",
                            "exit_code": 1
                        })
                    except Exception:
                        pass
                    import sys
                    sys.exit(1)
                # If not in worker mode, raise exception
                raise _e_init
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
