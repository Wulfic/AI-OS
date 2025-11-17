from __future__ import annotations

from typing import Optional, Callable, Dict, Any


def maybe_spawn_and_exit_if_parent(
    *,
    ddp: bool,
    device: str,
    torch,
    os,
    platform,
    world_size: Optional[int],
    strict: bool,
    worker_target: Callable[[int, int, str, Dict[str, Any]], None],
    spawn_kwargs: Dict[str, Any],
) -> bool:
    """Handle multi-GPU DDP initialization with support for both torchrun and internal spawn.

    Supports two modes:
    1. External launcher (torchrun/torch.distributed.launch): Detects env vars and continues
    2. Internal spawn: Spawns worker processes if AIOS_DDP_SPAWN=1 is set
    
    For GUI/wrapper compatibility, set AIOS_DDP_SPAWN=1 to enable internal spawn.
    For command-line usage, use torchrun or our simple_ddp_launcher.py script.
    
    Returns:
        True if parent process spawned workers (should exit)
        False if worker process or no spawning needed (continue to training)
    """
    try:
        # Prevent recursive spawning in worker processes
        if os.environ.get("AIOS_DDP_WORKER") == "1":
            return False  # Already in a spawned worker, don't spawn again
        
        if bool(ddp) and str(device).lower() in ("auto", "cuda") and torch.cuda.is_available():
            # Infer world_size from cuda_ids if not explicitly set
            if world_size is None or world_size <= 0:
                cuda_ids_str = spawn_kwargs.get("cuda_ids")
                if cuda_ids_str:
                    try:
                        cuda_ids_list = [x.strip() for x in str(cuda_ids_str).split(",") if x.strip()]
                        world_size = len(cuda_ids_list)
                    except Exception:
                        pass
                if not world_size or world_size <= 0:
                    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
            # Detect if we're already under a distributed launcher (torchrun)
            under_torchrun = False
            rank_env = None
            world_size_env = None
            
            try:
                rank_env = os.environ.get("RANK")
                world_size_env = os.environ.get("WORLD_SIZE")
                local_rank_env = os.environ.get("LOCAL_RANK")
                
                if rank_env is not None or local_rank_env is not None:
                    under_torchrun = True
                elif os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
                    # Could be pre-configured for spawn, check if RANK is set
                    if rank_env is None:
                        under_torchrun = False  # Probably spawn mode setup
                    else:
                        under_torchrun = True
            except Exception:
                under_torchrun = False
            
            if under_torchrun:
                # Running under torchrun or already in worker process - continue training
                from rich import print
                ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                ws = int(world_size_env) if world_size_env else world_size or ngpus
                rank = int(rank_env) if rank_env else 0
                print({
                    "ddp": "external_launcher_detected",
                    "rank": rank,
                    "world_size": ws,
                    "device_count": ngpus,
                    "mode": "torchrun"
                })
                return False  # Continue to training code
            
            # Check if internal spawn is requested (for GUI/wrapper compatibility)
            enable_spawn = os.environ.get("AIOS_DDP_SPAWN") == "1"
            
            if enable_spawn and world_size and int(world_size) > 1:
                # Internal spawn mode - spawn worker processes
                try:
                    from torch import multiprocessing as _mp
                    from pathlib import Path as _Path
                    from rich import print
                    
                    sys_name = platform.system()
                    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    ws = int(world_size)
                    
                    import tempfile
                    import time
                    
                    # Use unique temp file to avoid race conditions
                    init_dir = _Path(tempfile.gettempdir()) / "aios_ddp" / f"spawn_{int(time.time() * 1000)}"
                    init_dir.mkdir(parents=True, exist_ok=True)
                    init_file = str((init_dir / "init").absolute())
                    
                    # Ensure clean state
                    init_path = _Path(init_file)
                    if init_path.exists():
                        try:
                            init_path.unlink()
                            time.sleep(0.1)  # Brief delay for filesystem
                        except Exception:
                            pass
                    
                    # Backend selection
                    # Windows: Must use gloo (NCCL is Linux-only)
                    # Linux: Prefer nccl for CUDA, gloo as fallback
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
                    
                    # On Windows, force gloo regardless of CUDA availability
                    if sys_name.lower() == "windows":
                        backend = "gloo"
                    
                    # Allow override via environment variable
                    if os.environ.get("AIOS_DDP_BACKEND"):
                        backend = os.environ.get("AIOS_DDP_BACKEND")
                    
                    # CRITICAL: Clean environment of poisoned variables BEFORE spawning
                    # Docker Desktop and WSL2 can set MASTER_ADDR to kubernetes.docker.internal
                    # which causes gloo to fail with "makeDeviceForHostname: unsupported gloo device"
                    poisoned_vars = [
                        "KUBERNETES_SERVICE_HOST",
                        "KUBERNETES_SERVICE_PORT", 
                        "KUBERNETES_PORT",
                        "DOCKER_HOST",
                    ]
                    for var in poisoned_vars:
                        if var in os.environ:
                            del os.environ[var]
                    
                    # Set up clean environment for workers
                    # Force localhost for all ranks
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                    os.environ["AIOS_DDP_BACKEND"] = backend
                    os.environ["AIOS_DDP_INIT_FILE"] = init_file  # Pass init file path to workers
                    
                    # Set up progress file for monitoring
                    import tempfile
                    progress_file = os.path.join(tempfile.gettempdir(), f"aios_ddp_progress_{os.getpid()}.json")
                    os.environ["AIOS_DDP_PROGRESS_FILE"] = progress_file
                    
                    # CRITICAL: Set USE_LIBUV=0 BEFORE any torch imports in child processes
                    # PyTorch Windows builds don't support libuv - must disable early
                    os.environ["USE_LIBUV"] = "0"
                    
                    os.environ["AIOS_DDP_WORKER"] = "1"  # Mark as spawned worker process
                    os.environ["AIOS_DDP_STRICT"] = "1" if strict else "0"  # Pass strict mode to workers
                    
                    # CRITICAL: Force gloo to use TCP transport with IP addresses only
                    # Prevents reverse DNS that resolves 127.0.0.1 -> kubernetes.docker.internal from hosts file
                    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
                    os.environ["TP_VERBOSE"] = "0"
                    
                    # Don't set GLOO_SOCKET_IFNAME - let gloo auto-detect based on MASTER_ADDR
                    # Setting to empty string can cause interface detection issues
                    if "GLOO_SOCKET_IFNAME" in os.environ:
                        del os.environ["GLOO_SOCKET_IFNAME"]
                    
                    print({"ddp": "spawning_workers", "world_size": ws, "backend": backend, "device_count": ngpus})
                    
                    # Start progress monitor thread for GUI updates
                    import threading
                    import json
                    stop_monitor = threading.Event()
                    
                    def monitor_progress():
                        """Monitor progress file and print JSON updates for GUI consumers."""
                        last_step = -1
                        while not stop_monitor.is_set():
                            try:
                                if os.path.exists(progress_file):
                                    with open(progress_file, 'r', encoding='utf-8') as pf:
                                        data = json.load(pf)
                                    step = int(data.get("step") or 0)
                                    if step > last_step:
                                        last_step = step
                                        payload = {
                                            "event": "train",
                                            "step": step,
                                            "total_steps": data.get("total_steps"),
                                            "loss": data.get("loss"),
                                            "ddp": True,
                                            "rank": 0,
                                            "source": "progress_monitor"
                                        }
                                        print(json.dumps(payload), flush=True)
                            except Exception:
                                pass
                            stop_monitor.wait(0.5)  # Check every 500ms
                    
                    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                    monitor_thread.start()
                    
                    ctx = _mp.get_context("spawn")
                    procs = []
                    
                    for r in range(ws):
                        p = ctx.Process(
                            target=worker_target,
                            args=(r, ws, init_file, dict(spawn_kwargs)),
                            name=f"aios-ddp-rank{r}"
                        )
                        p.daemon = False
                        p.start()
                        procs.append(p)
                        print({"ddp": "worker_started", "rank": r, "pid": p.pid})
                    
                    # Wait for all workers with timeout
                    # First, give workers brief time to initialize DDP
                    # If any worker fails in first 30 seconds, likely a DDP init issue
                    import signal
                    import time
                    
                    ddp_init_timeout = 30  # seconds
                    training_timeout_minutes = 30
                    
                    # Check for early failures (DDP init issues)
                    print({"ddp": "waiting_for_init", "timeout_sec": ddp_init_timeout})
                    time.sleep(5)  # Give workers time to start
                    
                    early_failures = []
                    for i, p in enumerate(procs):
                        if not p.is_alive():
                            early_failures.append(i)
                            print({"ddp": "worker_early_exit", "rank": i, "exit_code": p.exitcode})
                    
                    if early_failures:
                        print({
                            "ddp": "early_failure_detected",
                            "failed_ranks": early_failures,
                            "likely_cause": "DDP initialization failure",
                            "action": "terminating_all_workers"
                        })
                        # Terminate all remaining workers
                        for i, p in enumerate(procs):
                            if p.is_alive():
                                try:
                                    p.terminate()
                                    p.join(timeout=5)
                                    if p.is_alive():
                                        p.kill()
                                except Exception:
                                    pass
                        
                        print("\n[red]❌ DDP initialization failed[/red]")
                        print("[yellow]Common causes on Windows:[/yellow]")
                        print("  1. PyTorch built without libuv/NCCL support")
                        print("  2. Firewall blocking port 29500")
                        print("  3. Docker Desktop hostname pollution")
                        print("\n[cyan]Solutions:[/cyan]")
                        print(f"  • Try single GPU: [green]--cuda-ids 0[/green]")
                        print(f"  • Check firewall: [green]Allow port 29500[/green]")
                        print(f"  • Review logs: [green]{os.environ.get('AIOS_DDP_LOG_DIR', 'N/A')}[/green]")
                        print()
                        
                        if strict:
                            import typer
                            raise typer.Exit(code=1)
                        else:
                            print({"ddp": "early_failure_fallback_single_gpu"})
                            return False
                    
                    print({"ddp": "init_successful", "all_workers_alive": True})
                    
                    # Now wait for training to complete
                    exit_codes = []
                    for i, p in enumerate(procs):
                        p.join(timeout=training_timeout_minutes * 60)  # 30 min timeout per worker
                        if p.is_alive():
                            print({"ddp": "worker_timeout", "rank": i, "terminating": True})
                            p.terminate()
                            p.join(timeout=10)
                            if p.is_alive():
                                p.kill()
                            exit_codes.append(-1)
                        else:
                            exit_codes.append(p.exitcode)
                        status = "completed" if p.exitcode == 0 else "failed"
                        print({"ddp": f"worker_{status}", "rank": i, "exit_code": p.exitcode})
                    
                    failed = [i for i, code in enumerate(exit_codes) if code != 0]
                    if failed:
                        print({"ddp": "completed_with_failures", "failed_ranks": failed})
                    else:
                        print({"ddp": "all_workers_completed"})
                    
                    # Stop progress monitor
                    stop_monitor.set()
                    monitor_thread.join(timeout=2)
                    
                    # Cleanup progress file
                    try:
                        if os.path.exists(progress_file):
                            os.remove(progress_file)
                    except Exception:
                        pass
                    
                    return True  # Parent exits after spawn
                    
                except Exception as e:
                    # Stop progress monitor on error
                    try:
                        stop_monitor.set()
                        monitor_thread.join(timeout=1)
                    except Exception:
                        pass
                    
                    # Cleanup progress file on error
                    try:
                        progress_file_local = os.environ.get("AIOS_DDP_PROGRESS_FILE")
                        if progress_file_local and os.path.exists(progress_file_local):
                            os.remove(progress_file_local)
                    except Exception:
                        pass
                    from rich import print
                    import traceback
                    print({"ddp": "spawn_failed", "error": str(e)[:200]})
                    if strict:
                        import typer
                        raise typer.Exit(code=1)
                    print({"ddp": "fallback_single_gpu", "reason": "spawn_error"})
                    return False
            else:
                # --ddp requested but no spawn enabled and not under torchrun
                from rich import print
                ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                sys_name = platform.system()
                
                print({"ddp": "requires_launcher", "gpus_available": ngpus})
                print("\n[yellow]Multi-GPU training requires a launcher:[/yellow]\n")
                
                if sys_name.lower() == "windows":
                    print("[cyan]Option 1 - Simple launcher (recommended):[/cyan]")
                    print(f"  python scripts/simple_ddp_launcher.py --nproc_per_node={ngpus} aios.cli.aios hrm-hf train-actv1 --ddp [green]<args>[/green]\n")
                    print("[cyan]Option 2 - Convenience script:[/cyan]")
                    print(f"  .\\scripts\\train_ddp.ps1 -NumGPUs {ngpus}\n")
                    print("[cyan]Option 3 - Enable internal spawn (for GUI/wrappers):[/cyan]")
                    print(f"  Set environment variable: AIOS_DDP_SPAWN=1\n")
                else:
                    print("[cyan]Option 1 - Simple launcher:[/cyan]")
                    print(f"  python scripts/simple_ddp_launcher.py --nproc_per_node={ngpus} aios.cli.aios hrm-hf train-actv1 --ddp [green]<args>[/green]\n")
                    print("[cyan]Option 2 - Convenience script:[/cyan]")
                    print(f"  bash scripts/train_ddp.sh --num-gpus {ngpus}\n")
                
                print("[yellow]See docs/MULTI_GPU_TRAINING.md for details[/yellow]\n")
                
                if strict:
                    import typer
                    raise typer.Exit(code=1)
                else:
                    print({"ddp": "fallback_single_gpu", "reason": "launcher_required"})
                    return False
        
        return False  # Not requesting DDP, or conditions not met
    except Exception as e:
        # Graceful fallback on any error
        try:
            from rich import print
            print({"ddp": "error", "exception": str(e)[:100], "fallback": "single_gpu"})
        except Exception:
            pass
        return False
