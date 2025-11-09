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
        if bool(ddp) and str(device).lower() in ("auto", "cuda") and torch.cuda.is_available():
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
                    
                    init_dir = _Path("artifacts/brains/actv1/_ddp")
                    init_dir.mkdir(parents=True, exist_ok=True)
                    init_file = str((init_dir / "init").absolute())
                    
                    # Clean up stale init file
                    init_path = _Path(init_file)
                    if init_path.exists():
                        try:
                            init_path.unlink()
                        except Exception:
                            pass
                    
                    # Backend selection
                    backend = "nccl"
                    if sys_name.lower() == "windows":
                        backend = "gloo"
                    elif os.environ.get("AIOS_DDP_BACKEND"):
                        backend = os.environ.get("AIOS_DDP_BACKEND")
                    
                    # Set up environment for workers
                    os.environ["MASTER_ADDR"] = "127.0.0.1"
                    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
                    os.environ["AIOS_DDP_BACKEND"] = backend
                    os.environ["USE_LIBUV"] = "0"  # Disable libuv for Windows compatibility
                    
                    print({"ddp": "spawning_workers", "world_size": ws, "backend": backend, "device_count": ngpus})
                    
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
                    
                    # Wait for all workers
                    exit_codes = []
                    for i, p in enumerate(procs):
                        p.join()
                        exit_codes.append(p.exitcode)
                        status = "completed" if p.exitcode == 0 else "failed"
                        print({"ddp": f"worker_{status}", "rank": i, "exit_code": p.exitcode})
                    
                    failed = [i for i, code in enumerate(exit_codes) if code != 0]
                    if failed:
                        print({"ddp": "completed_with_failures", "failed_ranks": failed})
                    else:
                        print({"ddp": "all_workers_completed"})
                    
                    return True  # Parent exits after spawn
                    
                except Exception as e:
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
