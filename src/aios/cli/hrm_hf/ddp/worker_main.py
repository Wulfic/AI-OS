"""
Minimal entry point for DDP worker processes.
This bypasses the full CLI to avoid heavy imports in spawn processes.
"""
from __future__ import annotations


def main(rank: int, world_size: int, init_file: str, config_dict: dict) -> None:
    """Minimal worker entry that only imports what's needed for training."""
    import os
    import warnings
    
    # CRITICAL: Clean environment FIRST, before any torch imports!
    # Docker Desktop poisons environment with kubernetes.docker.internal hostname
    # This must happen before PyTorch/gloo reads any environment variables
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
    
    # Set MASTER_ADDR immediately to prevent gloo from using system hostname
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")
    
    # CRITICAL: USE_LIBUV=0 should already be set by parent, but ensure it's set
    # PyTorch Windows builds don't support libuv
    os.environ["USE_LIBUV"] = "0"
    
    # CRITICAL: Force gloo to use TCP with IP addresses, not hostname resolution
    # This prevents reverse DNS lookup that resolves 127.0.0.1 -> kubernetes.docker.internal
    os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
    os.environ["TP_VERBOSE"] = "0"  # Suppress transport verbose logging
    
    # Don't set GLOO_SOCKET_IFNAME - let gloo auto-detect based on MASTER_ADDR=127.0.0.1
    # Setting to empty string can cause interface selection issues on Windows
    # Remove if present to avoid conflicts
    for ifname_var in ["GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_FAMILY"]:
        if ifname_var in os.environ:
            try:
                del os.environ[ifname_var]
            except Exception:
                pass
    
    # CRITICAL: Set CUDA allocator config FIRST, before any torch imports!
    # Note: expandable_segments not supported on Windows, but beneficial on Linux/Unix
    os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    
    # Suppress expandable_segments warning on Windows
    warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*", category=UserWarning)
    
    # Set threading env vars BEFORE any imports
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Set DDP environment
    os.environ["AIOS_DDP_RANK"] = str(rank)
    os.environ["AIOS_DDP_WORLD"] = str(world_size)
    os.environ["AIOS_DDP_INIT_FILE"] = str(init_file)
    
    # Set up logging if configured
    try:
        log_dir = os.environ.get("AIOS_DDP_LOG_DIR")
        if log_dir:
            from pathlib import Path
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"rank{rank}.out.log"
            
            import sys
            
            class Tee:
                def __init__(self, *streams):
                    self._streams = streams
                
                def write(self, s: str):
                    for st in self._streams:
                        try:
                            st.write(s)
                        except Exception:
                            pass
                
                def flush(self):
                    for st in self._streams:
                        try:
                            st.flush()
                        except Exception:
                            pass
            
            fh = open(log_path, "a", encoding="utf-8", buffering=1)
            sys.stdout = Tee(sys.stdout, fh)
            sys.stderr = Tee(sys.stderr, fh)
            try:
                fh.write(f"[AIOS][rank={rank}] logging initialized (world={world_size})\n")
            except Exception:
                pass
    except Exception:
        pass
    
    # Set CUDA device early
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    except Exception:
        pass
    
    # Suppress deprecation warnings before importing transformers
    import warnings
    warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
    
    # Deserialize TrainingConfig and update for DDP worker
    from aios.core.hrm_training import TrainingConfig
    config = TrainingConfig.from_dict(config_dict)
    
    # Override DDP settings for worker process
    config.ddp = False  # Don't re-spawn
    config.world_size = None
    config.device = "cuda"  # Workers always use CUDA
    # Route workers through the unified parallel training path so DDP uses the
    # same chunk-tracker-aware implementation as all other modes.
    try:
        config.parallel_independent = True
    except Exception:
        pass
    
    from aios.cli.hrm_hf.train_actv1 import train_actv1_impl
    train_actv1_impl(config=config)


if __name__ == "__main__":
    import sys
    # Entry point when called directly
    # Args: rank world_size init_file config_json
    if len(sys.argv) != 5:
        print("Usage: ddp_worker_main.py rank world_size init_file config_json")
        sys.exit(1)
    
    import json
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    init_file = sys.argv[3]
    
    with open(sys.argv[4], "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    
    main(rank, world_size, init_file, config_dict)

