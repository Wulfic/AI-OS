"""
Minimal entry point for DDP worker processes.
This bypasses the full CLI to avoid heavy imports in spawn processes.
"""
from __future__ import annotations


def main(rank: int, world_size: int, init_file: str, config_dict: dict) -> None:
    """Minimal worker entry that only imports what's needed for training."""
    import os
    
    # CRITICAL: Set CUDA allocator config FIRST, before any torch imports!
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:4'
    
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
            log_path = Path(log_dir) / f"rank{rank}.log"
            
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
    
    from aios.cli.hrm_hf.train_actv1 import train_actv1_impl
    train_actv1_impl(config=config)


if __name__ == "__main__":
    import sys
    # Entry point when called directly
    # Args: rank world_size init_file config_pickle
    if len(sys.argv) != 5:
        print("Usage: ddp_worker_main.py rank world_size init_file config_pickle")
        sys.exit(1)
    
    import pickle
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    init_file = sys.argv[3]
    
    with open(sys.argv[4], "rb") as f:
        config_dict = pickle.load(f)
    
    main(rank, world_size, init_file, config_dict)

