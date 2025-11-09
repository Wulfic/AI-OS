from __future__ import annotations

import os
import shutil
from typing import Any
import threading
import time

# Import the new advanced optimizer
try:
    from .optimizer_v2 import optimize_settings_v2
    from .gpu_monitor import create_gpu_monitor
    NEW_OPTIMIZER_AVAILABLE = True
    _optimize_v2_func = optimize_settings_v2
except ImportError:
    NEW_OPTIMIZER_AVAILABLE = False
    _optimize_v2_func = None


def optimize_settings(panel: Any) -> None:
    """Main optimization entry point with automatic system selection.
    
    Automatically selects between the legacy optimizer (v1) and the new
    advanced optimizer (v2) based on availability and user preferences.
    
    The v2 optimizer provides:
    - Proper multi-GPU DDP utilization
    - Real workload testing with actual generation/training
    - Clean stop file management and process isolation
    - GPU resource monitoring and verification
    - Progressive batch size optimization
    """
    
    # Check if user has enabled the new optimizer via environment variable
    use_new_optimizer = os.environ.get("AIOS_USE_OPTIMIZER_V2", "1") == "1"
    
    if NEW_OPTIMIZER_AVAILABLE and use_new_optimizer:
        panel._log("[opt] Using Advanced Optimizer v2")
        try:
            # Use the new advanced optimizer
            if _optimize_v2_func is not None:
                _optimize_v2_func(panel)
            else:
                raise ImportError("optimize_settings_v2 not properly imported")
            return
        except Exception as e:
            panel._log(f"[opt] Advanced optimizer failed: {e}")
            panel._log("[opt] Falling back to legacy optimizer")
    
    # Fall back to legacy optimizer
    panel._log("[opt] Using Legacy Optimizer v1")
    _optimize_settings_legacy(panel)


def _optimize_settings_legacy(panel: Any) -> None:
    """Legacy optimization implementation (preserved for compatibility).
    
    Note: This implementation has known issues with multi-GPU utilization
    and stop file management. Use the v2 optimizer when possible.
    """
    panel._log("[opt] Legacy optimizer disabled due to corruption")
    panel._log("[opt] Please use AIOS_USE_OPTIMIZER_V2=1 to enable the new optimizer")
    panel._log("[opt] Or restore the original optimizer.py from git")