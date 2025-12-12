"""Public API functions for unified optimizer - works across GUI, CLI, and TUI."""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, List, Optional

from .config import OptimizationConfig
from .optimizer import UnifiedOptimizer

logger = logging.getLogger(__name__)


def optimize_from_config(config: OptimizationConfig) -> Tuple[Dict[str, Any], UnifiedOptimizer]:
    """Main optimization entry point - works from any interface.
    
    Args:
        config: Optimization configuration
        
    Returns:
        Tuple of (results dictionary, optimizer instance)
    """
    optimizer = UnifiedOptimizer(config)
    results = optimizer.optimize()
    return results, optimizer


def optimize_from_dict(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], UnifiedOptimizer]:
    """Optimize from dictionary configuration.
    
    Args:
        config_dict: Configuration as dictionary
        
    Returns:
        Tuple of (results dictionary, optimizer instance)
    """
    config = OptimizationConfig(**config_dict)
    return optimize_from_config(config)


def optimize_from_gui(panel) -> Tuple[Dict[str, Any], UnifiedOptimizer]:
    """GUI adapter function - converts GUI panel to config.
    
    Args:
        panel: GUI panel instance with configuration
        
    Returns:
        Tuple of (results dictionary, optimizer instance)
    """
    
    def log_callback(message: str):
        panel._log(message)
    
    resources_panel = getattr(panel, '_resources_panel', None)

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            intval = int(value)
            return intval if intval > 0 else None
        except Exception:
            return None

    def _extract_target(util_map: Any) -> Optional[int]:
        if not isinstance(util_map, dict):
            return None
        values: List[int] = []
        for raw in util_map.values():
            coerc = _coerce_int(raw)
            if coerc is not None:
                values.append(coerc)
        return max(values) if values else None

    cuda_selected: List[int] = []
    target_util: Optional[int] = None
    device = "auto"

    if resources_panel is not None:
        try:
            rvals = resources_panel.get_values()
        except Exception:
            rvals = {}

        device = str(rvals.get("train_device", "auto")).strip() or "auto"

        cuda_selected = [
            int(i) for i in (rvals.get("train_cuda_selected") or [])
            if isinstance(i, (int, str)) and str(i).isdigit()
        ]

        target_util = _extract_target(rvals.get("train_cuda_util_pct"))
        if target_util is None:
            target_util = _coerce_int(rvals.get("gpu_util_pct"))

    cuda_devices_str = ",".join(str(i) for i in cuda_selected)

    def stop_callback() -> bool:
        return getattr(panel, '_stop_requested', False)
    
    # Get dataset file from panel
    dataset_file = "training_datasets/curated_datasets/test_sample.txt"
    if hasattr(panel, 'dataset_var'):
        try:
            user_dataset = panel.dataset_var.get().strip()
            if user_dataset:
                dataset_file = user_dataset
        except:
            pass
    
    # Extract configuration from GUI panel
    config = OptimizationConfig(
        model=getattr(panel, 'model_var', None) and panel.model_var.get() or "base_model",
        teacher_model=getattr(panel, 'teacher_var', None) and panel.teacher_var.get() or "",
        max_seq_len=int(getattr(panel, 'max_seq_var', None) and panel.max_seq_var.get() or "512"),
        dataset_file=dataset_file,
        log_callback=log_callback,
        stop_callback=stop_callback,
        test_duration=45,
        max_timeout=240,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        min_batch_size=1,
        max_batch_size=64,
        cuda_devices=cuda_devices_str,
        use_multi_gpu=len(cuda_selected) > 1,
        device=device or "auto",
        target_util=target_util,
        util_tolerance=5,
        monitor_interval=1.0
    )
    
    return optimize_from_config(config)


def optimize_cli(
    model: str = "base_model",
    teacher: str = "",
    max_seq: int = 512,
    test_duration: int = 45,
    batch_sizes: str = "1,2,4,8,16,32",
    output_dir: str = "artifacts/optimization",
    verbose: bool = True,
    device: str = "auto",
    cuda_devices: str = "",
    use_multi_gpu: bool = True,
    strict: bool = False,
    target_util: int = 90,
    util_tolerance: int = 5,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    growth_factor: float = 2.0,
    monitor_interval: float = 1.0
) -> Tuple[Dict[str, Any], UnifiedOptimizer]:
    """CLI optimization function.
    
    Args:
        model: Model path
        teacher: Teacher model path (defaults to model)
        max_seq: Maximum sequence length
        test_duration: Test duration in seconds
        batch_sizes: Comma-separated batch sizes to test
        output_dir: Output directory for results
        verbose: Enable verbose logging
        device: Device preference
        cuda_devices: CUDA device IDs (comma-separated)
        use_multi_gpu: Enable multi-GPU training
        strict: Strict device enforcement
        target_util: Target GPU utilization percentage
        util_tolerance: Utilization tolerance
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size
        growth_factor: Batch size growth factor
        monitor_interval: GPU monitoring interval
        
    Returns:
        Tuple of (results dictionary, optimizer instance)
    """
    
    def log_callback(message: str):
        if verbose:
            logger.info(message)

    batch_list = [int(x.strip()) for x in batch_sizes.split(",") if x.strip()]
    if not batch_list:
        batch_list = [1]

    config = OptimizationConfig(
        model=model,
        teacher_model=teacher or model,
        max_seq_len=max_seq,
        test_duration=test_duration,
        batch_sizes=batch_list,
        min_batch_size=min_batch_size,
        max_batch_size=max(max_batch_size, min_batch_size),
        batch_growth_factor=growth_factor,
        output_dir=output_dir,
        log_callback=log_callback,
        cuda_devices=cuda_devices,
        use_multi_gpu=use_multi_gpu,
        device=device,
        strict=strict,
        target_util=target_util,
        util_tolerance=util_tolerance,
        monitor_interval=monitor_interval
    )
    
    return optimize_from_config(config)
