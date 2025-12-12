"""GUI integration adapter for progressive optimizer."""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

from .models import OptimizationConfig
from .optimizer import ProgressiveOptimizer


def optimize_from_gui_progressive(panel) -> Dict[str, Any]:
    """
    GUI adapter for progressive optimizer.
    
    Extracts configuration from GUI panel and runs progressive optimization.
    Returns results dict that GUI can use to update checkboxes and settings.
    
    Args:
        panel: GUI panel instance with training configuration
        
    Returns:
        Results dictionary with optimization results and optimal settings
    """
    
    def log_callback(message: str):
        panel._log(message)
    
    def stop_callback() -> bool:
        return getattr(panel, '_stop_requested', False)
    
    # Extract configuration from panel
    model = getattr(panel, 'model_var', None) and panel.model_var.get() or "base_model"
    dataset_file = "training_datasets/curated_datasets/test_sample.txt"
    
    if hasattr(panel, 'dataset_var'):
        try:
            user_dataset = panel.dataset_var.get().strip()
            if user_dataset:
                dataset_file = user_dataset
        except:
            pass
    
    max_seq_len = 512
    if hasattr(panel, 'max_seq_var'):
        try:
            max_seq_len = int(panel.max_seq_var.get())
        except:
            pass
    
    # Get GPU configuration
    cuda_devices = ""
    use_multi_gpu = False
    
    resources_panel = getattr(panel, '_resources_panel', None)
    if resources_panel:
        try:
            rvals = resources_panel.get_values()
            cuda_selected = [
                int(i) for i in (rvals.get("train_cuda_selected") or [])
                if isinstance(i, (int, str)) and str(i).isdigit()
            ]
            cuda_devices = ",".join(str(i) for i in cuda_selected)
            use_multi_gpu = len(cuda_selected) > 1
        except:
            pass
    
    # Create configuration
    config = OptimizationConfig(
        model=model,
        dataset_file=dataset_file,
        max_seq_len=max_seq_len,
        train_steps=10,
        cuda_devices=cuda_devices,
        use_multi_gpu=use_multi_gpu,
        device="auto",
        test_duration=180,
        max_timeout=300,
        min_batch_size=1,
        max_batch_size=128,
        log_callback=log_callback,
        stop_callback=stop_callback
    )
    
    # Run optimization
    optimizer = ProgressiveOptimizer(config)
    results = optimizer.optimize()
    
    # Apply results to GUI if successful
    if results["success"] and results.get("optimal_level"):
        level = results["optimal_level"]
        batch = results["optimal_batch"]
        
        log_callback(f"\n📋 Applying optimal settings to GUI...")
        
        try:
            # Update batch size
            if hasattr(panel, 'batch_var'):
                panel.batch_var.set(str(batch))
                log_callback(f"   • Batch size: {batch}")
            
            # Update optimizations
            if hasattr(panel, 'use_grad_checkpoint_var'):
                panel.use_grad_checkpoint_var.set(level.gradient_checkpointing)
                log_callback(f"   • Gradient Checkpointing: {level.gradient_checkpointing}")
            
            if hasattr(panel, 'use_amp_var'):
                panel.use_amp_var.set(level.amp)
                log_callback(f"   • AMP: {level.amp}")
            
            if hasattr(panel, 'use_flashattn2_var'):
                panel.use_flashattn2_var.set(level.flashattn2)
                log_callback(f"   • FlashAttention-2: {level.flashattn2}")
            
            if hasattr(panel, 'use_lora_var'):
                panel.use_lora_var.set(level.lora)
                log_callback(f"   • LoRA: {level.lora}")
            
            if hasattr(panel, 'use_cpu_offload_var'):
                panel.use_cpu_offload_var.set(level.cpu_offload)
                log_callback(f"   • CPU Offload: {level.cpu_offload}")
            
            if hasattr(panel, 'zero_stage_var'):
                panel.zero_stage_var.set(level.zero_stage)
                log_callback(f"   • DeepSpeed ZeRO: {level.zero_stage}")
            
            if hasattr(panel, 'chunk_size_var') and level.chunk_size is not None:
                panel.chunk_size_var.set(str(level.chunk_size))
                log_callback(f"   • Chunk Size: {level.chunk_size}")
            
            log_callback("   ✅ GUI updated with optimal settings!")
            
        except Exception as e:
            log_callback(f"   ⚠️  Error updating GUI: {e}")
    
    return results
