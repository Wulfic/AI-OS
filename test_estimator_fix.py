"""Test fixed memory estimator with user's exact settings."""
import sys
sys.path.insert(0, r"c:\Users\tyler\Repos\AI-OS\src")

from aios.gui.components.hrm_training.memory_estimator import MemoryEstimator

# USER'S EXACT SETTINGS from screenshot
estimator = MemoryEstimator(
    total_params=54068226,
    hidden_size=256,
    num_layers=12,  # 6H + 6L
    batch_size=1,  # PER GPU
    seq_len=10000,
    use_amp=True,
    use_gradient_checkpointing=True,
    use_lora=False,
    use_cpu_offload=False,
    use_8bit_optimizer=False,
    zero_stage="zero1",  # ZeRO-1 as shown in screenshot
    num_gpus=2,  # System has 2 GPUs (but only trains on 1)
    use_chunking=True,
    chunk_size=4096,
)

vram = estimator.estimate_vram()

print("=" * 80)
print("FIXED ESTIMATE - NO MORE DIVISION BY NUM_GPUS FOR ACTIVATIONS!")
print("=" * 80)
print(f"Configuration:")
print(f"  Model: 54M params, Hidden: 256, Layers: 12")
print(f"  Training: Batch=1 (per GPU), SeqLen=10000, ChunkSize=4096")
print(f"  Optimizations: AMP, GradCheckpoint, ZeRO-1")
print(f"  System: 2 GPUs available (but likely training on 1)")
print()
print(f"Estimated Peak VRAM per GPU: {vram['total_gb']:.2f} GB")
print(f"Your Observed Usage:          ~15-20 GB (from screenshot)")
print()
print("Breakdown PER GPU:")
print(f"  Model:       {vram['model_gb']:.3f} GB (replicated on each GPU)")
print(f"  Optimizer:   {vram['optimizer_gb']:.3f} GB (partitioned by ZeRO-1)")
print(f"  Gradients:   {vram['gradients_gb']:.3f} GB (replicated)")
print(f"  Activations: {vram['activations_gb']:.3f} GB (NOT divided by GPUs!)")
print(f"  Overhead:    {vram['overhead_gb']:.3f} GB")
print()
print("IMPORTANT NOTES:")
print("  - This estimate is for ACTIVE GPU (the one actually training)")
print("  - If training on 1 GPU while 2 are available, second GPU stays idle")
print("  - The ~15-20 GB usage suggests extensive temporary buffers and")
print("    memory spikes during chunk overlap (backward + forward overlap)")
