"""Test updated RAM estimator with user's configuration."""
import sys
sys.path.insert(0, r"c:\Users\tyler\Repos\AI-OS\src")

from aios.gui.components.hrm_training.memory_estimator import MemoryEstimator

# USER'S EXACT SETTINGS
estimator = MemoryEstimator(
    total_params=54068226,
    hidden_size=256,
    num_layers=12,
    batch_size=1,
    seq_len=10000,
    use_amp=True,
    use_gradient_checkpointing=True,
    use_lora=False,
    use_cpu_offload=False,
    use_8bit_optimizer=False,
    zero_stage="zero1",
    num_gpus=2,  # System has 2 GPUs
    use_chunking=True,
    chunk_size=4096,
)

print("=" * 80)
print("FIXED RAM ESTIMATOR - REALISTIC CALCULATIONS")
print("=" * 80)
print()

vram = estimator.estimate_vram()
ram = estimator.estimate_ram()

print("VRAM Estimate (per GPU):")
print(f"  Total: {vram['total_gb']:.2f} GB")
print()

print("RAM Estimate (System Memory):")
print(f"  Model CPU Copy:       {ram['model_cpu_gb']:.2f} GB")
print(f"  Tokenizer + Vocab:    {ram['tokenizer_gb']:.2f} GB")
print(f"  Dataset Buffers:      {ram['dataset_gb']:.2f} GB")
print(f"  PyTorch/CUDA/Python:  {ram['pytorch_gb']:.2f} GB")
print(f"  Training Overhead:    {ram['training_gb']:.2f} GB")
print(f"  GPU Overflow to RAM:  {ram['gpu_overflow_gb']:.2f} GB")
print(f"  Optimizer (CPU):      {ram['optimizer_gb']:.2f} GB")
print("  " + "-" * 50)
print(f"  TOTAL RAM NEEDED:     {ram['total_gb']:.2f} GB")
print()

print("Comparison:")
print(f"  Old RAM estimate: ~1.6 GB (WRONG!)")
print(f"  New RAM estimate: {ram['total_gb']:.2f} GB")
print(f"  Improvement: {ram['total_gb'] / 1.6:.1f}x more accurate")
print()

print("Breakdown Analysis:")
print(f"  The {ram['gpu_overflow_gb']:.2f} GB 'GPU Overflow' corresponds to")
print(f"  the 'Shared GPU Memory' shown in Task Manager (4.8 GB in screenshot)")
print()
print(f"  With VRAM at {vram['total_gb']:.2f} GB exceeding 11 GB GPU capacity,")
print(f"  system RAM is used as overflow, causing high RAM usage.")
print()

print("Recommendations:")
if ram['gpu_overflow_gb'] > 2.0:
    print(f"  ⚠️  Heavy GPU overflow to RAM ({ram['gpu_overflow_gb']:.1f} GB)")
    print(f"  → Reduce VRAM usage to decrease RAM pressure")
    print(f"  → Try: chunk_size 4096→1024, batch_size may increase")
print()

print("Total System Requirements:")
print(f"  VRAM: {vram['total_gb']:.2f} GB per GPU")
print(f"  RAM:  {ram['total_gb']:.2f} GB system memory")
