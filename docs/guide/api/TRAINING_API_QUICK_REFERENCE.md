# Training API Quick Reference

## For CLI Users

### ✅ What Still Works (No Changes)

All standard training parameters work exactly as before:

```bash
# Basic training
aios hrm-hf train-actv1 \
    --model artifacts/hf_implant/base_model \
    --dataset-file training_data/my_data.txt \
    --max-seq-len 128 \
    --batch-size 8 \
    --steps 200

# With optimizations
aios hrm-hf train-actv1 \
    --model base_model \
    --dataset-file data.txt \
    --optimize \
    --gradient-checkpointing \
    --amp \
    --zero-stage zero2

# Multi-GPU DDP
aios hrm-hf train-actv1 \
    --model base_model \
    --dataset-file data.txt \
    --ddp \
    --cuda-ids 0,1 \
    --world-size 2
```


---

## For Python API Users

```python
from aios.core.hrm_training import TrainingConfig
from aios.cli.hrm_hf.train_actv1_impl import train_actv1_impl

# Create config object
config = TrainingConfig(
    model="artifacts/hf_implant/base_model",
    dataset_file="training_data/my_data.txt",
    max_seq_len=128,
    batch_size=8,
    steps=200,
    lr=2e-4,
    device="auto",
    # ... other parameters ...
)

# Validate (optional but recommended)
config.validate()

# Train
train_actv1_impl(config=config)
```

**Error you'll see**:
```
TypeError: train_actv1_impl() missing 1 required positional argument: 'config'
```

---

## Common Migration Examples

### Example 1: Basic Training Script

**Before**:
```python
from aios.cli.hrm_hf.train_actv1_impl import train_actv1_impl

train_actv1_impl(
    model="base_model",
    dataset_file="data.txt",
    max_seq_len=256,
    batch_size=16,
    steps=500,
)
```

**After**:
```python
from aios.core.hrm_training import TrainingConfig
from aios.cli.hrm_hf.train_actv1_impl import train_actv1_impl

config = TrainingConfig(
    model="base_model",
    dataset_file="data.txt",
    max_seq_len=256,
    batch_size=16,
    steps=500,
)
train_actv1_impl(config=config)
```

### Example 2: Programmatic Configuration

**Before**:
```python
# Build kwargs dynamically
kwargs = {
    "model": "base_model",
    "dataset_file": "data.txt",
}

if use_gpu:
    kwargs["device"] = "cuda"
    kwargs["ddp"] = True

train_actv1_impl(**kwargs)
```

**After**:
```python
# Build config dynamically (better type safety!)
config = TrainingConfig(
    model="base_model",
    dataset_file="data.txt",
)

if use_gpu:
    config.device = "cuda"
    config.ddp = True

train_actv1_impl(config=config)
```

### Example 3: Config Persistence

**New feature**: Save and load configurations!

```python
from aios.core.hrm_training import TrainingConfig

# Create config
config = TrainingConfig(
    model="base_model",
    dataset_file="data.txt",
    max_seq_len=1024,
    batch_size=4,
    steps=1000,
    optimize=True,
    zero_stage="zero2",
)

# Save to JSON
import json
with open("my_training_config.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)

# Load later
with open("my_training_config.json", "r") as f:
    config_dict = json.load(f)

config = TrainingConfig.from_dict(config_dict)
train_actv1_impl(config=config)
```

### Example 4: Config Validation

```python
from aios.core.hrm_training import TrainingConfig

# Create invalid config
config = TrainingConfig(
    model="base_model",
    dataset_file=None,  # ❌ Required!
    max_seq_len=128,
)

try:
    config.validate()
except ValueError as e:
    print(f"Config error: {e}")
    # Output: "Config error: dataset_file is required for training"
```

---

## TrainingConfig Parameters

All parameters available in TrainingConfig:

### Core Training:
- `model` (str) - HF model name or path
- `dataset_file` (str) - Training data file
- `max_seq_len` (int) - Sequence length, default: 128
- `batch_size` (int) - Batch size, default: 8
- `steps` (int) - Training steps, default: 200
- `lr` (float) - Learning rate, default: 2e-4
- `device` (str) - Device: auto|cpu|cuda|xpu|mps|dml, default: "auto"
- `halt_max_steps` (int) - Max ACT segments, default: 2
- `save_dir` (str) - Output directory, default: "training_data/actv1"

### Data Processing:
- `ascii_only` (bool) - Filter to ASCII-only lines, default: False
- `eval_file` (str|None) - Held-out eval file, default: None
- `eval_batches` (int) - Eval batches, default: 10
- `sys_mem_cap_pct` (int|None) - Memory cap %, default: None

### Training Control:
- `stop_file` (str|None) - Stop signal file, default: None
- `log_file` (str|None) - Metrics log file, default: None
- `student_init` (str|None) - Resume from checkpoint, default: None
- `iterate` (bool) - Loop training indefinitely, default: False

### Output Bundle:
- `brain_name` (str|None) - Brain bundle name, default: None
- `bundle_dir` (str) - Bundle directory, default: "artifacts/brains/actv1"

### Model Architecture:
- `h_layers` (int) - High-level layers, default: 2
- `l_layers` (int) - Low-level layers, default: 2
- `hidden_size` (int) - Hidden dimension, default: 512
- `expansion` (float) - FFN expansion, default: 2.0
- `num_heads` (int) - Attention heads, default: 8
- `h_cycles` (int) - High-level cycles, default: 2
- `l_cycles` (int) - Low-level cycles, default: 2
- `pos_encodings` (str) - Position encoding: rope|learned, default: "rope"

### Optimization:
- `optimize` (bool) - Auto-optimize for VRAM, default: False
- `gradient_checkpointing` (bool) - Enable grad checkpointing, default: True
- `use_amp` (bool) - Use mixed precision, default: True
- `use_cpu_offload` (bool) - Offload to CPU, default: False
- `zero_stage` (str) - DeepSpeed ZeRO: none|zero1|zero2|zero3, default: "none"

### Multi-GPU:
- `cuda_ids` (str|None) - CUDA device IDs (e.g., "0,1"), default: None
- `ddp` (bool) - Enable DDP, default: False
- `world_size` (int|None) - Number of GPUs, default: None
- `strict` (bool) - No device fallbacks, default: False

### Advanced:
- `kl` (float) - KL divergence scaling factor, default: 0.0
- `kl_temp` (float) - KL temperature annealing, default: 1.0

---

## Need Help?

### Check Configuration:
```python
from aios.core.hrm_training import TrainingConfig

config = TrainingConfig(model="test", dataset_file="data.txt")
print(config)  # Shows formatted summary
```

### Validate Before Training:
```python
try:
    config.validate()
    print("✓ Config is valid!")
except ValueError as e:
    print(f"✗ Config error: {e}")
```

### Get CLI Args:
```python
cli_args = config.to_cli_args()
print(" ".join(cli_args))  # See what CLI command this config represents
```

---

**Questions?** Check the full docs:
- [Advanced Features](../features/ADVANCED_FEATURES.md) - Training configuration options
- [Feature Combination Matrix](../features/FEATURE_COMBINATION_MATRIX.md) - What works together
- `src/aios/core/hrm_training/training_config.py` - Source code
