# Dataset Preprocessing

## Overview

The dataset preprocessing utility converts downloaded datasets into an optimized block-based structure for efficient training with accurate progress tracking.

## Why Preprocess?

**Without Preprocessing:**
- ❌ Slow or failed dataset size detection (especially on network drives)
- ❌ No block/chunk progress tracking
- ❌ Unpredictable performance on large datasets
- ❌ Shows "0/???" for chunks and blocks

**With Preprocessing:**
- ✅ Instant dataset size detection (reads metadata file)
- ✅ Accurate block and chunk progress tracking
- ✅ Consistent performance regardless of storage location
- ✅ Shows "15/25" for chunks, "2/10" for blocks
- ✅ Optimal for network drives and large datasets

## When to Preprocess

Preprocess datasets in these scenarios:
- Downloaded to network drives (Z:, mapped drives, NAS)
- Large datasets (>1GB, millions of samples)
- Datasets with many small files
- When training shows "epoch tracking disabled"

## Usage

### Command Line

```bash
# Basic preprocessing (100k samples per block)
aios hrm-hf preprocess-dataset Z:\training_datasets\tinystories

# Custom block size
aios hrm-hf preprocess-dataset ~/datasets/my_corpus --block-size 50000

# ASCII-only filtering
aios hrm-hf preprocess-dataset /data/multilingual --ascii-only

# Overwrite existing preprocessed structure
aios hrm-hf preprocess-dataset ./datasets/corpus --overwrite
```

### Python API

```python
from aios.cli.datasets.preprocess_dataset import preprocess_dataset

# Preprocess dataset
total_samples, samples_per_block, total_blocks = preprocess_dataset(
    dataset_path="Z:/training_datasets/tinystories",
    samples_per_block=100000,  # 100k samples per block
    ascii_only=False,
    overwrite=False
)

print(f"Preprocessed: {total_samples:,} samples in {total_blocks} blocks")
```

## Structure Created

```
dataset_name/
├── dataset_info.json     # Metadata (instant size detection)
├── raw/                  # Original files (preserved)
│   ├── file1.txt
│   ├── file2.txt
│   └── ...
├── block_0/              # First 100k samples
│   └── samples.txt       # One sample per line
├── block_1/              # Next 100k samples
│   └── samples.txt
├── block_2/
│   └── samples.txt
└── ...
```

### Metadata File (dataset_info.json)

```json
{
  "dataset_name": "tinystories",
  "total_samples": 2456789,
  "samples_per_block": 100000,
  "total_blocks": 25,
  "ascii_only": false,
  "preprocessed_by": "AI-OS dataset preprocessor",
  "structure": "block_N/samples.txt format"
}
```

## Supported Input Formats

The preprocessor automatically detects and handles:

### 1. HuggingFace Datasets
- Saved with `dataset.save_to_disk()`
- Contains `dataset_info.json`, `.arrow` files, or `data/` directory
- Extracts text from columns: text, content, sentence, article, etc.

### 2. Plain Text Files
- `.txt`, `.csv`, `.json`, `.jsonl` files
- Recursively scans subdirectories
- One sample per line

### 3. Mixed Directories
- Combination of text files and HF dataset files
- Automatically chooses best extraction method

## Training with Preprocessed Datasets

Once preprocessed, training automatically detects the structure:

```bash
# Just point to the preprocessed directory
aios hrm-hf train-actv1 --dataset-file Z:\training_datasets\tinystories --steps 1000

# Training output will show:
# ✓ Epoch tracking initialized
# ✓ Dataset: tinystories
# ✓ Total: 2,456,789 samples in 25 blocks
# ✓ Chunk: 15/25   Block: 2/25   Epoch: 0
```

## Parameters

### --block-size (default: 100000)
Number of samples per block. Larger blocks = fewer files but more memory per block load.

**Guidelines:**
- **Small datasets (<100k samples)**: Use 10000-50000
- **Medium datasets (100k-1M)**: Use 100000 (default)
- **Large datasets (>1M)**: Use 100000-200000

### --ascii-only
Filter to ASCII-only text, removing non-ASCII characters and samples.

**Use when:**
- Training English-only models
- Avoiding encoding issues
- Reducing dataset size

### --overwrite
Rebuild the preprocessed structure from scratch.

**Use when:**
- Updating after adding/removing raw files
- Changing block size
- Fixing corrupted structure

## Performance

### Before Preprocessing
```
Dataset: Z:\training_datasets\tinystories (network drive)
Detection: 45-120 seconds (or fails with timeout)
Progress: "Chunk: 0/???  Block: 0/???"
```

### After Preprocessing
```
Dataset: Z:\training_datasets\tinystories
Detection: <1 second (reads metadata file)
Progress: "Chunk: 15/25  Block: 2/25  Epoch: 0"
```

## Checking Status

```python
from aios.cli.datasets.preprocess_dataset import is_preprocessed, get_preprocessed_info

# Check if dataset is preprocessed
if is_preprocessed("Z:/training_datasets/tinystories"):
    print("Dataset is preprocessed!")
    
    # Get metadata
    info = get_preprocessed_info("Z:/training_datasets/tinystories")
    print(f"Samples: {info['total_samples']:,}")
    print(f"Blocks: {info['total_blocks']}")
```

## Troubleshooting

### "No text samples found"
- Check that raw files contain readable text
- Verify file extensions (.txt, .csv, .json, .jsonl)
- Try without `--ascii-only` flag

### "Preprocessed structure exists"
- Use `--overwrite` to rebuild
- Or delete `dataset_info.json` and `block_*` directories manually

### "Permission denied"
- Ensure write access to dataset directory
- Try running with elevated privileges
- Check network drive permissions

### Slow preprocessing
- Normal for large datasets (millions of samples)
- Progress shown every 100 files
- Consider preprocessing on local drive first, then moving

## Best Practices

1. **Preprocess once, train many times**
   - Preprocessing is one-time cost
   - Subsequent training runs are fast

2. **Keep raw files**
   - Original files moved to `raw/` subdirectory
   - Can rebuild anytime with `--overwrite`

3. **Use standard block size**
   - 100k samples per block works well for most datasets
   - Only adjust if specific memory constraints

4. **Preprocess before long training runs**
   - Ensures accurate progress tracking
   - Prevents "epoch tracking disabled" issues

5. **Version control metadata only**
   - Add `block_*/` to .gitignore
   - Keep `dataset_info.json` for reference
   - Raw files can be re-downloaded

## Integration with GUI

The GUI automatically detects preprocessed datasets:
- Shows accurate total blocks in "Training Progress"
- Displays chunk progress within current block
- Updates blocks as "X/Y" instead of "X"

No GUI changes needed - just preprocess the dataset and start training!
