# Automatic Dataset Preprocessing

## Overview

When training starts with a downloaded dataset, AI-OS now automatically checks if the dataset has been processed into the correct block form. If it hasn't, the system will automatically preprocess it before starting training.

## How It Works

### Before Training Starts

1. **Dataset Type Detection**: The system identifies the dataset type:
   - HuggingFace streaming datasets (`hf://...`): No preprocessing needed
   - Single text files: No preprocessing needed (treated as one block)
   - Directory datasets: Check for block structure

2. **Preprocessing Check**: For directory datasets, the system checks for:
   - `dataset_info.json` metadata file
   - `block_0/samples.txt` structure

3. **Automatic Preprocessing**: If the block structure is missing:
   - The system automatically runs preprocessing
   - Original files are moved to a `raw/` subdirectory
   - Dataset is split into blocks (default: 100,000 samples per block)
   - Metadata file is created for fast size detection

4. **Training Proceeds**: Once preprocessing is complete (or if already preprocessed), training starts normally

## Benefits

- **No Manual Step Required**: Users don't need to remember to preprocess datasets
- **Progress Tracking**: Preprocessed datasets enable accurate epoch tracking
- **Performance**: Block-based structure improves training performance on large datasets
- **Consistency**: All directory datasets are handled uniformly

## Example Output

When starting training with an unprocessed dataset:

```
============================================================
📦 Dataset preprocessing required
   Path: Z:\training_datasets\tinystories
   Block size: 100,000 samples per block
============================================================

📦 Preprocessing dataset: tinystories
   Block size: 100,000 samples
   Moving raw files to raw/ subdirectory...
   ✓ Moved 3 items to raw/
   Reading samples from raw files...
   ✓ Found 2,119,719 samples
   Creating 22 blocks...
   ✓ Block 0: 100,000 samples
   ✓ Block 1: 100,000 samples
   ...
   ✓ Block 21: 19,719 samples
   ✓ Created metadata file
✅ Preprocessing complete!
   Total: 2,119,719 samples in 22 blocks
============================================================

[INIT] Initializing BlockManager (async)...
...
```

## When Preprocessing is Skipped

- **HuggingFace streaming datasets**: Already chunked during download
- **Single files**: Small enough to load entirely
- **Already preprocessed datasets**: Detected and reused

## Manual Preprocessing

You can still manually preprocess datasets if desired:

```bash
aios hrm-hf preprocess-dataset Z:\training_datasets\tinystories
```

Options:
- `--block-size N`: Set samples per block (default: 100,000)
- `--ascii-only`: Filter to ASCII-only text
- `--overwrite`: Rebuild existing preprocessed structure

## Technical Details

### Files Created

After preprocessing, your dataset directory will contain:

```
dataset_name/
├── raw/                    # Original downloaded files
│   ├── file1.txt
│   └── file2.txt
├── block_0/                # First 100k samples
│   └── samples.txt
├── block_1/                # Next 100k samples
│   └── samples.txt
├── ...
└── dataset_info.json       # Metadata for fast detection
```

### Dataset Info Metadata

The `dataset_info.json` file contains:

```json
{
  "dataset_name": "tinystories",
  "total_samples": 2119719,
  "samples_per_block": 100000,
  "total_blocks": 22,
  "ascii_only": false,
  "preprocessed_by": "AI-OS dataset preprocessor",
  "structure": "block_N/samples.txt format"
}
```

## Implementation

The automatic preprocessing is implemented in:
- `src/aios/cli/datasets/dataset_validation.py`: Validation logic
- Integration points:
  - `src/aios/cli/hrm_hf/train_actv1.py`: DDP training path
  - `src/aios/cli/hrm_hf/parallel_training_v3.py`: Parallel training path

## Disabling Auto-Preprocessing

Currently, auto-preprocessing is always enabled for directory datasets. If you need to disable it:

1. Preprocess manually before training
2. Or use single file datasets instead of directories

Future versions may add a `--no-auto-preprocess` flag if needed.

## Error Handling

If preprocessing fails:
- Training is aborted with a clear error message
- The error includes details about what went wrong
- You can fix the issue and restart training

Example error:

```
❌ Dataset validation failed: Failed to preprocess dataset: No samples found in /path/to/dataset
```

## Compatibility

- **Windows**: ✅ Fully supported
- **Ubuntu/Linux**: ✅ Fully supported
- **macOS**: ✅ Fully supported

The preprocessing system works identically across all platforms.
