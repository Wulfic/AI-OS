# Dataset Download & Training Data Flow

This document describes how datasets are downloaded, stored, and loaded during training in AI-OS.

## Overview

AI-OS uses a hierarchical data structure optimized for memory efficiency and parallel training:

```
Dataset (Full dataset, e.g., 10M samples)
    └── Blocks (100k samples each, stored on disk)
        └── Chunks (4k samples each, loaded into RAM)
            └── Batches (8 samples, sent to GPU)
```

## Download System

### Streaming Block Writer

When downloading datasets, AI-OS uses streaming with automatic 100k block creation:

1. **Stream samples** from HuggingFace Hub (or other sources)
2. **Buffer samples** until 100k are collected
3. **Flush to disk** as a JSONL block file
4. **Repeat** until download complete

This provides:
- **Memory efficiency**: Only holds 100k samples at a time, not entire dataset
- **Resumability**: Blocks are saved progressively; partial downloads preserve completed blocks
- **Training-ready format**: Blocks match the training system's expected format

### Block Structure on Disk

```
dataset_name/
├── blocks/
│   ├── block_00000.jsonl  (100k samples)
│   ├── block_00001.jsonl  (100k samples)
│   ├── block_00002.jsonl  (up to 100k samples)
│   └── ...
└── block_manifest.json    (metadata about all blocks)
```

### Post-Processing

For datasets not downloaded in block format, use `process_raw_dataset_to_blocks()`:

```python
from aios.gui.components.dataset_download_panel.block_processor import (
    process_raw_dataset_to_blocks
)

block_info = process_raw_dataset_to_blocks(
    input_path=Path("raw_dataset/"),
    output_dir=Path("blocked_dataset/"),
    dataset_name="my_dataset",
    block_size=100000
)
```

## Training Data Flow

### 1. BlockManager

The `BlockManager` class handles loading blocks during training:

```python
block_manager = BlockManager(
    dataset_path="hf://dataset_name",
    samples_per_block=100000,     # 100k samples per block
    dataset_chunk_size=4000,      # 4k samples per chunk
)
```

**Key operations:**
- `get_block(block_id)` - Returns block metadata (not data!)
- `get_chunk(block_id, chunk_id, chunk_size)` - Loads specific chunk into RAM

### 2. Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│ DISK: Full dataset (blocks/block_*.jsonl)                       │
│   - All blocks stored as JSONL files                            │
│   - Only metadata loaded initially                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓ On-demand loading
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM RAM: Current chunk (4k samples, ~few MB)                 │
│   - BlockManager.get_chunk() loads one chunk at a time          │
│   - Chunk cache stores last 10 chunks for potential reuse       │
│   - Aggressive cache eviction to prevent memory growth          │
└─────────────────────────────────────────────────────────────────┘
                              ↓ Batching & tokenization
┌─────────────────────────────────────────────────────────────────┐
│ GPU VRAM: Current batch only (8 samples, tokenized tensors)     │
│   - Tokenized sequences moved to GPU                            │
│   - Forward/backward pass computed                              │
│   - Tensors released after each step                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Parallel GPU Training

For multi-GPU training, each GPU processes unique chunks:

```
Block 0 (100k samples, 25 chunks of 4k each):
├── Chunk 0  → GPU 0 claims and trains
├── Chunk 1  → GPU 1 claims and trains
├── Chunk 2  → GPU 0 claims and trains (after finishing Chunk 0)
├── Chunk 3  → GPU 1 claims and trains
└── ...
```

The `ChunkTracker` ensures:
- No duplicate training (each chunk trained exactly once)
- Progress persistence (resume from any point)
- Epoch detection (when all blocks/chunks processed)

### 4. Long Sequence Handling

For very long sequences (>10k tokens), chunked training splits sequences further:

```python
chunked_segment_rollout(
    model=model,
    batch=batch,              # Contains tokenized sequence
    max_segments=5,
    chunk_size=2048,          # Process 2k tokens at a time
)
```

This enables training on 100k+ token contexts with limited VRAM by:
- Processing sequence in 2k-token chunks
- Maintaining carry state across chunks
- Accumulating gradients
- Optional CPU offloading for extreme contexts

## Modality Filtering

The dataset search panel supports filtering by data modality:

| Modality | HF Filter Tag | Training Support |
|----------|--------------|------------------|
| Text | `modality:text` | ✅ Full support |
| Audio | `modality:audio` | ❌ Not yet |
| Image | `modality:image` | ❌ Not yet |
| Video | `modality:video` | ❌ Not yet |
| Tabular | `modality:tabular` | ❌ Not yet |
| Document | `modality:document` | ⚠️ Text extraction needed |
| Geospatial | `modality:geospatial` | ❌ Not yet |
| Time-series | `modality:timeseries` | ❌ Not yet |
| 3D | `modality:3d` | ❌ Not yet |

**Note:** Only "Text" datasets are currently supported for model training. Users can browse and download other modalities, but they won't work with the training system.

## Configuration Options

### Block/Chunk Sizes

```yaml
# In training config
samples_per_block: 100000   # Samples per downloaded block
dataset_chunk_size: 4000    # Samples loaded into RAM at once

# Recommendations by RAM:
# - 16GB RAM: chunk_size=2000-3000
# - 32GB RAM: chunk_size=4000 (default)
# - 64GB+ RAM: chunk_size=8000+
```

### Sequence Chunking (for long contexts)

```yaml
use_chunked_training: true  # Enable for long sequences
chunk_size: 2048           # Tokens per GPU chunk

# Recommendations by VRAM:
# - 10GB VRAM: chunk_size=1024
# - 20GB VRAM: chunk_size=2048 (default)
# - 24GB+ VRAM: chunk_size=4096
```

## Summary

The data flow in AI-OS is designed for:

1. **Efficient Downloads**: Streaming with progressive block saves
2. **Memory Efficiency**: Only load what's needed (chunks, not blocks)
3. **Parallel Training**: Unique chunk distribution across GPUs
4. **Long Context Support**: Chunked training for extreme sequence lengths
5. **Resumability**: Progress saved at block and chunk level
