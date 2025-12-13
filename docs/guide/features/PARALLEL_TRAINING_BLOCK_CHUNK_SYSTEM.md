# Parallel Training Data Distribution

Canonical source for how we stream and distribute data across GPUs without duplication. For the per-run chunk knob, see CONFIGURABLE_DATASET_CHUNK_SIZE.md.

## Overview

The system streams datasets in large blocks and serves only the requested chunks to each GPU. A shared tracker guarantees no two GPUs train the same chunk. It supports both HuggingFace streaming datasets and local files, with epoch detection and resume.

## Architecture

### Hierarchy

```
Dataset (e.g., 10M samples)
  ├─ Block 0 (≈100k samples) ─ streaming from HuggingFace or sliced from local files
  │   ├─ Chunk 0 (e.g., 4k samples) ─ GPU 0 trains
  │   ├─ Chunk 1 (e.g., 4k samples) ─ GPU 1 trains
  │   ├─ Chunk 2 (e.g., 4k samples) ─ GPU 0 trains
  │   └─ ...
  ├─ Block 1 (≈100k samples)
  │   └─ ...
  └─ ...

Epoch = one full pass through ALL blocks
```

### Key Components

1) BlockManager (`src/aios/cli/hrm_hf/block_manager.py`)
- Streams HuggingFace datasets in blocks (default ≈100k samples) and caches them on disk
- Loads only the requested chunk into memory on demand
- Prefetches metadata to detect last block
- Works with local files by slicing them into block-sized windows

2) ChunkTracker (state persisted in the brain bundle)
- Tracks which (block_id, chunk_id) were trained
- Prevents duplication across GPUs
- Tracks blocks visited per epoch and total steps
- Persists to `chunk_tracker_state.json` for resume

3) Parallel Control
- In parallel-independent mode, each GPU requests the next untrained chunk
- In DDP, data loading can still use block/chunk mechanics while gradients are synchronized

## Configuration surface

TrainingConfig defaults (see `src/aios/core/hrm_training/training_config/base_fields.py`):

- samples_per_block: 100000 (for HF streaming/local slicing; auto-detected/recorded)
- dataset_chunk_size: 4000 (per-iteration chunk size; user knob `--dataset-chunk-size`)
- stop_after_epoch: false (toggle via `--stop-after-epoch`)
- iterate: false (toggle via `--iterate`)

Notes:
- There is no `--samples-per-block` CLI flag. samples_per_block is chosen/detected inside dataset setup and recorded into metrics for UI.
- Adjust memory/throughput primarily via `--dataset-chunk-size`.

## Stopping conditions

1) Steps limit: `--steps N` stops after N steps.
2) Stop after epoch: `--stop-after-epoch` stops after a full dataset pass.
3) Iterate mode: `--iterate` loops indefinitely, rolling epochs.

## Windows PowerShell examples

Two-GPU parallel independent training on a HuggingFace dataset:

  .venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --parallel-independent --cuda-ids "0,1" --model gpt2 --dataset-file "hf://wikitext:wikitext-2-raw-v1:train" --dataset-chunk-size 4000 --steps 500 --batch-size 4 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

Stop after an epoch using a local dataset:

  .venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --parallel-independent --cuda-ids "0,1,2" --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --dataset-chunk-size 2000 --stop-after-epoch --batch-size 4 --steps 10000 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

Continuous iterate mode on two GPUs:

  .venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --parallel-independent --cuda-ids "0,1" --model gpt2 --dataset-file "hf://c4:en:train" --dataset-chunk-size 4000 --iterate --batch-size 4 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

## Chunk claiming (no-dup guarantee)

### No Duplication Guarantee

Each GPU claims chunks atomically from ChunkTracker:

```python
def get_next_untrained_chunk(block_id, total_chunks, gpu_id):
    with lock:
        for chunk_id in range(total_chunks):
            if (block_id, chunk_id) not in completed_chunks:
                # Mark as in-progress
                return chunk_id
        return None  # All chunks trained
```

Key points:
- Thread-safe: lock prevents races
- Atomic claiming: chunk is marked before training
- No duplicates: trained chunks are skipped
- Fair distribution: first-come-first-served

### Resume Capability

Training state saved to `chunk_tracker_state.json` (under the brain bundle):

```json
{
  "completed_chunks": [
    {"block_id": 0, "chunk_id": 0, "gpu_id": 0, "step": 125, "samples_trained": 4000},
    {"block_id": 0, "chunk_id": 1, "gpu_id": 1, "step": 127, "samples_trained": 4000}
  ],
  "current_epoch": 0,
  "blocks_this_epoch": [0, 1, 2],
  "total_steps": 250
}
```

On resume:
- Skips already-trained chunks
- Continues from last step count
- Maintains epoch tracking

## Epoch Detection

An **epoch** = Training on ALL blocks in the dataset once.

### Detection algorithm

```python
def check_epoch_complete(total_blocks):
  return len(blocks_this_epoch) >= total_blocks
```

When epoch completes:
1. ChunkTracker marks epoch complete
2. If `stop_after_epoch=True` → Stop training
3. If `iterate=True` → Start new epoch (reset block tracking)
4. Otherwise → Stop training

### Last block detection

BlockManager detects the last block by attempting to download block N+1:

```python
# Loading block 5
block_5 = load_block(5)  # Returns 100k samples
block_6 = load_block(6)  # Returns 0 samples → Last block detected

block_5.is_last_block = True
```

This works for:
- HuggingFace datasets: streaming ends naturally
- Local files: EOF reached
- Large datasets: consistent detection

## Performance Considerations

### Memory usage

Block metadata and chunks are cached. Only requested chunks are loaded into RAM at any time, keeping memory bounded mostly by `--dataset-chunk-size` and model batch/sequence length.

### Network I/O

HuggingFace streaming benefits from cached blocks on disk and metadata prefetching to hide latency.

### GPU Utilization

Optimal chunk size depends on VRAM and sequence length:
- Smaller chunks reduce VRAM but can increase coordination overhead
- Larger chunks improve throughput but increase memory
- Defaults: 4000 works well for 12–16 GB VRAM

## Troubleshooting

### Issue: GPUs training duplicate data

Cause: chunk tracker state not found/shared.

Solution: ensure a single brain bundle is used and that `chunk_tracker_state.json` is writable by all worker processes.

### Issue: Training stops prematurely

Check stopping conditions:
```python
config.steps  # max steps reached?
config.stop_after_epoch  # epoch completed?
```

Debug:
```python
stats = chunk_tracker.get_progress_stats()
print(stats)  # total_steps, blocks_this_epoch, current_epoch
```

### Issue: Epoch not completing

Possible causes:
1) total_blocks not detected yet (keep training until prefetch finishes)
2) Some blocks never requested (short runs)
3) ChunkTracker state corrupted

Solution:
```python
total_blocks = block_manager.get_total_blocks()
print(f"Total blocks: {total_blocks}")

stats = chunk_tracker.get_progress_stats()
print(f"Blocks this epoch: {stats['blocks_this_epoch']}")
```

### Issue: Out of memory

Reduce memory usage:
```python
config.dataset_chunk_size = 2000  # smaller chunk
config.batch_size = 4             # smaller batch
config.max_seq_len = 128          # shorter sequences
```

## System Architecture

### Block Management

```python
# Streams data in blocks from HuggingFace or local files
block_manager = BlockManager(dataset_file, samples_per_block=100k)

# Distributes chunks across GPUs without duplication
chunk_tracker = ChunkTracker(state_file)

# Features:
# - Full training progress tracking
# - Automatic epoch detection  
# - Resume capability from checkpoints
# - Support for all stopping conditions
```

### Key Components

1. **BlockManager**: Downloads and caches 100k-sample blocks from datasets
2. **ChunkTracker**: Tracks which chunks each GPU has processed
3. **State Persistence**: Saves progress to enable resuming training
4. **Epoch Detection**: Automatically detects when full dataset is processed

## Testing

### Test 1: No duplicate training

```python
# Track all chunks trained by each GPU
gpu0_chunks = set()
gpu1_chunks = set()

# After training
assert gpu0_chunks.isdisjoint(gpu1_chunks)  # No overlap
```

### Test 2: Epoch detection

```python
# Train with stop_after_epoch=True
config.stop_after_epoch = True

# Should stop when all blocks visited
final_stats = chunk_tracker.get_progress_stats()
assert final_stats['blocks_this_epoch'] == total_blocks
```

### Test 3: Resume capability

```python
# Train for 100 steps
config.steps = 100
run_training()

# Resume and train 100 more
config.steps = 200
run_training()

# Should have 200 total steps, no duplicate chunks
assert chunk_tracker.total_steps == 200
```

## Related: Configurable dataset chunk size

See CONFIGURABLE_DATASET_CHUNK_SIZE.md for usage guidance, examples, and how it interacts with batch size and sequence length.

## Future enhancements

1) Dynamic load balancing (fast GPUs get more chunks)
2) Chunk prioritization (curriculum)
3) Distributed tracker (multi-node)
4) Adaptive block sizing
5) Chunk prefetching
6) Partial-epoch checkpoints

## Summary

This streaming block/chunk system provides:

- No duplicate training across GPUs
- Proper block management for HF/local datasets
- Chunk-level tracking with resume
- Epoch detection and iterate mode
- Thread-safe coordination and persistence
- Memory-efficient operation by loading only the needed chunks

It replaces older approaches that loaded entire datasets into memory without proper progress tracking.
