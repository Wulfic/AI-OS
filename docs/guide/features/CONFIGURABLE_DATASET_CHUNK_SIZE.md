# Configurable Dataset Chunk Size (Sub-topic)
Last Updated: October 20, 2025

Status: Implemented and used by CLI/GUI

Canonical feature doc: `PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md`

## Overview

The dataset chunk size controls how many samples are processed per training cycle when using iterate-style loading. Smaller chunks reduce peak memory usage; larger chunks improve throughput at the cost of higher memory.

## How it works

- CLI flag: --dataset-chunk-size <int> (default: 4000)
- TrainingConfig field: dataset_chunk_size: int
- GUI: “Chunk size” field under Steps; the “Auto” button aligns steps to the chunk size

## Recommended values

- 8 GB VRAM: 2000–3000
- 12–16 GB VRAM: 4000 (default)
- 24 GB+ VRAM: 8000+

## CLI examples (Windows PowerShell)

- Low VRAM (8 GB)
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --dataset-chunk-size 2000 --steps 100
```

- Balanced (12 GB)
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --dataset-chunk-size 4000 --steps 100
```

- High VRAM (24 GB+)
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --dataset-chunk-size 8000 --steps 100
```

## Where it’s wired in

- src/aios/cli/hrm_hf_cli.py
  --dataset-chunk-size option passed into config
- src/aios/core/hrm_training/training_config/base_fields.py
  dataset_chunk_size: int = 4000
- src/aios/cli/hrm_hf/data.py, encoding.py, block_manager.py, train_actv1.py
  use dataset_chunk_size to bound lines read and chunk within blocks
- GUI
  - Variable: panel.dataset_chunk_size_var (default "4000")
  - Auto button in helpers.py sets steps to match chunk size

## Try it
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --dataset-chunk-size 4000 --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```
Expected: training parses and logs metrics; dataset loader will cap lines per cycle at 4000.

## Notes

- This setting affects data loading cadence and memory pressure, not model architecture.
- If you also use gradient accumulation, consider total tokens per optimizer step when tuning throughput vs memory.
