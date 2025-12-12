# Memory Optimization - AI-OS
Generated: December 12, 2025
Purpose: Techniques to reduce VRAM and enable larger models/contexts
Status: Implemented (some items require verification)

## Gradient Checkpointing
- Default: Enabled; CLI: `--gradient-checkpointing` / `--no-gradient-checkpointing`
- ~30–50% memory reduction; ~20% slowdown
- Applied during model setup in training

## Mixed Precision (AMP)
- Default: Enabled; CLI: `--amp` / `--no-amp`
- Uses autocast + GradScaler (FP16/BF16)
- ~40–50% memory reduction; 2–3x speedup

## 8-bit Optimizer
- CLI: `--use-8bit-optimizer`; uses `bitsandbytes` if available
- Quantizes optimizer states; large savings for 100M+ params
- Fallback to AdamW if bnb unavailable

## Chunked Training (Long Context)
- Config options exist: `use_chunked_training`, `chunk_size`
- Goal: split long sequences into chunks with accumulation
- Docs: PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md, CONFIGURABLE_DATASET_CHUNK_SIZE.md
- Status: Verification needed for training loop integration

## Dynamic Batch Size Reduction
- Auto-reduce batch size on CUDA OOM until 1; then fail if still OOM

## Gradient Accumulation
- CLI: `--gradient-accumulation-steps <n>` to simulate larger batches

## Attention Kernel Optimizations
- Uses FlashAttention when available; SDPA fallback
- See: FLASH_ATTENTION.md and FLASH_ATTENTION_VS_CHUNKING.md

## Example configurations (Windows PowerShell)

- Conservative VRAM profile:
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 1 --gradient-accumulation-steps 8 --gradient-checkpointing --amp --use-8bit-optimizer --log-file artifacts/brains/actv1/metrics.jsonl
```

- Long-context friendly (requires FA2-capable GPU or will fallback):
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 1 --gradient-checkpointing --amp --use-8bit-optimizer --window-size 2048 --log-file artifacts/brains/actv1/metrics.jsonl
```

## Verifying savings
- Watch metrics/logs for memory reports if emitted; compare throughput with/without options.
- For FlashAttention, see the Flash Attention doc for how to confirm activation vs fallback.

## Troubleshooting
- CUDA OOM: Reduce `--batch-size`, increase `--gradient-accumulation-steps`, or enable `--use-8bit-optimizer` and checkpointing.
- bitsandbytes missing: Install the extra or run without `--use-8bit-optimizer` (it will fallback).
- Unstable FP16: Try BF16 if supported (`--amp` selection is automatic) or temporarily `--no-amp`.

Related: Core Training, Model Architecture

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)