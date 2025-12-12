# Multi-GPU & Distributed
Generated: December 12, 2025
Purpose: How to use multiple GPUs with AI-OS training (DDP and Windows-compatible parallel mode)
Status: DDP supported via launcher or internal spawn; Windows defaults to parallel-independent mode. DeepSpeed flags exist but engine wiring is limited.

## Overview

AI-OS supports two multi-GPU modes:

1) DDP (torch.distributed) — classic data parallel synchronized training.
2) Parallel independent training — Windows-friendly mode that assigns distinct data chunks to each GPU without DDP synchronization, then aggregates progress/checkpoints.

On Windows, DDP often fails due to backend limitations. The code provides an internal spawn pathway and falls back to parallel-independent mode when needed.

Key flags in `aios hrm-hf train-actv1`:
- --ddp: enable DDP when a launcher or internal spawn is available
- --cuda-ids "0,1,...": which GPUs to use
- --world-size N: number of DDP processes/GPUs (optional; inferred from --cuda-ids)
- --parallel-independent: force the Windows-compatible multi-GPU path (no gradient sync)

Related files:
- CLI: `src/aios/cli/hrm_hf_cli.py` (options: --ddp, --world-size, --cuda-ids, --parallel-independent)
- DDP internals: `src/aios/cli/hrm_hf/ddp/utils.py`, `src/aios/cli/hrm_hf/ddp/worker_main.py`

## Using DDP

DDP requires either:
- An external launcher (recommended on Linux): torchrun/torch.distributed.run
- Internal spawn mode (set AIOS_DDP_SPAWN=1), useful on Windows/GUI

Windows PowerShell examples (GPU IDs 0 and 1):

1) Internal spawn (recommended on Windows)
- Set an environment variable for this PowerShell session and run training:

	$env:AIOS_DDP_SPAWN = "1"
	.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --ddp --cuda-ids "0,1" --world-size 2 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 10 --batch-size 4 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

Notes:
- The process will spawn per-rank workers and use the gloo backend on Windows.
- If early failures are detected during DDP init, training will fall back to single-GPU unless you passed --strict.

2) External launcher (Linux-oriented, shown for reference)
- torchrun --standalone --nproc_per_node 2 .venv/…/python -m aios.cli.aios hrm-hf train-actv1 --ddp --model gpt2 --dataset-file …
- On Windows, torchrun may still be unstable; prefer internal spawn or parallel-independent.

Verifying DDP:
- Logs will include keys like {"ddp": "external_launcher_detected"} or {"ddp": "spawning_workers"}.
- Each rank writes progress, and total throughput should scale with GPU count.

Troubleshooting DDP on Windows:
- If you see errors mentioning libuv/NCCL/backends or hostname resolution, the code automatically forces gloo, sets MASTER_ADDR=127.0.0.1, and disables libuv.
- Still failing? Try:
	- Close Docker Desktop (prevents hostname pollution)
	- Ensure firewall allows local port 29500
	- Set $env:AIOS_DDP_SPAWN = "1" and retry
	- Or use --parallel-independent instead of --ddp

## Parallel Independent Training (Windows-friendly)

This mode runs coordinated multi-GPU training without DDP. Each GPU trains on different dataset chunks (no duplication) and progress/checkpoints are tracked centrally. No gradient synchronization occurs, so results won’t be identical to DDP, but it achieves strong throughput on Windows.

Example (2 GPUs):

	.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --parallel-independent --cuda-ids "0,1" --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 200 --batch-size 4 --dataset-chunk-size 4000 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

Notes:
- Uses the block/chunk system to guarantee distinct chunks per GPU.
- Creates/updates `chunk_tracker_state.json` under the brain bundle for resume and epoch tracking.
- Ideal fallback when DDP isn’t viable on Windows.

## DeepSpeed ZeRO (status)

Flag surface exists: --zero-stage {none|zero1|zero2|zero3}
Configs present: config/deepspeed_zero1.json, config/deepspeed_zero2.json, config/deepspeed_zero3.json

Current state:
- The CLI and GUI expose ZeRO stage selection and use it for estimators/UI.
- `train_actv1` calls `initialize_deepspeed(...)`; when the `deepspeed` package is available and the requested stage is supported, the model is wrapped in a DeepSpeed engine automatically.
- The code falls back to standard optimizers if import or initialization fails, logging the reason.
- Treat ZeRO as experimental: there is minimal automated coverage, and successful initialization still depends on your environment (CUDA-only, compatible DeepSpeed build, etc.).

## Inputs and Outputs

Inputs (typical):
- --model <name or path>
- --dataset-file <path or hf://…>
- --steps, --batch-size, --halt-max-steps, etc.
- Multi-GPU picks: --ddp/--parallel-independent, --cuda-ids, --world-size

Outputs:
- Metrics JSONL: artifacts/brains/actv1/metrics.jsonl (configurable)
- Brain bundle(s): artifacts/brains/actv1/<brain-name>/
- chunk_tracker_state.json (parallel-independent runs)

## Quick starts (PowerShell)

DDP (internal spawn):

	$env:AIOS_DDP_SPAWN = "1"
	.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --ddp --cuda-ids "0,1" --world-size 2 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 5 --batch-size 2 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

Parallel independent (Windows-friendly):

	.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --parallel-independent --cuda-ids "0,1" --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 5 --batch-size 2 --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl

See also:
- Parallel Training Block/Chunk System
- Memory Optimization
- Core Training

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)