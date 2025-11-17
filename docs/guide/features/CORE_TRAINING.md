# Core Training - AI-OS
Generated: October 20, 2025
Purpose: Architecture and training engine for HRM
Status: Implemented

## Key Files
- `src/aios/core/hrm_training/training_config.py` – TrainingConfig (874+ lines)
- `src/aios/cli/hrm_hf/train_actv1.py` – Training loop (~2000 lines)
- `src/aios/core/hrm_engine.py` – Engine utilities

## Overview
The core training flow is exposed via the `aios hrm-hf train-actv1` command. It loads a base model and tokenizer, applies HRM training logic, logs metrics, writes checkpoints, and maintains a brain bundle directory under `artifacts/brains/actv1/`.

## Training Configuration
- Single source of truth for parameters
- Validation, type checking, CLI arg conversion, defaults, serialization

## Training Loop Features
- Gradient accumulation, loss/optimizer/scheduler
- Checkpoints, metrics logging, OOM handling, graceful stop file

## Brain Bundle System
Directory structure:
```
artifacts/brains/actv1/<brain-name>/
├─ config.json
├─ model.safetensors
├─ tokenizer.json
├─ metadata.json
├─ training_args.json
└─ checkpoints/
```
Features: auto-create, save model/tokenizer/config, resume

## Commands (CLI syntax)

You can run training either via the CLI entry point or directly through Python's module interface. On Windows, prefer PowerShell examples below.

### a) Direct CLI (named flags)
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1000 --batch-size 2 --halt-max-steps 1 --eval-batches 2 --log-file artifacts/brains/actv1/metrics.jsonl
```

### b) Module invocation (explicit model flag)
```powershell
.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1000 --batch-size 2 --halt-max-steps 1 --eval-batches 2 --log-file artifacts/brains/actv1/metrics.jsonl
```

### Key parameters (selection)
- Model selection: `--model <name_or_path>`
- Dataset: `--dataset-file <path>` (txt/jsonl); optional `--ascii-only`
- Steps and batching: `--steps <int>`, `--batch-size <int>`
- Halting: `--halt-max-steps <int>` (controls ACT halting behavior)
- Evaluation: `--eval-file <path>`, `--eval-batches <int>`
- Logging: `--log-file <path>` (JSONL)
- Iteration control: `--iterate`, `--stop-file <path>`
- Brain bundle: `--brain-name <str>`, `--bundle-dir <path>`
- Architecture knobs: `--h-layers`, `--l-layers`, `--hidden-size`, `--expansion`, `--num-heads`, `--h-cycles`, `--l-cycles`, `--window-size`, `--pos-encodings`
- Memory: `--gradient-checkpointing|--no-gradient-checkpointing`, `--amp|--no-amp`, `--use-8bit-optimizer`
- Multi-GPU: `--ddp`, `--cuda-ids <list>`, `--world-size <int>`
- DeepSpeed: `--zero-stage <none|zero1|zero2|zero3>` (uses configs in `config/`)
- Experts: `--expert-id <id>` (train/freeze expert-specific components)

Notes:
- Paths are relative to repo root unless absolute. PowerShell accepts forward slashes (`/`) in Python paths.
- For Windows shells, escape backslashes or quote paths with spaces.

## Iterate Mode
- `--iterate`: restart after completion with new shuffle; supports stop file

## Evaluation During Training
- `--eval-file`, `--eval-batches`
- Periodic eval, validation loss, perplexity, history

## Expert Training Mode
- `--expert-id <id>`: train individual expert, freeze base, save under `artifacts/experts/<id>/`
- Related: Dynamic Subbrains/MoE

## Inputs
- Dataset file(s): `training_data/curated_datasets/*.txt` (example set provided)
- Optional eval file: `training_data/eval_test_dataset.txt`
- Base model: HuggingFace hub id or local path (e.g., `gpt2` or `artifacts/hf_implant/base_model`)
- Tokenizer: auto-resolved from model or `artifacts/hf_implant/tokenizers`

## Outputs
- Brain bundle under `artifacts/brains/actv1/<brain-name>/`
- Metrics log (JSONL): default/explicit `artifacts/brains/actv1/metrics.jsonl`
- Checkpoints under the bundle `checkpoints/`
- Optional evaluation summaries in metrics/logs

## Try it: quick dry-run examples

These mirror VS Code tasks configured in this repo and are safe to run. Ensure your venv is active.

### Option 1: Direct CLI dry-run
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```

### Option 2: Module invocation
```powershell
.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```

### Option 3: Use VS Code Task
- Run: Tasks → "Run brief HRM CLI dry-run" or "Run HRM dry-run (module)"
- Expected outputs:
	- Metrics JSONL at `artifacts/brains/actv1/metrics.jsonl`
	- Brain bundle directories under `artifacts/brains/actv1/`
	- Console logs including training/eval step counts

## Usage Notes
- Use AMP and gradient checkpointing for memory savings
- Use 8-bit optimizer for larger models when bitsandbytes is available

Related: Memory Optimization, Model Architecture, Datasets, Tokenizers

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)

## Troubleshooting

- OOM (out of memory): lower `--batch-size`, `--max-seq-len`, or `--dataset-chunk-size`; enable `--gradient-checkpointing` and `--amp`; consider `--use-8bit-optimizer` if bitsandbytes is installed.
- FlashAttention: ensure `--use-flash-attn` and that your GPU supports it; otherwise it will fall back to SDPA.
- Multi-GPU on Windows: prefer `--parallel-independent` with `--cuda-ids`; DDP often fails on Windows. If you need DDP, set `$env:AIOS_DDP_SPAWN = "1"` before running with `--ddp`.
- Resume: when using parallel mode, `chunk_tracker_state.json` in the brain bundle enables resume; delete it if you want a fresh start.

See also:
- Parallel Training Block/Chunk System
- Multi-GPU & Distributed