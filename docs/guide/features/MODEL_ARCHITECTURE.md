# Model Architecture - AI-OS
Generated: December 12, 2025
Purpose: HRM model components and configurable parameters
Status: Implemented

## Files
- `src/aios/core/hrm_models/` – Core modules
- `src/aios/core/hrm.py` – Top-level assembly

## Hierarchical Structure
- High-level (H) and low-level (L) blocks
- Files: hierarchical_recurrence.py, higher_level_block.py, lower_level_block.py
- Params: `--h-layers`, `--l-layers`, `--h-cycles`, `--l-cycles`

## Adaptive Computation Time (ACT)
- File: act.py; param: `--halt-max-steps`

## Attention Mechanisms
- Files: attention.py, efficient_attention.py
- Types: MHA, sliding window, efficient variants
- Params: `--num-heads`, `--window-size`

## Position Encodings
- File: position_encoding.py
- Types: RoPE (default), sinusoidal, learned
- Param: `--pos-encodings`

## Feed-Forward Networks
- File: ffn.py; GLU; `--expansion`

## Residual Connections
- Throughout; includes LayerNorm and skip paths

## Selecting and validating architectures (CLI)

Architecture is configured through `aios hrm-hf train-actv1` flags. Typical flags include:
- Depth: `--h-layers`, `--l-layers`
- Cycles: `--h-cycles`, `--l-cycles`
- Width/heads: `--hidden-size`, `--num-heads`, `--expansion`
- Positional encodings: `--pos-encodings`
- Attention window: `--window-size` (sliding window) with FlashAttention or SDPA

### Example: small hierarchical model (Windows PowerShell)
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --h-layers 2 --l-layers 4 --hidden-size 512 --num-heads 8 --expansion 4 --pos-encodings rope --halt-max-steps 1 --log-file artifacts/brains/actv1/metrics.jsonl
```

### Validate model/tokenizer resolution
Use the same command with `--steps 1` to ensure it initializes and writes metrics/checkpoints under `artifacts/brains/actv1/`.

Notes:
- If you point `--model` to a local path, ensure tokenizer files are present or resolvable from HF.
- Windowed attention (`--window-size`) limits attention range; see Flash Attention docs for trade-offs.

Related: Memory Optimization, Dynamic Subbrains/MoE

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)