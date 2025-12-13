# Advanced Training Features
Last Updated: December 12, 2025
Purpose: Practical guide to advanced HRM-HF training flags, best‑known‑good combinations, platform notes, and troubleshooting.
Status: Implemented

## Overview
This guide consolidates all advanced knobs for the HRM-HF trainer (`aios hrm-hf train-actv1`) into one place. It groups options by function (attention/positional, memory & performance, precision/quantization, dataset streaming & chunking, distributed, MoE, PEFT, hot‑reload inference) and calls out compatibility constraints on Windows vs Linux.

Source of truth (CLI): `src/aios/cli/hrm_hf_cli.py` → command `train-actv1`
Configuration model: `src/aios/core/hrm_training/training_config.py`

See also:
- Memory & VRAM: [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md)
- FlashAttention: [FLASH_ATTENTION.md](./FLASH_ATTENTION.md) and [FLASH_ATTENTION_VS_CHUNKING.md](./FLASH_ATTENTION_VS_CHUNKING.md)
- Multi‑GPU & streaming: [MULTI_GPU_DISTRIBUTED.md](./MULTI_GPU_DISTRIBUTED.md), [PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md](./PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md)
- MoE: [DYNAMIC_SUBBRAINS_MOE.md](./DYNAMIC_SUBBRAINS_MOE.md)
- PEFT/LoRA: [LORA_PEFT.md](./LORA_PEFT.md)
- Core training entry: [CORE_TRAINING.md](./CORE_TRAINING.md)

## Prerequisites
- Windows PowerShell (pwsh), repo venv activated
- GPU and drivers installed. CUDA recommended on NVIDIA. DirectML supported for inference; training support varies.
- For 8‑bit optimizer and INT8/INT4 quantization: `bitsandbytes` with CUDA GPU. Windows support is limited—prefer Linux for 8/4‑bit.
- For FlashAttention 2: Ampere+ GPU; dedicated FA2 build required on most setups. Not typically available on Windows—falls back to SDPA.
- For DeepSpeed ZeRO: DeepSpeed installation (Linux recommended). ZeRO often not supported or unstable on Windows.

## Commands (CLI syntax)

### a) Direct CLI
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 200 --batch-size 8
```

### b) Module invocation (exact venv)
```powershell
.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 200 --batch-size 8
```

## Key option groups (with notes)

### Attention & positional
- `--use-flash-attn/--no-flash-attn`: Enable FlashAttention 2. Requires Ampere+ and proper install. On Windows, usually unavailable; will fall back to SDPA.
- `--window-size <int|None>`: Sliding‑window attention. Use 256–512 for extreme contexts (50–100k tokens) to reduce memory. Works with or without FlashAttention.
- `--pos-encodings rope|learned`: Choose positional encoding. Default: rope.

### Memory & performance
- `--gradient-checkpointing/--no-gradient-checkpointing`: ↓VRAM by ~30–50% at ~20% speed cost. Default: enabled.
- `--amp/--no-amp`: Mixed precision activations (FP16/BF16). Big savings with minimal quality impact. Default: enabled.
- `--cpu-offload/--no-cpu-offload`: Offload carry states to CPU between chunks for ultra‑long contexts (>500k). Slower; requires sufficient system RAM.
- `--use-8bit-optimizer`: Use bitsandbytes 8‑bit optimizer (~75% optimizer memory reduction). Requires CUDA + bitsandbytes.
- `--dataset-chunk-size <int>`: Samples per training cycle in iterate mode. Smaller uses less memory, larger is faster.

### Precision & quantization
- `--model-dtype fp32|fp16|bf16`: Weight precision when loading full‑precision models. Separate from AMP.
- `--load-in-8bit`: INT8 weight loading (75% memory reduction). Requires bitsandbytes + CUDA.
- `--load-in-4bit`: INT4 (QLoRA‑style) loading (≈87.5% memory reduction). Strongly pair with PEFT.
Notes:
- When `--load-in-8bit` or `--load-in-4bit` is set, the base weights load quantized; AMP still controls activation precision. On Windows, bitsandbytes support is limited—prefer Linux.

### Dataset streaming & chunked training
- `--use-chunked-training`: Split sequences into chunks to fit memory for long contexts.
- `--chunk-size <tokens>`: 1024–4096 typical. Smaller = less VRAM, slower.
- `--linear-dataset/--no-linear-dataset`: Linear order (default) enables progress tracking and resume.
- `--dataset-start-offset <int>`: Resume index for linear mode.
- `--iterate`: Repeat generate→train cycles until stopped.

### Distributed & multi‑GPU
- `--ddp`: Enable torch.distributed (CUDA only). Best on Linux.
- `--world-size <int>`: Number of processes/GPUs for DDP.
- `--cuda-ids "0,1"`: Pin devices explicitly.
- `--parallel-independent`: Windows‑friendly multi‑GPU alternative. Trains separate data blocks on different GPUs sequentially, then merges checkpoints. Bypasses DDP.
- `--zero-stage none|zero1|zero2|zero3`: DeepSpeed ZeRO. Requires DeepSpeed; Linux recommended.
- `--strict`: Disallow device fallbacks; fail fast on mismatches.

### MoE (Mixture of Experts)
- `--use-moe/--no-moe` (default: enabled)
- `--num-experts <int>`: Total experts (capacity vs VRAM trade‑off).
- `--num-experts-per-tok <int>`: Top‑k experts per token; lower = faster/less memory.
- `--moe-capacity-factor <float>`: Load‑balancing headroom.
- `--auto-adjust-lr/--no-auto-adjust-lr`: Auto reduce LR for MoE stability.
Tips: Start with 8 experts, top‑k=2, capacity 1.25; raise gradually.

### PEFT (parameter‑efficient fine‑tuning)
- `--use-peft/--no-peft` and `--peft-method lora|adalora|ia3|loha|lokr`
- `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--lora-target-modules "q_proj,v_proj"`
Best practice: With `--load-in-4bit`, enable PEFT and tune adapters only to keep memory low.

### Inference hot‑reload during training
- `--inference-device cuda:N`: Use a dedicated GPU for inference while training on another.
- `--hot-reload-steps <int>`: Frequency to reload inference model from checkpoints.

### Auto optimization
- `--optimize`: Auto‑find a stable combination for context length (up to ~100k) and batch size based on VRAM. May override `--max-seq-len` and `--batch-size`.

## Try it: minimal safe examples

### 1) Quick dry‑run (single GPU)
Runs 1 step with tiny batch to validate pipeline and logging.
```powershell
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-file training_data/curated_datasets/test_sample.txt `
	--steps 1 --batch-size 2 `
	--halt-max-steps 1 `
	--eval-batches 1 `
	--log-file artifacts/brains/actv1/metrics.jsonl
```
VS Code Task: Run “Run brief HRM CLI dry-run”.

Expected outputs:
- Metrics log appended at `artifacts/brains/actv1/metrics.jsonl`
- Checkpoints under `training_data/actv1` (default `--save-dir`)

### 2) Windows multi‑GPU without DDP
Use parallel‑independent with chunked training to reduce VRAM pressure.
```powershell
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-file training_data/curated_datasets/test_sample.txt `
	--parallel-independent `
	--use-chunked-training --chunk-size 2048 `
	--amp --gradient-checkpointing `
	--steps 200 --batch-size 4
```

### 3) QLoRA‑style PEFT on a single GPU (very low VRAM)
```powershell
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-file training_data/curated_datasets/test_sample.txt `
	--load-in-4bit --use-peft --peft-method lora `
	--lora-r 16 --lora-alpha 32 --lora-dropout 0.05 `
	--lora-target-modules "q_proj,v_proj" `
	--amp --gradient-checkpointing `
	--steps 200 --batch-size 8
```
Note: Requires bitsandbytes + CUDA; prefer Linux.

### 4) MoE enabled with conservative routing
```powershell
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-file training_data/curated_datasets/test_sample.txt `
	--use-moe --num-experts 8 --num-experts-per-tok 2 `
	--moe-capacity-factor 1.25 --auto-adjust-lr `
	--amp --gradient-checkpointing `
	--steps 200 --batch-size 8
```

## Compatibility, constraints, and tips

- FlashAttention 2:
	- Works best on Linux with an Ampere+ GPU and proper FA2 install.
	- On Windows, expect fallback to SDPA; performance gain may be limited.
- DeepSpeed ZeRO:
	- Requires DeepSpeed; Linux recommended. ZeRO not typically supported on Windows.
- DDP:
	- Best on Linux. On Windows, prefer `--parallel-independent`.
- Quantization:
	- `--load-in-4bit` pairs best with `--use-peft`. Keep base LM frozen; train adapters.
	- If bitsandbytes isn’t available, omit `--load-in-8bit/--load-in-4bit` and consider `--use-8bit-optimizer` only.
- Heads and shapes:
	- Ensure `--num-heads` divides `--hidden-size`. The validator will error otherwise.
- Chunked training:
	- For very long contexts, combine `--use-chunked-training`, `--chunk-size 1024–2048`, `--gradient-checkpointing`, and optionally `--cpu-offload`.
- Logging & resume:
	- Use `--log-file` for JSONL metrics; pair with `--linear-dataset` and `--dataset-start-offset` to resume deterministically.

## Troubleshooting

- “Configuration error …” on launch:
	- Check incompatible shapes (e.g., `num_heads` must divide `hidden_size`).
	- Remove `--use-flash-attn` if FA2 isn’t installed; it will fall back, but explicit removal helps isolate issues.
	- If using ZeRO on Windows, remove `--zero-stage`.
- “bitsandbytes not found” or CUDA errors:
	- Remove `--load-in-8bit/--load-in-4bit` and/or `--use-8bit-optimizer`, or switch to Linux with CUDA.
- DDP hang on Windows:
	- Switch to `--parallel-independent`. Verify `--cuda-ids` and drivers.
- OOM during long‑context training:
	- Lower `--chunk-size`, enable `--gradient-checkpointing`, ensure `--amp`, and reduce `--batch-size`.

## References
- CLI entry: `src/aios/cli/hrm_hf_cli.py` (train‑actv1)
- Config: `src/aios/core/hrm_training/training_config.py`
- Related docs:
	- [FLASH_ATTENTION.md](./FLASH_ATTENTION.md)
	- [FLASH_ATTENTION_VS_CHUNKING.md](./FLASH_ATTENTION_VS_CHUNKING.md)
	- [PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md](./PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md)
	- [MULTI_GPU_DISTRIBUTED.md](./MULTI_GPU_DISTRIBUTED.md)
	- [LORA_PEFT.md](./LORA_PEFT.md)
	- [DYNAMIC_SUBBRAINS_MOE.md](./DYNAMIC_SUBBRAINS_MOE.md)
	- [CORE_TRAINING.md](./CORE_TRAINING.md)
	- [CLI_COMMANDS.md](./CLI_COMMANDS.md)

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](./COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)