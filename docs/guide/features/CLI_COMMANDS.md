# CLI Commands - AI-OS
Generated: December 12, 2025
Purpose: Reference for the `aios` CLI and subcommands
Status: Implemented

## Overview

- Main entry point: `aios`
- File: `src/aios/cli/aios.py`

Sub-commands overview:
- hrm-hf – HuggingFace-based HRM training (see Core Training)
- brains – Brain management
- gui – Launch GUI (see GUI Features)
- status – System status
- datasets – Dataset management (see Datasets)
- cache – Cache management
- goals – Goals management
- eval – Evaluation utilities (see Advanced Features → Evaluation)
- artifacts – Artifacts management
- cleanup – Cleanup utilities
- crawl – Web crawling (see Tools & Integrations)
- optimization – Optimization utilities (see Memory Optimization)
- modelcard – Model card generation
- agent – Agent commands
- budgets – Budget management (see Advanced Features → Budgets)
- core – Core commands
- hf-cache – HuggingFace cache management
- dml – DirectML utilities

## HRM-HF Training

- Command: `aios hrm-hf`
- File: `src/aios/cli/hrm_hf_cli.py`
- Subcommand: `train-actv1` – Train HRM models with ACT v1
- File: `src/aios/cli/hrm_hf/train_actv1.py`
- Deep dive: See Core Training and Memory Optimization docs

Key parameters (selection):
- Model: `--model <name_or_path>`
- Brain naming: `--brain-name`, `--bundle-dir`
- Training control: `--steps`, `--batch-size`, `--lr`, `--max-seq-len`, `--iterate`, `--stop-file`, `--resume`, `--stop-after-epoch`
- Architecture: `--h-layers`, `--l-layers`, `--hidden-size`, `--expansion`, `--num-heads`, `--h-cycles`, `--l-cycles`, `--halt-max-steps`, `--window-size`, `--pos-encodings`
- Memory optimization: `--gradient-checkpointing|--no-gradient-checkpointing`, `--amp|--no-amp`, `--use-8bit-optimizer`, `--use-chunked-training`, `--chunk-size`, `--cpu-offload`
- Dataset: `--dataset-file`, `--ascii-only`, `--linear-dataset`, `--dataset-start-offset`, `--dataset-chunk-size`
- Evaluation: `--eval-file`, `--eval-batches`, `--log-file`
- Multi-GPU: `--ddp`, `--cuda-ids`, `--world-size`, `--parallel-independent`, `--strict`
- DeepSpeed: `--zero-stage <none|zero1|zero2|zero3>`
- MoE: `--use-moe`, `--num-experts`, `--num-experts-per-tok`, `--moe-capacity-factor`, `--auto-adjust-lr`
- PEFT: `--use-peft`, `--peft-method`, `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--lora-target-modules`
- Precision/Quant: `--model-dtype fp32|fp16|bf16`, `--load-in-8bit`, `--load-in-4bit`
- Inference hot‑reload: `--inference-device`, `--hot-reload-steps`

## Brains Management

- Command: `aios brains`
- File: `src/aios/cli/brains.py`
- Subcommands: list, load, info, delete, export, import
- Related: Core Training → Brain Bundle System, GUI Features → Brains Panel

## Datasets Management

- Command: `aios datasets`
- File: `src/aios/cli/datasets_cli.py`
- Features: list/download, scan, metadata, verification
- Related: Datasets doc

## Goals Management

- Command: `aios goals`
- File: `src/aios/cli/goals_cli.py`
- Create/list/activate goals, link to experts, goal-driven training.
- Related: Dynamic Subbrains/MoE and Advanced Features → Orchestrator

## Cache and HF Cache

- `aios cache` → Clear/show stats
- `aios hf-cache` → Location, move, clear, size reporting

## Evaluation

- Command: `aios eval`
- File: `src/aios/cli/eval_cli.py`
- Run evaluations, generate reports, compare models.
- Related: Advanced Features → Evaluation

## Crawling

- Command: `aios crawl`
- File: `src/aios/cli/crawl_cli.py`
- Web crawling, dataset generation from web.
- Related: Tools & Integrations

## Optimization

- Command: `aios optimization`
- File: `src/aios/cli/optimization_cli.py`
- Memory/VRAM estimation and parameter optimization.
- Related: Memory Optimization

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)