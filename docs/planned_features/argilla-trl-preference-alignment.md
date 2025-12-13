# Preference Alignment with Argilla and TRL

This document is an end-to-end integration guide and checklist for adding an open-source human-feedback workflow powered by Argilla and Hugging Face TRL into AI-OS. It covers the architecture, data flow, CLI and GUI elements, runbooks for Windows/Linux, acceptance criteria, and rollout.

## Outcomes and success criteria

- End-to-end: Annotators create preferences in Argilla → data exported → ingested to TRL schema → RM and DPO baselines trained → artifacts logged and evaluated → optional signals consumed by HRM.
- CLI: Three new commands available and documented: argilla-export, rm-train, dpo-train.
- GUI: Minimal “Alignment” section adds dataset stats, quick-run buttons (ingest/RM/DPO), and links to artifacts/logs.
- Validation: On a toy dataset, RM AUC > 0.6 and DPO improves win-rate vs reference ≥ 5% (see Acceptance Checklists).

## Architecture overview

Data flow (text diagram):
- Argilla (annotation UI) → Export (JSONL/Parquet)
- aios data argilla-export → TRL-ready splits under training_data/curated_datasets/preferences/{train,eval,test}.jsonl
- aios hrm-hf rm-train → artifacts/rm/<run_name>/ (model, tokenizer, metrics.jsonl)
- aios hrm-hf dpo-train → artifacts/dpo/<run_name>/ (policy, tokenizer, metrics.jsonl)
- Optional: Evaluation and HRM-side consumption or distillation in later PFs

## Scope

In scope:
- New CLI commands to train RM and DPO models using TRL on HF backbones.
- Data ingestion/export path from Argilla to TRL-ready datasets.
- Minimal GUI hooks: dataset stats, quick-run toggles, links to artifacts.
- Documentation and examples to run end-to-end on a small subset.

Out of scope (this PF):
- Direct TRL training on custom HRM architecture.
- Large UI rewrites; we add a minimal Alignment area and buttons.

## Repository integration points

- Data location: `training_data/curated_datasets/preferences/`
- New CLI modules:
  - `src/aios/cli/hrm_hf/rm_train.py` — wraps TRL RewardTrainer
  - `src/aios/cli/hrm_hf/dpo_train.py` — wraps TRL DPOTrainer
- Shared utils: `src/aios/data/argilla_ingest.py` — converts Argilla exports → TRL-ready datasets
- Config: either extend `TrainingConfig` or define small dataclasses local to each CLI for clarity
- Artifacts:
  - RM: `artifacts/rm/<name>/` (model, tokenizer, metrics.jsonl, config.json)
  - DPO: `artifacts/dpo/<name>/` (policy, tokenizer, metrics.jsonl, config.json)

## Prerequisites

- Python environment (see repo installers). Required packages (minimal):
  - torch (CUDA-enabled when using GPU), transformers, datasets, trl, accelerate, peft, evaluate, scikit-learn, pandas, pyarrow
  - Optional: bitsandbytes (8-bit), deepspeed (multi-GPU), wandb (telemetry)
- Argilla server for annotation (Docker or pip). Quick start:
  - Docker: `docker run -p 6900:6900 -e ARGILLA_ENABLE_TELEMETRY=0 argilla/argilla:latest`
  - Or follow Argilla docs; create a binary preference project and annotate a few pairs.
- Disk layout created:
  - `training_data/curated_datasets/preferences/`
  - `artifacts/{rm,dpo}/`

## Data formats and schema

Target TRL schema (JSONL or Arrow) with splits: `train.jsonl`, `eval.jsonl`, `test.jsonl`.
- DPO: Each row requires keys `prompt`, `chosen`, `rejected`.
- RM: Either the same triplet schema (derived pairs) or an explicit pair with labels; internally RM trainer can consume `(prompt, chosen, rejected)` by forming preference pairs.

Validation rules (ingestion command enforces):
- Non-empty strings for all required fields
- Reasonable length bounds (configurable), deduplication by hash
- Drop rows where `chosen == rejected`
- Compute and write a `schema_report.json` with counts and dropped reasons

## CLI specification (final)

1) Argilla export ingestion
- Command:
  - `aios data argilla-export --input <export.{jsonl,parquet}> --output training_data/curated_datasets/preferences --format trl` 
- Important flags:
  - `--train-ratio 0.9 --eval-ratio 0.05 --test-ratio 0.05`
  - `--min-prompt-toks 3 --max-prompt-toks 2048` (approx by chars if tokenizer not available)
  - `--dedup-by prompt+chosen+rejected` (hash function)
  - `--anonymize` (optional; integrates with Presidio later PF-006)
  - `--shuffle-seed 42`
- Output:
  - Writes `train.jsonl`, `eval.jsonl`, `test.jsonl` & `schema_report.json` under the output dir
  - Returns exit code 0 with summary printed; warns if <95% valid

2) Reward model training
- Command:
  - `aios hrm-hf rm-train --base-model <hf_model> --dataset-dir training_data/curated_datasets/preferences --output-dir artifacts/rm/<name> --epochs 1 --lr 2e-5 --batch-size 8`
- Additional flags:
  - `--max-steps <int>` (overrides epochs), `--gradient-accumulation 1`, `--bf16/--fp16`, `--lr-scheduler cosine`, `--warmup-ratio 0.03`, `--weight-decay 0.01`
  - `--max-length 1024`, `--truncation-right` (or left), `--eval-steps 100`, `--save-steps 200`
  - `--deepspeed config/deepspeed_zero2.json` (optional)
  - `--wandb <project>` (optional)
- Model:
  - HF `AutoModelForSequenceClassification` with 1-dim head or TRL RewardModel head
- Metrics:
  - AUC on eval pairs; accuracy of predicting `chosen > rejected`; loss curve
- Artifacts:
  - Model, tokenizer, `metrics.jsonl`, `config.json`, training logs

3) DPO baseline training
- Command:
  - `aios hrm-hf dpo-train --policy-model <hf_model> --ref-model <hf_model_or_policy_ckpt> --dataset-dir training_data/curated_datasets/preferences --output-dir artifacts/dpo/<name> --epochs 1 --lr 5e-6 --batch-size 4`
- Additional flags:
  - `--beta 0.1` (DPO temperature), `--label-smoothing 0.0`, `--max-length 1024`, `--gradient-accumulation 1`, `--bf16/--fp16`
  - `--lr-scheduler cosine`, `--warmup-ratio 0.03`, `--eval-steps 100`, `--save-steps 200`
  - `--deepspeed config/deepspeed_zero2.json` (optional), `--wandb <project>` (optional)
- Metrics:
  - Relative log-prob of chosen vs rejected; eval loss; win-rate on held-out
- Artifacts:
  - Policy model, tokenizer, `metrics.jsonl`, `config.json`, logs

## GUI integration (minimal, additive)

Add an “Alignment” section/panel in the HRM GUI (or CLI dashboard) with:
- Dataset card
  - Shows counts from `schema_report.json`: total, dropped by reason, splits
  - Buttons: “Re-ingest from Argilla export…”, “Open dataset folder”
- Quick actions
  - Run RM: pick base model, batch size, max steps, precision toggle; Start → streams `metrics.jsonl`
  - Run DPO: pick policy + reference; Start → streams `metrics.jsonl`
  - Links: “Open artifacts/rm/<name>”, “Open artifacts/dpo/<name>”, “View logs”
- Status + telemetry
  - Real-time tail of `metrics.jsonl` and last checkpoint info
  - Optional W&B run URL if enabled

Implementation notes:
- Wire buttons to the same CLI under the hood (subprocess) and follow our existing logging conventions.
- Use existing logging.yaml; stream both stdout and metrics JSONL.
- Non-blocking runs: spawn background process; allow stop/cancel.

## End-to-end runbook (Windows PowerShell)

1) Prepare environment
```powershell
# Activate repo venv if available
. .\.venv\Scripts\Activate.ps1

# Install required packages (example; adjust torch for your CUDA)
pip install transformers datasets trl accelerate peft evaluate scikit-learn pandas pyarrow
# Optional
pip install wandb bitsandbytes deepspeed
```

2) Start Argilla server and annotate a few pairs
```powershell
docker run --pull always -p 6900:6900 -e ARGILLA_ENABLE_TELEMETRY=0 argilla/argilla:latest
# Open http://localhost:6900, create a preference project, add prompt/chosen/rejected.
```

3) Export from Argilla to JSONL (via UI or API), then ingest
```powershell
aios data argilla-export --input path\to\export.jsonl --output training_data/curated_datasets/preferences --format trl --shuffle-seed 42
```

4) Train a small Reward Model on CPU (toy)
```powershell
aios hrm-hf rm-train --base-model gpt2 --dataset-dir training_data/curated_datasets/preferences --output-dir artifacts/rm/demo --max-steps 100 --batch-size 2 --eval-steps 20
```

5) Train a small DPO policy on CPU (toy)
```powershell
aios hrm-hf dpo-train --policy-model gpt2 --ref-model gpt2 --dataset-dir training_data/curated_datasets/preferences --output-dir artifacts/dpo/demo --max-steps 100 --batch-size 1 --eval-steps 20
```

6) Inspect outputs
- Open `artifacts/rm/demo/metrics.jsonl` and `artifacts/dpo/demo/metrics.jsonl` to verify loss decreasing.
- Check `schema_report.json` for ingestion health.

7) Optional GPU acceleration
- Install the correct torch build for your CUDA, add `--bf16` or `--fp16`, and optionally `--deepspeed config/deepspeed_zero2.json`.

## Directory structure conventions

```
training_data/
  curated_datasets/
    preferences/
      train.jsonl
      eval.jsonl
      test.jsonl
      schema_report.json
artifacts/
  rm/
    <run_name>/
      config.json
      metrics.jsonl
      tokenizer/
      model files…
  dpo/
    <run_name>/
      config.json
      metrics.jsonl
      tokenizer/
      model files…
```

## Acceptance checklists

Ingestion (argilla-export)
- [ ] >95% rows valid; <5% dropped for schema/length
- [ ] train/eval/test splits present
- [ ] `schema_report.json` written with counts by reason

Reward Model (rm-train)
- [ ] Training finishes 100 toy steps without errors on CPU
- [ ] AUC on eval set > 0.6 (toy acceptable)
- [ ] `metrics.jsonl` shows monotonic loss decrease on average
- [ ] Artifacts saved under `artifacts/rm/<name>/`

DPO (dpo-train)
- [ ] Training finishes 100 toy steps without errors on CPU
- [ ] Validation chosen log-prob improves vs reference ≥ 5% win-rate on held-out pairs
- [ ] `metrics.jsonl` present with eval loss improvements
- [ ] Artifacts saved under `artifacts/dpo/<name>/`

Docs & GUI
- [ ] This runbook executes end-to-end on CPU
- [ ] GUI Alignment panel shows dataset counts and can launch runs
- [ ] Links open artifacts and live logs

## Troubleshooting

- OOM on GPU: reduce `--batch-size`, increase `--gradient-accumulation`, enable `--fp16/--bf16`, or use `bitsandbytes` 8-bit
- Slow CPU runs: use tiny HF models (e.g., `sshleifer/tiny-gpt2`), reduce `--max-length`, `--max-steps`
- Tokenizer mismatch: ensure policy/ref and tokenizer directories align; remove stale cached tokenizers under `artifacts/hf_implant/tokenizers/` if needed
- Deepspeed import errors: reinstall with matching CUDA; fall back to single-GPU accelerate if blocked
- Dataset schema errors: open `schema_report.json`; fix malformed rows or re-export

## Security & privacy

- Avoid PII in free-text; if required, enable `--anonymize` (PF-006 Presidio integration)
- Do not upload private data to external services unless explicitly configured (disable telemetry by default)

## Telemetry (optional)

- Enable with `--wandb <project>`; include run name and tags like `pf-001`, `rm` or `dpo`
- Always write local `metrics.jsonl` for reproducibility

## Implementation plan (engineering)

Minimal code stubs to add:
- `src/aios/data/argilla_ingest.py`
  - read Argilla JSONL/Parquet → normalize → splits → schema report
- `src/aios/cli/hrm_hf/rm_train.py`
  - TRL RewardTrainer wrapper; HF model+tokenizer load; metrics logging
- `src/aios/cli/hrm_hf/dpo_train.py`
  - TRL DPOTrainer wrapper; policy+ref load; metrics logging

Config options:
- Either extend global `TrainingConfig` with rm/dpo sections or define lightweight local configs in each CLI module to reduce coupling for PF-001.

Logging:
- Reuse `logging.yaml`; write structured metrics to JSONL under the output dir.

## Milestones

- M1 (1–2 days): Ingestion and validation CLI; skeleton TRL trainers
- M2 (1–2 days): End-to-end toy run; artifact packaging; documentation and GUI buttons

## Go/No-Go criteria

- Go when all Acceptance checklists are satisfied on CPU; GPU smoke test completes with bf16/fp16; no critical regressions to existing HRM CLI.
- No-Go if ingestion validity < 90% or trainers fail to complete toy runs on CPU.

## Appendix A — Dataset examples

Sample DPO row (JSONL):
```
{"prompt": "Write a haiku about the moon.", "chosen": "Silver moon whispers\nTides hum ancient lullabies\nNight cradles the seas.", "rejected": "The moon is round. It is in the sky."}
```

## Appendix B — Suggested VS Code tasks (optional)

Add tasks to reproduce quick runs (mirroring existing HRM tasks):
- Run ingestion (argilla-export) with a sample file
- Run RM trainer with tiny model
- Run DPO trainer with tiny model

These tasks should log to `artifacts/{rm,dpo}/<name>/metrics.jsonl` and be discoverable from the Command Palette.

---

Notes:
- TRL is used for baselines and RM training; HRM remains the primary custom model. Distillation into HRM can be explored in a later PF.
