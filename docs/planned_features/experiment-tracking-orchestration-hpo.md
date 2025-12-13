# Experiment Tracking and Hyperparameter Optimization

### Summary

This PF introduces optional Weights & Biases (W&B) experiment tracking, Prefect-powered flows to orchestrate data → train → eval → package, and Optuna-based hyperparameter tuning. It includes both CLI and GUI surfaces and a full developer checklist to implement and verify end-to-end.

### Why this matters

- Make runs observable beyond local JSONL.
- Reproduce pipelines with Python-native orchestration.
- Systematically tune key knobs (LR, warmup, chunking, MoE stability) without guesswork.

---

## What ships in PF-004

In scope (new capabilities):
- W&B tracking: mirror metrics and upload artifacts from HRM training.
- Prefect flows: a simple, local-first flow for dataset prep → train → eval → package.
- Optuna autotune: `aios hrm-hf autotune` to search safe bounds with early-stopping.
- GUI hooks: toggles and forms to launch training with W&B, run flows, and kick off HPO.

Out of scope (future PF candidates):
- Enterprise schedulers (Airflow, K8s operators), distributed orchestration, and model registry integrations.

---

## Dependencies and installation

All features are optional and guarded by availability checks. Recommended extras per feature:

- Tracking: `wandb>=0.17`
- Orchestration: `prefect>=2.16`
- HPO: `optuna>=3.6`

Installation (PowerShell, Windows):

```powershell
# Activate venv if needed
. .venv\Scripts\Activate.ps1

# Install optional packages (any subset is fine)
pip install wandb prefect optuna
```

Note: W&B is optional and respects offline mode. Credentials are read from environment and W&B’s standard login flow.

---

## CLI design and UX

### 1) W&B flags in training

Command: `aios hrm-hf train-actv1`

New flags (all optional):
- `--wandb/--no-wandb` (default: no-wandb)
- `--wandb-project TEXT` (default: `aios-hrm`)
- `--wandb-entity TEXT` (optional)
- `--wandb-group TEXT` (optional)
- `--wandb-tags TEXT` (comma-separated)
- `--wandb-offline/--wandb-online` (default: online if logged in; else offline)
- `--wandb-run-name TEXT` (optional)

Behavior:
- If enabled, initialize run with `TrainingConfig.to_dict()` as config.
- Stream metrics each step and on eval; attach artifacts (metrics.jsonl, latest checkpoints, brain bundle, GPU metrics from `artifacts/optimization/*gpu_metrics*.jsonl` when present).
- Respect `WANDB_MODE=offline` and lack of network gracefully (no crash, local-only logging continues).

Examples:

```powershell
# Minimal live tracking
aios hrm-hf train-actv1 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 4 --wandb --wandb-project aios-hrm

# Offline (no network), with named run and tags
aios hrm-hf train-actv1 --dataset-file training_data/curated_datasets/test_sample.txt --steps 10 --batch-size 2 --wandb --wandb-offline --wandb-run-name dev-dryrun --wandb-tags smoke,debug
```

Fallback if `aios` entrypoint unavailable:

```powershell
.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --dataset-file training_data/curated_datasets/test_sample.txt --steps 10 --batch-size 2 --wandb
```

Metrics mirrored to W&B (initial set):
- step, train_loss, eval_loss, lr, tokens_per_sec, sec_per_step
- batch_size, max_seq_len, halt_max_steps
- memory: vram_alloc_gb (when available), cpu_ram_gb, gpu_overflow_gb (if detected)
- moe: load_balance_loss (when enabled), num_experts, num_experts_per_tok, capacity_factor

Artifacts:
- `metrics.jsonl` (if `--log-file` is set)
- last N checkpoints from `save_dir`
- packaged brain bundle under `bundle_dir/brain_name` (if used)

---

### 2) Prefect flow entry

Command: `aios flow hrm-train`

Purpose: End-to-end local flow for:
1) dataset prep (no-op if a plain text file is provided)
2) training via `hrm-hf train-actv1`
3) evaluation via `aios eval run` (when `--eval-file` is provided)
4) packaging into brain bundle (if `--brain-name` is provided)

Flags (representative):
- `--dataset-file PATH` (required)
- `--eval-file PATH` (optional)
- `--brain-name TEXT` (optional)
- `--wandb` (optional; passes through to train)
- Plus a passthrough `--train-args "..."` for advanced control

Examples:

```powershell
# Simple flow with W&B
aios flow hrm-train --dataset-file training_data/curated_datasets/test_sample.txt --wandb

# Full flow with eval + packaging
aios flow hrm-train --dataset-file training_data/curated_datasets/test_sample.txt --eval-file training_data/curated_datasets/test_sample.txt --brain-name demo-brain --wandb --train-args "--steps 100 --batch-size 4"
```

Implementation surfaces:
- `src/aios/flows/hrm_train_flow.py`: Prefect `@flow` and `@task`s (`prepare_dataset`, `train_model`, `run_eval`, `package_brain`).
- `src/aios/cli/flows_cli.py`: Typer command group that invokes Prefect flow (Python-native run; users don’t need a Prefect daemon).

---

### 3) Optuna HPO

Command: `aios hrm-hf autotune`

Core flags:
- `--trials INT` (default: 10)
- `--timeout-minutes INT` (optional)
- `--sampler {tpe,random}` (default: tpe)
- `--pruner {median,successive_halving,None}` (default: median)
- `--direction {minimize,maximize}` (default: minimize eval_loss)
- `--eval-batches INT` (default: 3 for quick signal)
- `--study-name TEXT` (optional, for resuming)
- `--storage TEXT` (optional, Optuna RDB string for persistence)
- `--seed INT` (optional; repeatable trials)
- Search-space overrides (optional):
	- `--lr-min 1e-6 --lr-max 1e-4`
	- `--warmup-min 20 --warmup-max 400`
	- `--chunk-choices 1024,2048,4096`
	- `--moe-balance-min 5e-3 --moe-balance-max 2e-2`

Behavior:
- Each trial runs a short `train-actv1` with trial params applied to `TrainingConfig`.
- OOM or fatal errors are caught; trial marked failed with readable metadata.
- Best trial is reported and optionally exported to a JSON config snippet.

Examples:

```powershell
# Fast 5-trial smoke test
aios hrm-hf autotune --dataset-file training_data/curated_datasets/test_sample.txt --trials 5 --eval-batches 2

# Longer tune with persistent study on SQLite
aios hrm-hf autotune --dataset-file training_data/curated_datasets/test_sample.txt --trials 30 --storage sqlite:///artifacts/optimization/autotune.db --study-name actv1-tune-v1 --seed 42 --wandb
```

Implementation surfaces:
- `src/aios/cli/hrm_hf/autotune.py`: Typer command; Optuna study setup; objective wrapper.
- `src/aios/hpo/spaces.py`: central search-space builders and safe bounds.
- `src/aios/hpo/objectives.py`: training/eval objective, robust exception handling.

---

## GUI design and UX

Locations: `src/aios/gui/components/`

Additions:
- Training panel: W&B toggle and advanced fields (project, entity, group, tags, run-name, offline). These map directly to CLI flags and `TrainingConfig` passthrough.
- Autotune panel: a small form for trial count, timeout, pruner, sampler, search-space limits, and a “Start Autotune” button that spawns `hrm-hf autotune` as a subprocess with a progress view and best-trial summary.
- Orchestration tab: run the `hrm-train` flow with inputs (dataset, eval, brain-name, W&B toggle). Show live logs and a link to Prefect UI (optional).

Suggested files:
- `hrm_training/wandb_fields.py` (reusable widget group)
- `hrm_training/autotune_panel.py`
- `flows/flow_runner.py` (thin wrapper to call Python flows or CLI)

Process notes:
- Use `TrainingConfig.to_cli_args()` for argument generation and append W&B/HPO/flow specific flags.
- Ensure long-running subprocesses drain stdout continuously to avoid deadlocks.
- Persist last-used settings in `~/.config/aios/gui_prefs.yaml` for convenience.

---

## Implementation plan (dev checklist)

1) W&B shim and wiring
- [ ] Create `src/aios/core/logging/wandb_logger.py` with a tiny adapter: `init(config: dict, flags)`, `log(dict, step)`, `log_artifact(path, name, type)`, `finish()`; internally no-op if W&B missing or disabled.
- [ ] Add CLI flags to `hrm_hf_cli.train_actv1` (see above) and plumb into `TrainingConfig` or function kwargs.
- [ ] In `train_actv1_impl`, when enabled:
	- [ ] init run with config
	- [ ] per-step: log metrics
	- [ ] on checkpoint/eval/end: upload artifacts
	- [ ] handle offline/no-auth gracefully

2) Prefect flow
- [ ] Add `src/aios/flows/hrm_train_flow.py` with `@task` steps and `@flow` wrapper.
- [ ] Add `src/aios/cli/flows_cli.py` exposing `aios flow hrm-train`.
- [ ] Optional: emit a short README in `docs/flows/HRM_TRAIN_FLOW.md` showing usage and Prefect UI link.

3) Optuna autotune
- [ ] Add `src/aios/cli/hrm_hf/autotune.py` Typer command and register in `hrm_hf_cli.register`.
- [ ] Implement `src/aios/hpo/spaces.py` and `src/aios/hpo/objectives.py`.
- [ ] Robust OOM capture: detect CUDA OOM and DML errors; mark trial failed with note.
- [ ] Emit `artifacts/optimization/autotune_<timestamp>.jsonl` with trial summaries.

4) Schema and config
- [ ] Update `TrainingConfig` with W&B-related passthrough fields only if needed, or treat as non-config CLI flags.
- [ ] Add env-driven defaults: `AIOS_WANDB_PROJECT`, `AIOS_WANDB_ENTITY`, `WANDB_MODE`.

5) Packaging and optional deps
- [ ] Add extras in `pyproject.toml`:
	- `[project.optional-dependencies] tracking = ["wandb>=0.17"]`
	- `orchestration = ["prefect>=2.16"]`
	- `hpo = ["optuna>=3.6"]`
- [ ] Document install snippets in `docs` and help texts.

6) Tests and dry-runs
- [ ] Unit test the W&B shim in no-op mode (no wandb installed) and with env offline.
- [ ] CLI smoke test: run 1-step training with `--log-file artifacts/brains/actv1/metrics.jsonl` (see VS Code tasks already present) and verify no exceptions when `--wandb` is toggled.
- [ ] HPO smoke: 2–3 trials, `--eval-batches 1`, ensure failure handling works.
- [ ] Flow smoke: run with toy dataset; verify all tasks execute and files exist.

---

## HPO search space (initial)

Primary knobs and safe ranges:
- `lr`: loguniform [1e-6, 1e-4] (reduced for MoE stability)
- `warmup_steps`: int [20, 400]
- `chunk_size`: categorical [1024, 2048, 4096]
- `moe_load_balance_loss_coef`: loguniform [5e-3, 2e-2]

Objective: minimize final eval loss on a small held-out slice (`--eval-batches` 2–5 for speed). Consider averaging 2 seeds in later iterations for stability.

Pruners/samplers:
- Default pruner: MedianPruner (fast convergence on short runs)
- Default sampler: TPE

Output:
- Best-trial JSON emitted to `artifacts/optimization/best_trial.json`
- Full trial history: `artifacts/optimization/autotune_*.jsonl`

---

## Observability and artifacts

Metrics source of truth remains JSONL when `--log-file` is supplied. W&B mirrors and aggregates these:
- Per-step metrics: train_loss, lr, throughput
- Per-eval metrics: eval_loss, ppl (if computed)
- System: memory stats, GPU overflow indicators when available

Artifacts to attach (when present):
- Checkpoints from `save_dir`
- Final brain bundle from `bundle_dir`
- `metrics.jsonl`, `gpu_metrics_*.jsonl`, model card HTML if generated

---

## Security and privacy

- W&B: Respect offline mode; never crash if not logged in or network blocked.
- Redact secrets from configs before logging.
- Allow disabling artifact uploads via `--no-wandb` or env `WANDB_MODE=disabled`.

---

## Troubleshooting

Common issues and fixes:

- No module named wandb/prefect/optuna
	- Install optional deps: `pip install wandb prefect optuna`

- W&B login required
	- Run `wandb login` or use `--wandb-offline` for offline runs.

- CUDA OOM during HPO
	- Optuna trial should be marked failed automatically; reduce `batch_size` or enable `--use-chunked-training` with smaller `--chunk-size`.

- Prefect UI not accessible
	- This PF uses in-process Python flows; the UI is optional. You can still use `prefect orion start` separately if you want dashboards.

Windows/PowerShell notes:
- Prefer the `aios` entrypoint; if missing, call the module with `.venv\Scripts\python.exe -m aios.cli.aios ...`.
- Paths in examples use forward slashes or PowerShell-friendly backslashes.

---

## Testing and acceptance criteria

W&B
- [ ] Enabling `--wandb` produces a new run with step/eval metrics.
- [ ] Artifacts (at least metrics.jsonl) upload at run end or are skipped gracefully if offline.

Prefect flow
- [ ] `aios flow hrm-train` completes end-to-end on a toy dataset.
- [ ] If `--eval-file` is given, eval metrics are produced.
- [ ] If `--brain-name` is given, a packaged bundle exists.

Optuna
- [ ] At least 5 trials complete; failures are recorded (not fatal for the command).
- [ ] Best-trial config improves eval loss vs default on the toy dataset.

---

## Milestones

- M1 (1–2 days): W&B logging shipped + docs; CLI flags integrated; GUI toggle wired.
- M2 (1–2 days): Prefect flow + CLI wrapper; Optuna autotune command + GUI panel; smoke tests and docs.

---

## Appendix A – Example quickstarts (copy/paste)

Dry-run training (JSONL only):

```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --halt-max-steps 1 --eval-batches 1 --log-file artifacts/brains/actv1/metrics.jsonl
```

Training with W&B (online):

```powershell
aios hrm-hf train-actv1 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 4 --wandb --wandb-project aios-hrm --log-file artifacts/brains/actv1/metrics.jsonl
```

Run the flow (with eval + packaging):

```powershell
aios flow hrm-train --dataset-file training_data/curated_datasets/test_sample.txt --eval-file training_data/curated_datasets/test_sample.txt --brain-name demo-brain --wandb --train-args "--steps 100 --batch-size 4"
```

Autotune (5 quick trials):

```powershell
aios hrm-hf autotune --dataset-file training_data/curated_datasets/test_sample.txt --trials 5 --eval-batches 2 --wandb
```

If the `aios` script is not available:

```powershell
.venv\Scripts\python.exe -m aios.cli.aios hrm-hf train-actv1 --dataset-file training_data/curated_datasets/test_sample.txt --steps 1 --batch-size 2 --log-file artifacts/brains/actv1/metrics.jsonl
```

---

## Appendix B – Proposed file map

- `src/aios/core/logging/wandb_logger.py` – W&B adapter shim (no-op when unavailable).
- `src/aios/flows/hrm_train_flow.py` – Prefect tasks and flow.
- `src/aios/cli/flows_cli.py` – Typer command `aios flow hrm-train`.
- `src/aios/cli/hrm_hf/autotune.py` – Typer command and Optuna objective.
- `src/aios/hpo/spaces.py` – reusable search spaces.
- `src/aios/hpo/objectives.py` – training objective with robust error handling.
- `src/aios/gui/components/hrm_training/wandb_fields.py` – W&B UI widgets.
- `src/aios/gui/components/hrm_training/autotune_panel.py` – GUI for HPO.
- `docs/flows/HRM_TRAIN_FLOW.md` – optional flow readme.
