# Dynamic Subbrains (Mixture of Experts)

Purpose: Sparse expert routing for efficiency and specialization. Includes expert metadata/registry, expert-only training, and goal-aware routing hooks.

Status: Implemented core MoE in ACTv1 + expert registry and expert-only training. Goal-aware biasing and full GUI management are WIP.

Key files:
- MoE layer and routing stats: `src/aios/core/hrm_models/moe_layer.py`
- Goal-aware router (biasing by goals): `src/aios/core/hrm_models/goal_aware_router.py`
- Expert metadata and registry: `src/aios/core/hrm_models/expert_metadata.py`
- ACTv1 model uses MoE by default: `src/aios/core/hrm_models/impl/hrm_act_v1.py` and `src/aios/core/brains/hf_brain.py`
- Training CLI flags (MoE and experts): `src/aios/cli/hrm_hf_cli.py`
- Expert-only training implementation: `src/aios/cli/hrm_hf/expert_training.py`
- Metrics logging (load balancing + expert usage): `src/aios/cli/hrm_hf/training_logic/train_epoch.py`
- GUI Subbrains Manager (WIP): `src/aios/gui/components/subbrains_manager_panel/`

See also:
- Core training: `CORE_TRAINING.md`
- Memory optimization (8-bit optimizer, AMP): `MEMORY_OPTIMIZATION.md`
- Multi-GPU/Windows-friendly parallel training: `MULTI_GPU_DISTRIBUTED.md`, `PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md`
- Goals CLI (link goals to experts): `CLI_COMMANDS.md` (Goals section)

## What you get
- Sparse MoE with top-k expert routing per token for ~75% compute reduction while increasing capacity.
- Automatic auxiliary load-balancing loss to prevent collapse.
- Periodic expert-usage metrics in your log file (routing probabilities, token counts).
- Expert-only training mode that produces standalone expert checkpoints and updates a persistent registry.
- Goal-aware router module (WIP hookup) to bias expert selection by active goals.

## Commands (PowerShell, Windows-first)

1) Train ACTv1 with MoE (default enabled)
- Flags come from `aios hrm-hf train-actv1`. MoE-related flags:
	- `--use-moe/--no-moe` (default: `--use-moe`)
	- `--num-experts <int>` (default: 8)
	- `--num-experts-per-tok <int>` (top-k, default: 2)
	- `--moe-capacity-factor <float>` (default: 1.25)
	- `--auto-adjust-lr/--no-auto-adjust-lr` (default: on; reduces LR for MoE stability)

Example (small dry-run, logs expert usage):

		aios hrm-hf train-actv1 `
			--model artifacts/hf_implant/base_model `
			--dataset-file training_data/curated_datasets/test_sample.txt `
			--steps 20 --batch-size 8 `
			--use-moe --num-experts 8 --num-experts-per-tok 2 --moe-capacity-factor 1.25 `
			--log-file artifacts/brains/actv1/metrics.jsonl

Disable MoE (train dense FFN instead):

		aios hrm-hf train-actv1 `
			--model artifacts/hf_implant/base_model `
			--dataset-file training_data/curated_datasets/test_sample.txt `
			--steps 20 --batch-size 8 `
			--no-moe `
			--log-file artifacts/brains/actv1/metrics.jsonl

Tips:
- Lower `--num-experts-per-tok` to 1 to reduce active compute/memory on very constrained GPUs.
- Keep `--auto-adjust-lr` enabled unless you know what you’re doing; MoE routers can be unstable at higher LR.

2) Train a standalone expert only (writes artifacts/experts and updates registry)
- Trigger by passing `--expert-id <string>` to `train-actv1`.
- Uses a lightweight FeedForward expert, saves as `.safetensors`, and writes/updates `artifacts/experts/registry.json`.

Example (quick expert build):

		aios hrm-hf train-actv1 `
			--model artifacts/hf_implant/base_model `
			--dataset-file training_data/curated_datasets/test_sample.txt `
			--steps 3 --batch-size 2 `
			--expert-id test-expert-004 `
			--default-goal "Improve summarization quality" `
			--log-file artifacts/experts/test-expert-004/metrics.jsonl

Outputs:
- `artifacts/experts/test-expert-004/expert.safetensors`
- `artifacts/experts/registry.json` (created or updated with metadata including `expert_id`, `name`, `goals`, `checkpoint_path`, `is_active/is_frozen`, hierarchy fields)

3) Link goals to experts (biasing signal for router)
- Goals live in the directives DB and can be associated with an expert.
- Commands are under `aios goals-*`.

Examples:

		# Add a goal and link to an expert immediately
		aios goals-add "Improve summarization quality" --expert-id test-expert-004

		# Link an existing goal to an expert
		aios goals-link-expert 42 test-expert-004

		# List active goals
		aios goals-list

		# List goals for an expert
		aios goals-list-for-expert test-expert-004

Notes:
- The `GoalAwareRouter` module supports biasing toward experts linked to active goals. Integration into the default training/inference loop is in progress; track `src/aios/core/hrm_models/goal_aware_router.py`.

## Inputs and Outputs

Inputs (training flags relevant to MoE/experts):
- `--use-moe`, `--num-experts`, `--num-experts-per-tok`, `--moe-capacity-factor`, `--auto-adjust-lr`
- Standard training knobs: `--max-seq-len`, `--batch-size`, `--steps`, `--lr`, `--amp`, `--gradient-checkpointing`, etc.
- Expert-only mode: `--expert-id <id>` plus optional `--default-goal` to seed goal linkage.

Outputs (files and metrics):
- Brain training logs: your `--log-file` JSONL includes, when MoE is enabled:
	- `lb_loss`: load-balancing loss value (applied internally; coef ~0.05)
	- Periodic `expert_usage` events with:
		- `avg_routing_prob`: average probability per expert
		- `token_counts`: tokens routed to each expert
		- `total_tokens`: total tokens seen when sampled
- Expert-only training:
	- `artifacts/experts/<expert-id>/expert.safetensors`
	- `artifacts/experts/registry.json` with entries like:
		- `expert_id`, `name`, `description`, `category`, `goals`, timestamps
		- `is_active`, `is_frozen`, `parent_expert_id`, `child_expert_ids`
		- `checkpoint_path`: e.g., `artifacts\\experts\\<expert-id>\\expert.safetensors`
		- `training_config`: hidden/intermediate sizes, steps, batch size, etc.

## How routing works (high level)
- Each MoE layer computes router logits over N experts and activates the top-k experts per token (`--num-experts-per-tok`).
- An auxiliary load-balancing loss is added to spread traffic across experts and avoid collapse. Metrics include `lb_loss` and `moe_layers` count.
- `expert_usage` entries in logs let you validate router health and specialization during training.

## GUI: Subbrains Manager (WIP)
- The panel shows the expert registry with counts and hierarchy and can refresh from disk:
	- Code: `src/aios/gui/components/subbrains_manager_panel/`
	- Data loader: `data_manager.py` reads `artifacts/experts/registry.json`
- Actions like create/delete/freeze are currently placeholders that print “CLI command needed”. Use CLI for expert training and goals linking.
- As features land, the panel will manage expert lifecycle and goal associations directly.

## Troubleshooting
- Training is unstable (NaNs/Inf) with MoE:
	- Keep `--auto-adjust-lr` enabled (default). It reduces LR for MoE automatically.
	- Lower base `--lr` and/or `--num-experts-per-tok`.
	- Ensure AMP/precision settings are stable (`--amp` by default; try `--model-dtype bf16` on supported GPUs).
- VRAM pressure with many experts:
	- Reduce `--num-experts` and/or set `--num-experts-per-tok 1`.
	- Use `--amp`, `--gradient-checkpointing`, and `--use-8bit-optimizer` (requires bitsandbytes).
- No expert usage metrics in log:
	- Ensure `--use-moe` is on.
	- `expert_usage` logs are periodic (every ~100 steps) and sampled from early layers; short runs may not emit them.
- Can’t find the expert registry:
	- Path: `artifacts/experts/registry.json`. It’s created on first expert-only training.

## Try it quickly
- Minimal MoE run with metrics:

		aios hrm-hf train-actv1 `
			--model artifacts/hf_implant/base_model `
			--dataset-file training_data/curated_datasets/test_sample.txt `
			--steps 30 --batch-size 4 `
			--use-moe --num-experts 8 --num-experts-per-tok 2 `
			--log-file artifacts/brains/actv1/metrics.jsonl

- Train one tiny expert and link a goal:

		aios hrm-hf train-actv1 `
			--model artifacts/hf_implant/base_model `
			--dataset-file training_data/curated_datasets/test_sample.txt `
			--steps 3 --batch-size 2 `
			--expert-id demo-expert-001 `
			--default-goal "Focus on troubleshooting clarity" `
			--log-file artifacts/experts/demo-expert-001/metrics.jsonl

		aios goals-list-for-expert demo-expert-001

## Notes and next steps
- Goal-aware router module exists and exposes bias controls; full wiring to training/inference loops and GUI controls is in progress.
- The GUI Subbrains Manager will gain create/delete/freeze operations backed by CLI endpoints.
- We’ll expose advanced router knobs (e.g., load-balance loss coef) once stabilized.

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)