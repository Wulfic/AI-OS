## Unlimiformer integration for unlimited-length inputs

This planned feature proposes integrating Unlimiformer (NeurIPS 2023) into AI‑OS to enable effectively unlimited-length inputs for supported Hugging Face models during evaluation/inference and optionally during training.

References
- Paper: “Unlimiformer: Long-Range Transformers with Unlimited Length Input” (Bertsch et al., 2023)
- Code: https://github.com/abertsch72/unlimiformer (MIT License)


### Why this matters
- Handle very long prompts (books, entire project repos, long transcripts) without truncation.
- Keep base attention unchanged; add retrieval-based attention above a chosen layer (“layer_begin”).
- Works with existing pretrained models (e.g., BART/T5 encoder–decoder; decoder-only models such as Llama‑2 according to the repo README) and can be used purely at evaluation time or for specialized training regimes.


## Scope and success criteria

In scope (Phase 1)
- Evaluation/inference integration for HF-backed “brains” (aios.core.hrm_models.hf_adapter) with decoder-only models (LLaMA family) using Unlimiformer’s retrieval at generation time.
- Early-stopping evaluation using Unlimiformer on long validation inputs.
- Configuration flags in AI‑OS to toggle Unlimiformer without breaking existing flows.

In scope (Phase 2)
- Training-time modes: “random-encoded” and “retrieval” training; alternating schedule as in the paper.
- Encoder–decoder support (e.g., BART/T5) on summarization-like datasets.

Out of scope (initially)
- Direct integration into the custom ACTV1 HRM model (non-HF architecture). This would require substantial porting of the attention hook path and is a potential Phase 3 exploration.

Success criteria
- P0: For a HF LLaMA‑family model configured in AI‑OS, users can enable Unlimiformer and successfully generate from inputs > base context (e.g., ≥ 100k tokens) with stable memory via FAISS datastore; outputs match expectations on long-doc summarization prompts.
- P1: Automated test(s) verify that enabling Unlimiformer leaves standard generation unchanged on short inputs (parity within tolerance) and does not regress normal inference when disabled.


## High-level design

Concept
- Unlimiformer augments a HF model with a retrieval path over the full input. Layers below `layer_begin` attend as usual over the last `context_window_size` tokens; from `layer_begin` upwards, cross/retrieval attention uses K‑NN over a datastore of hidden states (optionally via FAISS) to bring in relevant tokens from the entire (long) input.

Integration strategy in AI‑OS
1) HF integration layer
   - Extend or wrap `HFCausalLM_HRMAdapter` to optionally enable Unlimiformer for decoder-only models.
   - Provide a thin compatibility shim that mirrors the Unlimiformer repo’s expected arguments (e.g., `test_unlimiformer`, `layer_begin`, `use_datastore`, `gpu_index`, `gpu_datastore`, `index_devices`, `datastore_device`, `knn`).
   - Keep Unlimiformer disabled by default.

2) Vendor minimal Unlimiformer components
   - Add `aios.integrations.unlimiformer` module containing the minimal files required from the upstream `src/` (per MIT license), focusing on inference paths: model wrapper, indexing/datastore, and argument schema.
   - Avoid forking the training script initially; rely on our Typer CLI and config system.

3) Configuration plumbing
   - Add optional config keys under `brains.trainer_overrides` (HF mode) to toggle Unlimiformer and pass-through parameters.
   - Example keys (all optional):
     - `unlimiformer.enabled: false`
     - `unlimiformer.layer_begin: 16`
     - `unlimiformer.knn: true`
     - `unlimiformer.use_datastore: true`
     - `unlimiformer.gpu_datastore: true`
     - `unlimiformer.gpu_index: true`
     - `unlimiformer.index_devices: [1]`
     - `unlimiformer.datastore_device: 1`
     - `unlimiformer.context_window_size: 4096` (fallback sliding window for lower layers)
     - `unlimiformer.eval_max_source_length: 999999`
   - Defaults keep feature off, requiring no extra dependencies.

4) Dependency management
   - Add optional extra `[unlimiformer]` in `pyproject.toml` including: `faiss-cpu` (Windows-friendly), `faiss-gpu` (Linux/CUDA environments, optional), `transformers>=4.33`, and numpy/scipy as needed.
   - Detect platform; default to FAISS‑CPU on Windows; allow GPU index/datastore where supported.

5) Memory/runtimes
   - For very long inputs, use FAISS datastore (CPU by default) to keep GPU memory stable.
   - Provide flags to move index/datastore to GPU when memory allows.

6) UX
   - CLI switches added to relevant commands (e.g., `aios hrm-hf chat` or evaluation flows) to toggle Unlimiformer.
   - GUI toggle in HRM/HF panels: “Unlimited context (Unlimiformer)” with tooltips and safe defaults.


## Detailed design and APIs

### Module layout
- `src/aios/integrations/unlimiformer/`
  - `__init__.py`: feature gate and version checks
  - `adapter.py`: small wrapper class that:
    - Accepts a HF model and tokenizer
    - Builds/attaches Unlimiformer components
    - Exposes a `.generate(...)` and `.prepare_inputs_for_long_context(prefix, prompt, suffix)` helper
  - `datastore.py`: thin wrapper over FAISS index/datastore setup (CPU/GPU), device routing, and persistence
  - `args.py`: shared dataclass/TypedDict mapping AI‑OS config to Unlimiformer expected settings
  - `compat.py`: utilities to resolve `layer_begin` based on model depth and provide recommended defaults


### HF adapter extension points
Location: `src/aios/core/hrm_models/hf_adapter.py`

Add optional Unlimiformer activation path:
- New optional constructor parameter `unlimiformer: Optional[UnlimiformerConfigLike] = None`.
- When provided and `unlimiformer.enabled` is true, wrap the underlying `self.model` with the Unlimiformer augmentation via `aios.integrations.unlimiformer.adapter.enable(self.model, tokenizer, cfg)`.
- Ensure `forward(...)` and `generate(...)` continue to route through HF model seamlessly; Unlimiformer alters attention inside the model graph post-`layer_begin`.

Datastore/index devices
- Respect `datastore_device` and `index_devices` (GPU ids) when available; otherwise default to CPU FAISS.
- Expose graceful fallbacks on Windows to CPU FAISS.


### Configuration surface (proposed)
Under `brains.trainer_overrides` when `kind: hf`:

```
unlimiformer:
  enabled: false
  layer_begin: null          # null → auto (> 1/2 of total layers)
  context_window_size: 4096  # sliding window for lower layers
  knn: true
  use_datastore: true
  gpu_datastore: false       # default false on Windows
  gpu_index: false
  datastore_device: 1        # optional
  index_devices: [1]         # optional list
  eval_max_source_length: 999999
```

CLI flags (Typer) mirror the above and default to disabled.


### Training integration (Phase 2)
- Add a new command for HF fine‑tuning with long inputs using Unlimiformer:
  - `aios hrm-hf train-hf-unlimiformer` (or integrate into `training_cli.py`)
  - Modes: `--unlimiformer-training`, `--random-unlimiformer-training`, `--alternating` as in the repo
- Datasets: begin with summarization datasets (GovReport, SummScreen, BookSum) to mirror paper experiments.
- Early stopping: Evaluate with `test_unlimiformer` enabled to select checkpoints.


## Dependencies and compatibility
- Unlimiformer repo is MIT; vendoring minimal files is permissible with attribution.
- FAISS:
  - `faiss-cpu` works on Windows; `faiss-gpu` is Linux/CUDA‑oriented. Provide runtime detection and degrade to CPU index/datastore on unsupported platforms.
- Transformers version pinning: test with the current repo baseline; maintain a gated extra if newer transformers are required by Unlimiformer.


## Risks, constraints, and mitigations
- Windows GPU FAISS availability: default to CPU FAISS; add clear logs; allow GPU usage on Linux/CUDA.
- Model coverage: initial focus on LLaMA‑family decoder‑only; expand to encoder–decoder later.
- Performance variance: `layer_begin` is critical; provide auto‑heuristic (> half of total layers) and expose as a user setting.
- Memory pressure: ensure `use_datastore` defaults to true for very long inputs; include telemetry in logs.
- Maintenance: vendor minimal code only; isolate under `aios.integrations.unlimiformer` to avoid invasive changes.


## Test plan
Automated
- Unit test: enabling Unlimiformer on a short prompt yields outputs close to baseline generation (within tolerance on logits/perplexity or token parity for deterministic settings).
- E2E smoke: long input (> 64k tokens) generates without OOM, with FAISS CPU datastore, and token streaming stays responsive.

Manual
- LLaMA‑2 chat model: summarize a 100k‑token text with `unlimiformer.enabled=true`; compare quality and latency with and without GPU index/datastore.


## Milestones and timeline
- P0 (1–2 days)
  - Vendor minimal Unlimiformer inference code and feature‑gate in HF adapter; add config flags; CPU FAISS only; LLaMA‑family support.
  - Add docs and CLI flags; write basic unit/E2E smoke tests.
- P1 (3–5 days)
  - GPU datastore/index support on Linux/CUDA; GUI toggle; early‑stopping evaluation integration.
- P2 (1–2 weeks)
  - Training modes (random‑encoded, retrieval, alternating) for encoder–decoder tasks; sample recipes.
- P3 (exploratory)
  - Investigate feasibility of HRM (ACTV1) architecture adaptation.


## Rollout and observability
- Feature flag: off by default.
- Structured logs: write Unlimiformer settings and device placements; record datastore/index memory footprint and retrieval latency.
- Add a troubleshooting section in docs (common FAISS issues, Windows notes).


## Next actions (Phase 1)
1) Add optional dependency group `[unlimiformer]` to `pyproject.toml` (faiss-cpu; platform‑specific notes for faiss‑gpu).
2) Create `aios.integrations.unlimiformer` module and vendor minimal code.
3) Extend `HFCausalLM_HRMAdapter` to accept an optional `unlimiformer` config and wrap the model.
4) Add CLI and config flags; wire through `brains.trainer_overrides`.
5) Add unit and smoke tests; document usage and caveats.


---
Maintainers: Please comment on feasibility, preferred model targets for Phase 1, and any CI/platform constraints (especially Windows FAISS).


## Phase breakdown and checklists

Below are execution-ready checklists per phase with acceptance criteria and exit gates.

### Phase 0 — Repo readiness and scaffolding (0.5–1 day)

Checklist
- [ ] Add optional dependency extra `[unlimiformer]` (faiss-cpu; pin transformers range if needed)
- [ ] Create `src/aios/integrations/unlimiformer/` with MIT attribution NOTICE
- [ ] Define `UnlimiformerConfig` TypedDict and feature-gate helpers
- [ ] Add config block (disabled) under `brains.trainer_overrides.unlimiformer`
- [ ] Wire no-op read/validation in HF adapter (no behavior change yet)

Acceptance criteria
- [ ] App starts unchanged with default config
- [ ] CI/lint/tests remain green

Exit gate
- [ ] Feature is fully hidden behind `enabled: false`


### Phase 1 — Inference PoC (decoder-only, Windows-friendly) (1–2 days)

Checklist
- [ ] Vendor minimal Unlimiformer inference components (index/datastore + model hook)
- [ ] Implement `enable_on_model(model, tokenizer, cfg)` to attach retrieval attention from `layer_begin`
- [ ] Default to FAISS-CPU datastore/index on Windows; detect and log device placement
- [ ] Add CLI flags to evaluation/generation path to toggle Unlimiformer
- [ ] Provide an example: summarize a long text (≥100k tokens) with LLaMA‑family model
- [ ] Unit test: enabling Unlimiformer on short inputs ≈ baseline outputs
- [ ] E2E smoke: long input completes without OOM using FAISS-CPU; stream tokens

Acceptance criteria
- [ ] Long-input generation succeeds on Windows with FAISS-CPU
- [ ] Short-input parity within tolerance when Unlimiformer is enabled
- [ ] Clear logs for index/datastore size and latency

Exit gate
- [ ] Docs updated with Windows usage and limitations


### Phase 1.1 — Linux/CUDA GPU index/datastore (1–2 days)

Checklist
- [ ] Detect CUDA and allow `gpu_index`/`gpu_datastore`
- [ ] Validate FAISS-GPU install path with helpful errors/fallbacks
- [ ] Benchmark basic throughput vs CPU on the same prompt

Acceptance criteria
- [ ] GPU index/datastore operates with visible latency reduction vs CPU on eligible hardware
- [ ] Graceful fallback to CPU with warning logs when unavailable

Exit gate
- [ ] Docs include CUDA/GPU prerequisites and troubleshooting


### Phase 1.2 — GUI toggle + early-stopping evaluation (1–2 days)

Checklist
- [ ] Add GUI switch “Unlimited context (Unlimiformer)” under HF trainer settings
- [ ] Wire values: enabled, layer_begin, datastore/index devices
- [ ] Integrate Unlimiformer during evaluation used for early stopping (if configured)

Acceptance criteria
- [ ] GUI toggle persists to config and is honored by evaluation
- [ ] Early-stopping can leverage long-context evaluation without regressions

Exit gate
- [ ] UX doc and tooltip coverage for each setting


### Phase 2 — Training modes and encoder–decoder support (1–2 weeks)

Checklist
- [ ] Add CLI to enable training modes: `--unlimiformer-training`, `--random-unlimiformer-training`, `--alternating`
- [ ] Implement training-time hooks (sampling from full input, random-encoded mode)
- [ ] Add dataset recipes for GovReport/SummScreen/BookSum (encoder–decoder models)
- [ ] Early-stopping/eval configured with `test_unlimiformer`
- [ ] Add metrics logging for retrieval hits/latency during training

Acceptance criteria
- [ ] Reproduce representative improvements similar to paper on at least one dataset (relative, not exact)
- [ ] Training is stable with recommended defaults; resource use documented

Exit gate
- [ ] Docs include full training command examples and cost guidance


### Phase 3 — ACTV1 HRM exploration (exploratory, timeboxed)

Checklist
- [ ] Spike: map HRM attention blocks to identify hook points analogous to Unlimiformer
- [ ] Prototype minimal retrieval layer that augments HRM without changing base attention math
- [ ] Measure memory/perf impact on long sequences

Acceptance criteria
- [ ] Decision doc: feasible/not feasible with outline of required changes and risks

Exit gate
- [ ] If feasible, produce a separate follow-on feature spec; otherwise close spike with rationale


### Phase 4 — Hardening, docs, and demos (2–4 days)

Checklist
- [ ] Consolidate structured logs and counters: index size, retrieval QPS/latency, device placement
- [ ] Expand troubleshooting guide (FAISS install, GPU issues, Windows notes)
- [ ] Create repeatable demos (scripts + sample prompts) for long‑doc summarization and QA
- [ ] Finalize API stability and mark feature as beta (still default off)

Acceptance criteria
- [ ] Demo scripts run on Windows (CPU FAISS) and Linux/CUDA (GPU FAISS)
- [ ] Users can follow docs to reproduce results without support

Exit gate
- [ ] Sign-off from maintainers to publish feature docs


## Testing matrix (summary)

- Platforms
  - [ ] Windows 11 + CUDA GPU present → default CPU FAISS works; GPU path explicitly disabled by default
  - [ ] Linux + CUDA 12.x → CPU and GPU FAISS paths
  - [ ] macOS (CPU only) → CPU FAISS path
- Models
  - [ ] LLaMA‑family (decoder‑only) — Phase 1
  - [ ] BART/T5 (encoder–decoder) — Phase 2
- Inputs
  - [ ] Short (≤ 2k tokens) parity tests
  - [ ] Long (≥ 100k tokens) smoke + latency/throughput sampling
- Modes
  - [ ] Inference only
  - [ ] Training: random‑encoded, retrieval, alternating (Phase 2)

