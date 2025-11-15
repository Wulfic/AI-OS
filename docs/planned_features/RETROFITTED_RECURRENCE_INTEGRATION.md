# Retrofitted Recurrence Integration Plan

Status: Planning
Date: November 15, 2025
Priority: High
Owners: AI-OS Core (Brains, Training), Research Enablement

References:
- Paper: "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence" (McLeish et al., arXiv:2511.07384)
- Code: https://github.com/mcleish7/retrofitting-recurrence (Apache-2.0)

---

## Executive Summary
Retrofitted recurrence converts existing transformer checkpoints into depth-recurrent models that reuse a core block multiple times at inference. The approach enables test-time compute scaling without increasing parameter count, while training remains efficient through recurrence scheduling, truncated backpropagation, and data curricula. This plan describes how AI-OS will integrate the technique so that users can convert Hugging Face (HF) and ACTV1 brains, train them with efficient curricula, and expose configurable recurrence at inference through CLI, automation flows, and the GUI.

Key value:
- Unlock higher reasoning quality per FLOP for math and logic tasks handled by AI-OS HRM pathways.
- Provide adaptive test-time compute within existing orchestration workflows (auto-training, evaluation, deployment).
- Offer reusable tooling for other latent-thinking techniques (e.g., adaptive depth, mixture-of-recursions).

---

## Objectives and Scope
In scope (initial release):
- Tooling to surgically convert supported HF models (TinyLlama, Llama-3.2-1B, OLMo-2-1B) into recurrent `(prelude, recurrent block, coda)` configurations.
- Training pipeline updates for recurrence scheduling, truncated backpropagation, and Muon optimizer support within `src/aios/core/hrm_training`.
- Inference controls that let operators select test-time recurrence counts per request or via policy (CLI, GUI, automation jobs).
- Evaluation harness updates to run GSM8K, MATH, MMLU, Arc across recurrence depths with reproducible FLOP reporting.

Deferred (future phases):
- Recurrence-aware versions of the custom ACTV1 architecture.
- Automatic layer selection derived from pruning scores or learned gating.
- Adaptive stopping criteria (data-dependent exit) and reinforcement-style training loops.

Out of scope:
- Fundamental changes to tokenizer or dataset ingestion beyond recurrence curriculum needs.

---

## Research Insights That Inform the Design
Highlights from McLeish et al. (2025) and accompanying repo:
1. **Model Surgery Pattern:** Split pretrained decoder into prelude, recurrent block, and coda; add linear adapter to combine recurrent state with prelude output. Removing mid layers while reusing later layers yields efficient recurrent cores.
2. **Initialization Matters:** Starting from pretrained weights dramatically outperforms random init; our tooling must preserve layer stats and residual scaling.
3. **Recurrence Scheduling:** Linearly or one-minus-sqrt scheduling of mean recurrences during training reduces FLOPs without hurting convergence.
4. **Truncated Backprop:** Limit gradient flow through most recent recurring steps (default 8) to control memory and runtime.
5. **Muon Optimizer:** Muon stabilizes recurrent training better than AdamW*; implementations should support both for reproducibility.
6. **Data Curriculum:** Healing with near-original data before math-heavy fine-tuning recovers language quality.
7. **Metrics:** Evaluate per recurrence depth and per FLOP to quantify test-time scaling benefits.

---

## Target Capabilities for AI-OS
1. **Conversion Toolkit** (`src/aios/ml/recurrence_toolkit/`):
   - Layer selection DSL driven by YAML recipe files.
   - CLI: `aios recurrence convert --model-path ...` producing safetensors and config metadata.
   - Async file IO to avoid blocking GUI threads during conversion.
2. **Training Enhancements** (`src/aios/core/hrm_training/`):
   - Config objects for recurrence scheduling, truncated BPTT depth, Muon optimizer, Poisson-Lognormal sampling.
   - Parallel dataloader pipelines supporting mixed dataset phases (FineWeb, Nemotron math) with asynchronous prefetch.
3. **Inference Controls**:
   - Extend `src/aios/core/hrm_models/hf_adapter.py` and `src/aios/core/inference.py` to accept `recurrence_policy` (fixed, heuristic, dynamic).
   - CLI flag `--test-recursions N`, GUI slider, automation policies reading from deployment configs.
4. **Evaluation Suite** (`src/aios/core/evaluation/`):
   - Benchmark runners that sweep recurrence counts asynchronously and record FLOPs, accuracy, latency.
   - Integration with `tests/gui/` harness for regression checks.
5. **Observability**:
   - Structured logs capturing recurrence depth, effective FLOPs, truncated gradient depth, adapter stats.
   - Diagnostics stored under `artifacts/evaluation/recurrence/`.

---

## Architecture Overview
### Module Layout (proposed)
```
src/aios/ml/recurrence_toolkit/
    __init__.py
    conversion.py          # layer slicing, adapter injection
    config_schemas.py      # TypedDict / pydantic schema for recipes
    scheduler.py           # recurrence curricula helpers
    adapters/
        hf_llama.py
        hf_tinyllama.py
        hf_olmo.py
src/aios/core/hrm_training/
    recurrence_config.py   # new dataclasses for training params
    recurrence_loop.py     # truncated BPTT, Muon optimizer wrapper
src/aios/cli/
    recurrence_cli.py      # user-facing commands
```

### Data and Control Flow
```
User request (CLI/GUI)
    -> recurrence_cli.convert_async()  [asyncio task]
        -> load HF model + recipe
        -> conversion.py builds recurrent model + adapter tensors
        -> writes checkpoint + metadata to artifacts/brains/recurrence/

Training job (auto_training, hrm_hf_cli)
    -> load recurrence_config
    -> scheduler.sample_train_depth() yields mean recurrences per step
    -> recurrence_loop.step() executes truncated BPTT
    -> metrics pushed via communicator async channel

Inference request
    -> inference.Route selects recurrent brain
    -> recurrence_policy decides test recurrence per prompt
    -> hf_adapter.forward_with_recurrence()
    -> results streamed back via async generator
```

### Parallel and Asynchronous Execution
- Conversion runs in background asyncio tasks tied to `core/communicator` so GUI remains responsive; heavy tensor ops offloaded to process pool where available.
- Training dataloaders leverage `torch.distributed` with overlapping prefetch to hide IO latency; scheduler updates broadcast via async RPC.
- Evaluation sweeps spawn parallel tasks per recurrence depth (bounded concurrency) to maximize GPU throughput without violating memory caps.

---

## Implementation Phases
### Phase 0: Discovery and Scaffolding (1 week)
Checklist:
- [ ] Draft conversion recipes for TinyLlama, Llama-3.2-1B, OLMo.
- [ ] Add `recurrence` section to `config/default.yaml` with feature flag off.
- [ ] Prototype loading of Muon optimizer in existing training stack.
Exit criteria: feature gated scaffolding merged, no behavioral change when disabled.

### Phase 1: Conversion Toolkit MVP (1-2 weeks)
Checklist:
- [ ] Implement `recurrence_toolkit.conversion` following GitHub scripts (layer slicing, adapter injection).
- [ ] Support safetensors export with metadata JSON (prelude indices, adapter dims, state init variance).
- [ ] Provide CLI `aios recurrence convert` and Typer completion.
- [ ] Add unit tests referencing small HF fixtures.
Exit criteria: able to convert TinyLlama checkpoint and reload via HF adapter without training.

### Phase 2: Training Pipeline Integration (2-3 weeks)
Checklist:
- [ ] Extend `TrainingConfig` objects with recurrence parameters (mean depth schedule, truncated steps, optimizer choice).
- [ ] Implement Poisson-Lognormal sampler and scheduler utilities.
- [ ] Update `train_epoch.py` to handle truncated backprop and Muon optimizer hooks.
- [ ] Add dual-phase dataset orchestration (healing + task) with asynchronous dataset queue.
- [ ] Provide reproducible scripts for 50B token math curriculum.
Exit criteria: training loop runs recurrent TinyLlama and logs scheduled depths and FLOPs.

### Phase 3: Inference and Evaluation Controls (1-2 weeks)
Checklist:
- [ ] Extend HF adapter to accept recurrence depth override per forward pass.
- [ ] Add CLI/GUI controls (slider + numeric entry) to inference panels.
- [ ] Integrate evaluation sweeps into `aios evaluation run` command with concurrency limit.
- [ ] Implement policy hooks for auto-inference (e.g., escalate recurrence if confidence low).
Exit criteria: evaluation report comparing recurrence depths and baseline generated automatically.

### Phase 4: Observability, Docs, and Rollout (1 week)
Checklist:
- [ ] Structured logging and telemetry dashboards (Prometheus counters) for recurrence metrics.
- [ ] Documentation in `docs/guide/` describing workflow end-to-end.
- [ ] Migration guide for operators (how to convert existing HF brains).
- [ ] Feature flag default remains off pending beta feedback.
Exit criteria: docs published, monitoring pipelines validated on staging.

---

## Configuration and UX Integration
- **Config files:** add `brains.recurrence` block (conversion recipe path, default train/test recurrences, truncated depth, scheduler type).
- **CLI:**
  - `aios recurrence convert`
  - `aios hrm-train --train-recurrence <mean> --truncate-depth <k>`
  - `aios inference run --test-recurrence {auto|N}`
- **GUI:** new section in HRM panels with:
  - Model surgery recipe picker (loads YAML metadata).
  - Training scheduler controls (start, end recurrences, truncated steps).
  - Inference slider for recurrence depth; preview of estimated FLOPs.
- **Automation:** orchestrator supports recurrence options in job definitions; asynchronous status updates include recurrence-specific metrics.

---

## Data, Compute, and Scheduling Considerations
- Datasets: integrate FineWeb-Edu, Nemotron-CC-Math, Nemotron General through existing dataset registry with staged queues.
- Healing phase executed first (FineWeb) using async pipeline to reuse cached tokenization.
- Training hardware: optimize for 4x AMD MI300A nodes or equivalent; plan fallback for single-node RTX 4090 via reduced recurrence schedule.
- FLOP accounting: extend budget planner (`src/aios/core/budgets.py`) to use recurrent FLOP formula `(6*N1 + 2*N2)*tokens`.
- Checkpoint cadence: persist prelude, recurrent core, adapter, and metadata separately for selective updates.

---

## Testing and Validation Strategy
1. **Unit tests:**
   - Layer slicing correctness using synthetic transformer layers.
   - Scheduler outputs (mean, variance) vs analytic expectations.
2. **Integration tests:**
   - Conversion + reload smoke test in CI using TinyLlama toy checkpoint.
   - Training loop truncated backprop validation (ensuring gradients zero beyond window).
3. **Evaluation regression:**
   - Nightly job generating GSM8K accuracy table across recurrences; compare against baseline thresholds.
4. **Performance tests:**
   - Measure throughput vs baseline for recurrence schedule {1,2,4,8,16,32}; ensure scaling behavior matches paper trends.
5. **GUI tests:**
   - Selenium harness to toggle recurrence controls without deadlocking main loop (ensuring async tasks handled).

---

## Risks and Mitigations
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Conversion divergence from upstream scripts | Incorrect hidden state alignment | Cross-check hidden activations using repo comparison scripts; add automated numerical diff with tolerance <1e-5. |
| Muon optimizer availability on Windows | Training failure for Windows users | Keep AdamW* fallback; gate Muon behind optional extra, detect platform at runtime. |
| GPU memory pressure at high recurrence | OOM during evaluation | Default truncated depth to 8, expose config, add guard rails that cap recurrence based on available VRAM from `torch.cuda.mem_get_info`. |
| Asynchronous conversion blocking GUI | Poor UX | Run conversions in background process pool, surface progress via communicator events. |
| Licensing compliance | Legal risk when vendoring code | Preserve Apache-2.0 NOTICE, document provenance in `LICENSE_THIRD_PARTY.md`. |

---

## Success Metrics
- Converted recurrent model achieves >= +5 absolute GSM8K accuracy at test recurrence 32 compared to non-recurrent baseline (matching paper trends).
- Training throughput within 15 percent of target (Poisson-Lognormal schedule, truncated depth 8) relative to fixed depth baseline per FLOP.
- Inference latency scaling linear with recurrence depth and no crashes across 1,2,4,8,16,32 settings.
- Automation job success rate unchanged; logs include recurrence metrics in 95 percent of runs.
- User satisfaction: positive feedback from at least three internal evaluators after beta trial.

---

## Immediate Next Actions
1. Circulate this plan for maintainer review; collect sign-off on scope and resource estimates.
2. Draft initial conversion recipes by inspecting upstream scripts for TinyLlama and Llama.
3. Create feature flag entries and placeholder modules for recurrence toolkit.
4. Schedule spike to integrate Muon optimizer with current training loops and measure baseline stability.

---

## Appendix A: Open Questions
- Preferred format for storing recurrence metadata (YAML vs JSON) when publishing converted brains.
- Whether to support dynamic recurrence policies (confidence-based) in initial release or defer.
- Alignment with ACTV1 roadmap: should we plan for adapter hooks now or schedule follow-up RFC?

---

Document version: 0.1 (draft)