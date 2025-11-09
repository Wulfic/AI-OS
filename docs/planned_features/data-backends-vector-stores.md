## PF-005: Data backends and vector stores

This document is a comprehensive design and delivery guide for adding pluggable dataset backends (Hugging Face Datasets streaming, WebDataset shards) and a minimal vector store layer (Qdrant or LanceDB) to enable scalable data ingestion and future retrieval/memory features across CLI and GUI.

### Goals (What we ship)

- Add a first-class dataset backend abstraction used by ACTV1 training flows.
- Provide two new backends in addition to the current custom loader:
	- Hugging Face Datasets (streaming, with light filtering and caching knobs)
	- WebDataset shards (local/URL tar streams)
- Add a minimal vector store interface with swappable drivers (Qdrant, LanceDB) for embedding upsert/query.
- Expose all of the above via CLI and the HRM Training GUI panel with a clear, safe default path and backward compatibility.

### Motivation

- Improve scalability and reproducibility of dataset handling.
- Prepare for retrieval-augmented features and expert selection memories.

### Non-goals (Out of scope for PF-005)

- Full RAG pipelines, memory policies, or retrieval-integrated training loops.
- On-the-fly embedding model training; we only provide an API and a thin client.

---

## Architecture overview

### 1) Dataset backends

Introduce a small interface and pluggable implementations:

- Module: `src/aios/cli/hrm_hf/data_backends/`
	- `base.py`: defines the interface and utilities
	- `custom.py`: current file/dir/CSV/jsonl sampler (existing behavior)
	- `hf.py`: HF Datasets streaming loader
	- `webdataset.py`: WebDataset shard reader

Contract (Pythonic sketch):

```python
# src/aios/cli/hrm_hf/data_backends/base.py
from typing import Iterable, Protocol, Optional, Dict, Any

class SampleBatch(Protocol):
		input_ids: Any  # torch.LongTensor [B, T]
		attn_mask: Any  # Optional, same shape

class Backend(Protocol):
		def __init__(self, cfg: Dict[str, Any]): ...
		def iter_text(self) -> Iterable[str]: ...  # raw text stream (pre-tokenization)
		# Optional: direct batching/tokenization if desired
		def iter_batches(self) -> Iterable[SampleBatch]: ...
```

Integration point in training:

- In `src/aios/cli/hrm_hf/train_actv1.py` the data loader path delegates to a builder:
	- `from .data_backends import build_backend`
	- `backend = build_backend(config)`
	- Use `backend.iter_text()` and preserve existing tokenization/batching logic, unless `iter_batches()` is present and compatible.

### 2) Vector store layer

- Module: `src/aios/memory/vector_store.py`
	- Defines `VectorStoreClient` protocol and factory
	- Drivers: `qdrant.py`, `lancedb.py`

Contract (sketch):

```python
from typing import Sequence, Dict, Any, List, Tuple, Optional

class VectorStoreClient:
		def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metadata: Optional[Sequence[Dict[str, Any]]] = None) -> None: ...
		def query(self, vector: Sequence[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]: ...
		def delete(self, ids: Sequence[str]) -> None: ...
		def close(self) -> None: ...
```

Use cases now: basic smoke tests and future expert-selection memories. No training-time coupling required in PF-005.

---

## User-facing changes

### CLI additions (Typer)

Target: `src/aios/cli/hrm_hf_cli.py` → `train-actv1` command

New options (safe defaults maintain existing behavior):

- `--dataset-backend [custom|hf|webdataset]` (default: `custom`)
- Backend-specific flags:
	- HF Datasets:
		- `--hf-name TEXT` (e.g., "wikitext")
		- `--hf-config TEXT` (optional; e.g., "wikitext-103-raw-v1")
		- `--hf-split TEXT` (default: `train`)
		- `--hf-streaming/--no-hf-streaming` (default: enabled)
		- `--hf-cache-dir PATH` (optional)
		- `--hf-num-workers INT` (tokenization workers; default 2)
		- `--hf-token-env TEXT` (env var name for auth token; default `HUGGING_FACE_HUB_TOKEN`)
	- WebDataset:
		- `--wds-pattern TEXT` (e.g., `data/shards/shard-{000000..000099}.tar` or `https://.../shard-{000..099}.tar`)
		- `--wds-resampled/--no-wds-resampled` (default: false)
		- `--wds-shuffle INT` (buffer size; default 1000)
		- `--wds-decode [text|bytes]` (default: `text`)
		- `--wds-key TEXT` (key to read in tar, default: `txt`)

Backward compatibility:

- `--dataset-file` continues to work with `--dataset-backend=custom` (default).
- If users pass an `hf://dataset:config:split` value to `--dataset-file`, we auto-map to `--dataset-backend=hf` and parse parts (non-breaking convenience already used in GUI).

Examples:

```powershell
# Custom (unchanged)
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 10

# HF streaming
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-backend hf `
	--hf-name wikitext --hf-config wikitext-103-raw-v1 --hf-split train `
	--steps 10 --batch-size 2

# WebDataset shards (local)
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-backend webdataset `
	--wds-pattern "C:/data/shards/shard-{000000..000009}.tar" `
	--steps 10 --batch-size 2
```

Windows notes:

- Use PowerShell quoting as shown above. For large brace-globs, prefer quotes (") to prevent premature expansion.

### GUI additions (HRM Training Panel)

Target: `src/aios/gui/components/hrm_training_panel.py`

- Add a "Dataset backend" dropdown next to "Dataset file/dir" with values: `custom`, `hf`, `webdataset`.
- When `hf` is selected, show inline fields: Name, Config, Split, Streaming (checkbox), Cache dir.
- When `webdataset` is selected, show: Pattern, Resampled (checkbox), Shuffle buf, Decode, Key.
- Continue to support the existing "hf://…" entry shortcut. If user pastes `hf://<name>:<config>:<split>`, auto-select `hf` and populate fields.
- Persist new fields via `get_state()/set_state()` and include them when building `TrainingConfig`.

Behavioral notes:

- Keep dataset selection UX simple: either paste a path/URI or use the new dropdown to reveal backend-specific fields.
- Add small helper tooltips explaining streaming and shard patterns.
- Respect existing ASCII-only filter and memory estimator displays.

---

## Config and validation

Extend `TrainingConfig` (module: `src/aios/core/hrm_training/training_config.py`):

- New fields:
	- `dataset_backend: Literal["custom", "hf", "webdataset"] = "custom"`
	- HF: `hf_name: Optional[str]`, `hf_config: Optional[str]`, `hf_split: str = "train"`, `hf_streaming: bool = True`, `hf_cache_dir: Optional[str]`, `hf_num_workers: int = 2`, `hf_token_env: str = "HUGGING_FACE_HUB_TOKEN"`
	- WDS: `wds_pattern: Optional[str]`, `wds_resampled: bool = False`, `wds_shuffle: int = 1000`, `wds_decode: str = "text"`, `wds_key: str = "txt"`

Validation rules:

- If `dataset_backend == "custom"`, require `dataset_file` (existing behavior).
- If `dataset_backend == "hf"`, require `hf_name` (or parse from `dataset_file` if `hf://` URI). Validate split exists if metadata is available.
- If `dataset_backend == "webdataset"`, require `wds_pattern` and ensure it resolves to at least one shard (or allow late-binding with clear log warnings).
- ASCII filter, batching, and max sequence length behavior must remain unchanged from user’s POV.

---

## Implementation details

### A) Data backends package

Files to add:

- `src/aios/cli/hrm_hf/data_backends/base.py`
- `src/aios/cli/hrm_hf/data_backends/custom.py` (wrap current logic)
- `src/aios/cli/hrm_hf/data_backends/hf.py`
- `src/aios/cli/hrm_hf/data_backends/webdataset.py`
- `src/aios/cli/hrm_hf/data_backends/__init__.py` with `build_backend(config)`

Key behaviors:

- HF Datasets
	- Use `datasets.load_dataset(hf_name, hf_config, split=hf_split, streaming=hf_streaming)`
	- When streaming, iterate and yield `example["text"]` or join fields if text column not obvious (configurable via simple heuristic: prefer `text`, else `content`, else JSON stringify line)
	- Respect `ascii_only` and filtering already present in training pipeline
	- Support `hf_cache_dir` and `HUGGINGFACE_HUB_CACHE`/`HF_HOME`
	- Prefetch: use a small async queue with size 2–4 to smooth tokenizer throughput

- WebDataset
	- If `webdataset` lib is available: use `wds.WebDataset(pattern).shuffle(wds_shuffle)`; otherwise provide a minimal tar iterator (local-only) that reads `*.txt` entries matching key
	- Map `wds_decode": text|bytes` to return `str` or `bytes` and let tokenizer branch accordingly
	- Support `wds_resampled` with `wds.ResampledShards` when library is present

### B) Tokenization and batching

- Preserve current tokenization path to avoid regressions.
- If a backend exposes `iter_batches()` matching our `SampleBatch`, we can optionally bypass text tokenization, but keep this disabled by default in PF-005.
- Ensure deterministic seeding with existing RNG controls.

### C) HRM CLI wiring

- Update `src/aios/cli/hrm_hf_cli.py` to add new options and to pass them into `TrainingConfig`.
- Add validation hints in Typer help messages for Windows path quoting and HF auth.

### D) GUI wiring

- In `HRMTrainingPanel`, add a new `dataset_backend_var` and conditional UI stacks.
- Extend `build_training_config()` to populate the new `TrainingConfig` fields.
- Update `get_state()/set_state()` to persist and restore the new fields.
- Continue supporting the `hf://` inline format. If user switches backend manually, keep the inline field synchronized.

### E) Vector store minimal layer

Files to add:

- `src/aios/memory/vector_store.py` (interface + factory)
- `src/aios/memory/vector_stores/qdrant.py`
- `src/aios/memory/vector_stores/lancedb.py`

Functionality:

- `upsert(ids, vectors, metadata)` and `query(vector, top_k)` with cosine similarity.
- Qdrant: depend on `qdrant-client`; collection auto-create if missing; index on cosine.
- LanceDB: depend on `lancedb`; create table if missing; approximate query OK for starter.

Optional CLI (lightweight utility for smoke tests):

- `aios memory vs-upsert` and `aios memory vs-query` that call the above client; documented only, can be added in a follow-up small PR.

---

## Docker and local services

Add a Qdrant service snippet to `docker-compose.yml` (either appended or documented):

```yaml
services:
	qdrant:
		image: qdrant/qdrant:latest
		ports:
			- "6333:6333"
		volumes:
			- ./artifacts/qdrant:/qdrant/storage
		healthcheck:
			test: ["CMD", "wget", "-qO-", "http://localhost:6333/readyz"]
			interval: 10s
			timeout: 5s
			retries: 5
```

Windows quickstart:

```powershell
docker compose up -d qdrant
```

For LanceDB (no service), nothing to run; it’s an embedded store.

---

## Testing and acceptance criteria

What we test:

1) HF streaming backend
	 - Run a 1–2 step training invoking an HF dataset by name/config/split
	 - Verify tokenization continues to function; metrics JSONL emits at least one record
2) WebDataset backend (local shards)
	 - Create 1–2 tiny tar shards with small `*.txt` samples
	 - Run 1–2 step training using `--wds-pattern` and verify iteration + metrics
3) Vector store
	 - With Qdrant running or LanceDB selected, upsert 100 random 128D vectors, query a held-out vector, verify top-5 return with decreasing scores

Suggested smoke commands (PowerShell):

```powershell
# HF streaming smoke
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-backend hf --hf-name wikitext --hf-config wikitext-103-raw-v1 --hf-split train `
	--steps 1 --batch-size 2 --halt-max-steps 1 `
	--log-file artifacts/brains/actv1/metrics.jsonl

# WebDataset smoke (assuming shards exist)
aios hrm-hf train-actv1 --model gpt2 `
	--dataset-backend webdataset --wds-pattern "training_data/shards/shard-{000000..000001}.tar" `
	--steps 1 --batch-size 2 --halt-max-steps 1 `
	--log-file artifacts/brains/actv1/metrics.jsonl
```

Acceptance criteria (pass/fail):

- HF: Iterates and trains for 1 step on a public dataset split; no blocking warnings
- WDS: Iterates and trains for 1 step from shards; no blocking warnings
- VS: Upsert/query test returns top-5 with non-increasing similarity

---

## Rollout plan

1) M1 (2 days): Data backends + docs
	 - Implement backend package and CLI/GUI wiring
	 - Ship a doc with examples, and enable a dry-run path for each backend
	 - Add unit tests for HF URI parsing and WDS pattern parsing
2) M2 (1 day): Vector store + example
	 - Implement minimal drivers
	 - Add docker-compose snippet and a tiny usage doc

Feature flagging/back-compat:

- Default remains `custom` loader; no behavior change for current users.
- `hf://` path auto-detection provides an additive convenience, not a breaking change.

---

## Risks and mitigations

- Streaming backpressure: Use small prefetch queues and timeouts; allow `--hf-num-workers` to adjust throughput.
- HF auth/rate limits: Use `--hf-token-env` and document how to set `HUGGING_FACE_HUB_TOKEN`.
- WebDataset lib availability: Provide a minimal built-in tar reader for local shards when `webdataset` is not installed.
- Windows Docker friction: Provide LanceDB as an embedded alternative; document `docker compose up -d qdrant`.

---

## Checklists

### Engineering checklist

- [ ] Add `data_backends/` package and register `build_backend(config)`
- [ ] Extend `TrainingConfig` with new fields + validation
- [ ] Update `hrm_hf_cli.py` to surface new flags
- [ ] Wire `train_actv1_impl` to use backend builder
- [ ] Add GUI controls (dropdown + dynamic fields) and persist state
- [ ] Ensure `hf://` URI auto-maps to HF backend in both CLI and GUI
- [ ] Add `vector_store.py` interface and Qdrant/LanceDB drivers
- [ ] Document docker snippet and LanceDB alternative

### QA checklist

- [ ] CLI help shows new flags with clear descriptions
- [ ] Run HF smoke with 1 step; verify metrics JSONL appended
- [ ] Run WDS smoke with 1 step; verify metrics JSONL appended
- [ ] VS: upsert/query 100 embeddings; query returns 5 nearest with sensible scores
- [ ] GUI: switching backends updates visible fields and persists across restarts
- [ ] GUI: dataset estimator and memory estimator still render without errors

### Docs checklist

- [ ] Update quick starts (`docs/QUICK_START.md`, `docs/ACTV1_MOE_QUICK_START.md` [placeholder]) to mention backends
- [ ] Add a short “Data Backends” section in `docs/INDEX.md`
- [ ] Add a “Vector Store (starter)” section in `docs/INDEX.md`

---

## Appendix: mapping from dataset URIs to config

- `file:///path/to/data.txt` → backend=custom, dataset_file=that path
- `hf://<name>:<config>:<split>` → backend=hf, populate `hf_name`, `hf_config`, `hf_split`
- `wds://<pattern>` → backend=webdataset, `wds_pattern=<pattern>`

If users paste raw values (no scheme):

- If it looks like a local file/dir → `custom`
- If it matches `hf://` → `hf`
- If it contains `{000..999}.tar` or endswith `.tar` → suggest `webdataset`

This keeps the happy path minimal while enabling more scalable backends when needed.
