# Data Backends and Vector Stores

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
- Enable persistent cognitive memory for attention traces and crystallized motifs (see `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md`).

### Non-goals (Out of scope for PF-005)

- Full RAG pipelines, memory policies, or retrieval-integrated training loops.
- On-the-fly embedding model training; we only provide an API and a thin client.
- Cognitive memory integration (attention traces, crystallized motifs) is implemented separately in Phase 2.5 of `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` after both foundations are complete.

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

### 3) Cognitive memory integration (Future)

**Note**: This section describes planned integration with `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` (implemented in their Phase 2.5, dependent on PF-005 completion).

The `VectorStoreClient` provides the storage backend for two cognitive memory components:

#### A) Attention Trace Storage

**Purpose**: Persist high-salience attention patterns across training sessions.

**Wrapper class**: `TraceVectorStore`
```python
# src/aios/core/hrm_models/cognitive/vector_wrappers.py
from aios.memory.vector_store import VectorStoreClient

class TraceVectorStore:
    """Specialized wrapper for attention trace persistence."""
    
    def __init__(self, vector_client: VectorStoreClient, collection: str = "attention_traces"):
        self.client = vector_client
        self.collection = collection
        self.embedder = TraceEmbedder(embed_dim=128)  # Converts sparse trace to dense vector
    
    def upsert_traces(self, traces: List[AttentionTrace], training_step: int):
        """Store batch of attention traces."""
        ids = [f"trace_{t.layer_id}_{t.head_id}_{t.query_idx}_{t.key_idx}_{training_step}" 
               for t in traces]
        vectors = [self.embedder.embed(t) for t in traces]  # List[np.ndarray[128]]
        metadata = [{
            "layer": t.layer_id,
            "head": t.head_id,
            "query_idx": t.query_idx,
            "key_idx": t.key_idx,
            "salience": t.salience,
            "age": t.age,
            "training_step": training_step,
        } for t in traces]
        self.client.upsert(ids, vectors, metadata)
    
    def query_similar_traces(self, query_trace: AttentionTrace, top_k: int = 100):
        """Retrieve similar traces for warm-starting."""
        query_vec = self.embedder.embed(query_trace)
        return self.client.query(query_vec, top_k=top_k)
```

**Data flow**:
1. Training runs with `TraceManager` accumulating traces in RAM (~24 MB)
2. Every `trace_sync_interval` steps (e.g., 1000), `TraceManager.sync_to_vector_store()` calls `TraceVectorStore.upsert_traces()`
3. On training restart, `TraceManager.load_from_vector_store()` retrieves historical traces for warm start
4. Benefit: Cross-session learning - model remembers useful attention patterns from previous training runs

**Collections**: `attention_traces` (one per model or shared)

#### B) Crystallized Motif Storage

**Purpose**: Store and share high-utility expert routing patterns ("thought primitives").

**Wrapper class**: `MotifVectorStore`
```python
class MotifVectorStore:
    """Specialized wrapper for crystallized motif persistence."""
    
    def __init__(self, vector_client: VectorStoreClient, collection: str = "crystallized_motifs"):
        self.client = vector_client
        self.collection = collection
        self.embedder = MotifEmbedder(embed_dim=256)  # Encodes expert sequence to vector
    
    def upsert_motif(self, motif: CrystallizedMotif):
        """Store a single crystallized motif."""
        motif_id = f"motif_{motif.id}"
        vector = self.embedder.embed(motif)  # np.ndarray[256]
        metadata = {
            "motif_id": motif.id,
            "expert_sequence": motif.expert_sequence,  # [2, 7, 3, 1]
            "frequency": motif.count,
            "utility": motif.utility,
            "entropy": motif.entropy,
            "age": motif.age,
            "task_tags": motif.task_tags,  # ["retrieval", "QA"]
        }
        self.client.upsert([motif_id], [vector], [metadata])
    
    def query_motifs_for_task(self, task_tag: str, top_k: int = 10):
        """Retrieve best motifs for specific task type."""
        # Filter by task tag, return highest utility
        return self.client.query(
            vector=None,  # Use filter-only query if backend supports
            top_k=top_k,
            filter={"task_tags": task_tag}
        )
    
    def transfer_motifs_to_model(self, target_model_id: str, motif_ids: List[str]):
        """Enable cross-model motif sharing."""
        # Retrieve motifs and return for injection into target model's RoutingPathTree
        results = [self.client.query_by_id(mid) for mid in motif_ids]
        return [self._reconstruct_motif_from_metadata(r[2]) for r in results]
```

**Data flow**:
1. During training, `RoutingPathTree` tracks expert routing patterns
2. When motif crystallizes (high frequency + utility + stability), `RoutingPathTree.persist_motif()` calls `MotifVectorStore.upsert_motif()`
3. Other models can query motifs via `query_motifs_for_task()` for zero-shot transfer
4. Benefit: Multi-model collaboration - models share learned reasoning strategies

**Collections**: `crystallized_motifs` (shared across all models for collaborative learning)

#### C) Scalability Comparison

| Storage Mode | Capacity | Retrieval Speed | Persistence | Sharing |
|--------------|----------|-----------------|-------------|----------|
| **RAM-only** (baseline) | ~2M traces, 512 motifs | O(n) scan | None (lost on restart) | No |
| **Vector Store** (enhanced) | Billions of traces/motifs | O(log n) ANN | Persistent | Multi-model |

**Memory overhead**:
- RAM-only: 30 MB total
- With vector store: 30 MB (RAM) + 5 MB (embedding models) = 35 MB total
- External storage: Qdrant/LanceDB (on disk or service)

**Configuration** (see unified schema below):
```yaml
memory:
  vector_store:
    backend: "qdrant"  # Shared by datasets, traces, and motifs
  persistent_traces:
    persist_to_vector_store: true  # Enable cross-session persistence
    trace_sync_interval: 1000
  semantic_crystallization:
    persist_motifs: true  # Auto-save crystallized motifs
```

**Integration timeline**: Week 6.5-7.5 of Persistent Traces roadmap (after PF-005 vector store foundation is complete).

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

### F) Cognitive memory integration (Future)

**Note**: Implemented in `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` Phase 2.5 (Week 6.5-7.5), dependent on PF-005 completion.

Files to add by cognitive architecture team:

- `src/aios/core/hrm_models/cognitive/vector_wrappers.py` (`TraceVectorStore`, `MotifVectorStore`)
- `src/aios/core/hrm_models/cognitive/embedders.py` (`TraceEmbedder`, `MotifEmbedder`)

Integration points in existing files:

- `src/aios/core/hrm_models/cognitive/trace_manager.py`: Add `sync_to_vector_store()`, `load_from_vector_store()` methods
- `src/aios/core/hrm_models/cognitive/routing_tree.py`: Add `persist_motif()` method

See "Cognitive memory integration" section above for detailed specifications.

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

## Implementation roadmap

**Total timeline**: 6 weeks for core features + 1.5 weeks for cognitive memory integration (optional)

### Week 1-2: Core Infrastructure
**Owner**: Backend team  
**Deliverables**:
- [ ] Create `src/aios/memory/vector_store.py` with `VectorStoreClient` protocol
- [ ] Implement Qdrant driver (`src/aios/memory/vector_stores/qdrant.py`)
- [ ] Implement LanceDB driver (`src/aios/memory/vector_stores/lancedb.py`)
- [ ] Factory function `create_vector_store(backend, config)`
- [ ] Unit tests for upsert/query/delete operations
- [ ] Docker Compose service definition for Qdrant

**Acceptance**:
- Qdrant driver: Upsert 1000 vectors, query returns correct top-5
- LanceDB driver: Same test, embedded mode (no service)
- Both drivers pass same test suite (interface compliance)

### Week 3-4: Dataset Backend Integration
**Owner**: Training pipeline team  
**Deliverables**:
- [ ] Create `src/aios/cli/hrm_hf/data_backends/` package
- [ ] Implement `custom.py` (wrap existing file/dir/CSV logic)
- [ ] Implement `hf.py` (HuggingFace Datasets streaming)
- [ ] Implement `webdataset.py` (tar shard reader)
- [ ] Update `TrainingConfig` with new dataset backend fields
- [ ] CLI integration: `train-actv1 --dataset-backend hf --hf-name wikitext`
- [ ] Integration tests with small HF dataset and local tar shards

**Acceptance**:
- HF backend: Train 10 steps on wikitext, metrics JSONL contains 10 records
- WebDataset backend: Train 10 steps from 2 local tar shards, no errors
- Custom backend: Existing tests still pass (backward compatibility)

### Week 5-6: GUI and Production Hardening
**Owner**: GUI + DevOps  
**Deliverables**:
- [ ] GUI integration: `HRMTrainingPanel` dropdown for dataset backend selection
- [ ] Conditional UI fields for HF and WebDataset options
- [ ] Configuration validation (prevent missing required fields)
- [ ] Error handling for missing Qdrant service, HF auth failures, invalid shard patterns
- [ ] Documentation: User guide for dataset backends and vector store setup
- [ ] PowerShell examples for Windows users

**Acceptance**:
- GUI: Select HF backend, populate fields, launch training successfully
- Error handling: Missing Qdrant service shows clear error message
- Docs: New user can follow guide to set up Qdrant and run HF training

### Week 6.5-7.5: Cognitive Memory Integration (Optional)
**Prerequisites**: 
- PF-005 vector store complete (Weeks 1-6)
- Persistent Traces Phase 0-2 complete (see `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md`)

**Owner**: Cognitive architecture team  
**Deliverables**:
- [ ] Implement `TraceVectorStore` wrapper class
- [ ] Implement `MotifVectorStore` wrapper class
- [ ] Implement `TraceEmbedder` (sparse trace → 128D vector)
- [ ] Implement `MotifEmbedder` (expert sequence → 256D vector)
- [ ] Integration tests: Trace persistence/retrieval cycle
- [ ] Cross-model motif transfer test

**Acceptance**:
- TraceVectorStore: Persist 10K traces, reload, verify salience within 1% error
- MotifVectorStore: Query similar motifs, cosine similarity > 0.8 for same-task motifs
- Works with both Qdrant and LanceDB backends

**Notes**:
- This phase is implemented by the Persistent Traces team, not PF-005 core team
- See `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` Phase 2.5 for detailed spec
- Can be skipped if cognitive memory features not needed

### Week 8: Deployment and Monitoring (Final)
**Owner**: DevOps  
**Deliverables**:
- [ ] Production Qdrant deployment guide (cloud or self-hosted)
- [ ] Monitoring dashboard for vector store metrics (latency, storage size)
- [ ] Backup/restore scripts for Qdrant collections
- [ ] Performance benchmarks: Dataset iteration speed, vector query latency

**Acceptance**:
- Qdrant service runs in production with health checks
- Monitoring shows vector store query p95 latency < 50ms
- Backup script successfully exports and restores 1M vectors

---

## Coordination with Persistent Traces plan

**Parallel development strategy**:
- **Weeks 1-6**: PF-005 (this document) and Persistent Traces Phases 0-2 proceed **independently**
- **Week 6.5-7.5**: Integration phase requires **both systems complete**
- **Week 7+**: Persistent Traces continues independently, optionally using vector store

**Critical dependencies**:
1. `VectorStoreClient` interface must be finalized by end of Week 2
2. Qdrant or LanceDB must be deployable by end of Week 4
3. TraceVectorStore/MotifVectorStore depend on VectorStoreClient being stable

**Configuration namespace**:
Both plans share unified `memory:` section in `config/default.yaml` (see Unified Configuration Schema below).

**Cross-references**:
- For cognitive memory use cases, see: `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md`
- For trace/motif embedding specifications, see: `PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md` § Vector Store Integration
- For unified config schema, see: Unified Configuration Schema section below

---

## Unified configuration schema

**Location**: `config/default.yaml`

**Namespace**: All memory-related features unified under `memory:` top-level key to avoid conflicts.

```yaml
# Memory Architecture (Unified Schema)
# Covers: Dataset backends (PF-005), Vector stores (PF-005), 
#         Persistent traces (Cognitive), Semantic crystallization (Cognitive)
memory:
  # ============================================
  # Dataset Backends (PF-005)
  # ============================================
  dataset:
    backend: "custom"  # custom|hf|webdataset
    
    # HuggingFace Datasets streaming
    hf:
      name: null                      # e.g., "wikitext"
      config: null                    # e.g., "wikitext-103-raw-v1"
      split: "train"                  # train|validation|test
      streaming: true                 # Enable streaming mode
      cache_dir: null                 # Optional cache directory
      num_workers: 2                  # Tokenization workers
      token_env: "HUGGING_FACE_HUB_TOKEN"  # Auth token env var
    
    # WebDataset tar shards
    webdataset:
      pattern: null                   # e.g., "data/shards/shard-{000000..000099}.tar"
      resampled: false                # Use ResampledShards for infinite iteration
      shuffle: 1000                   # Shuffle buffer size
      decode: "text"                  # text|bytes
      key: "txt"                      # Key to extract from tar entries
  
  # ============================================
  # Vector Storage Backend (PF-005)
  # ============================================
  vector_store:
    backend: "qdrant"  # qdrant|lancedb|disabled
    
    # Qdrant configuration
    qdrant:
      host: "localhost"
      port: 6333
      collection_prefix: "aios_memory"  # Collections: aios_memory_traces, aios_memory_motifs
      api_key: null                     # Optional authentication
    
    # LanceDB configuration
    lancedb:
      path: "artifacts/memory/lancedb" # Embedded database path
  
  # ============================================
  # Persistent Attention Traces (Cognitive)
  # See: PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md
  # ============================================
  persistent_traces:
    enabled: false                    # Enable persistent trace memory
    
    # Core trace parameters
    sparsity: 0.001                   # Capture top 0.1% of attention edges
    quota_per_head: 2048              # Max traces per attention head
    salience_threshold: 0.05          # Minimum salience to persist
    retention_rate: 0.95              # λ (EMA momentum)
    decay_rate: 0.98                  # γ (forgetting rate for unused traces)
    bias_strength: 0.1                # α (injection strength into attention)
    update_interval: 100              # Steps between trace consolidation
    warmup_steps: 1000                # Standard attention before trace capture
    
    # Vector store integration (optional - requires vector_store.backend != disabled)
    persist_to_vector_store: false    # Enable cross-session persistence
    trace_sync_interval: 1000         # Steps between DB syncs
    embedding_dim: 128                # Trace embedding dimensionality
    warm_start: false                 # Load traces from DB on training start
    task_tag: null                    # Filter traces by task type ("QA", "generation", etc.)
  
  # ============================================
  # Semantic Crystallization (Cognitive)
  # See: PERSISTENT_TRACES_SEMANTIC_CRYSTALLIZATION.md
  # ============================================
  semantic_crystallization:
    enabled: false                    # Enable motif crystallization
    
    # Crystallization criteria
    min_frequency: 100                # f_min (minimum traversals)
    min_utility: 0.05                 # U_min (5% improvement over baseline)
    max_entropy: 1.0                  # H_max (routing stability threshold)
    min_age: 500                      # Temporal stability requirement (steps)
    max_motifs: 512                   # Hard limit on crystallized primitives
    
    # Vector store integration (optional)
    persist_motifs: false             # Auto-save crystallized motifs to DB
    motif_embedding_dim: 256          # Motif embedding dimensionality
    share_across_models: false        # Allow other models to query/reuse motifs
```

**Validation rules**:
- If `memory.dataset.backend == "hf"`, require `memory.dataset.hf.name`
- If `memory.dataset.backend == "webdataset"`, require `memory.dataset.webdataset.pattern`
- If `memory.persistent_traces.persist_to_vector_store == true`, require `memory.vector_store.backend != "disabled"`
- If `memory.semantic_crystallization.persist_motifs == true`, require `memory.vector_store.backend != "disabled"`

**Backward compatibility**:
- Existing `dataset_file` config maps to `memory.dataset.backend == "custom"`
- All `memory.*` fields default to disabled/conservative values
- No breaking changes to existing training configs

**Windows compatibility**:
- All paths use forward slashes internally, converted by PathLib
- PowerShell examples in documentation
- Qdrant via Docker Desktop (Windows native)
- LanceDB pure Python (Windows native)

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
