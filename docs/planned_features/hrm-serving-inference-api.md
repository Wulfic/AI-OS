## PF-003: Serving and inference integration (HRM + baselines)

### Summary

Provide a clean serving path for HRM and HF baselines. Ship a minimal FastAPI server for HRM inference and document how to deploy HF baselines on vLLM/TGI. For production-grade GPU fleets, outline Triton backend path.

### Motivation

- Benchmark and A/B test HRM vs standard transformer baselines.
- Enable downstream apps to use a stable `/generate` and `/loglikelihood` API.

### Scope

In scope:
- Minimal FastAPI server for HRM with batch generation and scoring endpoints.
- Configuration to run under Docker.
- Baseline serving docs for vLLM and TGI (HF models only).

Out of scope (for this PF):
- Triton custom backend implementation (design notes only).

### Integration points (repo)

- New module: `src/aios/serve/hrm_server.py` (FastAPI, uvicorn)
- Dockerfile target: `--target hrm-serve` (multi-stage)
- Scripts: `scripts/run_hrm_server.ps1` (Windows), `scripts/run_hrm_server.sh` (Linux)

### API contract (v1)

- POST `/generate`
  - Input: `{ prompts: string[], max_tokens?: number, temperature?: number, top_p?: number }`
  - Output: `{ completions: string[], usage?: { prompt_tokens, completion_tokens } }`

- POST `/loglikelihood`
  - Input: `{ prompts: string[], continuations: string[] }`
  - Output: `{ ll: number[] }`

---

## Comprehensive guide and checklist

This section expands PF-003 into an actionable implementation guide with CLI and GUI elements, deployment paths, and validation checklists.

### Deliverables

- HRM FastAPI server module: `src/aios/serve/hrm_server.py`
- Optional minimal UI for manual testing (Gradio or Streamlit)
- CLI launcher: `aios serve hrm` (via existing `aios.cli.aios` entrypoint)
- Dockerfile target: `hrm-serve` and helper scripts for Windows/Linux
- Baseline serving notes for vLLM and TGI
- Acceptance tests and operational runbooks

### Architecture overview

- Client (CLI/UI/SDK) → FastAPI app → HRM inference wrapper (tokenizer + model) → Torch device (CPU/GPU)
- Optional: request batcher (FIFO), configurable max batch size and max new tokens per request
- Observability: structured logs + optional Prometheus metrics endpoint
- Healthchecks: `/healthz` (process up), `/readyz` (model loaded)

### Data models and API details

- POST `/generate`
  - Request model (Pydantic):
    - `prompts: List[str]` (1..N)
    - `max_tokens: int = 128` (cap at server max, e.g., 1024)
    - `temperature: float = 0.7` (range [0, 2])
    - `top_p: float = 1.0` (range (0, 1])
    - `seed: Optional[int]` (optional for reproducibility)
    - `stop: Optional[List[str]]` (optional stop strings)
  - Response:
    - `completions: List[str]`
    - `usage?: { prompt_tokens: int, completion_tokens: int, total_tokens: int }`
  - Errors:
    - 400: validation (e.g., empty prompts, invalid ranges)
    - 429: rate limited (optional)
    - 500: internal error

- POST `/loglikelihood`
  - Request:
    - `prompts: List[str]`
    - `continuations: List[str]` (must match length of prompts)
  - Response:
    - `ll: List[float]` (sum of token log-probs of each continuation given prompt)
  - Errors: as above

- GET `/healthz`: `{ status: "ok" }`
- GET `/readyz`: `{ status: "ready" }` once model/tokenizer loaded
- Optional future: `/generate_stream` using server-sent events (not in v1 scope)

### HRM inference wrapper design

- Artifacts and paths
  - Tokenizer: reuse loading logic from training; prefer registry paths under `artifacts/hf_implant/tokenizers/` or configured `--tokenizer` path
  - Model weights: e.g., `artifacts/brains/actv1/actv1_student.pt` or `artifacts/hf_implant/q_head.pt` + base model
  - Config file (optional): `config/default.yaml` overrides

  - Batch tokenize `prompts` with truncation to model context window; return attention masks

- Decoding
  - Modes: greedy (temperature=0 or top_p=1, top_k=None), sampled (temperature>0 and/or top_p<1)
  - Respect `max_tokens` and `stop` strings; early stop if EOS token encountered
  - Use `torch.no_grad()` and AMP optional (`torch.cuda.amp.autocast`) when CUDA available

- Batching
  - Pad to max prompt length in batch; maintain mapping to return completions in order
  - Configurable `max_batch_size` to protect memory

- Device selection
  - Auto-detect CUDA/ROCm/MPS; environment variable `AIOS_DEVICE` to force: `cpu|cuda|mps`

- Pseudocode outline
  - `load_tokenizer()` → `load_model()` → set eval, move to device
  - For `/generate`: encode → loop new tokens → decode to strings → postprocess stop
  - For `/loglikelihood`: teacher-forcing forward over prompt+continuation and sum log-probs at continuation positions

### Configuration model

Precedence: CLI flags > Env vars > YAML config defaults.

- CLI flags (examples):
  - `--host 0.0.0.0 --port 8000`
  - `--model-path artifacts/brains/actv1/actv1_student.pt`
  - `--tokenizer-path artifacts/hf_implant/tokenizers/<name>`
  - `--device cpu|cuda|mps`
  - `--max-batch-size 16 --max-new-tokens 256 --context-window 2048`
  - `--enable-cors --cors-origins *`
  - `--log-level info` (honors `logging.yaml` if present)

- Env vars:
  - `AIOS_MODEL_PATH`, `AIOS_TOKENIZER_PATH`, `AIOS_DEVICE`, `AIOS_MAX_BATCH_SIZE`, `AIOS_MAX_NEW_TOKENS`, `AIOS_CORS_ORIGINS`

- YAML (optional): `config/default.yaml` → `serve.hrm` section

### CLI: serve commands

- Primary: `aios serve hrm` (wired via existing `aios.cli.aios`)
  - Example (Windows PowerShell):
    - `aios serve hrm --host 0.0.0.0 --port 8000 --model-path artifacts/brains/actv1/actv1_student.pt --tokenizer-path artifacts/hf_implant/tokenizers/base --device cpu`
  - Example (module invocation):
    - `.venv\Scripts\python.exe -m aios.serve.hrm_server --host 0.0.0.0 --port 8000`

- Admin helpers (optional):
  - `aios serve hrm --print-config`
  - `aios serve hrm --dry-run` (load-only, no HTTP)

### Minimal GUI (optional) for manual testing

- Choice: Gradio (simpler) or Streamlit
- Usage intent: quick sanity checks by product/QA without curl or code
- Proposed: `aios serve hrm --ui` opens a small panel:
  - Input textarea for prompt, sliders for `max_tokens`, `temperature`, `top_p`
  - Buttons "Generate" and "Score loglikelihood"
  - Display output text and usage metrics

- Local run (Streamlit example):
  - `python -m aios.serve.hrm_ui --server-url http://localhost:8000`

Note: The UI is helpful but not required for acceptance of PF-003. If code is deferred, include only docs and a future task for `hrm_ui.py`.

### Docker and containerization

- Dockerfile target: `hrm-serve` (multi-stage). Example build and run on Windows PowerShell:

```pwsh
# Build
docker build -t aios/hrm-serve:local --target hrm-serve .

# Run (CPU)
docker run --rm -p 8000:8000 ^
  -e AIOS_MODEL_PATH=artifacts/brains/actv1/actv1_student.pt ^
  -e AIOS_TOKENIZER_PATH=artifacts/hf_implant/tokenizers/base ^
  aios/hrm-serve:local

# Run (CUDA) – requires NVIDIA Container Toolkit
docker run --rm -p 8000:8000 --gpus all ^
  -e AIOS_DEVICE=cuda ^
  aios/hrm-serve:local
```

- docker-compose service snippet:

```yaml
services:
  hrm:
    image: aios/hrm-serve:local
    ports: ["8000:8000"]
    environment:
      AIOS_MODEL_PATH: artifacts/brains/actv1/actv1_student.pt
      AIOS_TOKENIZER_PATH: artifacts/hf_implant/tokenizers/base
      AIOS_DEVICE: cpu
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

- Helper scripts to add:
  - `scripts/run_hrm_server.ps1`
  - `scripts/run_hrm_server.sh`

### Baseline serving (A/B) with vLLM and TGI

- vLLM (HF checkpoint):

```pwsh
docker run --rm -p 8001:8000 ^
  -v $PWD\artifacts\hf_implant\base_model:/model ^
  vllm/vllm-openai:latest ^
  --model /model
```

Hit using OpenAI-compatible API:

```pwsh
curl -s http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{
  "model":"local",
  "prompt":"Hello",
  "max_tokens":16
}' | jq .
```

- Text Generation Inference (TGI):

```pwsh
docker run --rm -p 8002:80 ^
  -v $PWD\artifacts\hf_implant\base_model:/data ^
  ghcr.io/huggingface/text-generation-inference:latest ^
  --model-id /data
```

Hit using TGI API:

```pwsh
curl -s http://localhost:8002/generate -H "Content-Type: application/json" -d '{
  "inputs": "Hello",
  "parameters": {"max_new_tokens": 16, "temperature": 0.7, "top_p": 0.9}
}' | jq .
```

- Mapping to our contract
  - Our `/generate` → vLLM `/v1/completions` or OpenAI Chat; TGI `/generate`
  - For A/B, normalize fields: `max_tokens ↔ max_new_tokens`, `temperature`, `top_p`

### Observability and security

- Logging: use `logging.yaml` if present; otherwise default to INFO with request/response IDs (omit bodies in prod logs)
- Metrics: optional `/metrics` (Prometheus) for request counts/latency/tokens
- Tracing: optional OpenTelemetry if configured
- CORS: `--enable-cors` and whitelist origins
- Auth (optional): bearer token via `Authorization: Bearer <token>`; deny when missing

### Testing, acceptance, and checklists

Functional quick test (CPU):

```pwsh
# 1) Start server (example)
aios serve hrm --device cpu --port 8000 --model-path artifacts/brains/actv1/actv1_student.pt --tokenizer-path artifacts/hf_implant/tokenizers/base

# 2) Generate two prompts
curl -s http://localhost:8000/generate -H "Content-Type: application/json" -d '{
  "prompts": ["Hello", "Once upon a time"],
  "max_tokens": 8,
  "temperature": 0.7,
  "top_p": 0.9
}' | jq .

# 3) Loglikelihood
curl -s http://localhost:8000/loglikelihood -H "Content-Type: application/json" -d '{
  "prompts": ["Hello"],
  "continuations": [" world"]
}' | jq .
```

Readiness checklist

- [ ] Server starts and `/readyz` returns ready within 60s
- [ ] `/generate` returns completions for 2 prompts under 2s on CPU (sample model)
- [ ] `/loglikelihood` returns values and shape matches inputs
- [ ] Handles batch of 16 prompts without OOM; memory stable over 5 runs
- [ ] Error cases return 400/422 with clear message
- [ ] Logs include request IDs and timing
- [ ] Docker image builds and runs locally
- [ ] Optional UI launches and can call server
- [ ] A/B against vLLM/TGI produces comparable outputs with same seeds

Load and reliability

- [ ] Sustains 10 RPS with batch size 8 on CPU sample model (target; adjust per hardware)
- [ ] Backpressure: requests rejected with 429 when queue is full (if enabled)
- [ ] Graceful shutdown drains in-flight requests

### Production runbook (starter)

- Startup
  - Verify drivers (CUDA) or plan CPU
  - Warmup: send a 1-token request to pre-JIT kernels
  - Confirm `/readyz` and sample `/generate`

- Scaling
  - Horizontal: run multiple replicas behind a load balancer; sticky sessions not required
  - Vertical: tune `max_batch_size`, `max_new_tokens`, and context window to fit memory

- Troubleshooting
  - 500 at startup: verify paths for model/tokenizer; run with `--dry-run`
  - CUDA OOM: reduce batch size or `max_new_tokens`; ensure `torch.cuda.empty_cache()` between runs if needed
  - Slow responses: disable AMP on CPU; pin threads (`OMP_NUM_THREADS`)
  - CORS blocked: set `--enable-cors --cors-origins *` for dev only

### Triton backend (design notes, out of scope)

- Shape the HRM inference as a stateless backend with request batching and token cache
- Inputs: token IDs, attention mask, decode params; Outputs: next tokens/logprobs
- Consider KV cache management and paged attention for large contexts

---

### Implementation steps
1) HRM inference wrapper
- Reuse tokenizer loading from `train_actv1.py` helpers.
- Load `actv1_student.pt` and implement a simple forward for greedy/sampled decoding.
- Add batching and `torch.no_grad()`; AMP optional.

2) FastAPI app
- Create endpoints per contract; validate inputs; return JSON.
- Add health endpoint and basic logging.

3) Packaging
- Add new Docker target with a slim runtime (CUDA base optional).
- Provide PowerShell and Bash helpers to run locally.

4) Baseline docs
- Document how to start vLLM/TGI for a matching HF model and how to hit similar endpoints for A/B.

### Testing and acceptance criteria

- Local run: Start server, send `/generate` with 2 prompts, receive 2 completions within reasonable latency on CPU/GPU.
- Error handling: Invalid inputs return 400 with clear messages.
- Load: Handle batch of 16 prompts without crash; memory stable.

### Risks and mitigations

- HRM decoding path may need custom halting logic → start with simple decode, iterate.
- Windows GPU drivers: Recommend WSL2 or use CPU fallback for local dev.

### Milestones

- M1 (1–2 days): Minimal server + local run; sample client.
- M2 (1 day): Docker target and docs; baseline serving notes.
