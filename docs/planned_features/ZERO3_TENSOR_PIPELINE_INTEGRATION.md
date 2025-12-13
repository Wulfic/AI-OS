# ZeRO-3 Hybrid Parallel Inference

## Executive Summary
- Deliver a unified inference stack that combines DeepSpeed ZeRO stage 3 inference, tensor parallelism, and pipeline parallelism so the Chat and Evaluation panels can serve and benchmark 70B–1T parameter models on heterogeneous clusters.
- Stream model weights from CPU/NVMe while keeping GPU memory focused on activations and KV cache to sustain high-throughput batching without saturating a single device.[1][2]
- Introduce topology-aware orchestration so tensor sharding stays inside fast NVLink/PCIe domains, while pipeline stages stretch across nodes when memory pressure demands it.[7][8][10]
- Expose presets in the GUI that hide backend complexity but provide expert overrides for advanced operators.

## Goals & Success Criteria
- **Chat panel** can execute default prompts on >70B models with configurable ZeRO-3 + TP + PP and maintain <2.0× latency regression versus today at batch=1.
- **Evaluation panel** can launch lm-eval-harness jobs against GPTR-like workloads using the same distributed runtime with reproducible score parity (±0.1) versus reference single-node baselines.
- Provide battle-tested DeepSpeed config templates plus hardware compatibility checks for both Linux and Windows (native + WSL2) deployments.
- Ship dashboards that surface prefill/decode latency, NVMe queue depth, NCCL collective timing, and pipeline bubble percentage.

## Current State Baseline
### Chat panel
- Implements a Tkinter front end that delegates prompt handling to a synchronous `on_send` callback per session (`src/aios/gui/components/chat_panel.py`).
- Relies on a single threaded worker path; stopping a response toggles a thread flag but there is no notion of distributed shards or multi-rank coordination.
- Model lifecycle hooks (`on_load_brain`, `on_unload_model`) assume one device and no fine-grained partition control.

### Evaluation panel
- Uses `HarnessWrapper` and CLI orchestration to spawn lm-evaluation-harness runs (`src/aios/gui/components/evaluation_panel/panel_main.py`).
- Currently expects each evaluation job to run on a single process or coarse multi-GPU launch; there is no support for composing tensor/pipeline sub-topologies.
- Progress tracking is line-based and unaware of distributed stages, so pipeline stalls or NCCL retries are invisible to the UI.

## Technical Background
### ZeRO-3 inference
- Streams layers from CPU or NVMe via configurable buffers while freeing most GPU memory for activations and large batch sizes.[1][2][6]
- Prefetch buckets overlap storage transfers with compute; pinned memory improves PCIe bandwidth but is RAM intensive.[2]
- Optional KV-cache offload keeps decode latency manageable for long chats.[3]

### Tensor parallelism
- Splits weight matrices across GPUs, requiring collective ops (`all_gather`/`all_reduce`) each layer; best kept within intra-node high-bandwidth fabrics.[7][8][16]
- Communicator initialization must align with hardware topology; Megatron exposes APIs to bind ranks, priority, and NCCL tuning knobs.[10][13][15]

### Pipeline parallelism
- Breaks the network into sequential stages with micro-batch scheduling to reduce idle time.[7][9]
- Demands careful load balancing; DeepSpeed `PipelineModule` and Megatron virtual pipeline sizes mitigate uneven decoder blocks.[9][10]

### Hybrid (3D) deployment realities
- Production systems coordinate DP + TP + PP; ZeRO-3 provides the memory savings that make TP/PP practical at inference time.[1][7][12]
- DeepSpeed’s inference engine currently errors if pipeline checkpoints are loaded, so extending or bypassing that limit is a prerequisite.[4]
- FastGen’s SplitFuse scheduler improves mixed prompt/generation throughput and remains compatible with ZeRO partitioning.[11][12]

## Target Architecture
### High-level flow
```
GUI (Chat/Eval) → Inference Service Broker → Runtime Orchestrator
    → Cluster Topology Resolver → (ZeRO-3 Shard Manager + TP Group Manager + PP Scheduler)
        → DeepSpeed/Megatron runtime (per rank) → Transport metrics back to GUI
```

### Runtime orchestration
- Introduce a `HybridInferenceOrchestrator` core module that:
  - Creates process groups for data, tensor, and pipeline dimensions using Megatron parallel APIs when available.[10]
  - Builds DeepSpeed ZeRO-3 configs on the fly (CPU vs NVMe offload, buffer sizes, pinned memory) aligned with detected hardware.[2][6]
  - Wraps pipeline stages with either DeepSpeed PipelineModule (training-compatible) or, when unsupported, a Megatron/TensorRT-LLM compatible executor.
- Maintain topology definitions (JSON/YAML) mapping racks/nodes → NVLink islands; ensure TP ranks stay intra-node, PP extends inter-node as advised.[7][17]

### Chat panel integration
- Replace the single-threaded `on_send` path with a thin client that forwards requests to the orchestrator over an async RPC (gRPC or ZeroMQ) supporting streaming tokens.
- Add UI toggles for preset modes (`Balanced`, `Latency`, `Throughput`) that map to ZeRO buffer sizes, TP degree, and PP depth.
- Surface live telemetry (prefill vs decode latency, cache residency, throughput) fed from orchestrator metrics endpoints.

### Evaluation panel integration
- Extend benchmark runner to emit distributed launch descriptors (world size, TP, PP, ZeRO settings) and persist them in evaluation history for reproducibility.
- Provide batch planners that chunk benchmark suites into jobs fitting available GPU slots while reusing warmed caches when possible.
- Enhance progress tracker to consume JSON events reporting micro-batch advancement per pipeline stage, collective timings, and failure diagnostics.

### Configuration artifacts
- Ship curated DeepSpeed config templates (`config/deepspeed_zero3_tp{N}_pp{M}.json`) covering CPU and NVMe offload permutations.[2][6]
- Add GUI-level presets that resolve to those configs or auto-tune them based on detected GPU memory, NVMe bandwidth, and Windows vs Linux paths.
- Ensure Windows compatibility by supporting UNC-style NVMe paths, enumerating `cudaDeviceProp::integrated` flags via `nvidia-smi`, and providing WSL2-specific guidance.

## Implementation Roadmap
1. **Discovery & prerequisites (2 weeks)**
   - Audit current inference backends, identify assumption points about single GPU, and draft interface contracts for the orchestrator.
   - Extend hardware detection to report NVLink domains, PCIe bandwidth, NVMe IOPS, and OS-specific path nuances.
   - Prototype ZeRO-3 inference using DeepSpeedExamples baseline to validate offload throughput and pinned memory limits on Linux + Windows.[2][3]

2. **ZeRO-3 backend foundation (3 weeks)**
   - Refactor inference loader to produce dynamic ZeRO-3 configs (prefetch buckets, buffer counts, kv offload toggles).[1][2][6]
   - Add orchestrator service exposing REST/gRPC endpoints for session creation, token streaming, and job submission.
   - Integrate KV-cache residency controls and pinned memory safeties (auto downgrade when RAM low).[3]

3. **Tensor parallel integration (3 weeks)**
   - Embed Megatron/TensorRT communicator initialization to align TP groups with intra-node topology; expose TP degree selection in GUI expert drawer.[7][8][10][16]
   - Instrument NCCL timings and provide watchdogs for all-gather/all-reduce hotspots; bubble up alerts to the UI.
   - Update lm-eval harness wrapper to accept TP-aware checkpoints and environment vars (e.g., `TP_SIZE`, `CUDA_DEVICE_MAX_CONNECTIONS`).

4. **Pipeline parallel enablement (4 weeks)**
   - Fork or extend DeepSpeed inference engine to lift pipeline checkpoint restriction, or integrate Megatron pipeline executor when stage counts >1.[4][10]
   - Implement stage balancing heuristics using layer profiles; allow manual overrides for uneven decoder segments.[9]
   - Add pipeline-aware batching (micro-batch scheduling, SplitFuse integration) to keep stages saturated during chat streaming.[11][12]

5. **GUI wiring & UX (2 weeks)**
   - Wire chat panel to orchestrator via async client, add preset selectors, and display real-time metrics.
   - Update evaluation panel to configure distributed runs, visualize pipeline stage progress, and archive topology metadata with results.
   - Provide warning banners when operator-selected modes exceed detected hardware capabilities.

6. **Validation & hardening (3 weeks)**
   - Build automated smoke suites exercising different TP/PP/ZeRO combinations on Linux and Windows runners (local + WSL2).
   - Stress test with representative chat transcripts (long-context, multi-turn) and lm-eval benchmark sets to ensure score parity.
   - Document rollback paths and guardrails (e.g., fallback to single GPU when distributed bring-up fails).

## Observability & Tooling
- Collect metrics: NVMe queue depth, PCIe throughput, NCCL op duration, pipeline stage utilization, KV-cache hit rate.
- Publish results via Prometheus exporters and simple GUI graphs; alert when thresholds breached.
- Capture orchestrator logs with rank IDs; integrate with existing analytics event stream (`logs/diagnostics/analytics_events.jsonl`).

## Testing Strategy
- Unit tests for config generation (ZeRO buffers, TP topology mapping) across Windows/Linux path conventions.
- Integration tests launching mini GPT models (e.g., 7B, 13B, 70B) in various TP/PP configurations; assert token accuracy vs baseline outputs.
- Regression suite for evaluation panel to confirm harness results remain within tolerances and metadata persisted.

## Risks & Mitigations
- **DeepSpeed pipeline gap**: Current inference engine rejects PP checkpoints.[4] → Mitigate by upstream contribution or wrapping Megatron pipeline executor.
- **Bandwidth constraints**: NVMe and PCIe throughput may bottleneck ZeRO streaming.[1] → Autoscale prefetch buckets, provide operator warnings, document hardware minima.
- **Windows heterogeneity**: Driver and filesystem differences can destabilize NVMe offload. → Default to CPU offload on Windows, recommend WSL2 for NVMe, run CI coverage on both OS families.
- **Operational complexity**: Exposing too many knobs risks misconfiguration. → Offer curated presets and guardrails, hide advanced controls behind expert toggles.

## Open Questions
- Should we adopt DeepSpeed FastGen end-to-end or reuse existing batching logic with minimal changes?[11]
- Do we standardize on Megatron runtimes for both TP and PP, or keep DeepSpeed TP when PP depth is 1?
- What is the minimum hardware spec we officially support (GPU memory, NVMe latency, network link speed)?

## References
[1] https://www.deepspeed.ai/2022/09/09/zero-inference.html  
[2] https://raw.githubusercontent.com/microsoft/DeepSpeedExamples/master/inference/huggingface/zero_inference/README.md  
[3] https://raw.githubusercontent.com/microsoft/DeepSpeedExamples/master/inference/huggingface/zero_inference/run_model.py  
[4] https://raw.githubusercontent.com/microsoft/DeepSpeed/master/deepspeed/inference/engine.py  
[5] https://raw.githubusercontent.com/microsoft/DeepSpeed/master/deepspeed/inference/config.py  
[6] https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training  
[7] https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/performance/performance-tuning-guide/deciding-model-sharding-strategy.md#how-to-set-tensor-parallelism-and-pipeline-parallelism  
[8] https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/parallel-strategy.md#parallelism-in-tensorrt-llm  
[9] https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules  
[10] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L640-L948  
[11] https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md#3-dynamic-splitfuse-a-novel-prompt-and-generation-composition-strategy  
[12] https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md#2-existing-llm-serving-techniques-in-literature  
[13] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L181-L260  
[14] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L948-L1105  
[15] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/nccl_allocator.py#L1-L206  
[16] https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/tensor_parallel.py#L1-L214  
[17] https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/commands/serve.py#L269-L299  
[18] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L300-L520  
[19] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L900-L1050  
[20] https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L98-L164
