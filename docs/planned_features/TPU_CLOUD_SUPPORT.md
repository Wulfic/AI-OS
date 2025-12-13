# TPU and Cloud Support

**Status:** 📋 Planned  
**Priority:** High (Option A: Google Cloud TPU first)  
**Complexity:** High (multi-week)  
**Target Version:** Future Release  
**Created:** 2025-12-11  
**Owner:** @AI-OS-Team

## Overview

Add first-class support for TPUs (Google Cloud TPU VM) and cloud accelerators (AWS GPU/Trainium) across CLI, GUI, installers, and training/evaluation pipelines. Today AI-OS is CUDA-first; this plan introduces an accelerator abstraction that can target CUDA, XLA (TPU), and NeuronX (Trainium), and provides remote-execution hooks for cloud runs.

Option A is to ship TPU support first (Google Cloud TPU v4/v5e via torch_xla), then add cloud GPU/Trainium (AWS EC2 P4d/P5 + Trn1/Trn2) as a follow-on.

## Feasibility (no paid hardware access)

- **Local validation path:** torch_xla provides a CPU backend (`PJRT_DEVICE=CPU`) that can execute tiny models; we can use this to sanity-check adapter code, launcher plumbing, and config validation without TPU hardware. Performance will be extremely slow but sufficient for unit/integration tests.
- **Mocked device inventory:** Add mocks/fakes for device enumeration (XLA/Neuron) so CLI/GUI can be exercised in CI without accelerators present.
- **Dry-run launchers:** Implement `--dry-run` for cloud launch scripts to validate command generation, env vars, and bucket paths without actually creating instances.
- **Static analysis:** Dependency guards (optional extras), config schema validation, and type/unit tests around the accelerator abstraction give confidence before real hardware.
- **Community/manual verification:** Once merged, seek community testers with TPU/Trainium to confirm real throughput; keep feature gated/experimental until validated.

## Current State

- `config/default.yaml` hardcodes `train_device: cuda` and `run_device: cuda`; VRAM detection assumes CUDA.
- DeepSpeed configs exist for ZeRO stages (1–3) with CPU/NVMe offload but are CUDA-oriented.
- Mixed-vendor GPU and AMD/Intel GPU support plans exist (`MIXED_GPU_VENDOR_SUPPORT.md`, `AMD_INTEL_GPU_SUPPORT_FIX.md`), but no TPU/Neuron abstractions.
- Installers and GUI do not surface non-CUDA accelerators.

## Goals

### Must Have (TPU first)
- XLA device backend option (`train_device: xla`, `run_device: xla`) with torch_xla 2.2+.
- TPU VM local-mode training/eval support (single host, multi-core with `xmp.spawn`).
- TPU Pod slice support via PJRT multi-host launch (gcloud multi-ssh).
- Data/model I/O on GCS buckets with authenticated access.
- CLI/GUI surface for selecting `xla` and showing TPU topology (cores, slice type).
- Logging/metrics capture (torch_xla debug metrics) into existing diagnostics pipeline.

### Should Have (Cloud GPU/Trainium)
- AWS GPU presets (P4d/P5) with CUDA/NVLink awareness; AMI/DLC suggestions.
- Trainium (Trn1/Trn2) backend via torch-neuronx / NeuronX Distributed (NxD) for training; neuron runtimes for inference.
- Remote launcher that provisions/SSH’s into cloud instances (optionally user-provided) and runs AI-OS jobs headless.
- Artifact sync (checkpoints, logs) between cloud and local via S3/GCS.

### Nice to Have
- Cost/time estimator using known per-hour pricing and VRAM/throughput tables.
- Auto-select best accelerator given context length, batch, and budget.
- GUI wizards for cloud credential setup and instance selection.

### Out of Scope (for this feature)
- Azure TPU/NPUs; on-prem TPU pods; managed training services (Vertex AI, SageMaker) fully integrated (can be future work).
- Heterogeneous TPU+GPU simultaneous training.

## Design

### Accelerator Abstraction
- Introduce `AcceleratorType = ['cuda', 'xla', 'neuron', 'cpu']` with a thin adapter:
  - Device discovery: CUDA (torch.cuda), XLA (torch_xla.xla_model devices), Neuron (torch_neuronx / neuronx_distributed), CPU fallback.
  - Memory/properties API returning `{name, vendor, total_mem_mb, available_mem_mb, topology_info}`.
  - AMP/autocast helpers per backend.
- Reuse mixed-vendor detection patterns from `MIXED_GPU_VENDOR_SUPPORT.md`; extend to XLA/Neuron.

**Backend specifics to handle:**
- XLA: uses PJRT runtime; device strings `xla:0..N`; requires `xm.mark_step()` or `xm.optimizer_step`. Multi-core uses `xmp.spawn`; multi-host uses gcloud `--worker=all` wrapper. Some CUDA custom ops (FlashAttention, bitsandbytes) are unavailable—must feature-flag.
- Neuron (Trainium): uses `torch-neuronx` and `neuronx_distributed` (`import neuronx_distributed as nxd`); BF16 preferred, FP16 unsupported; ops coverage is narrower than CUDA—need safe fallbacks and op list checks. Launchers typically use `torchrun --nproc_per_node=<n_neuron_cores> --nnodes=<hosts>` with `NEURON_RT_NUM_CORES`.

### Configuration
- `config/default.yaml`: allow `train_device`/`run_device` values `cuda|xla|neuron|cpu`; add `cloud_provider: none|gcp|aws` and TPU/Neuron specific knobs (zone, instance type, slice, bucket paths).
- New TPU config section (e.g., `tpu:`): `{type: v5e-8|v4-8|v4-16, zone, project, service_account, pjrt_device: TPU, gcs_bucket}`.
- New AWS section (e.g., `aws:`): `{region, instance_type: p4d.24xlarge|p5.48xlarge|trn1.32xlarge, ami_or_dlc, s3_bucket}`.
- DeepSpeed: add TPU-friendly defaults (no CUDA custom ops; use bf16 where available); add NxD config templates for Trainium.

**Installer/optional deps:**
- Add optional extras: `xla` (torch_xla wheels pinned to torch version) and `neuron` (torch-neuronx, neuronx_distributed). Keep them out of default install to avoid heavy wheels.
- Installers/scripts should detect platform: on TPU VM, `pip install torch==<pin> torch_xla==<same>` or use prebuilt images; on Trainium, install Neuron repo key and `pip install torch-neuronx neuronx_distributed`.

### Training/Eval Pipeline Changes
- CLI (`aios train/eval`) to route device creation through accelerator adapter; when `xla`, wrap entrypoint with `PJRT_DEVICE=TPU` and `xmp.spawn` for multi-core.
- Data loaders: ensure `MpDeviceLoader` path for TPU; pin memory off by default for XLA.
- Checkpoint/optimizer state: confirm torch_xla `xm.save` for TPU; S3/GCS sync hooks for remote runs.
- Metrics: capture torch_xla metrics (`xm.get_metrics`) and include in diagnostics JSON.

### Remote Execution (Cloud)
- **GCP TPU VM**: Use gcloud CLI to create/start TPU VM (v5e-8, v4-8/16), install AI-OS deps, run training via SSH (`--worker=all` for pod slices). Provide scripts under `installers/scripts/cloud/`.
- **AWS GPU**: Provide user guidance to launch P4d/P5 with NVIDIA drivers + Deep Learning AMI/Container; optional helper script to SSH and run AI-OS headless.
- **AWS Trainium**: Optional pilot: install `torch-neuronx`, `neuronx_distributed`; use NxD launcher for multi-device; document limitations (FP16 unsupported, prefer bf16).
- Artifact handling: sync checkpoints/logs to S3/GCS; resume locally by download.

**Reference launch flows (sketch):**
- GCP TPU v5e-8 single host: `gcloud compute tpus tpu-vm create NAME --zone=Z --accelerator-type=v5e-8 --version=tpu-vm-v5-lite`; SSH, install deps, `PJRT_DEVICE=TPU python3 train.py` or `python3 -m torch_xla.distributed.xla_spawn --num_devices=8 train.py`.
- GCP TPU v4-16 pod slice: same create command with `--worker=all --command="PJRT_DEVICE=TPU python3 train.py"` to run on all hosts.
- AWS P4d/P5: recommend AWS Deep Learning AMI or DLC; run `torchrun --nproc_per_node=8 train.py` with NCCL/TCP tuned; ensure `NCCL_IB_HCA`, `NCCL_NET_GDR_LEVEL=PHB` set for good NVLink/ENA perf.
- AWS Trainium: install Neuron (`pip install torch-neuronx neuronx_distributed`), set `NEURON_CC_FLAGS="--model-type transformer"` as needed, launch with `torchrun --nproc_per_node=<cores> --nnodes=<hosts> --rdzv_backend=c10d --rdzv_endpoint=<host>:29400 train_neuron.py`.

## Phased Plan

### Phase 1 — TPU Local (TPU VM single host)
- Add accelerator abstraction module (`src/aios/core/accelerators.py`), integrate with CLI/GUI device display and `torch_info_cmd`.
- Add XLA backend support to training/eval launchers; ensure PJRT env + `xmp.spawn` path.
- Update config defaults to accept `xla`; add validation.
- Add doc/UX guidance for installing torch_xla on TPU VM (pip wheels or preinstalled images).

**Feasibility gates:** CPU/XLA backend smoke tests pass; dry-run launchers generate correct commands; config schema validation covers `xla` path.

### Phase 2 — TPU Pods & Cloud Ops (GCP)
- Add gcloud-based helper scripts to provision TPU VM, upload data, and start training across hosts (using `--worker=all`).
- Add bucket-based dataset/checkpoint paths; optional rsync/gsutil wrappers.
- GUI/CLI status surfaces TPU topology and logs; collect torch_xla metrics.

**Feasibility gates:** dry-run pod launch works; bucket sync tested locally; metrics collection path exercised with mock metrics.

### Phase 3 — AWS Cloud (GPU + Trainium)
- Add AWS config schema and guidance for P4d/P5 with DeepSpeed; confirm nvlink awareness in topology readout.
- (Pilot) Trainium path with torch-neuronx/NxD templates; detect Neuron devices; add warnings for unsupported ops.
- Add S3 sync helpers parallel to GCS flow.

**Feasibility gates:** dry-run AWS launcher emits correct commands; Neuron device detection mocked in CI; S3 sync script dry-run passes.

## Risks & Mitigations
- **Dependency weight**: torch_xla / torch-neuronx are large; gate installs behind optional extras and runtime detection.
- **Kernel incompatibilities**: Custom CUDA ops may fail on XLA/Neuron; provide backend-specific fallbacks and disable unsupported optimizations.
- **Network/egress costs**: Large dataset sync to cloud; add dry-run size estimator and warning prompts.
- **Security**: Manage cloud credentials via env/CLI; never store secrets in config; document least-privilege IAM roles.

## References (external)
- PyTorch/XLA TPU docs: https://docs.pytorch.org/xla/release/2.2/index.html
- AWS Neuron (Trainium/Inferentia) docs: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/
- AWS Deep Learning Containers/AMIs for PyTorch GPUs: https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2-tutorials-training.html
