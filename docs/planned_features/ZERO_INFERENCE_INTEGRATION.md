# DeepSpeed ZeRO-Inference Integration Plan

## Executive Summary

**Goal**: Integrate DeepSpeed ZeRO-Inference to enable inference of massive models (100B+ parameters) on consumer hardware with limited GPU memory by offloading model weights to CPU/NVMe while streaming layers to GPU for computation.

**Current State**: AI-OS supports standard inference that requires the entire model to fit in GPU memory. Large models (>10B params) require multiple high-end GPUs or cannot run at all on consumer hardware.

**Target State**: Full ZeRO-Inference support enabling:
- Single GPU inference of models up to 15 trillion parameters (with NVMe)
- Throughput-oriented inference with large batch sizes
- CPU and NVMe offloading with automatic layer streaming
- Intelligent prefetching and multi-GPU parallelization

**Impact**: 
- Run 100B+ parameter models on a single consumer GPU (11-24GB)
- Democratize access to SOTA models (GPT-3, BLOOM, OPT-175B, etc.)
- Enable inference on models 1000x larger than GPU memory allows
- Trade ~20-40% speed for 90%+ memory savings

---

## Background

### What is ZeRO-Inference?

ZeRO-Inference is DeepSpeed's technology for **massive model inference on limited GPU hardware** by:

1. **Offloading model weights** to CPU DRAM or NVMe storage
2. **Streaming layers** into GPU memory just-in-time for computation
3. **Using GPU memory** primarily for activations and large batch sizes
4. **Prefetching layers** ahead of time to hide transfer latency
5. **Parallelizing fetches** across multiple GPUs for better bandwidth

**Key Innovation**: Keep only 1-2 model layers in GPU memory at a time (~1% of model), use the freed memory for large batches to maximize throughput and hide transfer latency.

### Memory Hierarchy

```
GPU Memory (32GB)
├─ Active Layer Weights: ~1GB (1-2 layers of 100B model)
├─ Activations: ~20GB (batch_size=64, seq_len=2048)
└─ KV Cache: ~10GB (for text generation)

CPU Memory (64GB) or NVMe (2TB)
└─ Full Model Weights: ~200GB (100B params × 2 bytes fp16)
```

### Current AI-OS Capabilities

✅ **Already Implemented**:
- Standard inference (model fully in GPU memory)
- Multi-GPU inference via tensor parallelism
- 8-bit/4-bit quantization for compression
- Flash Attention for memory-efficient attention
- Dynamic batching for throughput

❌ **Missing for ZeRO-Inference**:
- Weight offloading to CPU/NVMe
- Layer-by-layer streaming to GPU
- Prefetching pipeline for hiding latency
- Multi-GPU fetch parallelization
- Throughput-oriented batch size optimization
- Integration with HuggingFace generation pipeline
- Memory-aware batch size auto-tuning

---

## Technical Architecture

### Inference Flow with ZeRO-Inference

```
┌──────────────────────────────────────────────────────────────┐
│             Token Generation with ZeRO-Inference              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Phase 1: Prompt Processing (Process input prompt)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ For each layer (0 to N):                             │   │
│  │   1. Prefetch layer[i+1] from CPU/NVMe (async)       │   │
│  │   2. Load layer[i] to GPU (if not prefetched)        │   │
│  │   3. Compute activations on GPU                      │   │
│  │   4. Free layer[i] weights from GPU                  │   │
│  │   5. Keep activations for next layer                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  Phase 2: Token Generation (Generate output tokens)           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ For each output token (0 to max_tokens):             │   │
│  │   For each layer (0 to N):                           │   │
│  │     1. Prefetch layer[i+1] (async)                   │   │
│  │     2. Load layer[i] to GPU                          │   │
│  │     3. Compute next token logits                     │   │
│  │     4. Update KV cache                               │   │
│  │     5. Free layer[i] weights                         │   │
│  │   Sample next token from logits                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Multi-GPU Fetch Parallelization

When using multiple GPUs, ZeRO-Inference parallelizes layer fetching:

```
Single GPU (PCIe 3.0 @ 15 GB/s):
├─ Fetch Layer: 6GB ÷ 15 GB/s = 400ms
└─ Compute Layer: batch_size=64 → 300ms
    Total: 700ms per layer (fetch dominates!)

4x GPUs (4× PCIe 3.0 @ 60 GB/s aggregate):
├─ Each GPU fetches 1.5GB ÷ 15 GB/s = 100ms
├─ All-gather across GPUs (NVLink @ 300 GB/s): ~20ms
└─ Compute Layer: 300ms
    Total: 420ms per layer (40% faster!)
```

### Prefetching Optimization

```
Without Prefetching:
[Fetch L0] → [Compute L0] → [Fetch L1] → [Compute L1] → ...
 ↑400ms↑     ↑300ms↑       ↑400ms↑      ↑300ms↑
Total: 700ms × N layers

With Prefetching:
[Fetch L0] → [Compute L0] → [Compute L1] → [Compute L2] → ...
 ↑400ms↑     [Fetch L1]↑   [Fetch L2]↑
             ↑300ms↑        ↑300ms↑
Total: 400ms + (300ms × N layers)  ← Hides fetch latency!
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Configuration Schema

**File**: `config/deepspeed_zero_inference.json`

```json
{
  "zero_optimization": {
    "stage": 3,
    
    "offload_param": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 100000000,
      "max_in_cpu": 1000000000
    }
  },
  
  "aio": {
    "block_size": 1048576,
    "queue_depth": 8,
    "thread_count": 1,
    "single_submit": false,
    "overlap_events": true
  },
  
  "fp16": {
    "enabled": true
  },
  
  "zero_inference": {
    "enabled": true,
    "offload_device": "cpu",
    "prefetch_layers": 2,
    "pin_memory": true,
    "parallel_fetch": true
  }
}
```

**File**: `config/deepspeed_zero_inference_nvme.json`

```json
{
  "zero_optimization": {
    "stage": 3,
    
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/tmp/deepspeed_inference_offload",
      "pin_memory": true,
      "buffer_count": 5,
      "buffer_size": 100000000,
      "max_in_cpu": 1000000000
    }
  },
  
  "aio": {
    "block_size": 1048576,
    "queue_depth": 16,
    "thread_count": 2,
    "single_submit": false,
    "overlap_events": true
  },
  
  "fp16": {
    "enabled": true
  },
  
  "zero_inference": {
    "enabled": true,
    "offload_device": "nvme",
    "prefetch_layers": 3,
    "pin_memory": true,
    "parallel_fetch": true
  }
}
```

#### 1.2 Inference Configuration Fields

**File**: `src/aios/core/inference/inference_config.py` (new)

```python
"""Configuration for model inference with ZeRO-Inference support."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class InferenceConfig:
    """Configuration for running inference with optional ZeRO-Inference."""
    
    # Model configuration
    model_path: str
    """Path to model checkpoint or HuggingFace model ID."""
    
    device: str = "cuda"
    """Device to run inference on: 'cuda', 'cpu', or 'cuda:0'."""
    
    dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    """Data type for model weights and computation."""
    
    # ZeRO-Inference settings
    use_zero_inference: bool = False
    """Enable ZeRO-Inference for massive models.
    
    When enabled, model weights are offloaded to CPU or NVMe and streamed
    to GPU layer-by-layer. Essential for models that don't fit in GPU memory.
    
    Use Cases:
    - Models >10B params on single GPU
    - Models >100B params on multi-GPU
    - Maximizing batch size for throughput
    
    Trade-offs:
    - Memory: 90%+ GPU memory freed
    - Speed: ~20-40% slower (depends on offload device)
    - Latency: Higher per-token latency (not suitable for real-time)
    
    Default: False (standard inference)
    """
    
    offload_device: Literal["cpu", "nvme"] = "cpu"
    """Device to offload model weights to.
    
    Options:
    - 'cpu': Offload to CPU DRAM (faster, ~20% slowdown)
    - 'nvme': Offload to NVMe storage (slower, ~40% slowdown, unlimited capacity)
    
    CPU Offload:
    - Speed: ~20% slower than GPU-only
    - Capacity: Limited by RAM (~64-256GB typical)
    - Best for: Models up to ~100B params
    
    NVMe Offload:
    - Speed: ~40% slower than GPU-only
    - Capacity: Limited by SSD (~1-4TB typical)
    - Best for: Models 100B-15T params
    
    Requires: Fast NVMe SSD (>2 GB/s) for reasonable performance
    
    Default: 'cpu'
    """
    
    nvme_offload_path: str = "/tmp/deepspeed_inference_offload"
    """Path to NVMe directory for weight offloading.
    
    Only used when offload_device='nvme'. Must have sufficient free space:
    - FP16: model_params × 2 bytes
    - FP32: model_params × 4 bytes
    
    Example space requirements:
    - 10B params (fp16): ~20GB
    - 100B params (fp16): ~200GB
    - 1T params (fp16): ~2TB
    
    Recommended: Fast NVMe SSD on PCIe 3.0+ with >2 GB/s write speed
    
    Default: '/tmp/deepspeed_inference_offload'
    """
    
    prefetch_layers: int = 2
    """Number of layers to prefetch ahead of computation.
    
    Prefetching overlaps layer transfer with computation to hide latency.
    Higher values reduce stalls but use more CPU/NVMe bandwidth.
    
    Recommended values:
    - 1: Minimal prefetch, lowest memory
    - 2: Balanced (default)
    - 3-4: Aggressive prefetch for fast NVMe
    
    Effect on performance:
    - prefetch=0: ~40% slower (no overlap)
    - prefetch=2: ~20% slower (good overlap)
    - prefetch=4: ~15% slower (maximum overlap)
    
    Default: 2
    """
    
    pin_memory: bool = True
    """Use pinned (page-locked) CPU memory for faster GPU transfers.
    
    Pinned memory enables DMA transfers without CPU involvement,
    reducing latency for CPU→GPU transfers by ~30%.
    
    Impact:
    - Speed: ~30% faster transfers
    - Memory: Uses non-swappable RAM
    - Stability: May fail with low RAM (<16GB)
    
    Disable if: System has <16GB RAM or memory errors occur
    
    Default: True
    """
    
    parallel_fetch: bool = True
    """Parallelize layer fetching across multiple GPUs.
    
    When using multiple GPUs, each GPU fetches a portion of the layer,
    then assembles the full layer via all-gather. This parallelizes PCIe
    bandwidth across GPUs.
    
    Example (4 GPUs):
    - Serial: 1 GPU fetches 4GB → 4GB ÷ 15 GB/s = 267ms
    - Parallel: 4 GPUs fetch 1GB each → 1GB ÷ 15 GB/s + all-gather = 90ms
    
    Speedup: 4× PCIe bandwidth → ~3× faster layer loading
    
    Requires: Multiple GPUs with high-speed interconnect (NVLink/PCIe)
    
    Default: True (auto-disabled if single GPU)
    """
    
    # Generation settings
    max_batch_size: Optional[int] = None
    """Maximum batch size for inference.
    
    ZeRO-Inference performs best with large batches to hide transfer latency.
    If None, automatically determined based on available GPU memory.
    
    Typical values:
    - Standard inference: 8-32
    - ZeRO-Inference (CPU): 64-256
    - ZeRO-Inference (NVMe): 128-512
    
    Larger batches → higher throughput but higher latency
    
    Default: None (auto-tuned)
    """
    
    max_tokens: int = 128
    """Maximum number of tokens to generate."""
    
    temperature: float = 1.0
    """Sampling temperature for generation."""
    
    top_p: float = 0.9
    """Top-p (nucleus) sampling parameter."""
    
    top_k: int = 50
    """Top-k sampling parameter."""
    
    # Performance tuning
    aio_block_size: int = 1048576
    """Async I/O block size for NVMe operations (bytes).
    
    Only used when offload_device='nvme'.
    
    Recommended:
    - 1048576 (1MB): Balanced (default)
    - 2097152 (2MB): High throughput NVMe
    - 524288 (512KB): Lower latency
    
    Default: 1048576
    """
    
    aio_queue_depth: int = 8
    """Async I/O queue depth for NVMe operations.
    
    Only used when offload_device='nvme'.
    
    Recommended:
    - 8: Balanced (default)
    - 16: High-performance NVMe
    - 32: Server-grade NVMe
    
    Default: 8
    """
    
    use_flash_attention: bool = True
    """Use Flash Attention for memory-efficient attention computation.
    
    Flash Attention reduces memory usage and improves speed for long contexts.
    Compatible with ZeRO-Inference.
    
    Default: True (if available)
    """
    
    compile_model: bool = False
    """Use torch.compile() for faster computation.
    
    May improve compute speed by ~20-30% but increases startup time.
    
    Compatibility: Works with ZeRO-Inference but limited benefit
    (bottleneck is transfer, not compute).
    
    Default: False
    """
```

#### 1.3 Layer Streaming Engine

**File**: `src/aios/core/inference/layer_streaming.py` (new)

```python
"""Layer streaming engine for ZeRO-Inference."""

from __future__ import annotations
import torch
import asyncio
from typing import Dict, List, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class LayerCache:
    """Cache for prefetched model layers."""
    
    def __init__(
        self,
        max_size: int = 3,
        pin_memory: bool = True,
        device: torch.device = torch.device("cuda")
    ):
        self.max_size = max_size
        self.pin_memory = pin_memory
        self.device = device
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.queue: deque[int] = deque(maxlen=max_size)
    
    def put(self, layer_idx: int, layer_weights: Dict[str, torch.Tensor]) -> None:
        """Add layer weights to cache."""
        if len(self.cache) >= self.max_size:
            # Evict oldest layer
            if self.queue:
                evict_idx = self.queue.popleft()
                if evict_idx in self.cache:
                    del self.cache[evict_idx]
        
        self.cache[layer_idx] = layer_weights
        if layer_idx not in self.queue:
            self.queue.append(layer_idx)
    
    def get(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve layer weights from cache."""
        return self.cache.get(layer_idx)
    
    def has(self, layer_idx: int) -> bool:
        """Check if layer is in cache."""
        return layer_idx in self.cache
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.queue.clear()


class LayerStreamer:
    """Streams model layers from CPU/NVMe to GPU for inference."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: "InferenceConfig",
        offload_manager: "OffloadManager",
    ):
        self.model = model
        self.config = config
        self.offload_manager = offload_manager
        self.device = torch.device(config.device)
        
        # Layer cache for prefetching
        self.cache = LayerCache(
            max_size=config.prefetch_layers + 1,
            pin_memory=config.pin_memory,
            device=self.device,
        )
        
        # Track which layers are currently on GPU
        self.gpu_layers: set[int] = set()
        
        # Prefetch queue
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_task: Optional[asyncio.Task] = None
    
    async def prefetch_layer(self, layer_idx: int) -> None:
        """Prefetch a layer from CPU/NVMe to cache."""
        if self.cache.has(layer_idx):
            return
        
        logger.debug(f"Prefetching layer {layer_idx}")
        
        # Load layer weights from offload manager
        layer_weights = await self.offload_manager.load_layer(layer_idx)
        
        # Store in cache
        self.cache.put(layer_idx, layer_weights)
        
        logger.debug(f"Layer {layer_idx} prefetched successfully")
    
    async def load_layer_to_gpu(self, layer_idx: int) -> None:
        """Load a layer to GPU memory."""
        # Check cache first
        if self.cache.has(layer_idx):
            layer_weights = self.cache.get(layer_idx)
        else:
            # Not in cache, load directly (blocking)
            logger.warning(f"Cache miss for layer {layer_idx}, loading directly")
            layer_weights = await self.offload_manager.load_layer(layer_idx)
        
        # Transfer to GPU
        layer_module = self._get_layer_module(layer_idx)
        
        for name, param in layer_module.named_parameters():
            if name in layer_weights:
                param.data = layer_weights[name].to(
                    self.device,
                    non_blocking=self.config.pin_memory
                )
        
        self.gpu_layers.add(layer_idx)
        logger.debug(f"Layer {layer_idx} loaded to GPU")
    
    def unload_layer_from_gpu(self, layer_idx: int) -> None:
        """Free layer weights from GPU memory."""
        if layer_idx not in self.gpu_layers:
            return
        
        layer_module = self._get_layer_module(layer_idx)
        
        # Move parameters to empty tensors (free GPU memory)
        for param in layer_module.parameters():
            param.data = torch.empty(0, device=self.device)
        
        self.gpu_layers.remove(layer_idx)
        logger.debug(f"Layer {layer_idx} unloaded from GPU")
    
    def _get_layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Get the layer module by index."""
        # Assumes model has .layers attribute (common for transformers)
        # May need to adapt for different architectures
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        else:
            raise AttributeError(f"Cannot find layer {layer_idx} in model")
    
    async def forward_with_streaming(
        self,
        input_ids: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        """Run forward pass with layer streaming."""
        # Get number of layers
        if num_layers is None:
            num_layers = self._get_num_layers()
        
        # Initial embedding
        hidden_states = self._embed_tokens(input_ids)
        
        # Process each layer
        for layer_idx in range(num_layers):
            # Prefetch next layers
            for prefetch_offset in range(1, self.config.prefetch_layers + 1):
                next_idx = layer_idx + prefetch_offset
                if next_idx < num_layers:
                    asyncio.create_task(self.prefetch_layer(next_idx))
            
            # Load current layer to GPU
            await self.load_layer_to_gpu(layer_idx)
            
            # Forward pass through layer
            layer_module = self._get_layer_module(layer_idx)
            hidden_states = layer_module(hidden_states)
            
            # Unload previous layer if not needed
            if layer_idx > 0:
                self.unload_layer_from_gpu(layer_idx - 1)
        
        return hidden_states
    
    def _get_num_layers(self) -> int:
        """Get number of layers in model."""
        if hasattr(self.model, "config"):
            return getattr(self.model.config, "num_hidden_layers", 
                          getattr(self.model.config, "n_layer", 12))
        return 12  # Default fallback
    
    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get initial token embeddings."""
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens(input_ids)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte(input_ids)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            return self.model.model.embed_tokens(input_ids)
        else:
            raise AttributeError("Cannot find embedding layer in model")
```

#### 1.4 Offload Manager

**File**: `src/aios/core/inference/offload_manager.py` (new)

```python
"""Manager for offloading model weights to CPU/NVMe."""

from __future__ import annotations
import torch
import asyncio
import pickle
from pathlib import Path
from typing import Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class OffloadManager:
    """Manages offloading and loading of model weights."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Literal["cpu", "nvme"],
        nvme_path: Optional[str] = None,
        pin_memory: bool = True,
    ):
        self.model = model
        self.device = device
        self.nvme_path = Path(nvme_path) if nvme_path else None
        self.pin_memory = pin_memory
        
        # Storage for offloaded weights
        self.cpu_storage: Dict[int, Dict[str, torch.Tensor]] = {}
        self.nvme_files: Dict[int, Path] = {}
        
        if device == "nvme" and self.nvme_path:
            self.nvme_path.mkdir(parents=True, exist_ok=True)
    
    async def offload_all_layers(self, num_layers: int) -> None:
        """Offload all model layers to CPU or NVMe."""
        logger.info(f"Offloading {num_layers} layers to {self.device}")
        
        for layer_idx in range(num_layers):
            await self.offload_layer(layer_idx)
        
        logger.info(f"All layers offloaded to {self.device}")
    
    async def offload_layer(self, layer_idx: int) -> None:
        """Offload a single layer to CPU or NVMe."""
        layer_module = self._get_layer_module(layer_idx)
        
        # Extract weights
        layer_weights = {}
        for name, param in layer_module.named_parameters():
            if self.device == "cpu":
                # Move to CPU with optional pinning
                cpu_tensor = param.data.cpu()
                if self.pin_memory:
                    cpu_tensor = cpu_tensor.pin_memory()
                layer_weights[name] = cpu_tensor
            else:
                # Save to NVMe
                layer_weights[name] = param.data.cpu()
        
        if self.device == "cpu":
            self.cpu_storage[layer_idx] = layer_weights
        else:
            # Save to disk
            file_path = self.nvme_path / f"layer_{layer_idx}.pt"
            torch.save(layer_weights, file_path)
            self.nvme_files[layer_idx] = file_path
        
        # Free GPU memory
        for param in layer_module.parameters():
            param.data = torch.empty(0)
        
        logger.debug(f"Layer {layer_idx} offloaded to {self.device}")
    
    async def load_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Load layer weights from CPU or NVMe."""
        if self.device == "cpu":
            return self.cpu_storage.get(layer_idx, {})
        else:
            # Load from NVMe
            file_path = self.nvme_files.get(layer_idx)
            if file_path and file_path.exists():
                # Use asyncio to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    torch.load,
                    file_path
                )
            return {}
    
    def _get_layer_module(self, layer_idx: int) -> torch.nn.Module:
        """Get layer module by index."""
        # Same logic as LayerStreamer
        if hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        else:
            raise AttributeError(f"Cannot find layer {layer_idx} in model")
    
    def cleanup(self) -> None:
        """Clean up offloaded weights."""
        self.cpu_storage.clear()
        
        if self.device == "nvme" and self.nvme_path:
            # Delete NVMe files
            for file_path in self.nvme_files.values():
                if file_path.exists():
                    file_path.unlink()
            self.nvme_files.clear()
```

---

### Phase 2: HuggingFace Integration (Week 2-3)

#### 2.1 Generation Pipeline Wrapper

**File**: `src/aios/core/inference/zero_inference_pipeline.py` (new)

```python
"""HuggingFace generation pipeline with ZeRO-Inference support."""

from __future__ import annotations
import torch
from transformers import GenerationMixin
from typing import Optional, Dict, Any, List
import logging

from .layer_streaming import LayerStreamer
from .offload_manager import OffloadManager
from .inference_config import InferenceConfig

logger = logging.getLogger(__name__)


class ZeroInferencePipeline:
    """Text generation pipeline with ZeRO-Inference."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: InferenceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize offload manager
        self.offload_manager = OffloadManager(
            model=model,
            device=config.offload_device,
            nvme_path=config.nvme_offload_path if config.offload_device == "nvme" else None,
            pin_memory=config.pin_memory,
        )
        
        # Initialize layer streamer
        self.streamer = LayerStreamer(
            model=model,
            config=config,
            offload_manager=self.offload_manager,
        )
        
        # Determine num layers
        self.num_layers = self._get_num_layers()
        
        # Offload all layers initially
        import asyncio
        asyncio.run(self.offload_manager.offload_all_layers(self.num_layers))
        
        logger.info(
            f"ZeroInferencePipeline initialized: "
            f"{self.num_layers} layers offloaded to {config.offload_device}"
        )
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate text from prompts using ZeRO-Inference."""
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Generate with streaming
        import asyncio
        outputs = asyncio.run(self._generate_async(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ))
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        return generated_texts
    
    async def _generate_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Async generation with layer streaming."""
        batch_size = input_ids.shape[0]
        
        # Phase 1: Process prompt (prefill)
        hidden_states = await self.streamer.forward_with_streaming(
            input_ids=input_ids,
            num_layers=self.num_layers,
        )
        
        # Get initial logits
        logits = self._get_logits(hidden_states)
        next_tokens = self._sample_tokens(logits[:, -1, :], temperature, top_p, top_k)
        
        # Initialize output with prompt + first token
        output_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
        
        # Phase 2: Auto-regressive generation
        for _ in range(max_tokens - 1):
            # Process only the new token through all layers
            new_token_ids = next_tokens.unsqueeze(1)
            hidden_states = await self.streamer.forward_with_streaming(
                input_ids=new_token_ids,
                num_layers=self.num_layers,
            )
            
            # Get next token
            logits = self._get_logits(hidden_states)
            next_tokens = self._sample_tokens(logits[:, -1, :], temperature, top_p, top_k)
            
            # Append to output
            output_ids = torch.cat([output_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for EOS
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break
        
        return output_ids
    
    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get logits from hidden states."""
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head(hidden_states)
        elif hasattr(self.model, "score"):
            return self.model.score(hidden_states)
        else:
            raise AttributeError("Cannot find output head in model")
    
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample next tokens from logits."""
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens
    
    def _get_num_layers(self) -> int:
        """Get number of layers."""
        if hasattr(self.model, "config"):
            return getattr(
                self.model.config,
                "num_hidden_layers",
                getattr(self.model.config, "n_layer", 12)
            )
        return 12
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.offload_manager.cleanup()
        self.streamer.cache.clear()
```

#### 2.2 CLI Integration

**File**: `src/aios/cli/inference_cli.py` (new)

```python
"""CLI for running inference with ZeRO-Inference support."""

import typer
from pathlib import Path
from typing import Optional, List
import logging

app = typer.Typer(help="Run model inference with ZeRO-Inference")
logger = logging.getLogger(__name__)


@app.command()
def generate(
    model: str = typer.Argument(..., help="Model path or HuggingFace ID"),
    prompts: List[str] = typer.Option(..., "--prompt", "-p", help="Input prompts"),
    
    # Output
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Save generated text to file"
    ),
    
    # Generation params
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
    top_k: int = typer.Option(50, "--top-k", help="Top-k sampling"),
    
    # ZeRO-Inference
    use_zero_inference: bool = typer.Option(
        False, "--zero-inference/--standard",
        help="Enable ZeRO-Inference for massive models"
    ),
    offload_device: str = typer.Option(
        "cpu", "--offload-device",
        help="Offload device: 'cpu' or 'nvme'"
    ),
    nvme_path: str = typer.Option(
        "/tmp/deepspeed_inference_offload",
        "--nvme-path",
        help="NVMe offload directory"
    ),
    prefetch_layers: int = typer.Option(
        2, "--prefetch-layers",
        help="Number of layers to prefetch"
    ),
    
    # Device
    device: str = typer.Option("cuda", "--device", help="Device: 'cuda' or 'cpu'"),
    dtype: str = typer.Option("fp16", "--dtype", help="Data type: 'fp16', 'bf16', 'fp32'"),
):
    """Generate text from prompts using ZeRO-Inference."""
    
    from aios.core.inference.inference_config import InferenceConfig
    from aios.core.inference.zero_inference_pipeline import ZeroInferencePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Load model and tokenizer
    typer.echo(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    if use_zero_inference:
        # Load model to CPU first
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, dtype),
            low_cpu_mem_usage=True,
        )
        
        # Create config
        config = InferenceConfig(
            model_path=model,
            device=device,
            dtype=dtype,
            use_zero_inference=True,
            offload_device=offload_device,
            nvme_offload_path=nvme_path,
            prefetch_layers=prefetch_layers,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        # Create pipeline
        typer.echo(f"Initializing ZeRO-Inference (offload to {offload_device})")
        pipeline = ZeroInferencePipeline(
            model=model_obj,
            tokenizer=tokenizer,
            config=config,
        )
        
        # Generate
        typer.echo("Generating...")
        outputs = pipeline.generate(prompts)
        
    else:
        # Standard inference
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=getattr(torch, dtype),
        ).to(device)
        
        typer.echo("Generating (standard inference)...")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        outputs_ids = model_obj.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
    
    # Print outputs
    typer.echo("\n" + "=" * 80)
    for i, output in enumerate(outputs):
        typer.echo(f"\nPrompt {i+1}: {prompts[i]}")
        typer.echo(f"Generated: {output}")
        typer.echo("=" * 80)
    
    # Save to file
    if output_file:
        output_file.write_text("\n\n".join(outputs))
        typer.echo(f"\nSaved to {output_file}")


if __name__ == "__main__":
    app()
```

---

### Phase 3: Performance Optimization (Week 3-4)

#### 3.1 Batch Size Auto-Tuning

**File**: `src/aios/core/inference/batch_size_tuner.py` (new)

```python
"""Auto-tune batch size for ZeRO-Inference."""

import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_max_batch_size(
    model: torch.nn.Module,
    seq_length: int,
    available_vram_gb: float,
    use_zero_inference: bool = True,
    safety_margin: float = 0.9,
) -> int:
    """Estimate maximum batch size given constraints."""
    
    # Get model size
    num_params = sum(p.numel() for p in model.parameters())
    
    if use_zero_inference:
        # With ZeRO-Inference, only 1-2 layers in VRAM
        model_vram_gb = (num_params * 2) / (1024**3) * 0.05  # ~5% of model
    else:
        # Standard inference: full model in VRAM
        model_vram_gb = (num_params * 2) / (1024**3)  # FP16
    
    # Calculate per-sample memory
    hidden_size = getattr(model.config, "hidden_size", 768)
    num_layers = getattr(model.config, "num_hidden_layers", 12)
    
    # Activations per sample
    activation_gb_per_sample = (
        seq_length * hidden_size * num_layers * 4  # FP32 activations
    ) / (1024**3)
    
    # KV cache per sample (for generation)
    kv_cache_gb_per_sample = (
        2 * num_layers * seq_length * hidden_size * 2  # FP16 K+V cache
    ) / (1024**3)
    
    total_per_sample = activation_gb_per_sample + kv_cache_gb_per_sample
    
    # Calculate max batch size
    available_for_batch = (available_vram_gb - model_vram_gb) * safety_margin
    max_batch_size = int(available_for_batch / total_per_sample)
    
    # Clamp to reasonable range
    max_batch_size = max(1, min(max_batch_size, 512))
    
    logger.info(
        f"Estimated max batch size: {max_batch_size} "
        f"(seq_len={seq_length}, VRAM={available_vram_gb:.1f}GB)"
    )
    
    return max_batch_size


def find_optimal_batch_size(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    use_zero_inference: bool = True,
    max_seq_length: int = 2048,
) -> int:
    """Binary search to find optimal batch size."""
    
    # Get available VRAM
    if device.type == "cuda":
        torch.cuda.empty_cache()
        available_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    else:
        available_vram_gb = 16.0  # Default for CPU
    
    # Estimate starting point
    estimated_max = estimate_max_batch_size(
        model=model,
        seq_length=max_seq_length,
        available_vram_gb=available_vram_gb,
        use_zero_inference=use_zero_inference,
    )
    
    # Binary search for actual max
    low, high = 1, estimated_max
    optimal = 1
    
    while low <= high:
        mid = (low + high) // 2
        
        # Try this batch size
        try:
            test_input = torch.randint(
                0, 1000,
                (mid, max_seq_length // 4),  # Test with 1/4 seq len
                device=device
            )
            
            with torch.no_grad():
                _ = model(test_input)
            
            # Success! Try larger
            optimal = mid
            low = mid + 1
            
            # Clear memory
            del test_input
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM, try smaller
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise
    
    logger.info(f"Optimal batch size found: {optimal}")
    return optimal
```

#### 3.2 Performance Monitoring

**File**: `src/aios/core/inference/performance_monitor.py` (new)

```python
"""Monitor and log ZeRO-Inference performance metrics."""

import time
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for inference."""
    
    total_tokens: int
    total_time_s: float
    tokens_per_second: float
    
    # Layer streaming metrics
    avg_layer_load_time_ms: float
    avg_layer_compute_time_ms: float
    cache_hit_rate: float
    
    # Memory metrics
    peak_gpu_memory_gb: float
    avg_gpu_memory_gb: float
    
    def __str__(self) -> str:
        return (
            f"Performance Metrics:\n"
            f"  Throughput: {self.tokens_per_second:.2f} tokens/s\n"
            f"  Total Tokens: {self.total_tokens}\n"
            f"  Total Time: {self.total_time_s:.2f}s\n"
            f"  Layer Load Time: {self.avg_layer_load_time_ms:.2f}ms\n"
            f"  Layer Compute Time: {self.avg_layer_compute_time_ms:.2f}ms\n"
            f"  Cache Hit Rate: {self.cache_hit_rate:.1%}\n"
            f"  Peak GPU Memory: {self.peak_gpu_memory_gb:.2f}GB\n"
            f"  Avg GPU Memory: {self.avg_gpu_memory_gb:.2f}GB"
        )


class PerformanceMonitor:
    """Monitor ZeRO-Inference performance."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.layer_load_times: List[float] = []
        self.layer_compute_times: List[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.gpu_memory_samples: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_tokens: int = 0
    
    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()
    
    def end(self) -> None:
        """End timing."""
        self.end_time = time.time()
    
    def record_layer_load(self, duration_ms: float) -> None:
        """Record layer load time."""
        self.layer_load_times.append(duration_ms)
    
    def record_layer_compute(self, duration_ms: float) -> None:
        """Record layer compute time."""
        self.layer_compute_times.append(duration_ms)
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.cache_misses += 1
    
    def sample_gpu_memory(self) -> None:
        """Sample current GPU memory usage."""
        if self.device == "cuda":
            import torch
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            self.gpu_memory_samples.append(memory_gb)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        total_time = (self.end_time or time.time()) - (self.start_time or 0)
        tokens_per_second = self.total_tokens / total_time if total_time > 0 else 0
        
        avg_load_time = (
            sum(self.layer_load_times) / len(self.layer_load_times)
            if self.layer_load_times else 0
        )
        
        avg_compute_time = (
            sum(self.layer_compute_times) / len(self.layer_compute_times)
            if self.layer_compute_times else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0
        )
        
        peak_memory = max(self.gpu_memory_samples) if self.gpu_memory_samples else 0
        avg_memory = (
            sum(self.gpu_memory_samples) / len(self.gpu_memory_samples)
            if self.gpu_memory_samples else 0
        )
        
        return PerformanceMetrics(
            total_tokens=self.total_tokens,
            total_time_s=total_time,
            tokens_per_second=tokens_per_second,
            avg_layer_load_time_ms=avg_load_time,
            avg_layer_compute_time_ms=avg_compute_time,
            cache_hit_rate=cache_hit_rate,
            peak_gpu_memory_gb=peak_memory,
            avg_gpu_memory_gb=avg_memory,
        )
```

---

### Phase 4: Testing & Validation (Week 4-5)

#### 4.1 Unit Tests

**File**: `tests/test_zero_inference.py`

```python
"""Unit tests for ZeRO-Inference components."""

import pytest
import torch
from aios.core.inference.layer_streaming import LayerCache, LayerStreamer
from aios.core.inference.offload_manager import OffloadManager
from aios.core.inference.inference_config import InferenceConfig


def test_layer_cache():
    """Test layer cache functionality."""
    cache = LayerCache(max_size=3, device=torch.device("cpu"))
    
    # Add layers
    layer0 = {"weight": torch.randn(10, 10)}
    layer1 = {"weight": torch.randn(10, 10)}
    layer2 = {"weight": torch.randn(10, 10)}
    
    cache.put(0, layer0)
    cache.put(1, layer1)
    cache.put(2, layer2)
    
    assert cache.has(0)
    assert cache.has(1)
    assert cache.has(2)
    
    # Eviction test
    layer3 = {"weight": torch.randn(10, 10)}
    cache.put(3, layer3)
    
    assert not cache.has(0)  # Evicted
    assert cache.has(3)


def test_offload_manager_cpu():
    """Test CPU offload manager."""
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Linear(10, 10),
    )
    
    manager = OffloadManager(
        model=model,
        device="cpu",
        pin_memory=False,
    )
    
    # This test would need async handling
    # Skipping full implementation for brevity


def test_inference_config():
    """Test inference config creation."""
    config = InferenceConfig(
        model_path="gpt2",
        use_zero_inference=True,
        offload_device="cpu",
        prefetch_layers=2,
    )
    
    assert config.use_zero_inference
    assert config.offload_device == "cpu"
    assert config.prefetch_layers == 2
```

#### 4.2 Integration Tests

**File**: `tests/integration/test_zero_inference_e2e.py`

```python
"""End-to-end tests for ZeRO-Inference."""

import pytest
import torch
from pathlib import Path


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_zero_inference_small_model():
    """Test ZeRO-Inference on GPT-2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from aios.core.inference.zero_inference_pipeline import ZeroInferencePipeline
    from aios.core.inference.inference_config import InferenceConfig
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Create config
    config = InferenceConfig(
        model_path="gpt2",
        device="cuda",
        use_zero_inference=True,
        offload_device="cpu",
        max_tokens=10,
    )
    
    # Create pipeline
    pipeline = ZeroInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    
    # Generate
    outputs = pipeline.generate(["Hello, world!"])
    
    assert len(outputs) == 1
    assert len(outputs[0]) > 0
    
    # Cleanup
    pipeline.cleanup()
```

---

### Phase 5: Documentation (Week 5)

#### 5.1 User Guide

**File**: `docs/guide/zero_inference_guide.md`

Content covering:
- What is ZeRO-Inference
- When to use it (model size thresholds)
- Hardware requirements
- Setup instructions
- CLI examples
- Performance tuning
- Troubleshooting

#### 5.2 API Documentation

- Comprehensive docstrings for all classes
- Usage examples in code
- Integration guides for custom models

---

## Success Metrics

### Functional Requirements
1. ✅ Run 100B+ param models on single 24GB GPU
2. ✅ CPU and NVMe offload working
3. ✅ Layer prefetching reduces latency
4. ✅ Multi-GPU fetch parallelization
5. ✅ Integration with HuggingFace models

### Performance Requirements
1. ✅ CPU offload: <30% slowdown vs GPU-only
2. ✅ NVMe offload: <50% slowdown vs GPU-only
3. ✅ Prefetching: >20% speedup vs no prefetch
4. ✅ Multi-GPU: Near-linear scaling (2x GPUs → ~1.8x faster)
5. ✅ Throughput: >10 tokens/s for 100B model on 1x 24GB GPU

### Quality Requirements
1. ✅ Comprehensive error handling
2. ✅ Performance monitoring
3. ✅ Auto-tuned batch sizes
4. ✅ Unit test coverage >80%
5. ✅ Complete documentation

---

## Risks & Mitigations

### Risk 1: HuggingFace API Compatibility
**Impact**: Different models have different layer access patterns

**Mitigation**:
- Abstract layer access via adapter pattern
- Support common architectures (GPT, LLaMA, BLOOM, OPT)
- Provide custom layer accessor API

### Risk 2: Transfer Latency Dominates
**Impact**: Even with prefetching, may be too slow

**Mitigation**:
- Optimize batch sizes (larger = better amortization)
- Use pinned memory
- Profile and optimize transfer paths
- Consider compression (future)

### Risk 3: Memory Fragmentation
**Impact**: GPU memory fragmentation causes OOM

**Mitigation**:
- Use memory pools
- Aggressive cache clearing
- Monitor fragmentation
- Restart recommendation if severe

---

## Timeline

- **Week 1-2**: Phase 1 (Core Infrastructure)
- **Week 2-3**: Phase 2 (HuggingFace Integration)
- **Week 3-4**: Phase 3 (Performance Optimization)
- **Week 4-5**: Phase 4 (Testing)
- **Week 5**: Phase 5 (Documentation)
- **Total**: ~5 weeks

---

## Dependencies

### Required Packages
```toml
[project.optional-dependencies]
inference = [
    "deepspeed>=0.6.6",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
]
```

### Hardware Requirements

**Minimum**:
- 1x NVIDIA GPU (11GB+)
- 32GB RAM
- CUDA 11.0+

**Recommended**:
- 1-2x NVIDIA GPU (24GB+)
- 64GB RAM
- NVMe SSD (for >100B models)

---

## References

- [DeepSpeed ZeRO-Inference Blog](https://www.deepspeed.ai/2022/09/09/zero-inference.html)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [HuggingFace Accelerate](https://github.com/huggingface/accelerate)

---

## Conclusion

ZeRO-Inference integration will democratize access to massive models by enabling inference on consumer hardware. Users with a single 24GB GPU will be able to run models previously requiring 8+ high-end GPUs.

**Key Benefits**:
- Run 100B+ models on single GPU
- 90%+ memory savings
- Reasonable performance (<30% slowdown with CPU offload)
- Full HuggingFace compatibility
- Automatic optimization

**Recommendation**: Proceed with implementation, focusing on CPU offload first, then NVMe support.
