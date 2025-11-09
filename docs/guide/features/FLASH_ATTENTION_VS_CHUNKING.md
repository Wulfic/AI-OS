# Flash Attention 2 vs Context Chunking: Technical Deep Dive

Note: Canonical source of truth for attention optimization in AI-OS. Window sizing guidance is a sub-topic; see `FLASH_ATTENTION.md`.

## Executive Summary

**Flash Attention 2** and **Context Chunking** solve different problems in the memory hierarchy:
- **Flash Attention 2**: Optimizes the *attention computation algorithm itself* (compute-level optimization)
- **Context Chunking**: Manages *sequence length* when even optimized attention can't fit in VRAM (data-level optimization)

They're complementary because Flash Attention 2 makes each chunk more efficient, and chunking makes Flash Attention 2 viable for extreme contexts.

---

## The Memory Problem: Understanding the Layers

### Standard Attention Memory Usage

For a sequence of length `N` with `d` dimensions:

```
Standard Attention Memory = O(N²)
```

**Example: 50K token sequence**
- Attention matrix: 50,000 × 50,000 = 2.5 billion entries
- At fp16 (2 bytes): **5GB just for attention scores**
- Plus gradients, activations, KV cache: **~20GB total**

This is why long contexts OOM (Out Of Memory).

---

## Solution 1: Flash Attention 2 (Algorithm Optimization)

### What It Does

Flash Attention 2 **changes how attention is computed** to avoid materializing the full N×N matrix in VRAM.

### Key Innovation: Tiling + Recomputation

Instead of computing the full attention matrix:
1. **Tiles attention into blocks** (e.g., 128×128 chunks)
2. **Streams through HBM** (High Bandwidth Memory) efficiently
3. **Recomputes values** instead of storing intermediate results
4. **Fuses operations** to minimize memory reads/writes

```python
# Standard Attention (simplified)
Q, K, V = input.chunk(3)
scores = Q @ K.T  # N×N matrix materialized in VRAM ❌
attn = softmax(scores)
output = attn @ V

# Flash Attention 2 (simplified concept)
output = flash_attn_func(Q, K, V)  # Never materializes N×N ✅
# Internally uses tiling: processes 128×128 blocks at a time
```

### Memory Reduction

```
Flash Attention 2 Memory = O(N)
```

**Same 50K example:**
- No full attention matrix stored
- Memory: **~2-4GB** instead of 20GB
- Can handle **2-3x longer contexts** with same VRAM

### What It DOESN'T Solve

Flash Attention 2 still needs to:
- Store full input sequence (50K tokens × embedding dimension)
- Store full gradients during backprop
- Store model activations for each token

**Limit: ~16K-32K tokens on consumer GPUs (24GB VRAM)**

---

## Solution 2: Context Chunking (Sequence Management)

### What It Does

Context Chunking **splits the sequence into smaller pieces** and processes them separately.

### How It Works

```python
# Without Chunking (standard)
input_tokens = [1, 2, 3, 4, ..., 50000]  # All at once
output = model(input_tokens)  # OOM! ❌

# With Chunking
chunk_1 = [1, 2, ..., 2048]      # Process first 2048 tokens
output_1, hidden_state_1 = model(chunk_1)

chunk_2 = [2049, 2050, ..., 4096]  # Process next 2048 tokens
output_2, hidden_state_2 = model(chunk_2, carry=hidden_state_1)

# Continue for all chunks...
```

### Key Features

1. **Recurrent State Passing**: Hidden states carry context between chunks
2. **Gradient Accumulation**: Gradients accumulated across chunks
3. **CPU Offloading**: Can move carry states to CPU for extreme contexts (100K+)

### Memory Reduction

```
Chunked Training Memory = O(chunk_size)
```

**Same 50K example with 2048 chunk size:**
- Only 2048 tokens in VRAM at once
- Rest stored in RAM or on disk
- Memory: **~1-2GB** (very conservative)

### What It DOESN'T Solve

- **Slower training**: Processing chunks sequentially has overhead (~10-20%)
- **Reduced parallelism**: Can't process all tokens in parallel
- **Potential context loss**: Chunks may have limited view of full context

---

## Why They're Complementary

### Scenario 1: Medium Context (8K-16K tokens)

**Flash Attention 2 ONLY:**
```
Sequence: 16K tokens
Flash Attention: 16K fits comfortably in VRAM ✅
Chunking: NOT NEEDED
Result: Fast, efficient training
```

### Scenario 2: Large Context (32K-64K tokens)

**Flash Attention 2 + Chunking:**
```
Sequence: 64K tokens
Flash Attention: Each chunk processed efficiently
Chunking: 64K ÷ 2048 = 32 chunks
Result: Feasible on 24GB GPU
```

Without Flash Attention:
- Standard attention on 2048-token chunks would still use more memory
- Training would be even slower

### Scenario 3: Extreme Context (100K+ tokens)

**Flash Attention 2 + Aggressive Chunking + CPU Offload:**
```
Sequence: 100K tokens
Flash Attention: Optimizes each 512-token chunk
Chunking: 100K ÷ 512 = 195 chunks
CPU Offload: Carry states stored in RAM
Result: Possible on consumer hardware (slow but works)
```

---

## Visual Comparison

### Memory Usage Hierarchy

```
Same 50K Token Sequence:

Standard Attention (NO CHUNKING)
├─ Attention Matrix: 5GB
├─ Activations: 8GB
├─ Gradients: 7GB
└─ Total: ~20GB ❌ OOM on 24GB GPU

Flash Attention 2 (NO CHUNKING)
├─ No Attention Matrix: 0GB (tiled)
├─ Activations: 2GB (optimized)
├─ Gradients: 2GB (optimized)
└─ Total: ~4GB ✅ Fits, but limited headroom

Standard Attention + CHUNKING (2048 chunks)
├─ Attention Matrix: 200MB (per chunk)
├─ Activations: 400MB (per chunk)
├─ Gradients: 300MB (per chunk)
└─ Total: ~1GB ✅ Fits, but SLOW

Flash Attention 2 + CHUNKING (2048 chunks)
├─ No Attention Matrix: 0GB
├─ Activations: 80MB (per chunk, optimized)
├─ Gradients: 60MB (per chunk, optimized)
└─ Total: ~200MB ✅ Fits, FASTER than standard chunking
```

---

## Practical Decision Matrix

| Context Length | GPU VRAM | Recommendation | Why |
|----------------|----------|----------------|-----|
| **< 4K tokens** | Any | Flash Attention 2 | No chunking needed, maximum speed |
| **4K-16K tokens** | 24GB+ | Flash Attention 2 | Fits comfortably without chunking |
| **4K-16K tokens** | 12GB | Flash Attention 2 + Optional Chunking | Test first; chunk if needed |
| **16K-32K tokens** | 24GB+ | Flash Attention 2 + Light Chunking (4096) | Balance speed and memory |
| **32K-64K tokens** | 24GB | Flash Attention 2 + Chunking (2048) | Flash Attention makes chunks efficient |
| **64K-100K tokens** | 24GB | Flash Attention 2 + Aggressive Chunking (512-1024) | Extreme context requires both |
| **100K+ tokens** | 24GB | Flash Attention 2 + Ultra-Aggressive Chunking (256-512) + CPU Offload | Maximum memory savings |

---

## Code Example: How They Work Together

```python
# In aios/core/hrm_models/impl/layers.py
class HRMAttention:
    def forward(self, hidden_states):
        # Flash Attention 2 optimizes THIS computation
        try:
            from flash_attn import flash_attn_func
            # Even with chunking, each chunk uses Flash Attention
            attn_output = flash_attn_func(q, k, v, causal=True)
        except:
            # Fallback to standard attention (slower)
            attn_output = F.scaled_dot_product_attention(q, k, v)
        
        return attn_output

# In aios/core/hrm_models/chunked_training.py
def chunked_segment_rollout(model, batch, chunk_size=2048):
    full_sequence = batch['input_ids']  # e.g., 50K tokens
    
    # Split into chunks
    for chunk_start in range(0, len(full_sequence), chunk_size):
        chunk = full_sequence[chunk_start:chunk_start + chunk_size]
        
        # Each chunk STILL uses Flash Attention inside the model!
        output, carry_state = model(chunk, carry_state=prev_carry)
        
        # Flash Attention makes THIS step 2-3x faster
        loss.backward()  # Gradient computation
        
    return total_loss
```

---

## Key Insight: Different Problems, Different Solutions

### Flash Attention 2 Solves:
❌ **Problem**: Attention computation is memory-inefficient (O(N²))  
✅ **Solution**: Smarter algorithm that avoids materializing full matrix (O(N))  
📊 **Impact**: 2-3x longer contexts with same VRAM  
⚡ **Speed**: Actually FASTER than standard attention  

### Context Chunking Solves:
❌ **Problem**: Even optimized attention can't fit 100K tokens in VRAM  
✅ **Solution**: Process sequence in smaller pieces  
📊 **Impact**: Unlimited context length (constrained by time, not memory)  
⚡ **Speed**: 10-20% slower due to sequential processing  

---

## Analogy: Moving a Mountain

**Problem**: Move 100 tons of rocks from point A to B

**Flash Attention 2 = Better Truck**
- Upgraded from 1-ton truck to 3-ton truck
- Each trip carries 3x more rocks
- Same number of trips needed, but faster per trip
- **Benefit**: Can move 3x more rocks in same time

**Context Chunking = Multiple Trips**
- Split 100 tons into 20 trips of 5 tons each
- Make multiple trips back and forth
- **Benefit**: Can move ANY amount (not limited by truck size)

**Both Together = Best Solution**
- Use the 3-ton truck (Flash Attention)
- Make fewer trips (Chunking with larger chunks)
- Result: Move 100 tons efficiently

---

## When to Use What

### ✅ Use Flash Attention 2 Alone
- Context ≤ 16K tokens on 24GB GPU
- You want maximum training speed
- You have compatible CUDA GPU

### ✅ Use Chunking Alone (Rare)
- Very old GPU without Flash Attention support
- CPU-only training (Flash Attention requires CUDA)
- Debugging/testing with small contexts

### ✅ Use Both Together (Common for Large Contexts)
- Context > 16K tokens
- Training on consumer GPUs (8-24GB VRAM)
- Want balance of memory efficiency and speed
- Extreme contexts (50K-100K+ tokens)

### ❌ Use Neither (Default for Short Contexts)
- Context ≤ 2K tokens
- Plenty of VRAM available
- Standard attention works fine

---

## Summary Table

| Feature | Flash Attention 2 | Context Chunking |
|---------|------------------|------------------|
| **Optimization Level** | Algorithm/Compute | Data/Sequence |
| **Memory Complexity** | O(N) from O(N²) | O(chunk_size) |
| **Speed Impact** | Faster (+20-30%) | Slower (-10-20%) |
| **Max Context Gain** | 2-3x | Unlimited |
| **Hardware Requirement** | CUDA GPU | Any |
| **When Needed** | Always beneficial | Only for very long contexts |
| **Typical Use** | 4K-32K contexts | 32K-100K+ contexts |

---

## Bottom Line

**Flash Attention 2** makes attention computation efficient.  
**Context Chunking** makes extremely long sequences feasible.

Together, they enable training with **50K-100K token contexts on consumer GPUs** that would otherwise require data center hardware.

**Your system now gives users full control** - they can enable chunking when needed, and Flash Attention 2 automatically optimizes whatever they choose to do. The "Optimize Settings" button helps find the sweet spot between memory and speed.

---

## How to switch between approaches (CLI)

- Full attention (no window, no explicit chunking flag):
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 2 --amp --gradient-checkpointing --log-file artifacts/brains/actv1/metrics.jsonl
```

- Sliding window attention (works with FA2 or SDPA):
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 2 --amp --gradient-checkpointing --window-size 2048 --log-file artifacts/brains/actv1/metrics.jsonl
```

- Long-context with dataset chunk cadence:
```powershell
aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 50 --batch-size 1 --amp --gradient-checkpointing --dataset-chunk-size 4000 --log-file artifacts/brains/actv1/metrics.jsonl
```

Notes:
- Windowed attention reduces attention range; dataset chunk size controls data loading/encoding cadence and memory pressure.
- The training loop may also use internal chunking for long sequences; see Configurable Dataset Chunk Size and Parallel Training Block/Chunk System.

## Measuring impact (Optimization CLI)

Use the optimization CLI to gather throughput/VRAM metrics and write results under `artifacts/optimization/`.

```powershell
aios optimize --model artifacts/hf_implant/base_model --batch-sizes "1,2,4,8" --test-duration 10 --output-dir artifacts/optimization
```

Outputs include JSON like `artifacts/optimization/results_<session>.json` and GPU metrics JSONL; compare runs with/without windowing or different dataset chunk sizes.
