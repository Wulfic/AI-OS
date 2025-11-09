# Flash Attention 2: Window Size Guide

This feature can be toggled via the training CLI or GUI:
- CLI: enable optimized kernels (if available) and optionally set a sliding window:
    ```powershell
    aios hrm-hf train-actv1 --model gpt2 --dataset-file training_data/curated_datasets/test_sample.txt --steps 10 --batch-size 1 --amp --gradient-checkpointing --window-size 2048 --log-file artifacts/brains/actv1/metrics.jsonl
    ```
    Note: FA2 usage is environment-dependent; when unavailable, PyTorch SDPA is used as a fallback. Windowing works with FA2 or SDPA.
- GUI: “FlashAttn-2” checkbox and “Window Size” field (see GUI Features → Training panel optimizations)

This page complements the canonical attention-optimization feature doc:
- Canonical: [FLASH_ATTENTION_VS_CHUNKING.md](FLASH_ATTENTION_VS_CHUNKING.md)

## What is Window Size?

**Window size** is NOT about enabling Flash Attention - it's about **limiting attention range** using a sliding window.

### Sliding Window Attention

Instead of each token attending to ALL previous tokens (full attention), it only attends to the N most recent tokens.

```
Full Attention (window_size = None or 0):
Token 1000 can attend to: Token 1, 2, 3, ..., 999, 1000 (all 1000 tokens)

Sliding Window (window_size = 512):
Token 1000 can attend to: Token 488, 489, ..., 999, 1000 (only 512 tokens)
```

## Why Use Sliding Window?

### Benefits
✅ **Reduced memory** - Less attention computation
✅ **Faster training** - Fewer attention scores to compute
✅ **Enables longer contexts** - Can fit more tokens in VRAM
✅ **Local coherence** - Most relevant context is usually recent

### Trade-offs
❌ **Limited long-range attention** - Can't see tokens outside window
❌ **May lose important context** - Earlier information might be needed
❌ **Not suitable for all tasks** - Some tasks need full context

## Choosing the Right Window Size

### Decision Matrix

| Context Length | Recommended Window | Reasoning |
|----------------|-------------------|-----------|
| **< 2K tokens** | `None` (full) | No need for windowing, fits easily |
| **2K-8K tokens** | `None` or `2048` | Full attention works fine |
| **8K-16K tokens** | `2048-4096` | Balance memory and context |
| **16K-32K tokens** | `1024-2048` | Need windowing for efficiency |
| **32K-64K tokens** | `512-1024` | Aggressive windowing needed |
| **64K-100K tokens** | `256-512` | Very aggressive windowing |
| **100K+ tokens** | `256` | Maximum memory savings |

### Rule of Thumb

```python
if context_length < 8192:
    window_size = None  # Full attention
elif context_length < 32768:
    window_size = 2048  # Moderate window
else:
    window_size = 512   # Aggressive window
```

## Window Size vs Context Length

**IMPORTANT**: Window size is NOT the same as max sequence length!

```
max_seq_len = 50000    # How many tokens to train on
window_size = 512      # How far each token can "see" back

Example with 50K tokens:
├─ Token 1:    Sees tokens 1 (only itself)
├─ Token 100:  Sees tokens 1-100 (all previous, window not limiting yet)
├─ Token 1000: Sees tokens 488-1000 (512 token window)
└─ Token 50000: Sees tokens 49488-50000 (512 token window)
```

## Practical Examples

### Example 1: Short Story (4K tokens)
```yaml
max_seq_len: 4096
window_size: None  # Full attention - story is short enough
use_flash_attn: True  # Enable Flash Attention for speed
```
**Result**: Each word can see the ENTIRE story

### Example 2: Long Document (32K tokens)
```yaml
max_seq_len: 32768
window_size: 2048  # Sliding window - see ~2K tokens back
use_flash_attn: True  # Enable Flash Attention for efficiency
```
**Result**: Each word sees ~2K tokens of recent context

### Example 3: Extreme Context (100K tokens)
```yaml
max_seq_len: 100000
window_size: 512   # Very limited window - memory constrained
use_flash_attn: True  # Enable Flash Attention
use_chunked_training: True  # Also enable chunking
chunk_size: 2048   # Process in 2048-token chunks
```
**Result**: Each word sees only ~500 tokens back, processed in chunks

## Window Size for Different Tasks

### Full Attention (window_size = None)
**Best for**:
- Short contexts (< 8K tokens)
- Tasks requiring global understanding
- Document classification
- Sentiment analysis

### Medium Window (1024-4096)
**Best for**:
- Long documents (8K-32K tokens)
- Story writing
- Technical documentation
- Most training scenarios

### Small Window (256-512)
**Best for**:
- Extreme contexts (50K+ tokens)
- Memory-constrained scenarios
- Stream-of-consciousness text
- Chat logs

## How Flash Attention Uses Window Size

### With Flash Attention 2 Enabled

```python
if window_size is not None:
    # Flash Attention uses efficient sliding window
    window = (window_size - 1, 0)  # Look back window_size-1 tokens
    output = flash_attn_func(q, k, v, causal=True, window_size=window)
else:
    # Flash Attention with full attention
    output = flash_attn_func(q, k, v, causal=True)
```

### Without Flash Attention (Fallback)

```python
if window_size is not None:
    # PyTorch SDPA with manual mask (less efficient)
    mask = create_sliding_window_mask(window_size)
    output = scaled_dot_product_attention(q, k, v, attn_mask=mask)
else:
    # PyTorch SDPA with full attention
    output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Flash Attention is MORE EFFICIENT at sliding windows** - another reason to use it!

## Common Misconceptions

### ❌ WRONG: "Window size is how many tokens I can train on"
✅ CORRECT: Window size is how far back each token can attend. You can train on 100K tokens with a 512 window.

### ❌ WRONG: "Larger window always better"
✅ CORRECT: Larger window uses more memory. Choose based on what your task needs and memory allows.

### ❌ WRONG: "Window size enables Flash Attention"
✅ CORRECT: Window size is a parameter TO Flash Attention. The checkbox enables it, window size configures it.

### ❌ WRONG: "I need window_size = max_seq_len"
✅ CORRECT: That's just full attention. Use `window_size = None` instead.

## Testing Window Sizes

### Start Conservative
1. Begin with **no window** (full attention) for short contexts
2. If OOM, enable window at **max_seq_len / 4**
3. Gradually reduce window if still OOM
4. Monitor training quality - smaller windows may reduce accuracy

### Monitor Impact
```python
# Log attention range
effective_context = min(window_size or max_seq_len, max_seq_len)
print(f"Each token attends to {effective_context} previous tokens")
```

## GUI Settings

### Flash Attention Checkbox
- **Checked**: Use Flash Attention 2 (if available)
- **Unchecked**: Use PyTorch SDPA fallback

### Window Size Entry
- **Empty or 0**: Full attention (no window)
- **256-8192**: Sliding window size in tokens
- **Default: 512**: Good balance for long contexts

### Recommended Combinations

```
Short context (< 8K):
☑ FlashAttn-2  Window: [    ] (empty/full attention)

Medium context (8K-32K):
☑ FlashAttn-2  Window: [2048]

Long context (32K-64K):
☑ FlashAttn-2  Window: [1024]
☑ Context Chunking  Chunk Size: [4096]

Extreme context (100K+):
☑ FlashAttn-2  Window: [512]
☑ Context Chunking  Chunk Size: [2048]
```

## Performance Impact

### Memory Usage (50K token sequence)

| Configuration | VRAM Usage | Speed |
|--------------|------------|-------|
| Full Attn, No Flash | ~20GB ❌ | Baseline |
| Full Attn + Flash | ~4GB ✅ | +30% faster |
| Window 2048 + Flash | ~2GB ✅ | +50% faster |
| Window 512 + Flash | ~1GB ✅ | +80% faster |

### Accuracy Impact

```
Window Size vs Task Performance:
- Full attention: 100% baseline accuracy
- Window 4096: 99-100% (minimal impact)
- Window 2048: 95-99% (slight impact on long-range tasks)
- Window 512: 90-95% (noticeable for tasks needing full context)
- Window 256: 85-90% (significant for most tasks)
```

## Summary

| Parameter | Purpose | Values | Default |
|-----------|---------|--------|---------|
| **use_flash_attn** | Enable Flash Attention | True/False | Should be True (GUI checkbox) |
| **window_size** | Sliding window size | None or 256-8192 | 512 |
| **max_seq_len** | Total sequence length | Any | 2048 |

**Key Insight**: Window size is about **local vs global attention**, not about enabling Flash Attention. The checkbox enables Flash Attention, the window size configures how it attends.

**Recommendation**: 
- Enable Flash Attention checkbox (for speed)
- Set window_size based on your context length and memory
- Use "Optimize Settings" button to find optimal values
