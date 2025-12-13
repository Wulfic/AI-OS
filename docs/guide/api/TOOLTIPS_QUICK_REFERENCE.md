# GUI Tooltips Reference

## Preset Section

### Preset Header
```
Quick architecture presets with pre-configured parameters.
Select Custom for full control over all parameters.
```

### 1M Preset
```
~1M parameters: Tiny model (hidden=256, 2+2 layers)
Fast training, minimal VRAM (~0.5 GB)
```

### 5M Preset
```
~5M parameters: Small model (hidden=512, 2+2 layers)
Good for testing and quick experiments (~1.5 GB)
```

### 10M Preset
```
~10M parameters: Medium model (hidden=768, 2+2 layers)
Balanced size/performance (~2.5 GB)
```

### 20M Preset
```
~20M parameters: Large model (hidden=1024, 2+2 layers)
Good quality, moderate VRAM (~4 GB)
```

### 50M Preset
```
~50M parameters: Very large (hidden=1536, 2+2 layers)
High quality, needs more VRAM (~7 GB)
```

### Custom Preset
```
Custom architecture: Configure all parameters manually.
Reveals advanced options for hidden size, layers, heads, etc.
```

## Brain Name Field
```
Unique name for this brain/model.
Will be saved to: artifacts/brains/actv1/{name}/
Use descriptive names like: large_context_v1, fast_inference, etc.
```

## Custom Architecture Fields

### Hidden Size
```
Model width / embedding dimension.
Larger = more expressive but more VRAM.
Must be divisible by num_heads.
Examples: 256, 512, 768, 1024, 1536, 2048
```

### H Layers
```
Number of Hierarchical reasoning layers.
Higher-level abstract processing.
More layers = deeper reasoning but slower.
Typical: 2-8 layers
```

### L Layers
```
Number of Local processing layers.
Lower-level detail processing.
More layers = better detail but slower.
Typical: 2-8 layers
```

### Num Heads
```
Number of attention heads per layer.
More heads = more parallel attention patterns.
Must evenly divide hidden_size.
Examples: 4, 8, 12, 16, 24, 32
```

### Expansion
```
Feed-forward network expansion factor.
FFN size = hidden_size × expansion.
Higher = more capacity but more VRAM.
Typical: 2.0-4.0
```

### H Cycles
```
Number of processing cycles per H layer.
More cycles = more refinement per layer.
Typical: 1-3 cycles
```

### L Cycles
```
Number of processing cycles per L layer.
More cycles = more refinement per layer.
Typical: 1-3 cycles
```

### Position Encoding
```
Position encoding method:

• rope (Rotary): Best for long contexts,
  relative positions, no learned params.
  RECOMMENDED for most use cases.

• learned: Absolute positions,
  trained embeddings, fixed max length.
```

## Visual Map

```
┌──────────────────────────────────────────────────────┐
│        Create New HRM Student                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Choose architecture preset: ← "Quick presets..."   │
│  ○ 1M        ← "~1M params: Tiny model..."          │
│  ○ 5M        ← "~5M params: Small model..."         │
│  ○ 10M       ← "~10M params: Medium model..."       │
│  ○ 20M       ← "~20M params: Large model..."        │
│  ○ 50M       ← "~50M params: Very large..."         │
│  ● Custom    ← "Custom architecture: Configure..."   │
│                                                      │
│  Brain name: [new_brain] ← "Unique name..."         │
│                                                      │
│  ┌─ Custom Architecture ─────────────────────────┐  │
│  │                                                │  │
│  │  Hidden size:  [512] ← "Model width..."       │  │
│  │                                                │  │
│  │  H layers:     [2]   ← "Hierarchical..."      │  │
│  │                                                │  │
│  │  L layers:     [2]   ← "Local processing..."  │  │
│  │                                                │  │
│  │  Num heads:    [8]   ← "Attention heads..."   │  │
│  │                                                │  │
│  │  Expansion:    [2.0] ← "FFN expansion..."     │  │
│  │                                                │  │
│  │  H cycles:     [2]   ← "Processing cycles..." │  │
│  │                                                │  │
│  │  L cycles:     [2]   ← "Processing cycles..." │  │
│  │                                                │  │
│  │  Pos encoding: [rope▼] ← "rope/learned/sincos"│  │
│  │                                                │  │
│  │  Note: DeepSpeed ZeRO can be selected in      │  │
│  │        the main training panel                 │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  [Create]  [Cancel]                                  │
└──────────────────────────────────────────────────────┘
```

## Hover Behavior
- Tooltips appear after **0.5 second** hover delay
- Tooltips stay visible while hovering
- Tooltips disappear when mouse moves away
- Multi-line tooltips are properly formatted
- All interactive elements have tooltips
