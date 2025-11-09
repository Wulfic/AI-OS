"""Baseline memory tests for different tokenizers and context sizes.

This module generates baseline test configurations to establish
accurate memory usage patterns across different:
- Tokenizers (GPT-2, Mistral, Qwen2.5, StarCoder2, CodeLLaMA)
- Context sizes (128, 512, 1024, 2048, 4096, 8192, 16384)
- Model sizes (small, medium, large)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_harness import TestConfiguration


# Tokenizer configurations
TOKENIZERS = {
    "gpt2": {
        "vocab_size": 50257,
        "typical_hidden": 768,
        "typical_layers": 12,
        "typical_heads": 12,
    },
    "mistral-7b": {
        "vocab_size": 32000,
        "typical_hidden": 4096,
        "typical_layers": 32,
        "typical_heads": 32,
    },
    "qwen2.5-7b": {
        "vocab_size": 151936,  # Large vocab!
        "typical_hidden": 3584,
        "typical_layers": 28,
        "typical_heads": 28,
    },
    "starcoder2": {
        "vocab_size": 49152,
        "typical_hidden": 6144,
        "typical_layers": 40,
        "typical_heads": 48,
    },
    "codellama": {
        "vocab_size": 32016,
        "typical_hidden": 4096,
        "typical_layers": 32,
        "typical_heads": 32,
    },
}

# Context sizes to test
# UPDATED: Smaller sizes to fit in 11GB VRAM for baseline testing
CONTEXT_SIZES = [128, 256, 512, 1024, 2048, 4096]

# Model size configurations (for HRM ACT-v1 architecture)
# UPDATED: Much smaller sizes to fit in 11GB VRAM for accurate baseline testing
MODEL_SIZES = {
    "micro": {
        "hidden_size": 128,
        "h_layers": 2,
        "l_layers": 2,
        "description": "Micro model for ultra-quick testing (~1M params)",
    },
    "tiny": {
        "hidden_size": 256,
        "h_layers": 2,
        "l_layers": 2,
        "description": "Tiny model for quick testing (~5-10M params)",
    },
    "small": {
        "hidden_size": 384,
        "h_layers": 3,
        "l_layers": 3,
        "description": "Small model (~20-30M params)",
    },
    "medium": {
        "hidden_size": 512,
        "h_layers": 4,
        "l_layers": 4,
        "description": "Medium model (~50-80M params)",
    },
    "large": {
        "hidden_size": 768,
        "h_layers": 6,
        "l_layers": 6,
        "description": "Large model (~150-200M params)",
    },
}


def calculate_hrm_params(
    hidden_size: int,
    h_layers: int,
    l_layers: int,
    vocab_size: int,
) -> int:
    """Calculate approximate HRM ACT-v1 parameter count.
    
    HRM has:
    - Embedding layer: vocab_size * hidden_size
    - H-level layers: h_layers * (transformer layer params)
    - L-level layers: l_layers * (transformer layer params)
    - LM head: hidden_size * vocab_size
    - Q-head: hidden_size * 2 (halt prediction)
    
    Transformer layer ≈ 12 * hidden_size^2 (attention + FFN)
    """
    embedding_params = vocab_size * hidden_size
    lm_head_params = hidden_size * vocab_size
    q_head_params = hidden_size * 2
    
    # Per-layer params (attention + FFN with expansion=2.0)
    per_layer_params = 12 * hidden_size * hidden_size
    
    total_layers = h_layers + l_layers
    layer_params = total_layers * per_layer_params
    
    total = embedding_params + layer_params + lm_head_params + q_head_params
    
    return total


def generate_baseline_configs(
    model_size: str = "micro",
    batch_size: int = 1,
    num_gpus: int = 1,
) -> List[TestConfiguration]:
    """Generate baseline test configurations.
    
    Args:
        model_size: Model size ("micro", "tiny", "small", "medium", "large")
        batch_size: Batch size for tests (default: 1 for safety)
        num_gpus: Number of GPUs
    
    Returns:
        List of TestConfiguration objects
    """
    configs = []
    
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {model_size}")
    
    model_config = MODEL_SIZES[model_size]
    hidden_size = model_config["hidden_size"]
    h_layers = model_config["h_layers"]
    l_layers = model_config["l_layers"]
    num_layers = h_layers + l_layers
    num_heads = max(4, hidden_size // 64)  # Standard ratio
    
    # Generate configs for each tokenizer and context size
    for tokenizer_name, tokenizer_info in TOKENIZERS.items():
        vocab_size = tokenizer_info["vocab_size"]
        
        # Calculate total params for this configuration
        total_params = calculate_hrm_params(
            hidden_size=hidden_size,
            h_layers=h_layers,
            l_layers=l_layers,
            vocab_size=vocab_size,
        )
        
        for seq_len in CONTEXT_SIZES:
            test_id = f"baseline_{model_size}_{tokenizer_name}_{seq_len}"
            test_name = f"Baseline: {model_size.title()} + {tokenizer_name} + {seq_len}"
            description = (
                f"Baseline test for {model_size} model ({total_params:,} params) "
                f"with {tokenizer_name} tokenizer (vocab={vocab_size:,}) "
                f"at context length {seq_len}"
            )
            
            config = TestConfiguration(
                model_name=f"hrm-actv1-{model_size}",
                tokenizer_name=tokenizer_name,
                total_params=total_params,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=1,  # Always use batch_size=1 for baseline tests
                num_gpus=num_gpus,
                # No optimizations for baseline
                use_amp=False,
                use_gradient_checkpointing=False,
                use_lora=False,
                lora_r=0,
                use_8bit_optimizer=False,
                offload_optimizer=False,
                zero_stage="none",
                use_chunking=False,
                chunk_size=None,
                test_id=test_id,
                test_name=test_name,
                description=description,
            )
            
            configs.append(config)
    
    return configs


def generate_quick_baseline_configs() -> List[TestConfiguration]:
    """Generate a quick subset of baseline configs for fast testing.
    
    Returns:
        List of TestConfiguration objects (reduced set)
    """
    configs = []
    
    # Only test micro model with 2 tokenizers and 3 context sizes
    # UPDATED: Use micro model and smaller contexts to fit in 11GB VRAM
    quick_tokenizers = ["gpt2", "qwen2.5-7b"]  # Small and large vocab
    quick_contexts = [256, 512, 1024]  # Very short, short, medium
    
    model_size = "micro"
    model_config = MODEL_SIZES[model_size]
    hidden_size = model_config["hidden_size"]
    h_layers = model_config["h_layers"]
    l_layers = model_config["l_layers"]
    num_layers = h_layers + l_layers
    num_heads = max(4, hidden_size // 64)
    
    for tokenizer_name in quick_tokenizers:
        if tokenizer_name not in TOKENIZERS:
            continue
        
        tokenizer_info = TOKENIZERS[tokenizer_name]
        vocab_size = tokenizer_info["vocab_size"]
        
        total_params = calculate_hrm_params(
            hidden_size=hidden_size,
            h_layers=h_layers,
            l_layers=l_layers,
            vocab_size=vocab_size,
        )
        
        for seq_len in quick_contexts:
            test_id = f"quick_baseline_{tokenizer_name}_{seq_len}"
            test_name = f"Quick Baseline: {tokenizer_name} + {seq_len}"
            description = (
                f"Quick baseline test with {tokenizer_name} "
                f"(vocab={vocab_size:,}) at context {seq_len}"
            )
            
            config = TestConfiguration(
                model_name=f"hrm-actv1-{model_size}",
                tokenizer_name=tokenizer_name,
                total_params=total_params,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                vocab_size=vocab_size,
                seq_len=seq_len,
                batch_size=1,  # Always use batch_size=1 for quick tests
                num_gpus=1,
                use_amp=False,
                use_gradient_checkpointing=False,
                use_lora=False,
                lora_r=0,
                use_8bit_optimizer=False,
                offload_optimizer=False,
                zero_stage="none",
                use_chunking=False,
                chunk_size=None,
                test_id=test_id,
                test_name=test_name,
                description=description,
            )
            
            configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Generate and display baseline configs
    print("="*80)
    print("FULL BASELINE TEST SUITE")
    print("="*80)
    
    for model_size in ["tiny", "small", "medium"]:
        configs = generate_baseline_configs(model_size=model_size)
        print(f"\n{model_size.upper()} Model: {len(configs)} configurations")
        print(f"  Total params range: {min(c.total_params for c in configs):,} - {max(c.total_params for c in configs):,}")
        print(f"  Context sizes: {sorted(set(c.seq_len for c in configs))}")
        print(f"  Tokenizers: {sorted(set(c.tokenizer_name for c in configs))}")
    
    print("\n" + "="*80)
    print("QUICK BASELINE TEST SUITE")
    print("="*80)
    
    quick_configs = generate_quick_baseline_configs()
    print(f"\nQuick tests: {len(quick_configs)} configurations")
    for config in quick_configs:
        print(f"  - {config.test_name} ({config.total_params:,} params)")
