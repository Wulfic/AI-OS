"""Optimization combination tests for memory estimation.

This module generates comprehensive test configurations covering
ALL combinations of memory optimizations:
- AMP (Automatic Mixed Precision)
- Gradient Checkpointing
- Chunked Sequence Processing
- LoRA/PEFT
- 8-bit Optimizer
- CPU Offload
- DeepSpeed ZeRO stages

Goal: Test every combination to measure actual memory impact
and validate estimation accuracy for each optimization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Iterator
from itertools import product

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_harness import TestConfiguration
from baseline_tests import calculate_hrm_params, TOKENIZERS


def generate_optimization_combinations(
    model_size_config: dict,
    tokenizer_name: str = "gpt2",
    seq_len: int = 2048,
    batch_size: int = 2,
    num_gpus: int = 1,
) -> List[TestConfiguration]:
    """Generate all optimization combinations for testing.
    
    Tests all 2^N combinations where N is number of optimizations.
    With 7 optimizations, this is 128 tests per configuration.
    
    Args:
        model_size_config: Dict with hidden_size, h_layers, l_layers
        tokenizer_name: Tokenizer to use
        seq_len: Sequence length
        batch_size: Batch size
        num_gpus: Number of GPUs
    
    Returns:
        List of TestConfiguration objects
    """
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
    
    tokenizer_info = TOKENIZERS[tokenizer_name]
    vocab_size = tokenizer_info["vocab_size"]
    
    hidden_size = model_size_config["hidden_size"]
    h_layers = model_size_config["h_layers"]
    l_layers = model_size_config["l_layers"]
    num_layers = h_layers + l_layers
    num_heads = max(4, hidden_size // 64)
    
    total_params = calculate_hrm_params(
        hidden_size=hidden_size,
        h_layers=h_layers,
        l_layers=l_layers,
        vocab_size=vocab_size,
    )
    
    # Define optimization flags to test
    # Each will be tested as True/False
    optimizations = {
        "amp": [False, True],
        "gradient_checkpointing": [False, True],
        "chunking": [False, True],
        "lora": [False, True],
        "8bit_optimizer": [False, True],
        "cpu_offload": [False, True],
        # ZeRO stages: none, zero1, zero2, zero3
        "zero_stage": ["none", "zero1", "zero2", "zero3"] if num_gpus > 1 else ["none"],
    }
    
    configs = []
    
    # Generate all combinations
    # Use itertools.product to get Cartesian product
    keys = list(optimizations.keys())
    for values in product(*[optimizations[k] for k in keys]):
        opt_dict = dict(zip(keys, values))
        
        # Create optimization flags
        use_amp = opt_dict["amp"]
        use_gradient_checkpointing = opt_dict["gradient_checkpointing"]
        use_chunking = opt_dict["chunking"]
        use_lora = opt_dict["lora"]
        use_8bit_optimizer = opt_dict["8bit_optimizer"]
        offload_optimizer = opt_dict["cpu_offload"]
        zero_stage = opt_dict["zero_stage"]
        
        # LoRA configuration
        lora_r = 16 if use_lora else 0
        
        # Chunk size (auto-calculated if chunking enabled)
        chunk_size = None if use_chunking else None  # Auto-calculate
        
        # Build test ID and name
        opt_flags = []
        if use_amp:
            opt_flags.append("amp")
        if use_gradient_checkpointing:
            opt_flags.append("gradckpt")
        if use_chunking:
            opt_flags.append("chunk")
        if use_lora:
            opt_flags.append(f"lora{lora_r}")
        if use_8bit_optimizer:
            opt_flags.append("8bitopt")
        if offload_optimizer:
            opt_flags.append("cpuoff")
        if zero_stage != "none":
            opt_flags.append(zero_stage)
        
        if not opt_flags:
            opt_flags.append("baseline")
        
        opt_str = "_".join(opt_flags)
        
        test_id = f"opt_{tokenizer_name}_{seq_len}_{opt_str}"
        test_name = f"Optimization: {opt_str}"
        
        description = (
            f"Testing optimization combination: {', '.join(opt_flags)} "
            f"with {tokenizer_name} (vocab={vocab_size:,}) "
            f"at context {seq_len}, {total_params:,} params"
        )
        
        config = TestConfiguration(
            model_name=f"hrm-actv1-test",
            tokenizer_name=tokenizer_name,
            total_params=total_params,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
            num_gpus=num_gpus,
            use_amp=use_amp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_lora=use_lora,
            lora_r=lora_r,
            use_8bit_optimizer=use_8bit_optimizer,
            offload_optimizer=offload_optimizer,
            zero_stage=zero_stage,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            test_id=test_id,
            test_name=test_name,
            description=description,
        )
        
        configs.append(config)
    
    return configs


def generate_critical_optimization_tests(
    model_size_config: dict,
    tokenizer_name: str = "gpt2",
    seq_len: int = 2048,
) -> List[TestConfiguration]:
    """Generate critical optimization test cases.
    
    Instead of testing all 2^7 = 128 combinations, test only the most
    common and important combinations that users actually use.
    
    Args:
        model_size_config: Dict with hidden_size, h_layers, l_layers
        tokenizer_name: Tokenizer to use
        seq_len: Sequence length
    
    Returns:
        List of critical TestConfiguration objects
    """
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
    
    tokenizer_info = TOKENIZERS[tokenizer_name]
    vocab_size = tokenizer_info["vocab_size"]
    
    hidden_size = model_size_config["hidden_size"]
    h_layers = model_size_config["h_layers"]
    l_layers = model_size_config["l_layers"]
    num_layers = h_layers + l_layers
    num_heads = max(4, hidden_size // 64)
    
    total_params = calculate_hrm_params(
        hidden_size=hidden_size,
        h_layers=h_layers,
        l_layers=l_layers,
        vocab_size=vocab_size,
    )
    
    # Define critical optimization combinations
    critical_combos = [
        # 1. Baseline (no optimizations)
        {
            "name": "baseline",
            "amp": False,
            "grad_ckpt": False,
            "chunking": False,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 2. AMP only (most common single optimization)
        {
            "name": "amp_only",
            "amp": True,
            "grad_ckpt": False,
            "chunking": False,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 3. AMP + Gradient Checkpointing (common combo)
        {
            "name": "amp_gradckpt",
            "amp": True,
            "grad_ckpt": True,
            "chunking": False,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 4. AMP + Chunking (for long context)
        {
            "name": "amp_chunking",
            "amp": True,
            "grad_ckpt": False,
            "chunking": True,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 5. AMP + Gradient Checkpointing + Chunking (aggressive memory saving)
        {
            "name": "amp_gradckpt_chunking",
            "amp": True,
            "grad_ckpt": True,
            "chunking": True,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 6. LoRA fine-tuning (parameter efficient)
        {
            "name": "lora_amp",
            "amp": True,
            "grad_ckpt": False,
            "chunking": False,
            "lora": True,
            "8bit_opt": False,
            "cpu_off": False,
            "zero": "none",
        },
        # 7. 8-bit optimizer (memory efficient optimizer)
        {
            "name": "amp_8bitopt",
            "amp": True,
            "grad_ckpt": False,
            "chunking": False,
            "lora": False,
            "8bit_opt": True,
            "cpu_off": False,
            "zero": "none",
        },
        # 8. CPU offload (for constrained VRAM)
        {
            "name": "amp_cpuoff",
            "amp": True,
            "grad_ckpt": False,
            "chunking": False,
            "lora": False,
            "8bit_opt": False,
            "cpu_off": True,
            "zero": "none",
        },
        # 9. Kitchen sink (all optimizations)
        {
            "name": "all_optimizations",
            "amp": True,
            "grad_ckpt": True,
            "chunking": True,
            "lora": True,
            "8bit_opt": True,
            "cpu_off": True,
            "zero": "none",
        },
        # 10. LoRA + Everything (realistic heavy optimization)
        {
            "name": "lora_heavy",
            "amp": True,
            "grad_ckpt": True,
            "chunking": True,
            "lora": True,
            "8bit_opt": True,
            "cpu_off": False,  # LoRA usually doesn't need CPU offload
            "zero": "none",
        },
    ]
    
    configs = []
    
    for combo in critical_combos:
        name = combo["name"]
        use_amp = combo["amp"]
        use_gradient_checkpointing = combo["grad_ckpt"]
        use_chunking = combo["chunking"]
        use_lora = combo["lora"]
        use_8bit_optimizer = combo["8bit_opt"]
        offload_optimizer = combo["cpu_off"]
        zero_stage = combo["zero"]
        
        lora_r = 16 if use_lora else 0
        chunk_size = None if use_chunking else None
        
        test_id = f"critical_{tokenizer_name}_{seq_len}_{name}"
        test_name = f"Critical: {name}"
        
        description = (
            f"Critical optimization test: {name} "
            f"with {tokenizer_name} (vocab={vocab_size:,}) "
            f"at context {seq_len}, {total_params:,} params"
        )
        
        config = TestConfiguration(
            model_name=f"hrm-actv1-test",
            tokenizer_name=tokenizer_name,
            total_params=total_params,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=2,
            num_gpus=1,
            use_amp=use_amp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_lora=use_lora,
            lora_r=lora_r,
            use_8bit_optimizer=use_8bit_optimizer,
            offload_optimizer=offload_optimizer,
            zero_stage=zero_stage,
            use_chunking=use_chunking,
            chunk_size=chunk_size,
            test_id=test_id,
            test_name=test_name,
            description=description,
        )
        
        configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Demo: Generate optimization tests
    from baseline_tests import MODEL_SIZES
    
    print("="*80)
    print("OPTIMIZATION COMBINATION TESTS")
    print("="*80)
    
    # Full combination test
    model_config = MODEL_SIZES["tiny"]
    full_configs = generate_optimization_combinations(
        model_size_config=model_config,
        tokenizer_name="gpt2",
        seq_len=1024,
    )
    
    print(f"\nFull combinations (tiny model, GPT-2, 1024 context):")
    print(f"  Total configurations: {len(full_configs)}")
    
    # Count by optimization type
    amp_count = sum(1 for c in full_configs if c.use_amp)
    grad_ckpt_count = sum(1 for c in full_configs if c.use_gradient_checkpointing)
    chunking_count = sum(1 for c in full_configs if c.use_chunking)
    lora_count = sum(1 for c in full_configs if c.use_lora)
    
    print(f"  With AMP: {amp_count}")
    print(f"  With Gradient Checkpointing: {grad_ckpt_count}")
    print(f"  With Chunking: {chunking_count}")
    print(f"  With LoRA: {lora_count}")
    
    print("\n" + "="*80)
    print("CRITICAL OPTIMIZATION TESTS")
    print("="*80)
    
    critical_configs = generate_critical_optimization_tests(
        model_size_config=model_config,
        tokenizer_name="gpt2",
        seq_len=1024,
    )
    
    print(f"\nCritical tests (tiny model, GPT-2, 1024 context):")
    print(f"  Total configurations: {len(critical_configs)}")
    print("\nTest cases:")
    for config in critical_configs:
        print(f"  - {config.test_name}")
