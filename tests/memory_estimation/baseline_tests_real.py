"""Real Baseline Test Configurations.

Generates test configurations for actual HRM training tests using train_actv1_impl().
Unlike baseline_tests.py which uses synthetic SimpleTransformer, this generates configs
for the real ACT V1 HRM architecture.

Test Matrix Strategy:
- Model sizes: Based on actual ACT V1 params (h_layers, l_layers, hidden_size)
- Tokenizers: All 5 production tokenizers
- Context sizes: Progressive from 128 to 4096+ tokens
- Batch sizes: Conservative (start with 1)
- Optimizations: Test with/without MoE, gradient checkpointing, AMP

The goal is to gather ACCURATE data including:
- Real model architecture (not synthetic transformer)
- Dataset loading overhead
- Tokenizer overhead
- All actual training optimizations
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test_harness_real import RealTestConfig


# ============================================================================
# Model Size Definitions (ACT V1 Architecture)
# ============================================================================

# These model sizes are based on the actual ACT V1 HRM architecture
# and are sized to fit on an 11GB VRAM GPU (RTX 2080 Ti, 3060, etc.)
#
# Parameter count estimation for ACT V1:
# - Each H-layer: ~2-4M params per layer (depends on hidden_size)
# - Each L-layer: ~2-4M params per layer
# - MoE: Multiplies FFN params by num_experts (but only 2/8 active)
# - Total: (h_layers + l_layers) × per_layer_params
#
# Examples:
# - micro: 1h/1l, 128 hidden → ~1-2M params
# - tiny: 1h/1l, 256 hidden → ~3-5M params  
# - small: 2h/2l, 384 hidden → ~15-20M params
# - medium: 2h/2l, 512 hidden → ~30-40M params
# - large: 3h/3l, 768 hidden → ~80-100M params

REAL_MODEL_SIZES = {
    "micro": {
        "h_layers": 1,
        "l_layers": 1,
        "hidden_size": 128,
        "num_heads": 8,
        "description": "Micro model (~1-2M params) - fastest testing",
    },
    "tiny": {
        "h_layers": 1,
        "l_layers": 1,
        "hidden_size": 256,
        "num_heads": 8,
        "description": "Tiny model (~3-5M params) - quick validation",
    },
    "small": {
        "h_layers": 2,
        "l_layers": 2,
        "hidden_size": 384,
        "num_heads": 8,
        "description": "Small model (~15-20M params) - baseline size",
    },
    "medium": {
        "h_layers": 2,
        "l_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "description": "Medium model (~30-40M params) - production size",
    },
    "large": {
        "h_layers": 3,
        "l_layers": 3,
        "hidden_size": 768,
        "num_heads": 8,
        "description": "Large model (~80-100M params) - max for 11GB VRAM",
    },
}


# ============================================================================
# Tokenizer Definitions
# ============================================================================

# All production tokenizers used in AI-OS
# These represent different vocabulary sizes and encoding strategies
REAL_TOKENIZERS = {
    # General purpose tokenizers
    "gpt2": {
        "model_name": "artifacts/hf_implant/tokenizers/gpt2",
        "vocab_size": 50257,
        "description": "GPT-2 tokenizer (50K vocab) - baseline",
    },
    "mistral-7b": {
        "model_name": "artifacts/hf_implant/tokenizers/mistral-7b",
        "vocab_size": 32000,
        "description": "Mistral 7B tokenizer (32K vocab) - efficient",
    },
    "qwen2.5-7b": {
        "model_name": "artifacts/hf_implant/tokenizers/qwen2.5-7b",
        "vocab_size": 151665,
        "description": "Qwen 2.5 tokenizer (152K vocab) - largest",
    },
    "phi3-mini": {
        "model_name": "artifacts/hf_implant/tokenizers/phi3-mini",
        "vocab_size": 32011,
        "description": "Phi-3 Mini tokenizer (32K vocab) - small model optimized",
    },
    "llava-1.5": {
        "model_name": "artifacts/hf_implant/tokenizers/llava-1.5",
        "vocab_size": 32002,
        "description": "LLaVA 1.5 tokenizer (32K vocab) - multimodal",
    },
    
    # Code-focused tokenizers
    "starcoder2": {
        "model_name": "artifacts/hf_implant/tokenizers/starcoder2",
        "vocab_size": 49152,
        "description": "StarCoder2 tokenizer (49K vocab) - code-focused",
    },
    "codellama": {
        "model_name": "artifacts/hf_implant/tokenizers/codellama",
        "vocab_size": 32016,
        "description": "CodeLlama tokenizer (32K vocab) - code-optimized",
    },
    "deepseek-coder-v2": {
        "model_name": "artifacts/hf_implant/tokenizers/deepseek-coder-v2",
        "vocab_size": 100018,
        "description": "DeepSeek Coder V2 tokenizer (100K vocab) - code with large vocab",
    },
    
    # Vision tokenizers
    "clip-vit": {
        "model_name": "artifacts/hf_implant/tokenizers/clip-vit",
        "vocab_size": 49408,
        "description": "CLIP ViT tokenizer (49K vocab) - vision-language",
    },
    "siglip": {
        "model_name": "artifacts/hf_implant/tokenizers/siglip",
        "vocab_size": 32000,
        "description": "SigLIP tokenizer (32K vocab) - vision-language",
    },
    
    # Domain-specific (scientific/medical/legal) tokenizers
    "biobert": {
        "model_name": "artifacts/hf_implant/tokenizers/biobert",
        "vocab_size": 28996,
        "description": "BioBERT tokenizer (29K vocab) - biomedical",
    },
    "scibert": {
        "model_name": "artifacts/hf_implant/tokenizers/scibert",
        "vocab_size": 31090,
        "description": "SciBERT tokenizer (31K vocab) - scientific papers",
    },
    "finbert": {
        "model_name": "artifacts/hf_implant/tokenizers/finbert",
        "vocab_size": 30522,
        "description": "FinBERT tokenizer (31K vocab) - financial",
    },
    "legal-bert": {
        "model_name": "artifacts/hf_implant/tokenizers/legal-bert",
        "vocab_size": 30522,
        "description": "Legal-BERT tokenizer (31K vocab) - legal documents",
    },
}


# ============================================================================
# Context Size Ranges
# ============================================================================

# Context sizes to test (tokens)
# Start conservative, expand as we validate VRAM capacity
CONTEXT_SIZES_QUICK = [128, 512, 1024]  # Quick validation
CONTEXT_SIZES_STANDARD = [128, 256, 512, 1024, 2048, 4096]  # Standard baseline
CONTEXT_SIZES_EXTENDED = [128, 256, 512, 1024, 2048, 4096, 8192]  # Push limits


# ============================================================================
# Test Configuration Generators
# ============================================================================

def generate_quick_baseline_configs() -> list[RealTestConfig]:
    """Generate quick baseline test configs for validation.
    
    Test matrix: 1 tokenizer × 3 contexts × 2 model sizes = 6 tests
    
    Purpose: Validate that real test harness works correctly before running
    extensive test suite.
    
    Returns:
        List of 6 test configurations
    """
    configs = []
    
    # Use tiny and small models for quick tests
    model_size_keys = ["tiny", "small"]
    
    # Use GPT-2 tokenizer only for quick validation (avoid tokenizer mismatch issues)
    tokenizer_keys = ["gpt2"]
    
    # Use 3 context sizes
    context_sizes = CONTEXT_SIZES_QUICK
    
    for model_size_key in model_size_keys:
        model_params = REAL_MODEL_SIZES[model_size_key]
        
        for tokenizer_key in tokenizer_keys:
            tokenizer = REAL_TOKENIZERS[tokenizer_key]
            
            for context_size in context_sizes:
                config = RealTestConfig(
                    model_name=tokenizer["model_name"],
                    h_layers=model_params["h_layers"],
                    l_layers=model_params["l_layers"],
                    hidden_size=model_params["hidden_size"],
                    num_heads=model_params["num_heads"],
                    context_size=context_size,
                    batch_size=1,  # Conservative
                    use_moe=True,  # Default enabled
                    num_experts=8,
                    num_experts_per_tok=2,
                    gradient_checkpointing=True,
                    use_amp=True,
                )
                configs.append(config)
    
    return configs


def generate_standard_baseline_configs() -> list[RealTestConfig]:
    """Generate standard baseline test configs.
    
    Test matrix: 5 tokenizers × 6 contexts × 3 model sizes = 90 tests
    
    Purpose: Comprehensive baseline covering all tokenizers, common context sizes,
    and representative model sizes.
    
    Returns:
        List of 90 test configurations
    """
    configs = []
    
    # Use small, medium, large models
    model_size_keys = ["small", "medium", "large"]
    
    # All tokenizers
    tokenizer_keys = list(REAL_TOKENIZERS.keys())
    
    # Standard context sizes
    context_sizes = CONTEXT_SIZES_STANDARD
    
    for model_size_key in model_size_keys:
        model_params = REAL_MODEL_SIZES[model_size_key]
        
        for tokenizer_key in tokenizer_keys:
            tokenizer = REAL_TOKENIZERS[tokenizer_key]
            
            for context_size in context_sizes:
                config = RealTestConfig(
                    model_name=tokenizer["model_name"],
                    h_layers=model_params["h_layers"],
                    l_layers=model_params["l_layers"],
                    hidden_size=model_params["hidden_size"],
                    num_heads=model_params["num_heads"],
                    context_size=context_size,
                    batch_size=1,
                    use_moe=True,
                    num_experts=8,
                    num_experts_per_tok=2,
                    gradient_checkpointing=True,
                    use_amp=True,
                )
                configs.append(config)
    
    return configs


def generate_full_baseline_configs() -> list[RealTestConfig]:
    """Generate full baseline test configs.
    
    Test matrix: 5 tokenizers × 6 contexts × 5 model sizes = 150 tests
    
    Purpose: Exhaustive baseline covering all model sizes and context ranges.
    
    Returns:
        List of 150 test configurations
    """
    configs = []
    
    # All model sizes
    model_size_keys = list(REAL_MODEL_SIZES.keys())
    
    # All tokenizers
    tokenizer_keys = list(REAL_TOKENIZERS.keys())
    
    # Standard context sizes
    context_sizes = CONTEXT_SIZES_STANDARD
    
    for model_size_key in model_size_keys:
        model_params = REAL_MODEL_SIZES[model_size_key]
        
        for tokenizer_key in tokenizer_keys:
            tokenizer = REAL_TOKENIZERS[tokenizer_key]
            
            for context_size in context_sizes:
                config = RealTestConfig(
                    model_name=tokenizer["model_name"],
                    h_layers=model_params["h_layers"],
                    l_layers=model_params["l_layers"],
                    hidden_size=model_params["hidden_size"],
                    num_heads=model_params["num_heads"],
                    context_size=context_size,
                    batch_size=1,
                    use_moe=True,
                    num_experts=8,
                    num_experts_per_tok=2,
                    gradient_checkpointing=True,
                    use_amp=True,
                )
                configs.append(config)
    
    return configs


def generate_optimization_comparison_configs() -> list[RealTestConfig]:
    """Generate configs to compare different optimization strategies.
    
    Test matrix: 1 tokenizer × 3 contexts × 1 model × 16 optimization combos = 48 tests
    
    Optimization combinations tested:
    1. Baseline: No optimizations
    2. MoE only
    3. Gradient checkpointing only
    4. AMP only
    5. MoE + gradient checkpointing
    6. MoE + AMP
    7. Gradient checkpointing + AMP
    8. All optimizations (MoE + gradient checkpointing + AMP)
    
    Each combination tested on both:
    - Single GPU (cuda:1)
    - Parallel (will be handled by separate runner)
    
    Purpose: Understand VRAM impact of each optimization individually and combined.
    
    Returns:
        List of 48 test configurations
    """
    configs = []
    
    # Use medium model for optimization tests
    model_params = REAL_MODEL_SIZES["medium"]
    
    # Use gpt2 tokenizer (baseline)
    tokenizer = REAL_TOKENIZERS["gpt2"]
    
    # Test at 3 context sizes
    context_sizes = [512, 1024, 2048]
    
    # Define optimization combinations
    optimization_combos = [
        # (use_moe, gradient_checkpointing, use_amp, description)
        (False, False, False, "baseline_no_opts"),
        (True, False, False, "moe_only"),
        (False, True, False, "gradcheck_only"),
        (False, False, True, "amp_only"),
        (True, True, False, "moe_gradcheck"),
        (True, False, True, "moe_amp"),
        (False, True, True, "gradcheck_amp"),
        (True, True, True, "all_opts"),
    ]
    
    for context_size in context_sizes:
        for use_moe, grad_check, use_amp, desc in optimization_combos:
            config = RealTestConfig(
                model_name=tokenizer["model_name"],
                h_layers=model_params["h_layers"],
                l_layers=model_params["l_layers"],
                hidden_size=model_params["hidden_size"],
                num_heads=model_params["num_heads"],
                context_size=context_size,
                batch_size=1,
                use_moe=use_moe,
                num_experts=8 if use_moe else 8,  # Keep same for consistency
                num_experts_per_tok=2 if use_moe else 2,
                gradient_checkpointing=grad_check,
                use_amp=use_amp,
            )
            configs.append(config)
    
    return configs


def generate_parallel_gpu_configs() -> list[RealTestConfig]:
    """Generate configs for parallel GPU testing.
    
    Test matrix: 5 tokenizers × 3 contexts × 3 model sizes = 45 tests
    
    These configs will be run with parallel_independent=True to test
    multi-GPU training on both GPUs simultaneously.
    
    Purpose: Measure VRAM usage during parallel training and validate
    memory estimates for multi-GPU scenarios.
    
    Returns:
        List of 45 test configurations for parallel testing
    """
    configs = []
    
    # Use small, medium, large models for parallel tests
    model_size_keys = ["small", "medium", "large"]
    
    # All tokenizers
    tokenizer_keys = list(REAL_TOKENIZERS.keys())
    
    # Test at 3 context sizes (keep conservative for parallel)
    context_sizes = [512, 1024, 2048]
    
    for model_size_key in model_size_keys:
        model_params = REAL_MODEL_SIZES[model_size_key]
        
        for tokenizer_key in tokenizer_keys:
            tokenizer = REAL_TOKENIZERS[tokenizer_key]
            
            for context_size in context_sizes:
                config = RealTestConfig(
                    model_name=tokenizer["model_name"],
                    h_layers=model_params["h_layers"],
                    l_layers=model_params["l_layers"],
                    hidden_size=model_params["hidden_size"],
                    num_heads=model_params["num_heads"],
                    context_size=context_size,
                    batch_size=1,
                    use_moe=True,  # Default enabled
                    num_experts=8,
                    num_experts_per_tok=2,
                    gradient_checkpointing=True,
                    use_amp=True,
                    device="cuda:0,1",  # Mark for parallel testing
                )
                configs.append(config)
    
    return configs


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_test_duration(num_tests: int, avg_seconds_per_test: float = 30.0) -> str:
    """Estimate total test duration.
    
    Args:
        num_tests: Number of tests to run
        avg_seconds_per_test: Average seconds per test (default: 30s)
        
    Returns:
        Human-readable duration estimate
    """
    total_seconds = num_tests * avg_seconds_per_test
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    if hours > 0:
        return f"~{hours}h {minutes}m"
    else:
        return f"~{minutes}m"


def print_test_summary(configs: list[RealTestConfig], suite_name: str):
    """Print summary of test configurations.
    
    Args:
        configs: List of test configs
        suite_name: Name of the test suite
    """
    print(f"\n{'='*80}")
    print(f"{suite_name} Test Suite Summary")
    print(f"{'='*80}")
    print(f"Total tests: {len(configs)}")
    print(f"Estimated duration: {estimate_test_duration(len(configs))}")
    print()
    
    # Count by model size
    model_counts = {}
    for config in configs:
        key = f"{config.h_layers}h/{config.l_layers}l/{config.hidden_size}d"
        model_counts[key] = model_counts.get(key, 0) + 1
    
    print("Model sizes:")
    for model_key, count in sorted(model_counts.items()):
        print(f"  {model_key}: {count} tests")
    
    # Count by context size
    context_counts = {}
    for config in configs:
        context_counts[config.context_size] = context_counts.get(config.context_size, 0) + 1
    
    print("\nContext sizes:")
    for context, count in sorted(context_counts.items()):
        print(f"  {context} tokens: {count} tests")
    
    print(f"{'='*80}\n")


# ============================================================================
# Main (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    print("Real Baseline Test Configuration Generator")
    print()
    
    # Generate and display all test suites
    quick_configs = generate_quick_baseline_configs()
    print_test_summary(quick_configs, "Quick Baseline")
    
    standard_configs = generate_standard_baseline_configs()
    print_test_summary(standard_configs, "Standard Baseline")
    
    full_configs = generate_full_baseline_configs()
    print_test_summary(full_configs, "Full Baseline")
    
    opt_configs = generate_optimization_comparison_configs()
    print_test_summary(opt_configs, "Optimization Comparison")
    
    parallel_configs = generate_parallel_gpu_configs()
    print_test_summary(parallel_configs, "Parallel GPU")
    
    print("\nComplete test plan:")
    print(f"  Quick: {len(quick_configs)} tests (~{len(quick_configs) * 0.5:.0f}m)")
    print(f"  Standard: {len(standard_configs)} tests (~{len(standard_configs) * 0.5:.0f}m)")
    print(f"  Full: {len(full_configs)} tests (~{len(full_configs) * 0.5:.0f}m)")
    print(f"  Optimization: {len(opt_configs)} tests (~{len(opt_configs) * 0.5:.0f}m)")
    print(f"  Parallel: {len(parallel_configs)} tests (~{len(parallel_configs) * 1:.0f}m)")
    print(f"  TOTAL: {len(quick_configs) + len(standard_configs) + len(full_configs) + len(opt_configs) + len(parallel_configs)} tests")
    
    print("\nNext steps:")
    print("1. Run quick baseline: python run_real_tests.py --suite quick")
    print("2. Run optimization tests: python run_real_tests.py --suite optimization")
    print("3. Run standard baseline: python run_real_tests.py --suite standard")
    print("4. Run full baseline: python run_real_tests.py --suite full")

    print("5. Run parallel tests: python run_real_tests.py --suite parallel")
>'
    rint("6. Analyze ALL data and update VRAM/RAM estimators for 95%+ accuracy")
  4/  
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+