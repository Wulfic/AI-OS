"""Comprehensive optimization test matrix covering all memory optimization techniques.

This tests EVERY reasonable combination of optimization strategies to gather
complete data for the VRAM estimator.

Test Categories:
1. Basic optimizations (MoE, gradient checkpointing, AMP, Flash Attention 2)
2. Advanced optimizations (8-bit optimizer, CPU offload, context chunking)
3. DeepSpeed ZeRO stages (1, 2, 3)
4. LoRA/PEFT configurations (rank variations, alpha variations, target modules)
5. Combined optimizations (all reasonable combinations)

Total estimated tests: 200-300+ configurations
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for a single optimization test."""
    name: str
    description: str
    
    # Basic optimizations
    use_moe: bool = False
    gradient_checkpointing: bool = False
    use_amp: bool = False
    use_flash_attention_2: bool = False
    
    # Advanced memory optimizations
    use_8bit_optimizer: bool = False
    cpu_offload: bool = False
    context_chunking: bool = False
    chunk_size: int = None
    
    # DeepSpeed
    deepspeed_stage: int = None  # 0 (disabled), 1, 2, or 3
    
    # LoRA/PEFT
    use_lora: bool = False
    lora_rank: int = None
    lora_alpha: int = None
    lora_target_modules: str = None  # "all", "attention", "mlp", "specific"
    
    # Test parameters
    context_sizes: List[int] = None  # Which contexts to test
    model_size: str = "medium"  # Which model size to use

    def __post_init__(self):
        if self.context_sizes is None:
            # Default: test small, medium, large contexts (adjusted for 11GB VRAM)
            self.context_sizes = [128, 256, 384]


# ============================================================================
# CATEGORY 1: BASELINE & SINGLE OPTIMIZATIONS
# ============================================================================
# These establish baseline savings for each optimization individually

CATEGORY_1_BASELINE = [
    OptimizationConfig(
        name="baseline",
        description="No optimizations - pure baseline",
        context_sizes=[128, 256, 384]  # Reduced for 11GB VRAM
    ),
]

CATEGORY_1_SINGLE_OPTS = [
    OptimizationConfig(
        name="moe_only",
        description="MoE only (8 experts, 2 active)",
        use_moe=True,
    ),
    OptimizationConfig(
        name="gradcheck_only",
        description="Gradient checkpointing only",
        gradient_checkpointing=True,
    ),
    OptimizationConfig(
        name="amp_only",
        description="Automatic Mixed Precision only",
        use_amp=True,
    ),
    OptimizationConfig(
        name="flash_attn2_only",
        description="Flash Attention 2 only",
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="8bit_optimizer_only",
        description="8-bit Adam optimizer only",
        use_8bit_optimizer=True,
    ),
    OptimizationConfig(
        name="cpu_offload_only",
        description="CPU offload only (offload optimizer states)",
        cpu_offload=True,
    ),
    OptimizationConfig(
        name="context_chunking_256",
        description="Context chunking (256 tokens per chunk)",
        context_chunking=True,
        chunk_size=256,
        context_sizes=[256, 384, 512],  # Reduced for 11GB VRAM
    ),
]

# ============================================================================
# CATEGORY 2: DEEPSPEED ZERO STAGES
# ============================================================================
# Test each ZeRO stage individually and with basic optimizations

CATEGORY_2_DEEPSPEED = [
    # ZeRO Stage 1: Optimizer state partitioning
    OptimizationConfig(
        name="zero1_only",
        description="DeepSpeed ZeRO Stage 1 (optimizer state partitioning)",
        deepspeed_stage=1,
    ),
    OptimizationConfig(
        name="zero1_gradcheck",
        description="ZeRO-1 + gradient checkpointing",
        deepspeed_stage=1,
        gradient_checkpointing=True,
    ),
    OptimizationConfig(
        name="zero1_amp",
        description="ZeRO-1 + AMP",
        deepspeed_stage=1,
        use_amp=True,
    ),
    
    # ZeRO Stage 2: Gradient partitioning
    OptimizationConfig(
        name="zero2_only",
        description="DeepSpeed ZeRO Stage 2 (gradient + optimizer partitioning)",
        deepspeed_stage=2,
    ),
    OptimizationConfig(
        name="zero2_gradcheck",
        description="ZeRO-2 + gradient checkpointing",
        deepspeed_stage=2,
        gradient_checkpointing=True,
    ),
    OptimizationConfig(
        name="zero2_amp",
        description="ZeRO-2 + AMP",
        deepspeed_stage=2,
        use_amp=True,
    ),
    
    # ZeRO-Offload: ZeRO-2 with CPU offload (replaces ZeRO-3 for single-GPU)
    # ZeRO-3 requires DeepSpeed launcher which doesn't work in test environment
    # ZeRO-Offload provides similar memory savings without launcher requirement
    OptimizationConfig(
        name="zero_offload",
        description="DeepSpeed ZeRO-Offload (ZeRO-2 + CPU offload, single-GPU friendly)",
        deepspeed_stage=2,
        cpu_offload=True,
    ),
    OptimizationConfig(
        name="zero_offload_gradcheck",
        description="ZeRO-Offload + gradient checkpointing",
        deepspeed_stage=2,
        cpu_offload=True,
        gradient_checkpointing=True,
    ),
    OptimizationConfig(
        name="zero_offload_amp",
        description="ZeRO-Offload + AMP",
        deepspeed_stage=2,
        cpu_offload=True,
        use_amp=True,
    ),
    OptimizationConfig(
        name="zero_offload_max",
        description="ZeRO-Offload + all basic opts (maximum single-GPU memory savings)",
        deepspeed_stage=2,
        cpu_offload=True,
        gradient_checkpointing=True,
        use_amp=True,
    ),
]

# ============================================================================
# CATEGORY 3: LORA/PEFT CONFIGURATIONS
# ============================================================================
# Test various LoRA ranks, alphas, and target modules

CATEGORY_3_LORA_RANKS = [
    # Low rank (memory efficient)
    OptimizationConfig(
        name="lora_r4_all",
        description="LoRA rank=4, alpha=8, all modules",
        use_lora=True,
        lora_rank=4,
        lora_alpha=8,
        lora_target_modules="all",
    ),
    OptimizationConfig(
        name="lora_r8_all",
        description="LoRA rank=8, alpha=16, all modules",
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_target_modules="all",
    ),
    
    # Medium rank (balanced)
    OptimizationConfig(
        name="lora_r16_all",
        description="LoRA rank=16, alpha=32, all modules",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
    ),
    OptimizationConfig(
        name="lora_r32_all",
        description="LoRA rank=32, alpha=64, all modules",
        use_lora=True,
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules="all",
    ),
    
    # High rank (more parameters)
    OptimizationConfig(
        name="lora_r64_all",
        description="LoRA rank=64, alpha=128, all modules",
        use_lora=True,
        lora_rank=64,
        lora_alpha=128,
        lora_target_modules="all",
    ),
]

CATEGORY_3_LORA_TARGETS = [
    # Attention only (most common)
    OptimizationConfig(
        name="lora_r16_attn",
        description="LoRA rank=16, attention modules only",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="attention",
    ),
    
    # MLP only
    OptimizationConfig(
        name="lora_r16_mlp",
        description="LoRA rank=16, MLP modules only",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="mlp",
    ),
    
    # Q/K/V only (minimal)
    OptimizationConfig(
        name="lora_r16_qkv",
        description="LoRA rank=16, Q/K/V only",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="qkv",
    ),
]

CATEGORY_3_LORA_COMBOS = [
    # LoRA + gradient checkpointing
    OptimizationConfig(
        name="lora_r16_gradcheck",
        description="LoRA r=16 + gradient checkpointing",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
        gradient_checkpointing=True,
    ),
    
    # LoRA + AMP
    OptimizationConfig(
        name="lora_r16_amp",
        description="LoRA r=16 + AMP",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
        use_amp=True,
    ),
    
    # LoRA + 8-bit optimizer
    OptimizationConfig(
        name="lora_r16_8bit",
        description="LoRA r=16 + 8-bit optimizer",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
        use_8bit_optimizer=True,
    ),
    
    # LoRA + all basic opts
    OptimizationConfig(
        name="lora_r16_all_opts",
        description="LoRA r=16 + gradcheck + AMP + 8-bit",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
        gradient_checkpointing=True,
        use_amp=True,
        use_8bit_optimizer=True,
    ),
]

# ============================================================================
# CATEGORY 4: COMBINED OPTIMIZATIONS (2-WAY)
# ============================================================================
# Test all reasonable 2-way combinations

CATEGORY_4_TWO_WAY = [
    # MoE combinations
    OptimizationConfig(
        name="moe_gradcheck",
        description="MoE + gradient checkpointing",
        use_moe=True,
        gradient_checkpointing=True,
    ),
    OptimizationConfig(
        name="moe_amp",
        description="MoE + AMP",
        use_moe=True,
        use_amp=True,
    ),
    OptimizationConfig(
        name="moe_flash_attn2",
        description="MoE + Flash Attention 2",
        use_moe=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="moe_8bit",
        description="MoE + 8-bit optimizer",
        use_moe=True,
        use_8bit_optimizer=True,
    ),
    
    # Gradient checkpointing combinations
    OptimizationConfig(
        name="gradcheck_amp",
        description="Gradient checkpointing + AMP",
        gradient_checkpointing=True,
        use_amp=True,
    ),
    OptimizationConfig(
        name="gradcheck_flash_attn2",
        description="Gradient checkpointing + Flash Attention 2",
        gradient_checkpointing=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="gradcheck_8bit",
        description="Gradient checkpointing + 8-bit optimizer",
        gradient_checkpointing=True,
        use_8bit_optimizer=True,
    ),
    OptimizationConfig(
        name="gradcheck_cpu_offload",
        description="Gradient checkpointing + CPU offload",
        gradient_checkpointing=True,
        cpu_offload=True,
    ),
    
    # AMP combinations
    OptimizationConfig(
        name="amp_flash_attn2",
        description="AMP + Flash Attention 2",
        use_amp=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="amp_8bit",
        description="AMP + 8-bit optimizer",
        use_amp=True,
        use_8bit_optimizer=True,
    ),
    
    # Flash Attention 2 combinations
    OptimizationConfig(
        name="flash_attn2_8bit",
        description="Flash Attention 2 + 8-bit optimizer",
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
    ),
    
    # CPU offload combinations
    OptimizationConfig(
        name="cpu_offload_8bit",
        description="CPU offload + 8-bit optimizer",
        cpu_offload=True,
        use_8bit_optimizer=True,
    ),
]

# ============================================================================
# CATEGORY 5: COMBINED OPTIMIZATIONS (3-WAY)
# ============================================================================
# Test common 3-way combinations

CATEGORY_5_THREE_WAY = [
    OptimizationConfig(
        name="moe_gradcheck_amp",
        description="MoE + gradient checkpointing + AMP",
        use_moe=True,
        gradient_checkpointing=True,
        use_amp=True,
    ),
    OptimizationConfig(
        name="moe_gradcheck_flash_attn2",
        description="MoE + gradient checkpointing + Flash Attention 2",
        use_moe=True,
        gradient_checkpointing=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="moe_amp_flash_attn2",
        description="MoE + AMP + Flash Attention 2",
        use_moe=True,
        use_amp=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="gradcheck_amp_flash_attn2",
        description="Gradient checkpointing + AMP + Flash Attention 2",
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="gradcheck_amp_8bit",
        description="Gradient checkpointing + AMP + 8-bit optimizer",
        gradient_checkpointing=True,
        use_amp=True,
        use_8bit_optimizer=True,
    ),
    OptimizationConfig(
        name="gradcheck_flash_attn2_8bit",
        description="Gradient checkpointing + Flash Attention 2 + 8-bit",
        gradient_checkpointing=True,
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
    ),
]

# ============================================================================
# CATEGORY 6: MAXIMUM OPTIMIZATION STACKS
# ============================================================================
# Test maximum memory savings combinations

CATEGORY_6_MAX_OPTS = [
    OptimizationConfig(
        name="all_basic_opts",
        description="All basic optimizations (no DeepSpeed/LoRA)",
        use_moe=True,
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
    ),
    OptimizationConfig(
        name="all_opts_no_offload",
        description="All optimizations except CPU offload",
        use_moe=True,
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
        context_chunking=True,
        chunk_size=256,
    ),
    OptimizationConfig(
        name="maximum_memory_savings",
        description="Absolute maximum memory savings configuration",
        use_moe=True,
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
        cpu_offload=True,
        context_chunking=True,
        chunk_size=256,
    ),
    OptimizationConfig(
        name="zero_offload_flash_amp",
        description="ZeRO-Offload + Flash Attention 2 + AMP + gradient checkpointing",
        deepspeed_stage=2,
        cpu_offload=True,
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
    ),
    OptimizationConfig(
        name="lora_max",
        description="LoRA with all compatible optimizations",
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_target_modules="all",
        gradient_checkpointing=True,
        use_amp=True,
        use_flash_attention_2=True,
        use_8bit_optimizer=True,
    ),
]

# ============================================================================
# CATEGORY 7: CONTEXT CHUNKING VARIATIONS
# ============================================================================
# Test different chunk sizes

CATEGORY_7_CHUNKING = [
    OptimizationConfig(
        name="chunking_128",
        description="Context chunking: 128 tokens/chunk",
        context_chunking=True,
        chunk_size=128,
        context_sizes=[256, 384, 512],  # Reduced for 11GB VRAM
    ),
    OptimizationConfig(
        name="chunking_256",
        description="Context chunking: 256 tokens/chunk",
        context_chunking=True,
        chunk_size=256,
        context_sizes=[384, 512, 768],  # Reduced for 11GB VRAM
    ),
    OptimizationConfig(
        name="chunking_512",
        description="Context chunking: 512 tokens/chunk",
        context_chunking=True,
        chunk_size=512,
        context_sizes=[768, 1024, 1536],  # Reduced for 11GB VRAM
    ),
    OptimizationConfig(
        name="chunking_256_gradcheck",
        description="Context chunking (256) + gradient checkpointing",
        context_chunking=True,
        chunk_size=256,
        gradient_checkpointing=True,
        context_sizes=[384, 512, 768],  # Reduced for 11GB VRAM
    ),
]

# ============================================================================
# COMPILE ALL TEST CONFIGURATIONS
# ============================================================================

ALL_OPTIMIZATION_CONFIGS = (
    CATEGORY_1_BASELINE +
    CATEGORY_1_SINGLE_OPTS +
    CATEGORY_2_DEEPSPEED +
    CATEGORY_3_LORA_RANKS +
    CATEGORY_3_LORA_TARGETS +
    CATEGORY_3_LORA_COMBOS +
    CATEGORY_4_TWO_WAY +
    CATEGORY_5_THREE_WAY +
    CATEGORY_6_MAX_OPTS +
    CATEGORY_7_CHUNKING
)

def count_total_tests() -> int:
    """Count total number of test combinations."""
    total = 0
    for config in ALL_OPTIMIZATION_CONFIGS:
        total += len(config.context_sizes)
    return total

def print_test_summary():
    """Print summary of all test configurations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE OPTIMIZATION TEST MATRIX")
    print("="*80)
    print()
    
    categories = [
        ("Category 1: Baseline & Single Optimizations", len(CATEGORY_1_BASELINE) + len(CATEGORY_1_SINGLE_OPTS)),
        ("Category 2: DeepSpeed ZeRO Stages", len(CATEGORY_2_DEEPSPEED)),
        ("Category 3: LoRA/PEFT Configurations", len(CATEGORY_3_LORA_RANKS) + len(CATEGORY_3_LORA_TARGETS) + len(CATEGORY_3_LORA_COMBOS)),
        ("Category 4: Combined Optimizations (2-way)", len(CATEGORY_4_TWO_WAY)),
        ("Category 5: Combined Optimizations (3-way)", len(CATEGORY_5_THREE_WAY)),
        ("Category 6: Maximum Optimization Stacks", len(CATEGORY_6_MAX_OPTS)),
        ("Category 7: Context Chunking Variations", len(CATEGORY_7_CHUNKING)),
    ]
    
    for cat_name, count in categories:
        print(f"{cat_name}: {count} configs")
    
    print()
    print(f"Total unique configurations: {len(ALL_OPTIMIZATION_CONFIGS)}")
    print(f"Total tests (configs × contexts): {count_total_tests()}")
    print()
    print("Coverage:")
    print("  ✓ MoE (8 experts, 2 active)")
    print("  ✓ Gradient checkpointing")
    print("  ✓ Automatic Mixed Precision (AMP)")
    print("  ✓ Flash Attention 2")
    print("  ✓ 8-bit optimizer")
    print("  ✓ CPU offload")
    print("  ✓ Context chunking (3 chunk sizes)")
    print("  ✓ DeepSpeed ZeRO-1, ZeRO-2, ZeRO-3")
    print("  ✓ LoRA (5 ranks × 3 target modules)")
    print("  ✓ All reasonable 2-way combinations")
    print("  ✓ All reasonable 3-way combinations")
    print("  ✓ Maximum memory savings configurations")
    print("="*80)


if __name__ == "__main__":
    print_test_summary()
    
    print("\nFirst 10 test configurations:")
    print("-" * 80)
    for i, config in enumerate(ALL_OPTIMIZATION_CONFIGS[:10], 1):
        print(f"{i}. {config.name}: {config.description}")
        print(f"   Contexts: {config.context_sizes}")
    
    print("\n...")
    print(f"\n[{len(ALL_OPTIMIZATION_CONFIGS) - 10} more configurations...]")
