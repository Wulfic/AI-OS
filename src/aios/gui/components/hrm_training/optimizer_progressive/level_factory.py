"""Factory for creating optimization levels to test."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .models import OptimizationLevel, OptimizationConfig


def create_optimization_levels(
    config: "OptimizationConfig",
    log_func=None
) -> List["OptimizationLevel"]:
    """Create the progression of optimization levels to test.
    
    ENHANCED: Now tests various LoRA configurations based on available VRAM.
    Selection rules align with the canonical LoRA/PEFT guide:
    docs/guide/features/LORA_PEFT_COMPREHENSIVE_ANALYSIS.md (see Parameter Impact
    and Configuration Presets) for choosing appropriate LoRA ranks, modules, and methods.
    
    IMPORTANT: For long sequences (>2048), start with AMP enabled by default!
    FP32 attention memory scales O(n²) and becomes prohibitive.
    
    For long sequences (>8192), also test multiple chunk sizes to find optimal balance.
    
    Args:
        config: Optimization configuration
        log_func: Optional logging function
        
    Returns:
        List of OptimizationLevel objects to test
    """
    from .models import OptimizationLevel, LoRAConfig
    from .lora_factory import create_lora_configs, estimate_available_vram_gb, get_lora_config_description
    
    def log(msg: str):
        if log_func:
            log_func(msg)
    
    # Estimate available VRAM for LoRA configuration selection
    available_vram_gb = estimate_available_vram_gb()
    log(f"[opt] Estimated available VRAM: {available_vram_gb:.1f} GB")
    
    # Get LoRA configurations to test based on VRAM
    lora_configs = create_lora_configs(
        available_vram_gb=available_vram_gb,
        dataset_size=None,  # Could be enhanced to parse dataset file
        task_complexity="medium"  # Could be a config parameter
    )
    
    log(f"[opt] Will test {len(lora_configs)} LoRA configurations based on VRAM:")
    for lora_cfg in lora_configs:
        desc = get_lora_config_description(lora_cfg)
        log(f"[opt]   {desc.split(chr(10))[0]}")  # First line only
    
    # For long contexts, skip FP32-only testing (it will OOM)
    force_amp = config.max_seq_len > 2048
    needs_chunking = config.max_seq_len > 8192
    
    # Chunk sizes to test for long sequences (largest to smallest for best performance)
    # Standard sizes: 4096, 2048, 1024
    # Exhaustive sizes (final resort): 512, 256, 128
    chunk_sizes = [4096, 2048, 1024] if needs_chunking else [None]
    
    if force_amp:
        log(f"[opt] Long sequence detected ({config.max_seq_len} tokens)")
        log(f"[opt] Auto-enabling AMP for all tests (FP32 would require ~{(config.max_seq_len**2 * 4) / 1e9:.1f} GB just for attention!)")
    
    if needs_chunking:
        log(f"[opt] Very long sequence - will test chunk sizes: {chunk_sizes}")
    
    levels = []
    
    # ========================================================================
    # LEVEL GENERATION STRATEGY (ENHANCED WITH LORA CONFIGS)
    # ========================================================================
    # 1. Test baseline (no PEFT) with minimal optimizations
    # 2. Test each LoRA configuration with GradCP + AMP
    # 3. If needed, test aggressive optimizations (CPU offload, ZeRO stages)
    # ========================================================================
    
    # Level 1: Baseline without PEFT (only if not forcing AMP)
    if not force_amp:
        for chunk_size in chunk_sizes:
            level_num = len(levels) + 1
            name = "Baseline (No PEFT)"
            if chunk_size is not None:
                name = f"{name} (chunk={chunk_size})"
            
            levels.append(OptimizationLevel(
                name=f"Level {level_num}: {name}",
                gradient_checkpointing=True,
                amp=False,
                flashattn2=False,
                lora_config=LoRAConfig(enabled=False),
                cpu_offload=False,
                zero_stage="none",
                chunk_size=chunk_size
            ))
    
    # Level 2-N: Test each LoRA configuration with GradCP + AMP
    for lora_cfg in lora_configs:
        if not lora_cfg.enabled and not force_amp:
            # Skip disabled config if we already tested it as baseline
            continue
        
        for chunk_size in chunk_sizes:
            level_num = len(levels) + 1
            if lora_cfg.enabled:
                name = f"GradCP + AMP + {lora_cfg}"
            else:
                name = "GradCP + AMP (No PEFT)"
            
            if chunk_size is not None:
                name = f"{name} (chunk={chunk_size})"
            
            levels.append(OptimizationLevel(
                name=f"Level {level_num}: {name}",
                gradient_checkpointing=True,
                amp=True,
                flashattn2=False,
                lora_config=lora_cfg,
                cpu_offload=False,
                zero_stage="none",
                chunk_size=chunk_size
            ))
    
    # Level N+1 onwards: Aggressive optimizations (only if LoRA enabled)
    # Test with the most promising LoRA config (usually r=16 balanced or first enabled config)
    best_lora_for_aggressive = next((cfg for cfg in lora_configs if cfg.enabled), None)
    
    if best_lora_for_aggressive:
        # Add CPU offload level
        for chunk_size in chunk_sizes:
            level_num = len(levels) + 1
            name = f"GradCP + AMP + {best_lora_for_aggressive} + CPUOff"
            if chunk_size is not None:
                name = f"{name} (chunk={chunk_size})"
            
            levels.append(OptimizationLevel(
                name=f"Level {level_num}: {name}",
                gradient_checkpointing=True,
                amp=True,
                flashattn2=False,
                lora_config=best_lora_for_aggressive,
                cpu_offload=True,
                zero_stage="none",
                chunk_size=chunk_size
            ))
        
        # Add ZeRO stages for extreme cases
        for zero_stage in ["zero1", "zero2", "zero3"]:
            for chunk_size in chunk_sizes:
                level_num = len(levels) + 1
                name = f"Full + {zero_stage.upper()}"
                if chunk_size is not None:
                    name = f"{name} (chunk={chunk_size})"
                
                levels.append(OptimizationLevel(
                    name=f"Level {level_num}: {name}",
                    gradient_checkpointing=True,
                    amp=True,
                    flashattn2=False,
                    lora_config=best_lora_for_aggressive,
                    cpu_offload=True,
                    zero_stage=zero_stage,
                    chunk_size=chunk_size
                ))
    
    log(f"[opt] Generated {len(levels)} optimization levels to test")
    
    return levels


def create_exhaustive_levels(
    previous_levels: List["OptimizationLevel"],
    config: "OptimizationConfig"
) -> List["OptimizationLevel"]:
    """Create exhaustive chunk size levels for final resort testing.
    
    Args:
        previous_levels: Previously tested optimization levels
        config: Optimization configuration
        
    Returns:
        List of OptimizationLevel objects with tiny chunk sizes
    """
    from .models import OptimizationLevel, LoRAConfig
    from .lora_factory import estimate_available_vram_gb, create_lora_configs
    
    exhaustive_chunk_sizes = [512, 256, 128]
    levels = []
    
    # Use the most aggressive optimization (Full + ZeRO-3 with best LoRA)
    # Get a LoRA config to use
    available_vram_gb = estimate_available_vram_gb()
    lora_configs = create_lora_configs(available_vram_gb=available_vram_gb)
    best_lora = next((cfg for cfg in lora_configs if cfg.enabled), LoRAConfig(enabled=False))
    
    for chunk_size in exhaustive_chunk_sizes:
        level_num = 100 + len(levels)  # Start from 100 to distinguish exhaustive
        name = f"Full + ZeRO-3"
        if best_lora.enabled:
            name += f" + {best_lora}"
        name += f" (chunk={chunk_size}) [EXHAUSTIVE]"
        
        level = OptimizationLevel(
            name=f"Level {level_num}: {name}",
            gradient_checkpointing=True,
            amp=True,
            flashattn2=False,
            lora_config=best_lora,
            cpu_offload=True,
            zero_stage="zero3",
            chunk_size=chunk_size
        )
        levels.append(level)
    
    return levels
