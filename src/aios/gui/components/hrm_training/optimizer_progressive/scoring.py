"""Scoring system for optimization results.

Scores optimization results based on:
- Throughput (steps/sec)
- Memory efficiency
- Expected quality (see canonical LoRA/PEFT guide:
    docs/guide/features/LORA_PEFT_COMPREHENSIVE_ANALYSIS.md)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import OptimizationLevel


def score_optimization_result(
    level: OptimizationLevel,
    batch_size: int,
    throughput: float,
    memory_percent: float,
    prioritize_quality: bool = True
) -> float:
    """
    Score an optimization result considering multiple factors.
    
    Uses quality estimates consistent with the canonical LoRA/PEFT guide to
    factor in expected model quality when selecting the best configuration.
    
    Args:
        level: Optimization level tested
        batch_size: Batch size used
        throughput: Training throughput in steps/sec
        memory_percent: GPU memory utilization percentage
        prioritize_quality: If True, weight quality more heavily
        
    Returns:
        Score (higher is better, typically 0-200 range)
    """
    
    # Component 1: Throughput score (0-100 range typically)
    # Higher throughput = better training speed
    throughput_score = throughput * 100
    
    # Component 2: Memory efficiency bonus (0-50 range)
    # Sweet spot is 85-95% utilization
    if 85 <= memory_percent <= 95:
        memory_bonus = 50  # Perfect utilization
    elif memory_percent < 85:
        # Penalty for underutilization (wasting GPU)
        memory_bonus = (memory_percent / 85) * 50
    else:
        # Penalty for over-utilization (risk of OOM)
        memory_bonus = max(0, 50 - (memory_percent - 95) * 5)
    
    # Component 3: Quality score (0-100 range)
    # Get expected quality from LoRA configuration
    if level.lora_config:
        quality_score = level.lora_config.expected_quality_percent
    else:
        quality_score = 100.0  # Full fine-tuning baseline
    
    # Component 4: Batch size bonus (0-20 range)
    # Larger batch sizes are generally more stable
    batch_bonus = min(20, batch_size * 2.5)
    
    # Combine scores with weights
    if prioritize_quality:
        # Weight: 30% throughput, 15% memory, 50% quality, 5% batch
        total_score = (
            throughput_score * 0.30 +
            memory_bonus * 0.15 +
            quality_score * 0.50 +
            batch_bonus * 0.05
        )
    else:
        # Weight: 50% throughput, 25% memory, 20% quality, 5% batch
        total_score = (
            throughput_score * 0.50 +
            memory_bonus * 0.25 +
            quality_score * 0.20 +
            batch_bonus * 0.05
        )
    
    return total_score


def get_quality_tier(level: OptimizationLevel) -> str:
    """Get quality tier description for a level.
    
    Args:
        level: Optimization level
        
    Returns:
        Quality tier string (e.g., "Excellent", "High", "Good")
    """
    if not level.lora_config or not level.lora_config.enabled:
        return "Maximum (Full FT)"
    
    quality = level.lora_config.expected_quality_percent
    
    if quality >= 99:
        return "Excellent (99%+)"
    elif quality >= 97:
        return "High (97-99%)"
    elif quality >= 95:
        return "Good (95-97%)"
    elif quality >= 90:
        return "Fair (90-95%)"
    else:
        return "Acceptable (85-90%)"


def format_result_summary(
    level: OptimizationLevel,
    batch_size: int,
    throughput: float,
    memory_percent: float,
    score: float
) -> str:
    """Format a result summary for display.
    
    Args:
        level: Optimization level
        batch_size: Batch size
        throughput: Throughput in steps/sec
        memory_percent: Memory utilization percentage
        score: Overall score
        
    Returns:
        Formatted summary string
    """
    quality_tier = get_quality_tier(level)
    
    lines = [
        f"Configuration: {level}",
        f"  Batch size: {batch_size}",
        f"  Throughput: {throughput:.2f} steps/sec",
        f"  Memory: {memory_percent:.1f}%",
        f"  Quality: {quality_tier}",
        f"  Score: {score:.1f}",
    ]
    
    if level.lora_config and level.lora_config.enabled:
        lines.append(f"  Trainable params: ~{level.lora_config.estimated_params / 1_000_000:.1f}M")
        lines.append(f"  VRAM overhead: +{level.lora_config.estimated_vram_overhead_gb:.1f} GB")
    
    return "\n".join(lines)
