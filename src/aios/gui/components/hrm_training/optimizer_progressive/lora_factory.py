"""Factory for creating LoRA configurations based on available resources.

References:
- Canonical LoRA/PEFT guide: docs/guide/features/LORA_PEFT_COMPREHENSIVE_ANALYSIS.md
    (see Parameter Impact, Target Modules Explained, and Configuration Presets)
"""

from __future__ import annotations

from typing import List, Optional
import torch

from .models import LoRAConfig


def estimate_available_vram_gb() -> float:
    """Estimate available VRAM across all GPUs.
    
    Returns:
        Available VRAM in GB
    """
    if not torch.cuda.is_available():
        return 0.0
    
    total_available = 0.0
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / (1024 ** 3)
        
        # Conservatively estimate available (assume 85% usable)
        available_gb = total_gb * 0.85
        total_available += available_gb
    
    return total_available


def create_lora_configs(
    available_vram_gb: Optional[float] = None,
    dataset_size: Optional[int] = None,
    task_complexity: str = "medium"
) -> List[LoRAConfig]:
    """
    Create LoRA configurations to test based on available resources.
    
    Uses decision rules summarized in the canonical LoRA/PEFT guide
    (docs/guide/features/LORA_PEFT_COMPREHENSIVE_ANALYSIS.md):
    - < 8 GB: IA3 or LoRA r=8 minimal
    - 8-10 GB: LoRA r=16 minimal
    - 10-14 GB: LoRA r=16 balanced
    - 14-20 GB: LoRA r=32 balanced/full
    - 20+ GB: LoRA r=64 full or full fine-tuning
    
    Args:
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        dataset_size: Number of samples in dataset (for dropout tuning)
        task_complexity: "simple", "medium", "complex", "very_complex"
    
    Returns:
        List of LoRAConfig objects to test
    """
    
    # Auto-detect VRAM if not provided
    if available_vram_gb is None:
        available_vram_gb = estimate_available_vram_gb()
    
    # Determine dropout based on dataset size
    if dataset_size:
        if dataset_size > 100_000:
            dropout = 0.0
        elif dataset_size > 10_000:
            dropout = 0.05
        elif dataset_size > 1_000:
            dropout = 0.1
        else:
            dropout = 0.2
    else:
        dropout = 0.05  # Default from matrix
    
    configs = []
    
    # Always test NO PEFT as baseline for comparison
    configs.append(LoRAConfig(enabled=False))
    
    # ====================================================================
    # VRAM-Based Configuration Selection
    # From the canonical LoRA/PEFT guide's quick decision rules
    # ====================================================================
    
    if available_vram_gb < 8:
        # < 8 GB: IA3 Minimal or LoRA r=8 Minimal
        # Expected: 85-95% quality
        configs.extend([
            LoRAConfig(
                enabled=True,
                method="ia3",
                target_modules="q_proj,v_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=8,
                alpha=16,
                dropout=dropout,
                target_modules="q_proj,v_proj"
            )
        ])
        
    elif available_vram_gb < 10:
        # 8-10 GB: LoRA r=8 Minimal and r=16 Minimal
        # Expected: 90-98% quality
        configs.extend([
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=8,
                alpha=16,
                dropout=dropout,
                target_modules="q_proj,v_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=16,
                alpha=32,
                dropout=dropout,
                target_modules="q_proj,v_proj"
            )
        ])
        
    elif available_vram_gb < 14:
        # 10-14 GB: LoRA r=16 Minimal and Balanced
        # Expected: 95-99% quality
        configs.extend([
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=16,
                alpha=32,
                dropout=dropout,
                target_modules="q_proj,v_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=16,
                alpha=32,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj"
            )
        ])
        
    elif available_vram_gb < 20:
        # 14-20 GB: LoRA r=16 Balanced, r=32 Balanced, and r=32 Full
        # Expected: 97-99.5% quality
        configs.extend([
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=16,
                alpha=32,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=32,
                alpha=64,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=32,
                alpha=64,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
            )
        ])
        
    else:
        # 20+ GB: LoRA r=32 Full, r=64 Full, and AdaLoRA for research
        # Expected: 99%+ quality
        configs.extend([
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=32,
                alpha=64,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
            ),
            LoRAConfig(
                enabled=True,
                method="lora",
                rank=64,
                alpha=128,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
            )
        ])
        
        # For complex tasks, test AdaLoRA for slight quality boost
        if task_complexity in ["complex", "very_complex"]:
            configs.append(LoRAConfig(
                enabled=True,
                method="adalora",
                rank=32,
                alpha=64,
                dropout=dropout,
                target_modules="q_proj,k_proj,v_proj,o_proj"
            ))
    
    return configs


def get_lora_config_description(config: LoRAConfig) -> str:
    """Get a detailed description of a LoRA configuration.
    
    Args:
        config: LoRA configuration
        
    Returns:
        Multi-line description string
    """
    if not config.enabled:
        return "No PEFT (Full Fine-Tuning)\n  - All parameters trainable\n  - Maximum quality (100%)\n  - Highest VRAM usage"
    
    desc = [f"{config}"]
    desc.append(f"  - Trainable params: ~{config.estimated_params / 1_000_000:.1f}M")
    desc.append(f"  - VRAM overhead: +{config.estimated_vram_overhead_gb:.1f} GB")
    desc.append(f"  - Expected quality: {config.expected_quality_percent:.1f}%")
    
    return "\n".join(desc)
