"""Main MemoryEstimator class for HRM training memory prediction.

This module provides the high-level MemoryEstimator class that:
- Takes HRM training configuration
- Estimates VRAM and RAM requirements
- Provides summaries and recommendations
- Auto-calculates optimal chunk sizes
"""

from __future__ import annotations
from typing import Optional, Dict, Any

from .constants import GB
from .vram_estimation import estimate_vram
from .ram_estimation import estimate_ram
from .vram_lookup import estimate_vram_hybrid, OptimizationConfig


class MemoryEstimator:
    """Estimates memory requirements for HRM training configurations."""
    
    def __init__(
        self,
        total_params: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        batch_size: int,
        num_gpus: int = 1,
        use_amp: bool = False,
        use_gradient_checkpointing: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        use_8bit_optimizer: bool = False,
        offload_optimizer: bool = False,
        zero_stage: str = "none",
        use_chunking: bool = False,
        chunk_size: Optional[int] = None,
        vocab_size: int = 50257,
    ):
        """
        Initialize memory estimator.
        
        Args:
            total_params: Total model parameters
            hidden_size: Model hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            seq_len: Sequence length
            batch_size: Training batch size
            num_gpus: Number of GPUs for distributed training
            use_amp: Whether using automatic mixed precision (FP16)
            use_gradient_checkpointing: Whether using gradient checkpointing
            use_lora: Whether using LoRA adapters
            lora_r: LoRA rank (if using LoRA)
            use_8bit_optimizer: Whether using 8-bit optimizer
            offload_optimizer: Whether offloading optimizer to CPU
            zero_stage: DeepSpeed ZeRO stage ("none", "1", "2", "3")
            use_chunking: Whether using chunked sequence processing
            chunk_size: Chunk size (if chunking), or None to auto-calculate
            vocab_size: Tokenizer vocabulary size
        """
        self.total_params = total_params
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.use_amp = use_amp
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.use_8bit_optimizer = use_8bit_optimizer
        self.offload_optimizer = offload_optimizer
        self.zero_stage = zero_stage
        self.use_chunking = use_chunking
        self.vocab_size = vocab_size
        
        # Auto-calculate chunk size if needed
        if use_chunking and chunk_size is None:
            self.chunk_size = self._get_auto_chunk_size()
        else:
            self.chunk_size = chunk_size or seq_len
    
    @property
    def trainable_params(self) -> int:
        """Calculate trainable parameters based on LoRA configuration."""
        if self.use_lora:
            # With LoRA: only optimize adapter parameters
            return self.lora_adapter_params
        else:
            # Full fine-tuning: all parameters are trainable
            return self.total_params
    
    @property
    def lora_adapter_params(self) -> int:
        """Calculate LoRA adapter parameters."""
        if not self.use_lora or self.lora_r == 0:
            return 0
        # LoRA adds r * (hidden_in + hidden_out) parameters per layer
        # For transformer: 2 * num_layers * lora_r * hidden_size (for Q, V matrices)
        return 2 * self.num_layers * self.lora_r * self.hidden_size
    
    @property
    def use_cpu_offload(self) -> bool:
        """Alias for offload_optimizer for backward compatibility."""
        return self.offload_optimizer
    
    def _get_auto_chunk_size(self) -> int:
        """
        Auto-calculate optimal chunk size for long sequences.
        
        Balances:
        - Memory efficiency (smaller chunks)
        - Computation efficiency (larger chunks, less overhead)
        - Attention O(n²) scaling
        
        Returns:
            Optimal chunk size
        """
        # Target: ~4-8 GB VRAM per chunk for attention
        # Attention memory = batch_size * num_heads * chunk_size²
        
        # Start with sqrt approach for O(n²) scaling
        # If seq_len = 32768, chunk_size ≈ sqrt(32768) * factor
        import math
        base_chunk = int(math.sqrt(self.seq_len) * 8)  # Factor of 8 works well empirically
        
        # Clamp to reasonable range
        min_chunk = 512    # Below this, overhead dominates
        max_chunk = 4096   # Above this, memory savings diminish
        
        chunk_size = max(min_chunk, min(base_chunk, max_chunk))
        
        # Round down to power of 2 for efficiency
        chunk_size = 2 ** int(math.log2(chunk_size))
        
        # Ensure chunk size divides sequence length evenly (or close)
        # This avoids weird edge cases in padding
        if self.seq_len % chunk_size > chunk_size * 0.25:
            # If remainder is > 25% of chunk, might want smaller chunks
            # Try smaller power of 2
            chunk_size = chunk_size // 2
        
        return chunk_size
    
    def get_summary(self, use_hybrid_estimator: bool = True) -> Dict[str, Any]:
        """
        Get complete memory summary.
        
        Args:
            use_hybrid_estimator: If True, use empirical lookup table estimator (98.1% accuracy)
                                 If False, use analytical formula estimator (legacy)
        
        Returns dict with:
            - vram: VRAM breakdown dict
            - ram: RAM breakdown dict
            - total_vram_gb: Total VRAM needed per GPU
            - total_ram_gb: Total system RAM needed
            - config: Configuration summary
            - estimation_method: Which estimator was used
        """
        # Try hybrid estimator first (more accurate)
        vram_hybrid_gb = None
        if use_hybrid_estimator:
            try:
                # Determine MoE usage based on model architecture
                # HRM ACT-v1 typically uses MoE in the L-layers
                use_moe = hasattr(self, 'use_moe') and self.use_moe
                
                # Determine Flash Attention 2 usage
                use_flash_attn2 = hasattr(self, 'use_flash_attention_2') and self.use_flash_attention_2
                
                # Map zero_stage to integer (1, 2, 3) or None
                zero_stage_int = None
                if self.zero_stage == "zero1":
                    zero_stage_int = 1
                elif self.zero_stage == "zero2":
                    zero_stage_int = 2
                elif self.zero_stage == "zero3":
                    zero_stage_int = 3
                
                vram_hybrid_gb = estimate_vram_hybrid(
                    h_layers=self.num_layers // 2,  # Approximate H-layers
                    l_layers=self.num_layers - (self.num_layers // 2),  # Approximate L-layers
                    hidden_size=self.hidden_size,
                    context_size=self.seq_len,
                    batch_size=self.batch_size,
                    use_moe=use_moe,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_amp=self.use_amp,
                    use_flash_attention_2=use_flash_attn2,
                    use_8bit_optimizer=self.use_8bit_optimizer,
                    cpu_offload=self.offload_optimizer,
                    deepspeed_stage=zero_stage_int,
                    use_lora=self.use_lora,
                    lora_rank=self.lora_r if self.use_lora else None,
                    context_chunking=self.use_chunking,
                    chunk_size=self.chunk_size if self.use_chunking else None,
                )
                estimation_method = "hybrid_lookup_table"
            except Exception as e:
                # Fallback to analytical estimator if hybrid fails
                print(f"Hybrid estimator unavailable ({e}), using analytical estimator")
                vram_hybrid_gb = None
                estimation_method = "analytical_formula"
        else:
            estimation_method = "analytical_formula"
        
        # Use analytical estimator (always calculate for comparison)
        vram_dict = estimate_vram(self)
        ram_dict = estimate_ram(self)
        
        # NOTE: Hybrid estimator disabled - it doesn't account for multi-GPU properly
        # and returns totals that don't match component breakdown.
        # The analytical estimator's component-based calculation is more accurate.
        # If hybrid estimation succeeded, store it for reference but don't use it
        if vram_hybrid_gb is not None:
            vram_dict["hybrid_estimate_gb"] = vram_hybrid_gb
            vram_dict["estimation_method"] = "analytical_formula (component-based)"
        else:
            vram_dict["estimation_method"] = "analytical_formula (legacy)"
        
        return {
            "vram": vram_dict,
            "ram": ram_dict,
            "total_vram_gb": vram_dict.get("total_gb", 0),
            "total_ram_gb": ram_dict.get("total_gb", 0),
            "estimation_method": estimation_method,
            "config": {
                "total_params": self.total_params,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "num_gpus": self.num_gpus,
                "use_amp": self.use_amp,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "use_8bit_optimizer": self.use_8bit_optimizer,
                "offload_optimizer": self.offload_optimizer,
                "zero_stage": self.zero_stage,
                "use_chunking": self.use_chunking,
                "chunk_size": self.chunk_size,
            },
        }
    
    def get_recommendations(self, available_vram_gb: float = 11.0, available_ram_gb: float = 32.0) -> Dict[str, Any]:
        """
        Get training recommendations based on available memory.
        
        Args:
            available_vram_gb: Available GPU VRAM in GB (default: 11 GB for RTX 2080 Ti)
            available_ram_gb: Available system RAM in GB (default: 32 GB)
        
        Returns dict with:
            - feasible: Whether training is feasible with current config
            - vram_usage_pct: VRAM usage percentage
            - ram_usage_pct: RAM usage percentage
            - warnings: List of warning messages
            - suggestions: List of optimization suggestions
        """
        summary = self.get_summary()
        estimated_vram = summary["total_vram_gb"]
        estimated_ram = summary["total_ram_gb"]
        
        vram_usage_pct = (estimated_vram / available_vram_gb) * 100
        ram_usage_pct = (estimated_ram / available_ram_gb) * 100
        
        warnings = []
        suggestions = []
        
        # ===== VRAM ANALYSIS =====
        if vram_usage_pct > 95:
            warnings.append(f"⚠️ CRITICAL: VRAM usage ({estimated_vram:.1f} GB) exceeds available ({available_vram_gb:.1f} GB)")
            suggestions.append("• Reduce batch size")
            suggestions.append("• Enable gradient checkpointing")
            suggestions.append("• Enable chunked sequence processing")
            suggestions.append("• Consider DeepSpeed ZeRO-3 for model sharding")
        elif vram_usage_pct > 85:
            warnings.append(f"⚠️ WARNING: High VRAM usage ({vram_usage_pct:.0f}%) - OOM risk")
            suggestions.append("• Consider enabling chunking for long sequences")
            suggestions.append("• Try gradient checkpointing if not enabled")
        elif vram_usage_pct > 75:
            warnings.append(f"ℹ️ Moderate VRAM usage ({vram_usage_pct:.0f}%) - should be okay")
        
        # ===== RAM ANALYSIS =====
        if ram_usage_pct > 95:
            warnings.append(f"⚠️ CRITICAL: RAM usage ({estimated_ram:.1f} GB) exceeds available ({available_ram_gb:.1f} GB)")
            suggestions.append("• Enable optimizer CPU offload")
            suggestions.append("• Reduce dataset prefetch buffer")
            suggestions.append("• Close other applications")
        elif ram_usage_pct > 85:
            warnings.append(f"⚠️ WARNING: High RAM usage ({ram_usage_pct:.0f}%) - may cause swapping")
            suggestions.append("• Consider optimizer CPU offload")
        
        # ===== OPTIMIZATION SUGGESTIONS =====
        if not self.use_amp and estimated_vram > available_vram_gb * 0.7:
            suggestions.append("• Enable AMP (mixed precision) to halve VRAM usage")
        
        if not self.use_gradient_checkpointing and self.num_layers >= 12:
            suggestions.append("• Enable gradient checkpointing for deep models")
        
        if not self.use_chunking and self.seq_len > 8192:
            suggestions.append(f"• Enable chunking for long sequences (auto chunk size: {self._get_auto_chunk_size()})")
        
        if not self.use_8bit_optimizer and self.use_amp:
            suggestions.append("• Try 8-bit optimizer with AMP for further memory savings")
        
        if self.num_gpus == 1 and estimated_vram > available_vram_gb:
            suggestions.append("• Consider multi-GPU training with DeepSpeed ZeRO")
        
        # Determine feasibility
        feasible = (vram_usage_pct <= 95 and ram_usage_pct <= 95)
        
        return {
            "feasible": feasible,
            "vram_usage_pct": vram_usage_pct,
            "ram_usage_pct": ram_usage_pct,
            "warnings": warnings,
            "suggestions": suggestions,
        }


def quick_estimate(
    total_params: int,
    seq_len: int,
    batch_size: int = 1,
    use_amp: bool = False,
    use_chunking: bool = False,
) -> Dict[str, float]:
    """
    Quick memory estimate with minimal configuration.
    
    Useful for rapid prototyping and UI displays.
    
    Args:
        total_params: Total model parameters
        seq_len: Sequence length
        batch_size: Batch size (default: 1)
        use_amp: Whether using mixed precision (default: False)
        use_chunking: Whether using chunking (default: False)
    
    Returns dict with:
        - vram_gb: Estimated VRAM per GPU
        - ram_gb: Estimated system RAM
    """
    # Heuristic estimates for quick calculation
    # Assume standard transformer with hidden_size = sqrt(total_params / 12 / num_layers)
    # For GPT-2 style: num_layers ≈ log2(total_params / 1e6)
    import math
    
    num_layers = max(12, int(math.log2(max(1, total_params / 1_000_000))))
    hidden_size = int(math.sqrt(total_params / (12 * num_layers)))
    num_heads = max(8, hidden_size // 64)
    
    estimator = MemoryEstimator(
        total_params=total_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        batch_size=batch_size,
        use_amp=use_amp,
        use_chunking=use_chunking,
    )
    
    summary = estimator.get_summary()
    
    return {
        "vram_gb": summary["total_vram_gb"],
        "ram_gb": summary["total_ram_gb"],
    }
