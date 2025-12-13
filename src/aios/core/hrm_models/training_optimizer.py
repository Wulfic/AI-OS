"""
Intelligent training optimization that finds the best configuration for available hardware.

This module analyzes GPU memory, model size, and training objectives to automatically
configure optimal settings for training runs.

Optimization priorities:
1. Maximize context length (up to 100K tokens)
2. Maximize batch size (after context is optimized)
3. Select best optimizer and memory strategy (DeepSpeed, ZeRO, etc.)
"""

from __future__ import annotations

import logging
import os
import gc
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Result of training optimization analysis."""
    
    # Context and batch settings
    context_length: int
    batch_size: int
    chunk_size: int
    
    # Optimizer settings
    optimizer_type: str  # "adamw", "adam", "sgd", etc.
    use_deepspeed: bool
    deepspeed_stage: int  # 0, 1, 2, or 3
    
    # Memory optimizations
    use_amp: bool
    use_gradient_checkpointing: bool
    gradient_accumulation_steps: int
    
    # Additional settings
    use_cpu_offload: bool
    estimated_vram_gb: float
    available_vram_gb: float
    optimization_score: float  # 0-100, higher is better
    warnings: List[str]
    recommendations: List[str]


class TrainingOptimizer:
    """Intelligent optimizer that finds best training configuration."""
    
    def __init__(self):
        self.min_context_length = 4000
        self.max_context_length = 100000
        self.target_vram_usage = 0.85  # Use 85% of available VRAM
        
    def detect_available_vram(self) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Detect available VRAM across all GPUs (CUDA, XPU).
        
        Returns:
            (total_vram_gb, list of GPU info dicts with keys:
             id, backend, name, total_gb, allocated_gb, reserved_gb, available_gb, vendor)
        """
        from aios.core.gpu_vendor import identify_gpu_vendor
        
        gpus: List[Dict[str, Any]] = []
        total_vram = 0.0
        
        # Check for ROCm build
        rocm = False
        try:
            rocm = bool(getattr(torch.version, "hip", None))
        except Exception:
            pass
        
        # Detect CUDA devices (NVIDIA + AMD ROCm)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detecting VRAM across {num_gpus} CUDA GPU(s)...")
            
            for gpu_id in range(num_gpus):
                try:
                    props = torch.cuda.get_device_properties(gpu_id)
                    total_gb = props.total_memory / (1024 ** 3)
                    
                    # Get current allocation
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                    available_gb = total_gb - reserved_gb
                    
                    vendor = identify_gpu_vendor(props.name, check_rocm=rocm)
                    
                    gpu_info = {
                        "id": gpu_id,
                        "backend": "cuda",
                        "name": props.name,
                        "vendor": vendor,
                        "total_gb": total_gb,
                        "allocated_gb": allocated_gb,
                        "reserved_gb": reserved_gb,
                        "available_gb": available_gb,
                    }
                    
                    logger.info(
                        f"[{vendor}] GPU {gpu_id}: {props.name}, "
                        f"Total: {total_gb:.1f} GB, "
                        f"Available: {available_gb:.1f} GB, "
                        f"Allocated: {allocated_gb:.1f} GB"
                    )
                    
                    gpus.append(gpu_info)
                    total_vram += total_gb
                except Exception as e:
                    logger.warning(f"Failed to get CUDA device {gpu_id} properties: {e}")
        
        # Detect Intel XPU devices
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                xpu_count = torch.xpu.device_count()
                logger.info(f"Detecting VRAM across {xpu_count} Intel XPU device(s)...")
                
                for xpu_id in range(xpu_count):
                    try:
                        props = torch.xpu.get_device_properties(xpu_id)
                        total_gb = getattr(props, "total_memory", 0) / (1024 ** 3)
                        
                        # XPU memory info
                        try:
                            free_mem, total_mem = torch.xpu.mem_get_info(xpu_id)
                            available_gb = free_mem / (1024 ** 3)
                            reserved_gb = (total_mem - free_mem) / (1024 ** 3)
                        except Exception:
                            available_gb = total_gb * 0.9  # Estimate 90% available
                            reserved_gb = total_gb * 0.1
                        
                        gpu_info = {
                            "id": len(gpus),  # Global ID across all backends
                            "backend": "xpu",
                            "name": getattr(props, "name", f"Intel XPU {xpu_id}"),
                            "vendor": "Intel",
                            "total_gb": total_gb,
                            "allocated_gb": 0.0,  # Not available via public API
                            "reserved_gb": reserved_gb,
                            "available_gb": available_gb,
                        }
                        
                        logger.info(
                            f"[Intel] XPU {xpu_id}: {gpu_info['name']}, "
                            f"Total: {total_gb:.1f} GB, "
                            f"Available: {available_gb:.1f} GB"
                        )
                        
                        gpus.append(gpu_info)
                        total_vram += total_gb
                    except Exception as e:
                        logger.warning(f"Failed to get XPU device {xpu_id} properties: {e}")
        except ImportError:
            # intel-extension-for-pytorch not installed
            pass
        except Exception as e:
            logger.debug(f"XPU detection warning: {e}")
        
        if not gpus:
            logger.info("No GPUs detected (CUDA or XPU)")
            return 0.0, []
        
        logger.info(f"Total VRAM across all GPUs: {total_vram:.1f} GB")
        return total_vram, gpus
    
    def estimate_memory_usage(
        self,
        model_params: int,
        context_length: int,
        batch_size: int,
        hidden_size: int,
        num_layers: int,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = True,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Estimate VRAM usage for given configuration.
        
        Returns dict with memory breakdown in GB.
        """
        # Model parameters (always FP32)
        model_gb = (model_params * 4) / (1024 ** 3)
        
        # Optimizer states (AdamW = 2x params in FP32)
        optimizer_gb = (model_params * 2 * 4) / (1024 ** 3)
        
        # Gradients (FP32)
        gradients_gb = (model_params * 4) / (1024 ** 3)
        
        # Activations (per micro-batch, affected by AMP and checkpointing)
        bytes_per_element = 2 if use_amp else 4  # FP16 vs FP32
        
        # Estimate activation memory per layer
        # Rough estimate: batch × seq × hidden × num_layers × components
        activation_elements = batch_size * context_length * hidden_size * num_layers * 8
        activations_gb = (activation_elements * bytes_per_element) / (1024 ** 3)
        
        # Gradient checkpointing saves ~50% of activation memory
        if use_gradient_checkpointing:
            activations_gb *= 0.5
        
        # Gradient accumulation reduces per-step activation memory
        activations_gb /= gradient_accumulation_steps
        
        # Logits output
        vocab_size = 50257  # Standard tokenizer vocab size
        logits_gb = (batch_size * context_length * vocab_size * bytes_per_element) / (1024 ** 3)
        
        # CUDA overhead (15%)
        base_total = model_gb + optimizer_gb + gradients_gb + activations_gb + logits_gb
        cuda_overhead_gb = base_total * 0.15
        
        total_gb = base_total + cuda_overhead_gb
        
        return {
            "model_gb": model_gb,
            "optimizer_gb": optimizer_gb,
            "gradients_gb": gradients_gb,
            "activations_gb": activations_gb,
            "logits_gb": logits_gb,
            "cuda_overhead_gb": cuda_overhead_gb,
            "total_gb": total_gb,
        }
    
    def find_optimal_context(
        self,
        available_vram_gb: float,
        model_params: int,
        hidden_size: int,
        num_layers: int,
        min_context: Optional[int] = None,
        max_context: Optional[int] = None,
    ) -> Tuple[int, int, Dict[str, float]]:
        """
        Binary search to find maximum context length that fits in VRAM.
        
        Returns:
            (optimal_context, optimal_chunk_size, memory_estimate)
        """
        min_ctx = min_context or self.min_context_length
        max_ctx = max_context or self.max_context_length
        
        # Target: use 85% of available VRAM
        target_vram = available_vram_gb * self.target_vram_usage
        
        best_context = min_ctx
        best_estimate = None
        
        # Binary search for maximum context that fits
        low, high = min_ctx, max_ctx
        
        while low <= high:
            mid = (low + high) // 2
            
            # Round to nearest 1000 for cleaner numbers
            mid = (mid // 1000) * 1000
            if mid < min_ctx:
                mid = min_ctx
            
            # Estimate with chunking (aggressive for large contexts)
            chunk_size = self._calculate_chunk_size(mid)
            
            estimate = self.estimate_memory_usage(
                model_params=model_params,
                context_length=chunk_size,  # Only one chunk active at a time!
                batch_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                use_amp=True,
                use_gradient_checkpointing=True,
            )
            
            if estimate["total_gb"] <= target_vram:
                # This context fits, try larger
                best_context = mid
                best_estimate = estimate
                low = mid + 1000
            else:
                # Too large, try smaller
                high = mid - 1000
        
        # Calculate optimal chunk size for best context
        chunk_size = self._calculate_chunk_size(best_context)
        
        return best_context, chunk_size, best_estimate or {}
    
    def _calculate_chunk_size(self, context_length: int) -> int:
        """Calculate optimal chunk size based on context length."""
        if context_length >= 500000:
            return 192
        elif context_length >= 200000:
            return 256
        elif context_length >= 100000:
            return 384
        elif context_length >= 50000:
            return 512
        elif context_length >= 20000:
            return 640
        elif context_length >= 10000:
            return 768
        elif context_length >= 8000:
            return 1024
        else:
            return 2048
    
    def find_optimal_batch_size(
        self,
        available_vram_gb: float,
        model_params: int,
        hidden_size: int,
        num_layers: int,
        context_length: int,
        chunk_size: int,
    ) -> Tuple[int, int]:
        """
        Find optimal batch size and gradient accumulation steps.
        
        Returns:
            (batch_size, gradient_accumulation_steps)
        """
        target_vram = available_vram_gb * self.target_vram_usage
        
        # Try batch sizes: 1, 2, 4, 8
        for batch_size in [8, 4, 2, 1]:
            estimate = self.estimate_memory_usage(
                model_params=model_params,
                context_length=chunk_size,
                batch_size=batch_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                use_amp=True,
                use_gradient_checkpointing=True,
            )
            
            if estimate["total_gb"] <= target_vram:
                # This batch size fits
                # Use gradient accumulation to simulate larger batches
                grad_accum = max(1, 8 // batch_size)
                return batch_size, grad_accum
        
        # If even batch_size=1 is tight, use gradient accumulation
        return 1, 8
    
    def select_optimizer_strategy(
        self,
        available_vram_gb: float,
        num_gpus: int,
        model_params: int,
        context_length: int,
    ) -> Dict[str, Any]:
        """
        Select best optimizer and DeepSpeed configuration.
        
        Returns dict with optimizer settings.
        """
        strategy = {
            "optimizer_type": "adamw",
            "use_deepspeed": False,
            "deepspeed_stage": 0,
            "use_cpu_offload": False,
            "reason": "",
        }
        
        # Single GPU - no DeepSpeed needed
        if num_gpus <= 1:
            strategy["reason"] = "Single GPU - using standard AdamW"
            return strategy
        
        # Multi-GPU with limited VRAM per GPU
        vram_per_gpu = available_vram_gb / num_gpus
        
        if vram_per_gpu < 12:
            # Limited VRAM - use ZeRO Stage 2 or 3
            if model_params > 500_000_000:
                # Very large model - use ZeRO Stage 3 (partition everything)
                strategy["use_deepspeed"] = True
                strategy["deepspeed_stage"] = 3
                strategy["use_cpu_offload"] = True
                strategy["reason"] = "Large model + limited VRAM - using ZeRO Stage 3 with CPU offload"
            else:
                # Medium model - use ZeRO Stage 2 (partition optimizer states + gradients)
                strategy["use_deepspeed"] = True
                strategy["deepspeed_stage"] = 2
                strategy["reason"] = "Multi-GPU with limited VRAM - using ZeRO Stage 2"
        else:
            # Sufficient VRAM - use ZeRO Stage 1 (partition optimizer states only)
            strategy["use_deepspeed"] = True
            strategy["deepspeed_stage"] = 1
            strategy["reason"] = "Multi-GPU with good VRAM - using ZeRO Stage 1"
        
        return strategy
    
    def optimize(
        self,
        model_params: int,
        hidden_size: int,
        num_layers: int,
        min_context: Optional[int] = None,
        max_context: Optional[int] = None,
        prioritize_context: bool = True,
    ) -> OptimizationConfig:
        """
        Main optimization function - finds best configuration.
        
        Args:
            model_params: Total model parameters
            hidden_size: Model hidden dimension
            num_layers: Total number of layers (H + L)
            min_context: Minimum context length (default: 4000)
            max_context: Maximum context length (default: 100000)
            prioritize_context: If True, maximize context first, then batch
        
        Returns:
            OptimizationConfig with optimal settings
        """
        warnings = []
        recommendations = []
        
        # Detect available hardware
        total_vram, gpus = self.detect_available_vram()
        num_gpus = len(gpus)
        
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs detected")
        
        # Use minimum VRAM across GPUs (bottleneck)
        min_vram = min(gpu["available_gb"] for gpu in gpus)
        
        logger.info({
            "optimization_started": True,
            "num_gpus": num_gpus,
            "total_vram_gb": round(total_vram, 2),
            "min_gpu_vram_gb": round(min_vram, 2),
            "model_params": model_params,
        })
        
        # Step 1: Find optimal context length
        optimal_context, chunk_size, memory_est = self.find_optimal_context(
            available_vram_gb=min_vram,
            model_params=model_params,
            hidden_size=hidden_size,
            num_layers=num_layers,
            min_context=min_context,
            max_context=max_context,
        )
        
        logger.info({
            "optimal_context_found": True,
            "context_length": optimal_context,
            "chunk_size": chunk_size,
            "estimated_vram_gb": round(memory_est.get("total_gb", 0), 2),
        })
        
        # Step 2: Find optimal batch size
        batch_size, grad_accum = self.find_optimal_batch_size(
            available_vram_gb=min_vram,
            model_params=model_params,
            hidden_size=hidden_size,
            num_layers=num_layers,
            context_length=optimal_context,
            chunk_size=chunk_size,
        )
        
        logger.info({
            "optimal_batch_found": True,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
        })
        
        # Step 3: Select optimizer strategy
        optimizer_config = self.select_optimizer_strategy(
            available_vram_gb=total_vram,
            num_gpus=num_gpus,
            model_params=model_params,
            context_length=optimal_context,
        )
        
        print({
            "optimizer_selected": True,
            "strategy": optimizer_config["optimizer_type"],
            "deepspeed": optimizer_config["use_deepspeed"],
            "deepspeed_stage": optimizer_config["deepspeed_stage"],
            "reason": optimizer_config["reason"],
        })
        
        # Generate warnings and recommendations
        if optimal_context < 10000:
            warnings.append(f"Context length limited to {optimal_context} due to VRAM constraints")
            recommendations.append("Consider using a smaller model or fewer layers")
        
        if optimal_context >= 50000:
            recommendations.append(f"Excellent! Can train with {optimal_context} token context")
        
        if batch_size == 1 and grad_accum > 1:
            recommendations.append(f"Using gradient accumulation ({grad_accum} steps) to simulate larger batch")
        
        if optimizer_config["use_cpu_offload"]:
            warnings.append("CPU offloading enabled - training will be slower but use less VRAM")
        
        # Calculate optimization score (0-100)
        context_score = min(100, (optimal_context / 100000) * 70)  # 70% weight on context
        batch_score = min(30, (batch_size / 8) * 30)  # 30% weight on batch size
        optimization_score = context_score + batch_score
        
        return OptimizationConfig(
            context_length=optimal_context,
            batch_size=batch_size,
            chunk_size=chunk_size,
            optimizer_type=optimizer_config["optimizer_type"],
            use_deepspeed=optimizer_config["use_deepspeed"],
            deepspeed_stage=optimizer_config["deepspeed_stage"],
            use_amp=True,
            use_gradient_checkpointing=True,
            gradient_accumulation_steps=grad_accum,
            use_cpu_offload=optimizer_config["use_cpu_offload"],
            estimated_vram_gb=memory_est.get("total_gb", 0),
            available_vram_gb=min_vram,
            optimization_score=optimization_score,
            warnings=warnings,
            recommendations=recommendations,
        )


def optimize_training_config(
    model_params: int,
    hidden_size: int,
    num_layers: int,
    min_context: int = 4000,
    max_context: int = 100000,
) -> OptimizationConfig:
    """
    Convenience function to optimize training configuration.
    
    Example:
        config = optimize_training_config(
            model_params=87_115_778,
            hidden_size=512,
            num_layers=16,
        )
        logger.info(f"Use context: {config.context_length}")
        logger.info(f"Use batch: {config.batch_size}")
    """
    optimizer = TrainingOptimizer()
    return optimizer.optimize(
        model_params=model_params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        min_context=min_context,
        max_context=max_context,
    )
