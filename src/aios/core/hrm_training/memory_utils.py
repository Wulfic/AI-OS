"""
Memory tracking and profiling utilities for HRM training.

Provides detailed visibility into memory usage, optimization overhead,
and utilization metrics during training.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Captures memory state at a specific point in training."""
    name: str
    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "checkpoint": self.name,
            "timestamp": self.timestamp,
            "allocated_gb": round(self.allocated_bytes / 1024**3, 3),
            "reserved_gb": round(self.reserved_bytes / 1024**3, 3),
            "max_allocated_gb": round(self.max_allocated_bytes / 1024**3, 3),
            "device": self.device,
            **self.metadata
        }


class MemoryTracker:
    """
    Tracks GPU memory usage across training lifecycle.
    
    Usage:
        tracker = MemoryTracker(device='cuda:0')
        tracker.snapshot('after_model_creation')
        tracker.snapshot('after_optimizer_creation')
        print(tracker.get_report())
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.snapshots: List[MemorySnapshot] = []
        self.enabled = self._check_cuda_available()
        
        if self.enabled:
            logger.info(f"Memory tracking enabled for device: {device}")
        else:
            logger.warning(f"Memory tracking disabled - CUDA not available on {device}")
        
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for memory tracking."""
        try:
            import torch
            available = torch.cuda.is_available() and 'cuda' in self.device
            if not available:
                logger.debug("CUDA not available or device is not CUDA")
            return available
        except Exception as e:
            logger.warning(f"Failed to check CUDA availability: {e}")
            return False
    
    def snapshot(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[MemorySnapshot]:
        """
        Capture current memory state.
        
        Args:
            name: Descriptive name for this checkpoint
            metadata: Additional context to include in snapshot
            
        Returns:
            MemorySnapshot object or None if CUDA unavailable
        """
        if not self.enabled:
            return None
            
        try:
            import torch
            
            # Synchronize to ensure accurate measurements
            torch.cuda.synchronize()
            
            snapshot = MemorySnapshot(
                name=name,
                timestamp=time.time(),
                allocated_bytes=torch.cuda.memory_allocated(self.device),
                reserved_bytes=torch.cuda.memory_reserved(self.device),
                max_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                device=self.device,
                metadata=metadata or {}
            )
            
            self.snapshots.append(snapshot)
            
            # Log snapshot details
            allocated_gb = snapshot.allocated_bytes / (1024**3)
            reserved_gb = snapshot.reserved_bytes / (1024**3)
            logger.debug(
                f"Memory snapshot '{name}': "
                f"allocated={allocated_gb:.2f}GB, reserved={reserved_gb:.2f}GB"
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot '{name}': {e}")
            logger.error({"memory_snapshot_error": str(e), "checkpoint": name})
            return None
    
    def get_delta(self, from_name: str, to_name: str) -> Optional[Dict[str, Any]]:
        """
        Calculate memory difference between two snapshots.
        
        Args:
            from_name: Starting checkpoint name
            to_name: Ending checkpoint name
            
        Returns:
            Dictionary with delta statistics
        """
        from_snap = next((s for s in self.snapshots if s.name == from_name), None)
        to_snap = next((s for s in self.snapshots if s.name == to_name), None)
        
        if not from_snap or not to_snap:
            return None
        
        delta_allocated = to_snap.allocated_bytes - from_snap.allocated_bytes
        delta_reserved = to_snap.reserved_bytes - from_snap.reserved_bytes
        
        return {
            "from": from_name,
            "to": to_name,
            "delta_allocated_mb": round(delta_allocated / 1024**2, 2),
            "delta_reserved_mb": round(delta_reserved / 1024**2, 2),
            "delta_allocated_gb": round(delta_allocated / 1024**3, 3),
            "duration_sec": round(to_snap.timestamp - from_snap.timestamp, 3),
            "allocated_before_gb": round(from_snap.allocated_bytes / 1024**3, 3),
            "allocated_after_gb": round(to_snap.allocated_bytes / 1024**3, 3),
        }
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.
        
        Returns:
            Dictionary with full memory profile
        """
        if not self.snapshots:
            return {"error": "No snapshots captured"}
        
        report = {
            "device": self.device,
            "total_snapshots": len(self.snapshots),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "deltas": []
        }
        
        # Calculate deltas between consecutive snapshots
        for i in range(1, len(self.snapshots)):
            delta = self.get_delta(self.snapshots[i-1].name, self.snapshots[i].name)
            if delta:
                report["deltas"].append(delta)
        
        # Add peak memory info
        if self.snapshots:
            peak_snapshot = max(self.snapshots, key=lambda s: s.allocated_bytes)
            report["peak_memory"] = {
                "checkpoint": peak_snapshot.name,
                "allocated_gb": round(peak_snapshot.allocated_bytes / 1024**3, 3),
                "reserved_gb": round(peak_snapshot.reserved_bytes / 1024**3, 3),
            }
        
        return report
    
    def log_current(self, label: str = "current", extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log current memory state without creating a snapshot.
        
        Args:
            label: Label for this measurement
            extra_info: Additional information to include
            
        Returns:
            Dictionary with current memory stats
        """
        if not self.enabled:
            return {"error": "CUDA not available"}
        
        try:
            import torch
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            
            # Get total device memory
            device_props = torch.cuda.get_device_properties(self.device)
            total_memory = device_props.total_memory
            
            result = {
                "label": label,
                "allocated_gb": round(allocated / 1024**3, 3),
                "reserved_gb": round(reserved / 1024**3, 3),
                "max_allocated_gb": round(max_allocated / 1024**3, 3),
                "total_gb": round(total_memory / 1024**3, 3),
                "utilization_pct": round((allocated / total_memory) * 100, 1),
                "fragmentation_mb": round((reserved - allocated) / 1024**2, 1),
            }
            
            if extra_info:
                result.update(extra_info)
            
            # Log the memory state
            logger.info(
                f"Memory [{label}]: {result['allocated_gb']:.2f}/{result['total_gb']:.2f} GB "
                f"({result['utilization_pct']:.1f}% utilized), "
                f"reserved={result['reserved_gb']:.2f} GB, "
                f"fragmentation={result['fragmentation_mb']:.1f} MB"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to log current memory for '{label}': {e}")
            return {"error": str(e), "label": label}


def estimate_model_memory(
    num_parameters: int,
    precision: str = 'fp16',
    include_gradients: bool = True,
    include_optimizer: bool = True,
    optimizer_type: str = 'adamw'
) -> Dict[str, Any]:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_parameters: Total number of model parameters
        precision: 'fp32', 'fp16', or 'bf16'
        include_gradients: Whether to include gradient memory
        include_optimizer: Whether to include optimizer state memory
        optimizer_type: 'adamw', 'adamw8bit', 'sgd'
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
    }.get(precision, 2)
    
    model_memory = num_parameters * bytes_per_param
    gradient_memory = num_parameters * bytes_per_param if include_gradients else 0
    
    # Optimizer state memory
    optimizer_memory = 0
    if include_optimizer:
        if optimizer_type == 'adamw':
            # AdamW stores: momentum (4 bytes) + variance (4 bytes) per param
            optimizer_memory = num_parameters * 8
        elif optimizer_type == 'adamw8bit':
            # 8-bit optimizer: ~1 byte per param for quantized states
            optimizer_memory = num_parameters * 1
        elif optimizer_type == 'sgd':
            # SGD with momentum: 4 bytes per param
            optimizer_memory = num_parameters * 4
    
    total_memory = model_memory + gradient_memory + optimizer_memory
    
    return {
        "model_gb": round(model_memory / 1024**3, 3),
        "gradients_gb": round(gradient_memory / 1024**3, 3),
        "optimizer_gb": round(optimizer_memory / 1024**3, 3),
        "total_gb": round(total_memory / 1024**3, 3),
        "parameters": num_parameters,
        "precision": precision,
        "optimizer": optimizer_type,
    }


def estimate_activation_memory(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    gradient_checkpointing: bool = False,
    precision: str = 'fp16'
) -> Dict[str, Any]:
    """
    Estimate activation memory for transformer-based models.
    
    Args:
        batch_size: Training batch size
        sequence_length: Maximum sequence length
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        gradient_checkpointing: Whether gradient checkpointing is enabled
        precision: 'fp32', 'fp16', or 'bf16'
        
    Returns:
        Dictionary with activation memory estimates
    """
    bytes_per_element = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
    }.get(precision, 2)
    
    # Attention activations: Q, K, V matrices
    attention_memory = batch_size * sequence_length * hidden_size * bytes_per_element * 3
    
    # Attention scores: (batch, heads, seq_len, seq_len)
    attention_scores = batch_size * num_heads * sequence_length * sequence_length * bytes_per_element
    
    # Feed-forward activations (typically 4x hidden_size)
    ffn_memory = batch_size * sequence_length * hidden_size * 4 * bytes_per_element
    
    # Per layer memory
    per_layer = attention_memory + attention_scores + ffn_memory
    
    # With gradient checkpointing, only store activations for 1 layer at a time
    if gradient_checkpointing:
        total_activations = per_layer * 2  # Forward + backward for 1 layer
        saving_factor = num_layers / 2
    else:
        total_activations = per_layer * num_layers
        saving_factor = 1
    
    return {
        "activation_gb": round(total_activations / 1024**3, 3),
        "per_layer_mb": round(per_layer / 1024**2, 1),
        "attention_mb": round((attention_memory + attention_scores) / 1024**2, 1),
        "ffn_mb": round(ffn_memory / 1024**2, 1),
        "gradient_checkpointing": gradient_checkpointing,
        "memory_saved_gb": round((per_layer * (num_layers - 2)) / 1024**3, 3) if gradient_checkpointing else 0,
        "saving_factor": round(saving_factor, 1),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def log_optimization_summary(
    model_memory_gb: float,
    use_8bit_optimizer: bool = False,
    gradient_checkpointing: bool = False,
    use_amp: bool = False,
    use_chunked_training: bool = False,
    chunk_size: Optional[int] = None,
    zero_stage: str = "none",
    num_gpus: int = 1
) -> Dict[str, Any]:
    """
    Generate summary of active optimizations and their expected impact.
    
    Returns:
        Dictionary with optimization summary
    """
    optimizations = []
    total_savings_gb = 0
    
    if use_8bit_optimizer:
        # 8-bit optimizer saves ~75% of optimizer memory (typically 8 bytes -> 2 bytes per param)
        optimizer_savings = model_memory_gb * 0.75
        total_savings_gb += optimizer_savings
        optimizations.append({
            "name": "8-bit Optimizer",
            "enabled": True,
            "savings_gb": round(optimizer_savings, 3),
            "description": "Quantized optimizer states (75% memory reduction)"
        })
    
    if gradient_checkpointing:
        # Gradient checkpointing saves ~num_layers-2 layers worth of activations
        # Approximate as 50% of activation memory
        checkpoint_savings = model_memory_gb * 0.5
        total_savings_gb += checkpoint_savings
        optimizations.append({
            "name": "Gradient Checkpointing",
            "enabled": True,
            "savings_gb": round(checkpoint_savings, 3),
            "description": "Recompute activations during backward pass"
        })
    
    if use_amp:
        # AMP saves ~50% of model + activation memory
        amp_savings = model_memory_gb * 0.5
        total_savings_gb += amp_savings
        optimizations.append({
            "name": "Mixed Precision (AMP)",
            "enabled": True,
            "savings_gb": round(amp_savings, 3),
            "description": "FP16/BF16 computation (50% memory reduction)"
        })
    
    if use_chunked_training and chunk_size:
        optimizations.append({
            "name": "Chunked Training",
            "enabled": True,
            "chunk_size": chunk_size,
            "description": f"Process sequence in {chunk_size}-token chunks"
        })
    
    if zero_stage != "none" and num_gpus > 1:
        stage_num = int(zero_stage.replace("zero", "")) if "zero" in zero_stage else 0
        if stage_num == 3:
            # ZeRO-3 partitions model across GPUs
            zero_savings = model_memory_gb * (1 - 1/num_gpus)
            total_savings_gb += zero_savings
            optimizations.append({
                "name": f"DeepSpeed ZeRO-{stage_num}",
                "enabled": True,
                "savings_gb": round(zero_savings, 3),
                "num_gpus": num_gpus,
                "description": f"Model partitioned across {num_gpus} GPUs"
            })
        elif stage_num == 2:
            # ZeRO-2 partitions optimizer + gradients
            zero_savings = model_memory_gb * 0.5 * (1 - 1/num_gpus)
            total_savings_gb += zero_savings
            optimizations.append({
                "name": f"DeepSpeed ZeRO-{stage_num}",
                "enabled": True,
                "savings_gb": round(zero_savings, 3),
                "num_gpus": num_gpus,
                "description": f"Optimizer+gradients partitioned across {num_gpus} GPUs"
            })
    
    return {
        "optimizations": optimizations,
        "total_estimated_savings_gb": round(total_savings_gb, 3),
        "baseline_memory_gb": round(model_memory_gb, 3),
        "expected_memory_gb": round(max(0, model_memory_gb - total_savings_gb), 3),
    }
