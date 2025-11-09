"""Core test harness for memory estimation accuracy testing.

This module provides the infrastructure to:
1. Run actual training with memory profiling
2. Compare actual vs estimated memory usage
3. Store results for analysis
4. Calculate accuracy metrics
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import sys

try:
    import torch
    import psutil
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_cached_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float


@dataclass
class TestConfiguration:
    """Configuration for a memory test."""
    # Model configuration
    model_name: str
    tokenizer_name: str
    total_params: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    
    # Training configuration
    seq_len: int
    batch_size: int
    num_gpus: int
    
    # Optimization flags
    use_amp: bool
    use_gradient_checkpointing: bool
    use_lora: bool
    lora_r: int
    use_8bit_optimizer: bool
    offload_optimizer: bool
    zero_stage: str
    use_chunking: bool
    chunk_size: Optional[int]
    
    # Test metadata
    test_id: str
    test_name: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestConfiguration":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TestResult:
    """Result of a memory test."""
    config: TestConfiguration
    
    # Estimated memory
    estimated_vram_gb: float
    estimated_ram_gb: float
    estimation_breakdown: Dict[str, Any]
    
    # Actual measured memory
    actual_peak_vram_allocated_gb: float
    actual_peak_vram_reserved_gb: float
    actual_peak_ram_gb: float
    
    # Memory snapshots over time
    snapshots: List[MemorySnapshot]
    
    # Accuracy metrics
    vram_accuracy_pct: float  # 100 - abs(estimated - actual) / actual * 100
    ram_accuracy_pct: float
    vram_error_gb: float  # estimated - actual (positive = overestimate)
    ram_error_gb: float
    
    # Test metadata
    test_duration_sec: float
    success: bool
    error_message: Optional[str]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict
        data = asdict(self)
        # Config is already a dict from asdict
        # Snapshots are already dicts from asdict
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create from dictionary."""
        # Convert nested objects
        data["config"] = TestConfiguration.from_dict(data["config"])
        data["snapshots"] = [MemorySnapshot(**s) for s in data["snapshots"]]
        return cls(**data)


class MemoryProfiler:
    """Profiles memory usage during training."""
    
    def __init__(self):
        """Initialize profiler."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for memory profiling")
        
        self.snapshots: List[MemorySnapshot] = []
        self.peak_vram_allocated = 0.0
        self.peak_vram_reserved = 0.0
        self.peak_ram = 0.0
        self.process = psutil.Process()
        
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_cached = torch.cuda.memory_cached() / (1024**3) if hasattr(torch.cuda, "memory_cached") else gpu_reserved
        else:
            gpu_allocated = gpu_reserved = gpu_cached = 0.0
        
        # System RAM
        mem_info = self.process.memory_info()
        ram_used = mem_info.rss / (1024**3)  # Resident Set Size
        
        vm = psutil.virtual_memory()
        ram_available = vm.available / (1024**3)
        ram_percent = vm.percent
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated_gb=gpu_allocated,
            gpu_reserved_gb=gpu_reserved,
            gpu_cached_gb=gpu_cached,
            ram_used_gb=ram_used,
            ram_available_gb=ram_available,
            ram_percent=ram_percent,
        )
        
        # Track peaks
        self.peak_vram_allocated = max(self.peak_vram_allocated, gpu_allocated)
        self.peak_vram_reserved = max(self.peak_vram_reserved, gpu_reserved)
        self.peak_ram = max(self.peak_ram, ram_used)
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_peaks(self) -> Dict[str, float]:
        """Get peak memory usage."""
        # Also check PyTorch's internal peak stats
        if torch.cuda.is_available():
            torch_peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            torch_peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
            
            # Use maximum from both sources
            peak_vram_allocated = max(self.peak_vram_allocated, torch_peak_allocated)
            peak_vram_reserved = max(self.peak_vram_reserved, torch_peak_reserved)
        else:
            peak_vram_allocated = self.peak_vram_allocated
            peak_vram_reserved = self.peak_vram_reserved
        
        return {
            "peak_vram_allocated_gb": peak_vram_allocated,
            "peak_vram_reserved_gb": peak_vram_reserved,
            "peak_ram_gb": self.peak_ram,
        }
    
    def reset(self):
        """Reset profiler state."""
        self.snapshots.clear()
        self.peak_vram_allocated = 0.0
        self.peak_vram_reserved = 0.0
        self.peak_ram = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        gc.collect()


class MemoryTestHarness:
    """Main test harness for memory estimation testing."""
    
    def __init__(self, results_dir: str = "artifacts/memory_tests"):
        """Initialize test harness.
        
        Args:
            results_dir: Directory to store test results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and psutil are required for memory testing")
        
        self.profiler = MemoryProfiler()
        self.current_test: Optional[TestConfiguration] = None
    
    def run_test(
        self,
        config: TestConfiguration,
        training_steps: int = 10,
        profile_interval_sec: float = 0.5,
    ) -> TestResult:
        """Run a single memory test.
        
        Args:
            config: Test configuration
            training_steps: Number of training steps to run
            profile_interval_sec: Interval between memory snapshots
        
        Returns:
            TestResult with measurements and accuracy metrics
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for memory testing")
        
        print(f"\n{'='*80}")
        print(f"Running test: {config.test_name}")
        print(f"Description: {config.description}")
        print(f"{'='*80}\n")
        
        self.current_test = config
        start_time = time.time()
        
        # Get estimation
        from aios.gui.components.hrm_training.memory_estimator import MemoryEstimator
        
        estimator = MemoryEstimator(
            total_params=config.total_params,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            num_gpus=config.num_gpus,
            use_amp=config.use_amp,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            use_8bit_optimizer=config.use_8bit_optimizer,
            offload_optimizer=config.offload_optimizer,
            zero_stage=config.zero_stage,
            use_chunking=config.use_chunking,
            chunk_size=config.chunk_size,
            vocab_size=config.vocab_size,
        )
        
        summary = estimator.get_summary()
        estimated_vram = summary["total_vram_gb"]
        estimated_ram = summary["total_ram_gb"]
        
        print(f"Estimated VRAM: {estimated_vram:.2f} GB")
        print(f"Estimated RAM: {estimated_ram:.2f} GB\n")
        
        # Reset profiler
        self.profiler.reset()
        
        try:
            # Run actual training with profiling
            actual_peaks = self._run_training_with_profiling(
                config=config,
                training_steps=training_steps,
                profile_interval_sec=profile_interval_sec,
            )
            
            success = True
            error_message = None
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            success = False
            error_message = str(e)
            actual_peaks = self.profiler.get_peaks()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate accuracy metrics
        actual_vram = actual_peaks["peak_vram_reserved_gb"]  # Use reserved (closer to estimation)
        actual_ram = actual_peaks["peak_ram_gb"]
        
        vram_error = estimated_vram - actual_vram
        ram_error = estimated_ram - actual_ram
        
        # Accuracy = 100 - abs(error) / actual * 100
        # But handle edge cases where actual might be very small
        if actual_vram > 0.1:
            vram_accuracy = 100 - (abs(vram_error) / actual_vram * 100)
        else:
            vram_accuracy = 0.0  # Can't measure accuracy if no actual usage
        
        if actual_ram > 0.5:
            ram_accuracy = 100 - (abs(ram_error) / actual_ram * 100)
        else:
            ram_accuracy = 0.0
        
        print(f"\n{'='*80}")
        print(f"Test Results:")
        print(f"{'='*80}")
        print(f"Actual VRAM (reserved): {actual_vram:.2f} GB")
        print(f"Actual VRAM (allocated): {actual_peaks['peak_vram_allocated_gb']:.2f} GB")
        print(f"Actual RAM: {actual_ram:.2f} GB")
        print(f"\nVRAM Error: {vram_error:+.2f} GB ({vram_error/actual_vram*100:+.1f}%)")
        print(f"RAM Error: {ram_error:+.2f} GB ({ram_error/actual_ram*100:+.1f}%)")
        print(f"\nVRAM Accuracy: {vram_accuracy:.1f}%")
        print(f"RAM Accuracy: {ram_accuracy:.1f}%")
        print(f"Duration: {duration:.1f}s")
        print(f"Success: {success}")
        print(f"{'='*80}\n")
        
        result = TestResult(
            config=config,
            estimated_vram_gb=estimated_vram,
            estimated_ram_gb=estimated_ram,
            estimation_breakdown=summary,
            actual_peak_vram_allocated_gb=actual_peaks["peak_vram_allocated_gb"],
            actual_peak_vram_reserved_gb=actual_vram,
            actual_peak_ram_gb=actual_ram,
            snapshots=self.profiler.snapshots,
            vram_accuracy_pct=vram_accuracy,
            ram_accuracy_pct=ram_accuracy,
            vram_error_gb=vram_error,
            ram_error_gb=ram_error,
            test_duration_sec=duration,
            success=success,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
        )
        
        # Save result
        self._save_result(result)
        
        return result
    
    def _run_training_with_profiling(
        self,
        config: TestConfiguration,
        training_steps: int,
        profile_interval_sec: float,
    ) -> Dict[str, float]:
        """Run actual training with memory profiling.
        
        This is the core function that runs real training to measure memory.
        """
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        
        # Create a TRULY VANILLA model with NO automatic optimizations
        # PyTorch 2.0+ uses Flash Attention/SDPA by default in TransformerEncoderLayer
        # We need to disable this to get true baseline memory measurements!
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                
                # Build layers manually to avoid automatic optimizations
                self.layers = nn.ModuleList([
                    self._make_vanilla_layer(hidden_size, num_heads)
                    for _ in range(num_layers)
                ])
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def _make_vanilla_layer(self, hidden_size, num_heads):
                """Create a vanilla transformer layer without Flash Attention."""
                layer = nn.ModuleDict({
                    'attn_q': nn.Linear(hidden_size, hidden_size),
                    'attn_k': nn.Linear(hidden_size, hidden_size),
                    'attn_v': nn.Linear(hidden_size, hidden_size),
                    'attn_out': nn.Linear(hidden_size, hidden_size),
                    'norm1': nn.LayerNorm(hidden_size),
                    'ffn1': nn.Linear(hidden_size, hidden_size * 4),
                    'ffn2': nn.Linear(hidden_size * 4, hidden_size),
                    'norm2': nn.LayerNorm(hidden_size),
                })
                return layer
            
            def forward(self, input_ids):
                x = self.embed(input_ids)
                
                for layer in self.layers:
                    # Vanilla attention (NO Flash Attention, NO SDPA)
                    # This uses standard O(n²) attention
                    residual = x
                    x = layer['norm1'](x)
                    
                    batch, seq_len, hidden = x.shape
                    head_dim = self.hidden_size // self.num_heads
                    
                    # Compute Q, K, V
                    q = layer['attn_q'](x).view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
                    k = layer['attn_k'](x).view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
                    v = layer['attn_v'](x).view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
                    
                    # Vanilla attention calculation - O(n²) memory!
                    # This creates the full [batch, heads, seq, seq] attention matrix
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                    attn_weights = torch.softmax(scores, dim=-1)
                    attn_output = torch.matmul(attn_weights, v)
                    
                    # Reshape and project
                    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
                    x = layer['attn_out'](attn_output)
                    x = x + residual
                    
                    # FFN
                    residual = x
                    x = layer['norm2'](x)
                    x = layer['ffn2'](torch.relu(layer['ffn1'](x)))
                    x = x + residual
                
                return self.lm_head(x)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Creating model with {config.total_params:,} parameters...")
        model = SimpleTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        ).to(device)
        
        # Take initial snapshot
        self.profiler.snapshot()
        
        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # Snapshot after optimizer creation
        self.profiler.snapshot()
        
        # Mixed precision setup
        if config.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        print(f"Running {training_steps} training steps...")
        
        last_snapshot_time = time.time()
        
        for step in range(training_steps):
            # Create dummy batch
            batch = torch.randint(
                0, config.vocab_size,
                (config.batch_size, config.seq_len),
                device=device,
            )
            labels = batch.clone()
            
            # Forward pass
            if config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss = nn.functional.cross_entropy(
                        outputs.view(-1, config.vocab_size),
                        labels.view(-1),
                    )
            else:
                outputs = model(batch)
                loss = nn.functional.cross_entropy(
                    outputs.view(-1, config.vocab_size),
                    labels.view(-1),
                )
            
            # Take snapshot during forward pass
            current_time = time.time()
            if current_time - last_snapshot_time >= profile_interval_sec:
                self.profiler.snapshot()
                last_snapshot_time = current_time
            
            # Backward pass
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Take snapshot after backward
            current_time = time.time()
            if current_time - last_snapshot_time >= profile_interval_sec:
                self.profiler.snapshot()
                last_snapshot_time = current_time
            
            print(f"  Step {step+1}/{training_steps} - Loss: {loss.item():.4f}")
        
        # Final snapshot
        self.profiler.snapshot()
        
        # Get peak stats
        peaks = self.profiler.get_peaks()
        
        # Cleanup
        del model
        del optimizer
        if config.use_amp:
            del scaler
        torch.cuda.empty_cache()
        gc.collect()
        
        return peaks
    
    def _save_result(self, result: TestResult):
        """Save test result to file."""
        # Save individual result
        result_file = self.results_dir / f"{result.config.test_id}.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"✅ Result saved to: {result_file}")
        
        # Append to combined results
        combined_file = self.results_dir / "all_results.jsonl"
        with open(combined_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
    
    def load_results(self) -> List[TestResult]:
        """Load all test results."""
        results = []
        
        combined_file = self.results_dir / "all_results.jsonl"
        if combined_file.exists():
            with open(combined_file) as f:
                for line in f:
                    data = json.loads(line)
                    results.append(TestResult.from_dict(data))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all test results."""
        results = self.load_results()
        
        if not results:
            return {"message": "No test results found"}
        
        # Filter successful tests
        successful = [r for r in results if r.success]
        
        if not successful:
            return {"message": "No successful test results found"}
        
        # Calculate statistics
        vram_accuracies = [r.vram_accuracy_pct for r in successful]
        ram_accuracies = [r.ram_accuracy_pct for r in successful]
        vram_errors = [r.vram_error_gb for r in successful]
        ram_errors = [r.ram_error_gb for r in successful]
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful),
            "failed_tests": len(results) - len(successful),
            "vram_accuracy": {
                "mean": sum(vram_accuracies) / len(vram_accuracies),
                "min": min(vram_accuracies),
                "max": max(vram_accuracies),
                "std": self._std(vram_accuracies),
            },
            "ram_accuracy": {
                "mean": sum(ram_accuracies) / len(ram_accuracies),
                "min": min(ram_accuracies),
                "max": max(ram_accuracies),
                "std": self._std(ram_accuracies),
            },
            "vram_error_gb": {
                "mean": sum(vram_errors) / len(vram_errors),
                "min": min(vram_errors),
                "max": max(vram_errors),
                "std": self._std(vram_errors),
            },
            "ram_error_gb": {
                "mean": sum(ram_errors) / len(ram_errors),
                "min": min(ram_errors),
                "max": max(ram_errors),
                "std": self._std(ram_errors),
            },
        }
    
    @staticmethod
    def _std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


if __name__ == "__main__":
    # Quick test
    harness = MemoryTestHarness()
    
    config = TestConfiguration(
        model_name="test-tiny",
        tokenizer_name="gpt2",
        total_params=1_000_000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        vocab_size=50257,
        seq_len=512,
        batch_size=2,
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
        test_id="quick_test",
        test_name="Quick Test",
        description="Quick test of memory profiling",
    )
    
    result = harness.run_test(config, training_steps=5)
    
    stats = harness.get_statistics()
    print("\n" + "="*80)
    print("Overall Statistics:")
    print("="*80)
    print(json.dumps(stats, indent=2))
