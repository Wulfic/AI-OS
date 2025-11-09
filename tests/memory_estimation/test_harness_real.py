"""Real HRM Training Test Harness.

This test harness runs ACTUAL train_actv1_impl() calls to measure real VRAM/RAM usage
during HRM training. Unlike test_harness.py which uses synthetic SimpleTransformer models,
this harness captures realistic memory usage including:
- Dataset loading and tokenization overhead
- Real ACT V1 HRM architecture (H/L layers, MoE, carry states, halt heads)
- Optimizer states
- Training data buffers
- CUDA context overhead
- All actual training optimizations

Usage:
    python test_harness_real.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aios.core.hrm_training.training_config import TrainingConfig
from aios.cli.hrm_hf.train_actv1 import train_actv1_impl


# ============================================================================
# Real Test Configuration
# ============================================================================

@dataclass
class RealTestConfig:
    """Configuration for a real HRM training test.
    
    This is a simplified config focused on the parameters that affect VRAM usage.
    All other training parameters are set to minimal/safe defaults.
    """
    # Model identification
    model_name: str  # e.g., "gpt2", "mistral-7b-v0.3", "qwen2.5-7b"
    
    # Architecture (ACT V1 HRM specific)
    h_layers: int  # Number of high-level layers
    l_layers: int  # Number of low-level layers
    hidden_size: int  # Hidden dimension (affects params: 512→~50M, 768→~87M)
    num_heads: int  # Attention heads (must divide hidden_size evenly)
    
    # Context and batch
    context_size: int  # max_seq_len (tokens)
    batch_size: int = 1  # Usually 1 for testing
    
    # Training steps
    steps: int = 10  # Enough steps to get stable VRAM measurements
    halt_max_steps: int = 10  # Max steps per sample
    eval_batches: int = 1  # Minimal eval
    
    # Optimization flags
    use_moe: bool = True  # Mixture of Experts
    num_experts: int = 8  # Number of MoE experts
    num_experts_per_tok: int = 2  # Active experts per token
    gradient_checkpointing: bool = True  # Trade compute for memory
    use_amp: bool = True  # Automatic Mixed Precision (FP16)
    use_flash_attention_2: bool = False  # Flash Attention 2 (faster attention)
    use_8bit_optimizer: bool = False  # 8-bit optimizer (bitsandbytes)
    cpu_offload: bool = False  # CPU offload for optimizer states
    context_chunking: bool = False  # Process context in chunks
    chunk_size: Optional[int] = None  # Chunk size if chunking enabled
    deepspeed_stage: Optional[int] = None  # DeepSpeed ZeRO stage (1, 2, or 3)
    use_lora: bool = False  # LoRA/PEFT adapter
    lora_rank: int = 8  # LoRA rank (r)
    lora_alpha: int = 16  # LoRA alpha (scaling)
    lora_dropout: float = 0.1  # LoRA dropout
    lora_target_modules: str = "q_proj,v_proj"  # Comma-separated target modules
    
    # Hardware
    device: str = "cuda:1"  # GPU 1 for testing (GPU 0 has OS overhead)
    
    # Data
    dataset_file: str = "training_data/curated_datasets/test_sample.txt"
    
    # Model cache (using project logs directory)
    # Get project root dynamically
    _project_root = Path(__file__).parent.parent.parent
    model_cache_dir: str = str(_project_root / "logs" / "memory_tests" / "model_cache")
    temp_dir_base: str = str(_project_root / "logs" / "memory_tests" / "temp")
    
    def get_dataset_file(self) -> str:
        """Get appropriate dataset file for the tokenizer.
        
        Different tokenizers may pre-tokenize the same text differently,
        so we use raw text and let each tokenizer process it fresh.
        """
        return self.dataset_file
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# Model Cache Management
# ============================================================================

def get_or_create_cached_model(test_config: RealTestConfig) -> str:
    """Get or create a cached model configuration for this tokenizer and architecture.
    
    Each tokenizer has a different vocab_size, which requires a different model.
    We cache these models to avoid recreating them for every test.
    
    Cache key is: tokenizer_name + h_layers + l_layers + hidden_size + use_moe
    
    Args:
        test_config: Test configuration
        
    Returns:
        Path to the cached model directory
    """
    # Create cache directory if needed
    cache_dir = Path(test_config.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key from model parameters
    # Sanitize tokenizer name for filesystem
    tokenizer_safe_name = test_config.model_name.replace("/", "_").replace(".", "_")
    moe_suffix = f"_moe{test_config.num_experts}x{test_config.num_experts_per_tok}" if test_config.use_moe else "_nomoe"
    cache_key = f"{tokenizer_safe_name}_h{test_config.h_layers}l{test_config.l_layers}_hid{test_config.hidden_size}{moe_suffix}"
    
    model_cache_path = cache_dir / cache_key
    
    # If cached model exists, return it
    if model_cache_path.exists():
        print(f"[ModelCache] Using cached model: {cache_key}")
        return str(model_cache_path)
    
    # Otherwise, create a new model
    print(f"[ModelCache] Creating new model: {cache_key}")
    
    # Load tokenizer to get vocab_size
    try:
        # Try loading from local cache first to avoid network issues
        print(f"[ModelCache] Loading tokenizer: {test_config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            test_config.model_name, 
            use_fast=True,
            local_files_only=False  # Allow network fallback
        )
        vocab_size = len(tokenizer)
        print(f"[ModelCache] Tokenizer vocab_size: {vocab_size}")
    except Exception as e:
        print(f"[ModelCache] ERROR loading tokenizer {test_config.model_name}: {e}")
        raise
    
    # Create model config directory
    model_cache_path.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer to model directory
    tokenizer.save_pretrained(str(model_cache_path))
    
    # Create config.json for HRM model
    model_config = {
        "model_type": "hrm_actv1",
        "architectures": ["HierarchicalReasoningModel_ACTV1"],
        "vocab_size": vocab_size,
        "h_layers": test_config.h_layers,
        "l_layers": test_config.l_layers,
        "hidden_size": test_config.hidden_size,
        "num_heads": test_config.num_heads,
        "expansion": 2.0,
        "h_cycles": 2,
        "l_cycles": 2,
        "use_moe": test_config.use_moe,
        "num_experts": test_config.num_experts if test_config.use_moe else 0,
        "num_experts_per_tok": test_config.num_experts_per_tok if test_config.use_moe else 0,
        "tokenizer_name": test_config.model_name,
        "max_seq_len": 4096,
    }
    
    config_path = model_cache_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"[ModelCache] Saved config to: {config_path}")
    
    return str(model_cache_path)


# ============================================================================
# Training Config Builder
# ============================================================================

def create_training_config(test_config: RealTestConfig) -> tuple[TrainingConfig, str]:
    """Build a TrainingConfig from a RealTestConfig.
    
    This maps our simplified test config to the full ~100 parameter TrainingConfig
    used by actual HRM training.
    
    Args:
        test_config: Simplified test configuration
        
    Returns:
        Tuple of (TrainingConfig, temp_dir_path) - temp_dir should be cleaned up after use
    """
    # Get or create cached model for this tokenizer/architecture
    model_path = get_or_create_cached_model(test_config)
    
    # Create temporary directory for training artifacts on Z drive
    # This prevents filling up C drive with test data
    import os
    temp_base = Path(test_config.temp_dir_base)
    temp_base.mkdir(parents=True, exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="hrm_test_", dir=str(temp_base))
    save_dir = str(Path(temp_dir) / "checkpoints")
    log_file = str(Path(temp_dir) / "metrics.jsonl")
    
    # Build full training config with minimal safe defaults
    config = TrainingConfig(
        # Model and data
        model=model_path,  # Use cached model path, not tokenizer name
        dataset_file=test_config.dataset_file,
        dataset_chunk_size=4000,  # Default
        max_seq_len=test_config.context_size,
        batch_size=test_config.batch_size,
        steps=test_config.steps,
        
        # Optimization
        lr=2e-4,  # Default, will be auto-adjusted for MoE
        halt_max_steps=test_config.halt_max_steps,
        gradient_checkpointing=test_config.gradient_checkpointing,
        use_amp=test_config.use_amp,
        use_cpu_offload=test_config.cpu_offload,
        use_8bit_optimizer=test_config.use_8bit_optimizer,
        use_chunked_training=test_config.context_chunking,
        chunk_size=test_config.chunk_size if test_config.chunk_size else 2048,
        sys_mem_cap_pct=None,
        
        # Architecture
        h_layers=test_config.h_layers,
        l_layers=test_config.l_layers,
        hidden_size=test_config.hidden_size,
        expansion=2.0,  # Default FFN expansion
        num_heads=test_config.num_heads,
        h_cycles=2,  # Default
        l_cycles=2,  # Default
        pos_encodings="rope",  # Default, recommended
        use_flash_attn=test_config.use_flash_attention_2,
        window_size=None,  # Full attention
        
        # MoE
        use_moe=test_config.use_moe,
        num_experts=test_config.num_experts,
        num_experts_per_tok=test_config.num_experts_per_tok,
        moe_capacity_factor=1.25,
        auto_adjust_moe_lr=True,
        
        # Device and distributed
        device=test_config.device,
        cuda_ids=test_config.device.split(":")[-1] if ":" in test_config.device else None,  # Extract device ID
        ddp=False,  # No distributed
        world_size=None,
        strict=False,  # Allow fallbacks
        parallel_independent=False,
        
        # IO
        save_dir=save_dir,
        bundle_dir=temp_dir,
        log_file=log_file,
        stop_file=None,
        eval_file=None,
        eval_batches=test_config.eval_batches,
        student_init=None,
        brain_name=None,
        default_goal=None,
        expert_id=None,
        
        # Advanced
        iterate=False,  # Single epoch
        stop_after_epoch=False,
        resume=False,
        linear_dataset=True,
        dataset_start_offset=0,
        optimize=False,  # CRITICAL: No auto-optimization
        ascii_only=False,
        
        # DeepSpeed ZeRO
        zero_stage=f"zero{test_config.deepspeed_stage}" if test_config.deepspeed_stage else "none",
        
        # PEFT (LoRA)
        use_peft=test_config.use_lora,
        peft_method="lora",
        lora_r=test_config.lora_rank,
        lora_alpha=test_config.lora_alpha,
        lora_dropout=test_config.lora_dropout,
        lora_target_modules=test_config.lora_target_modules,
        
        # Model precision
        model_dtype="fp32",  # Full precision model
        load_in_8bit=False,
        load_in_4bit=False,
        
        # Multi-GPU inference
        inference_device=None,
        hot_reload_steps=0,
        
        # Block streaming
        samples_per_block=100000,
    )
    
    return config, temp_dir


# ============================================================================
# Memory Measurement
# ============================================================================

class MemoryTracker:
    """Track CUDA memory usage at different training stages."""
    
    def __init__(self, device: str = "cuda:1"):
        # Initialize CUDA and validate device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
            
        # Parse device index
        if device.startswith("cuda:"):
            device_idx = int(device.split(":")[1])
        else:
            device_idx = 0
            
        # Validate device index
        if device_idx >= torch.cuda.device_count():
            # Prefer GPU 1 if available (GPU 0 has OS overhead), else use GPU 0
            fallback_idx = 1 if torch.cuda.device_count() > 1 else 0
            print(f"Warning: Device {device} not available. Using cuda:{fallback_idx} instead.")
            device_idx = fallback_idx
            device = f"cuda:{fallback_idx}"
        
        # Set as current device
        torch.cuda.set_device(device_idx)
            
        self.device = torch.device(device)
        self.device_id = device_idx
        self.measurements: dict[str, float] = {}
        
    def reset(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.empty_cache()
            
    def measure(self, stage: str) -> float:
        """Measure current VRAM usage and store it.
        
        Args:
            stage: Name of the training stage (e.g., "baseline", "model_loaded")
            
        Returns:
            Current allocated memory in bytes
        """
        if not torch.cuda.is_available():
            return 0.0
        
        # Force synchronization to ensure all operations completed
        torch.cuda.synchronize(self.device_id)
        
        # Get current allocated memory
        allocated = torch.cuda.memory_allocated(self.device_id)
        peak = torch.cuda.max_memory_allocated(self.device_id)
        
        self.measurements[stage] = allocated
        self.measurements[f"{stage}_peak"] = peak
        
        return allocated
        
    def get_peak(self) -> float:
        """Get peak memory usage across all stages.
        
        Returns:
            Peak allocated memory in bytes
        """
        if not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.max_memory_allocated(self.device_id)
        
    def to_dict(self) -> dict:
        """Convert measurements to dictionary."""
        return self.measurements.copy()


# ============================================================================
# Test Execution
# ============================================================================

@dataclass
class RealTestResult:
    """Results from a real HRM training test."""
    # Test config
    config: dict
    
    # Actual memory usage (bytes)
    actual_vram_bytes: float
    actual_ram_bytes: float
    
    # Memory breakdown (bytes)
    vram_baseline: float  # CUDA initialized, no model
    vram_model_loaded: float  # Model created, no optimizer
    vram_optimizer_created: float  # Optimizer added
    vram_data_loaded: float  # Dataset loaded
    vram_training_peak: float  # Peak during training
    
    # Estimated memory (from vram_estimation.py)
    estimated_vram_bytes: float
    estimated_ram_bytes: float
    
    # Accuracy metrics
    vram_accuracy: float  # 1.0 - abs(estimated - actual) / actual
    ram_accuracy: float
    
    # Execution metadata
    test_duration_seconds: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


def run_real_training_test(
    test_config: RealTestConfig,
    verbose: bool = True,
) -> RealTestResult:
    """Run a real HRM training test and measure memory usage.
    
    This is the core test function that:
    1. Builds TrainingConfig from test config
    2. Measures memory at key stages
    3. Runs actual train_actv1_impl()
    4. Collects memory measurements
    5. Compares against estimator predictions
    
    Args:
        test_config: Test configuration
        verbose: Print progress messages
        
    Returns:
        Test result with actual vs estimated memory
    """
    import datetime
    import signal
    
    start_time = time.time()
    timestamp = datetime.datetime.now().isoformat()
    temp_dir = None  # Initialize for cleanup
    
    # Handler for graceful interrupt
    def signal_handler(signum, frame):
        if verbose:
            print(f"\n[!] Received interrupt signal, cleaning up...")
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise KeyboardInterrupt("Test interrupted by signal")
    
    # Register signal handler (on Windows, only SIGINT is available)
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Set minimal logging to reduce file I/O interrupts during testing
        os.environ["AIOS_MINIMAL_LOGGING"] = "1"
        
        # Initialize memory tracker
        tracker = MemoryTracker(device=test_config.device)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running real training test:")
            print(f"  Model: {test_config.model_name}")
            print(f"  Architecture: {test_config.h_layers}h/{test_config.l_layers}l, "
                  f"hidden={test_config.hidden_size}, heads={test_config.num_heads}")
            print(f"  Context: {test_config.context_size}, Batch: {test_config.batch_size}")
            print(f"  MoE: {test_config.use_moe} ({test_config.num_experts} experts, "
                  f"{test_config.num_experts_per_tok} active)")
            print(f"{'='*80}")
        
        # Stage 1: Baseline (CUDA initialized, no model)
        tracker.reset()
        baseline = tracker.measure("baseline")
        if verbose:
            print(f"[Stage 1/5] Baseline CUDA memory: {baseline / 1024**2:.1f} MB")
        
        # Stage 2: Create training config
        training_config, temp_dir = create_training_config(test_config)
        if verbose:
            print(f"[Stage 2/5] Created training config")
        
        # Stage 3: Run actual training
        # NOTE: train_actv1_impl() will:
        # - Load tokenizer
        # - Load dataset
        # - Build model
        # - Create optimizer
        # - Run training step
        # We measure peak memory after this completes
        if verbose:
            print(f"[Stage 3/5] Starting actual HRM training...")
        
        # Run training (will take 10-60 seconds depending on model size)
        train_actv1_impl(training_config)
        
        # Stage 4: Measure peak memory
        training_peak = tracker.get_peak()
        if verbose:
            print(f"[Stage 4/5] Training completed")
            print(f"  Peak VRAM: {training_peak / 1024**3:.2f} GB")
        
        # Stage 5: Calculate actual memory usage
        # NOTE: We're NOT calling estimators here - just recording raw data
        # Once we have ALL test data, we'll analyze and update estimators
        actual_vram = training_peak - baseline  # Subtract baseline CUDA overhead
        # NOTE: RAM measurement deferred - would require psutil dependency and adds complexity
        # For VRAM-focused testing, this is acceptable; RAM is primarily system memory overhead
        actual_ram = 0.0
        
        duration = time.time() - start_time
        
        if verbose:
            print(f"[Stage 5/5] Test completed in {duration:.1f}s")
            print(f"  Actual VRAM: {actual_vram / 1024**3:.2f} GB")
        
        # Create result with raw measurements only
        # NOTE: Intermediate measurements (model loaded, optimizer created, data loaded) are
        # deferred to avoid complexity. Peak VRAM is the critical metric for sizing decisions.
        result = RealTestResult(
            config=test_config.to_dict(),
            actual_vram_bytes=actual_vram,
            actual_ram_bytes=actual_ram,
            vram_baseline=baseline,
            vram_model_loaded=0.0,
            vram_optimizer_created=0.0,
            vram_data_loaded=0.0,
            vram_training_peak=training_peak,
            estimated_vram_bytes=0.0,  # Not using estimator yet
            estimated_ram_bytes=0.0,  # Not using estimator yet
            vram_accuracy=0.0,  # Will calculate after estimator is updated
            ram_accuracy=0.0,  # Will calculate after estimator is updated
            test_duration_seconds=duration,
            timestamp=timestamp,
            success=True,
        )
        
        # Clean up temp directory to save disk space
        try:
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                if verbose:
                    print(f"[Cleanup] Removed temp directory: {temp_dir}")
        except Exception as cleanup_error:
            if verbose:
                print(f"[Cleanup] Warning: Failed to remove temp directory: {cleanup_error}")
        
        # Restore signal handler
        signal.signal(signal.SIGINT, old_handler)
        
        return result
        
    except KeyboardInterrupt:
        # Handle graceful interrupt
        duration = time.time() - start_time
        
        if verbose:
            print(f"\n[!] Test interrupted by user/system")
        
        # Clean up temp directory
        try:
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                if verbose:
                    print(f"[Cleanup] Removed temp directory: {temp_dir}")
        except:
            pass
        
        # Restore signal handler
        signal.signal(signal.SIGINT, old_handler)
        
        # Return interrupted result
        result = RealTestResult(
            config=test_config.to_dict(),
            actual_vram_bytes=0.0,
            actual_ram_bytes=0.0,
            vram_baseline=0.0,
            vram_model_loaded=0.0,
            vram_optimizer_created=0.0,
            vram_data_loaded=0.0,
            vram_training_peak=0.0,
            estimated_vram_bytes=0.0,
            estimated_ram_bytes=0.0,
            vram_accuracy=0.0,
            ram_accuracy=0.0,
            test_duration_seconds=duration,
            timestamp=timestamp,
            success=False,
            error_message="Test interrupted"
        )
        return result
        
    except Exception as e:
        import traceback
        duration = time.time() - start_time
        
        # Clean up temp directory even on failure
        try:
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        if verbose:
            print(f"\n[X] Test FAILED:")
            print(f"  Error: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()
        
        # Return error result
        result = RealTestResult(
            config=test_config.to_dict(),
            actual_vram_bytes=0.0,
            actual_ram_bytes=0.0,
            vram_baseline=0.0,
            vram_model_loaded=0.0,
            vram_optimizer_created=0.0,
            vram_data_loaded=0.0,
            vram_training_peak=0.0,
            estimated_vram_bytes=0.0,
            estimated_ram_bytes=0.0,
            vram_accuracy=0.0,
            ram_accuracy=0.0,
            test_duration_seconds=duration,
            timestamp=timestamp,
            success=False,
            error_message=str(e),
        )
        
        return result


# ============================================================================
# Main (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    # Quick test: tiny model, small context
    test_config = RealTestConfig(
        model_name="gpt2",
        h_layers=1,
        l_layers=1,
        hidden_size=256,
        num_heads=8,
        context_size=128,
        batch_size=1,
    )
    
    print("Running standalone test of real HRM training harness...")
    result = run_real_training_test(test_config, verbose=True)
    
    if result.success:
        print(f"\n✅ Test PASSED")
        print(f"Peak VRAM: {result.vram_training_peak / 1024**3:.2f} GB")
    else:
        print(f"\n❌ Test FAILED: {result.error_message}")
