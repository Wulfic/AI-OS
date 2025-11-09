"""Data models for progressive optimization system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, List
import uuid


@dataclass
class LoRAConfig:
    """Configuration for a specific LoRA/PEFT setup.
    
    Based on parameter impact analysis summarized in the canonical guide:
    docs/guide/features/LORA_PEFT_COMPREHENSIVE_ANALYSIS.md
    """
    enabled: bool = False
    method: str = "lora"  # "lora", "adalora", "ia3"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "q_proj,v_proj"
    
    @property
    def estimated_params(self) -> int:
        """Estimate trainable parameters based on config.
        
        Returns parameter counts following patterns described in the canonical
        LoRA/PEFT guide (see Parameter Impact and Target Modules Explained).
        """
        if not self.enabled:
            return 0
        
        if self.method == "ia3":
            return 100_000  # ~100K for IA3 minimal
        
        # LoRA/AdaLoRA parameter estimates
        params_map = {
            ("minimal", 4): 250_000,
            ("minimal", 8): 500_000,
            ("minimal", 16): 1_000_000,
            ("minimal", 32): 2_000_000,
            ("minimal", 64): 4_000_000,
            ("balanced", 16): 2_000_000,
            ("balanced", 32): 4_000_000,
            ("balanced", 64): 8_000_000,
            ("full", 16): 6_000_000,
            ("full", 32): 12_000_000,
            ("full", 64): 24_000_000,
        }
        
        # Determine module set
        modules = set(self.target_modules.split(","))
        if modules <= {"q_proj", "v_proj"}:
            module_set = "minimal"
        elif modules <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
            module_set = "balanced"
        else:
            module_set = "full"
        
        return params_map.get((module_set, self.rank), 1_000_000)
    
    @property
    def estimated_vram_overhead_gb(self) -> float:
        """Estimate VRAM overhead (see canonical LoRA/PEFT guide)."""
        if not self.enabled:
            return 0.0
        
        if self.method == "ia3":
            return 1.0  # IA3 has minimal overhead
        
        vram_map = {
            4: 1.0,
            8: 1.5,
            16: 2.5,
            32: 4.0,
            64: 7.0,
        }
        return vram_map.get(self.rank, 2.5)
    
    @property
    def expected_quality_percent(self) -> float:
        """Expected quality percentage (see canonical LoRA/PEFT guide)."""
        if not self.enabled:
            return 100.0  # Full fine-tuning baseline
        
        if self.method == "ia3":
            return 87.5  # 85-90%
        elif self.method == "adalora":
            return 98.75  # 98-99.5%
        
        # LoRA quality estimates based on rank and modules
        if self.rank >= 64:
            return 99.25  # 99%+
        elif self.rank >= 32:
            if "gate_proj" in self.target_modules:
                return 99.0  # Full modules
            else:
                return 98.75  # Balanced
        elif self.rank >= 16:
            if "o_proj" in self.target_modules:
                return 98.0  # Balanced
            else:
                return 96.5  # Minimal
        else:  # r=8
            return 93.5  # 92-95%
    
    def to_cli_args(self) -> List[str]:
        """Convert to CLI arguments."""
        if not self.enabled:
            return ["--no-peft"]
        
        args = [
            "--use-peft",
            "--peft-method", self.method,
        ]
        
        if self.method in ["lora", "adalora"]:
            args.extend([
                "--lora-r", str(self.rank),
                "--lora-alpha", str(self.alpha),
                "--lora-dropout", str(self.dropout),
            ])
        
        args.extend(["--lora-target-modules", self.target_modules])
        
        return args
    
    def __str__(self) -> str:
        """Human-readable description."""
        if not self.enabled:
            return "No PEFT"
        
        if self.method == "ia3":
            modules_count = len(self.target_modules.split(","))
            return f"IA3 ({modules_count} modules)"
        
        # Determine module shorthand
        modules = self.target_modules.split(",")
        if len(modules) <= 2:
            modules_short = "minimal"
        elif len(modules) <= 4:
            modules_short = "balanced"
        else:
            modules_short = "full"
        
        return f"{self.method.upper()} r={self.rank} α={self.alpha} {modules_short}"


@dataclass
class OptimizationLevel:
    """Represents a specific combination of optimizations to test."""
    
    name: str
    gradient_checkpointing: bool = True  # Always enabled as baseline
    amp: bool = False
    flashattn2: bool = False
    lora_config: Optional[LoRAConfig] = None  # LoRA configuration (replaces bool lora)
    cpu_offload: bool = False
    zero_stage: str = "none"  # "none", "zero1", "zero2", "zero3"
    chunk_size: Optional[int] = None  # Chunk size for long sequences (None = auto)
    
    @property
    def lora(self) -> bool:
        """Backward compatibility - returns True if LoRA is enabled."""
        return self.lora_config is not None and self.lora_config.enabled
    
    def to_cli_args(self) -> List[str]:
        """Convert this optimization level to CLI arguments."""
        args = []
        
        # Gradient checkpointing (always enabled)
        if self.gradient_checkpointing:
            args.append("--gradient-checkpointing")
        else:
            args.append("--no-gradient-checkpointing")
        
        # AMP
        if self.amp:
            args.append("--amp")
        else:
            args.append("--no-amp")
        
        # FlashAttention-2 - Note: Not a CLI flag, handled automatically by the model
        # This is kept for tracking optimization levels but doesn't generate CLI args
        
        # LoRA (via PEFT) - NOW CONFIGURABLE
        if self.lora_config:
            args.extend(self.lora_config.to_cli_args())
        else:
            args.append("--no-peft")
        
        # CPU Offload
        if self.cpu_offload:
            args.append("--cpu-offload")
        else:
            args.append("--no-cpu-offload")
        
        # DeepSpeed ZeRO
        if self.zero_stage != "none":
            args.extend(["--zero-stage", self.zero_stage])
        
        # Chunk size (for long sequences)
        if self.chunk_size is not None:
            args.extend(["--chunk-size", str(self.chunk_size)])
        
        return args
    
    def __str__(self) -> str:
        """Human-readable description of this optimization level."""
        parts = []
        if self.gradient_checkpointing:
            parts.append("GradCP")
        if self.amp:
            parts.append("AMP")
        if self.lora_config and self.lora_config.enabled:
            parts.append(str(self.lora_config))
        if self.cpu_offload:
            parts.append("CPUOff")
        if self.zero_stage != "none":
            parts.append(self.zero_stage.upper())
        if self.chunk_size is not None:
            parts.append(f"chunk={self.chunk_size}")
        
        return " + ".join(parts) if parts else "baseline"


@dataclass
class BatchTestResult:
    """Result of testing a specific batch size with an optimization level."""
    
    batch_size: int
    success: bool
    is_oom: bool
    memory_percent: float
    throughput: float
    duration: float
    exit_code: int
    error_message: str = ""
    
    @property
    def has_memory_headroom(self) -> bool:
        """True if there's room to increase batch size."""
        return self.memory_percent < 95.0


@dataclass
class OptimizationConfig:
    """Configuration for progressive optimization."""
    
    # Model and training data
    model: str
    dataset_file: str
    max_seq_len: int = 512
    train_steps: int = 10
    
    # GPU configuration
    cuda_devices: str = ""
    use_multi_gpu: bool = False
    device: str = "auto"
    
    # Test configuration
    test_duration: int = 180  # seconds per test (3 min for large context + streaming)
    max_timeout: int = 300   # max time to wait for subprocess
    
    # Batch size limits
    min_batch_size: int = 1
    max_batch_size: int = 128
    
    # Callbacks
    log_callback: Optional[Callable[[str], None]] = None
    stop_callback: Optional[Callable[[], bool]] = None
    
    # Output
    output_dir: str = "artifacts/optimization"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
