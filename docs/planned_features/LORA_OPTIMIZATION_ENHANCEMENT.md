# LoRA Parameter Optimization Enhancement

**Date:** October 19, 2025  
**Status:** Planned Enhancement  
**Target:** Progressive Optimizer

---

## Problem Statement

The Progressive Optimizer currently tests LoRA ON/OFF but uses **hardcoded** parameters:
- `lora_r = 16` (rank)
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `lora_target_modules = "q_proj,v_proj"` (minimal)

This ignores the canonical guide **features/LORA_PEFT.md** which shows:
- Different ranks (4, 8, 16, 32, 64) have massive impact on quality (85-99.5%)
- Module combinations affect trainable params (1M to 24M)
- VRAM requirements vary significantly (1-7 GB overhead)

---

## Goal

Enhance Progressive Optimizer to intelligently test LoRA configurations based on:
1. **Available VRAM** - Test appropriate ranks and module combinations
2. **Dataset size** - Adjust dropout accordingly
3. **Task complexity** - Use quality/efficiency trade-offs from matrix
4. **Performance metrics** - Select based on throughput and quality

---

## Design

### 1. LoRA Configuration Dataclass

```python
@dataclass
class LoRAConfig:
    """Configuration for a specific LoRA/PEFT setup."""
    enabled: bool = False
    method: str = "lora"  # "lora", "adalora", "ia3"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "q_proj,v_proj"
    
    @property
    def estimated_params(self) -> int:
        """Estimate trainable parameters based on config."""
    # From canonical LoRA/PEFT guide decision rules
        params_map = {
            ("minimal", 4): 250_000,
            ("minimal", 8): 500_000,
            ("minimal", 16): 1_000_000,
            ("minimal", 32): 2_000_000,
            ("minimal", 64): 4_000_000,
            ("balanced", 16): 2_000_000,
            ("balanced", 32): 4_000_000,
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
        vram_map = {
            4: 1.0,
            8: 1.5,
            16: 2.5,
            32: 4.0,
            64: 7.0,
        }
        return vram_map.get(self.rank, 2.5)
    
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
            return f"IA3 {self.target_modules}"
        
        modules_short = "minimal" if "q_proj,v_proj" == self.target_modules else (
            "balanced" if len(self.target_modules.split(",")) <= 4 else "full"
        )
        return f"{self.method.upper()} r={self.rank} α={self.alpha} {modules_short}"
```

---

### 2. LoRA Configuration Factory

Based on the canonical **features/LORA_PEFT.md** decision rules:

```python
def create_lora_configs(
    available_vram_gb: float,
    dataset_size: Optional[int] = None,
    task_complexity: str = "medium"
) -> List[LoRAConfig]:
    """
    Create LoRA configurations to test based on available resources.
    
    Uses decision rules from the canonical LoRA/PEFT guide
    
    Args:
        available_vram_gb: Available VRAM in GB
        dataset_size: Number of samples in dataset (for dropout tuning)
        task_complexity: "simple", "medium", "complex", "very_complex"
    
    Returns:
        List of LoRAConfig objects to test
    """
    configs = []
    
    # Always test NO PEFT as baseline
    configs.append(LoRAConfig(enabled=False))
    
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
        dropout = 0.05  # Default
    
    # ====================================================================
    # VRAM-Based Configuration Selection
    # From canonical guide Quick Decision rules
    # ====================================================================
    
    if available_vram_gb < 8:
        # < 8 GB: IA3 Minimal or LoRA r=8 Minimal
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
        # 8-10 GB: LoRA r=8 or r=16 Minimal
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
        # 14-20 GB: LoRA r=16 Balanced and r=32 Balanced/Full
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
        # 20+ GB: LoRA r=32, r=64 Full, or consider full fine-tuning
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
        
        # For research: test AdaLoRA
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
```

---

### 3. Enhanced OptimizationLevel

Modify existing `OptimizationLevel` to include LoRA config:

```python
@dataclass
class OptimizationLevel:
    """Represents a specific combination of optimizations to test."""
    
    name: str
    gradient_checkpointing: bool = True
    amp: bool = False
    flashattn2: bool = False
    lora_config: Optional[LoRAConfig] = None  # ← NEW
    cpu_offload: bool = False
    zero_stage: str = "none"
    chunk_size: Optional[int] = None
    
    @property
    def lora(self) -> bool:
        """Backward compatibility."""
        return self.lora_config is not None and self.lora_config.enabled
    
    def to_cli_args(self) -> List[str]:
        """Convert this optimization level to CLI arguments."""
        args = []
        
        # ... existing args ...
        
        # LoRA (via PEFT) - NOW CONFIGURABLE
        if self.lora_config:
            args.extend(self.lora_config.to_cli_args())
        else:
            args.append("--no-peft")
        
        # ... rest of args ...
        
        return args
    
    def __str__(self) -> str:
        """Human-readable description."""
        parts = []
        if self.gradient_checkpointing:
            parts.append("GradCP")
        if self.amp:
            parts.append("AMP")
        if self.lora_config:
            parts.append(str(self.lora_config))
        # ... rest ...
        
        return " + ".join(parts) if parts else "baseline"
```

---

### 4. Updated Level Factory

```python
def create_optimization_levels(
    config: "OptimizationConfig",
    log_func=None
) -> List["OptimizationLevel"]:
    """Create the progression of optimization levels to test."""
    
    # Estimate available VRAM (simplified - improve with actual detection)
    available_vram_gb = estimate_available_vram()
    
    # Get LoRA configurations to test
    lora_configs = create_lora_configs(
        available_vram_gb=available_vram_gb,
        dataset_size=None,  # Could parse from dataset file
        task_complexity="medium"  # Could be a config parameter
    )
    
    levels = []
    
    # Level 1: Baseline (no PEFT)
    base_config = {
        "name": "Baseline (No PEFT)",
        "gradient_checkpointing": True,
        "amp": True,
        "lora_config": LoRAConfig(enabled=False),
        "cpu_offload": False,
        "zero_stage": "none"
    }
    levels.append(OptimizationLevel(**base_config))
    
    # Level 2-N: Test each LoRA configuration
    for idx, lora_cfg in enumerate(lora_configs):
        if not lora_cfg.enabled:
            continue  # Skip disabled config (already tested as baseline)
        
        level_config = {
            "name": f"Level {len(levels) + 1}: GradCP + AMP + {lora_cfg}",
            "gradient_checkpointing": True,
            "amp": True,
            "lora_config": lora_cfg,
            "cpu_offload": False,
            "zero_stage": "none"
        }
        levels.append(OptimizationLevel(**level_config))
    
    # Additional levels with CPU offload and ZeRO stages
    # (only for the best performing LoRA config if needed)
    
    return levels
```

---

### 5. Result Scoring

Enhanced scoring to consider quality estimates from matrix:

```python
def score_optimization_result(
    level: OptimizationLevel,
    batch_size: int,
    throughput: float,
    memory_percent: float
) -> float:
    """
    Score an optimization result considering:
    - Throughput (steps/sec)
    - Memory efficiency
    - Expected quality (from canonical LoRA/PEFT guide)
    
    Returns:
        Score (higher is better)
    """
    
    # Base score from throughput
    throughput_score = throughput * 100
    
    # Memory efficiency bonus (prefer using GPU fully)
    if 85 <= memory_percent <= 95:
        memory_bonus = 50  # Sweet spot
    elif memory_percent < 85:
        memory_bonus = (memory_percent / 85) * 50  # Penalty for underutilization
    else:
        memory_bonus = 0  # Using too much memory
    
    # Quality estimate (see canonical LoRA/PEFT guide)
    quality_score = 0
    if level.lora_config and level.lora_config.enabled:
        # Quality estimates from matrix
        if level.lora_config.method == "ia3":
            quality_score = 87.5  # 85-90%
        elif level.lora_config.method == "adalora":
            quality_score = 98.75  # 98-99.5%
        else:  # LoRA
            # Based on rank and modules
            if level.lora_config.rank >= 64:
                quality_score = 99.25  # 99%+
            elif level.lora_config.rank >= 32:
                if "gate_proj" in level.lora_config.target_modules:
                    quality_score = 99.0  # Full modules
                else:
                    quality_score = 98.75  # Balanced
            elif level.lora_config.rank >= 16:
                if "o_proj" in level.lora_config.target_modules:
                    quality_score = 98.0  # Balanced
                else:
                    quality_score = 96.5  # Minimal
            else:  # r=8
                quality_score = 93.5  # 92-95%
    else:
        quality_score = 100  # Full fine-tuning baseline
    
    # Combine scores
    # Weight: 40% throughput, 20% memory, 40% quality
    total_score = (
        throughput_score * 0.4 +
        memory_bonus * 0.2 +
        quality_score * 0.4
    )
    
    return total_score
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (2-3 hours)
1. ✅ Create `LoRAConfig` dataclass
2. ✅ Create `create_lora_configs()` factory function
3. ✅ Update `OptimizationLevel` to include `lora_config`
4. ✅ Update `to_cli_args()` to use LoRA config

### Phase 2: Level Generation (1-2 hours)
1. ✅ Update `create_optimization_levels()` to generate LoRA variants
2. ✅ Add VRAM detection helper
3. ✅ Add dataset size estimation helper
4. ✅ Test level generation logic

### Phase 3: Scoring System (1 hour)
1. ✅ Implement `score_optimization_result()` function
2. ✅ Update result selection to use new scoring
3. ✅ Add quality estimates to result output

### Phase 4: Testing (2-3 hours)
1. ✅ Test with low VRAM scenario (< 8 GB)
2. ✅ Test with medium VRAM scenario (10-14 GB)
3. ✅ Test with high VRAM scenario (20+ GB)
4. ✅ Verify CLI arg generation
5. ✅ Verify result scoring

### Phase 5: Documentation (1 hour)
1. ✅ Update Progressive Optimizer docstrings
2. ✅ Add examples to features/LORA_PEFT.md
3. ✅ Update OPTIMIZER_SYSTEMS_EXPLAINED.md

---

## Expected Results

### Before Enhancement
```
Testing Level 3: GradCP + AMP + LoRA
- LoRA: r=16, alpha=32, dropout=0.05, modules=q_proj,v_proj
- Result: 95-98% quality
```

### After Enhancement
```
Testing Level 2: GradCP + AMP + LoRA r=8 minimal
- LoRA: r=8, alpha=16, dropout=0.05, modules=q_proj,v_proj
- Result: 92-95% quality, Score: 85.2

Testing Level 3: GradCP + AMP + LoRA r=16 minimal
- LoRA: r=16, alpha=32, dropout=0.05, modules=q_proj,v_proj
- Result: 95-98% quality, Score: 92.1

Testing Level 4: GradCP + AMP + LoRA r=16 balanced
- LoRA: r=16, alpha=32, dropout=0.05, modules=q_proj,k_proj,v_proj,o_proj
- Result: 97-99% quality, Score: 96.8 ⭐ BEST

Selected: Level 4 (highest score considering throughput, memory, and quality)
```

---

## Files to Modify

1. `src/aios/gui/components/hrm_training/optimizer_progressive/models.py`
   - Add `LoRAConfig` dataclass
   - Update `OptimizationLevel` to include `lora_config`

2. `src/aios/gui/components/hrm_training/optimizer_progressive/level_factory.py`
   - Add `create_lora_configs()` function
   - Update `create_optimization_levels()` to use LoRA configs
   - Add VRAM detection helper

3. `src/aios/gui/components/hrm_training/optimizer_progressive/optimizer.py`
   - Update result scoring to use new `score_optimization_result()`
   - Add quality estimates to output

4. `docs/guide/features/LORA_PEFT.md`
   - Add section on automatic optimization
   - Add examples of optimizer output

---

## Future Enhancements

### Adaptive Tuning
- Monitor loss during optimization
- If loss plateau, automatically try higher rank
- If training unstable, automatically reduce alpha

### Dataset Analysis
- Parse dataset file to estimate size
- Adjust dropout based on actual sample count
- Detect task complexity from dataset content

### Multi-Objective Optimization
- Allow user to prioritize: speed vs quality vs memory
- Use Pareto optimization to find trade-offs
- Generate multiple recommendations

---

## References

- **features/LORA_PEFT.md** - Parameter impact analysis and decision rules
- **OPTIMIZER_SYSTEMS_EXPLAINED.md** - Overview of all optimization systems
- `src/aios/gui/components/hrm_training/optimizer_progressive/` - Current implementation

---

**Status:** Ready for implementation  
**Estimated Time:** 7-11 hours total  
**Priority:** High (significantly improves training quality)
