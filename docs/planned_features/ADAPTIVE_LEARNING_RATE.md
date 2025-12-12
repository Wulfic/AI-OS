# Adaptive Learning Rate System

## Overview

A loss-reactive learning rate scheduler that dynamically adjusts the learning rate during training based on actual training dynamics. Unlike fixed schedules (cosine, step decay), this system monitors training progress and makes intelligent decisions to increase or decrease LR in real-time.

## Motivation

### Current State
The existing "Auto-adjust LR" feature is a **one-time pre-flight check** that clamps the learning rate for MoE models before training starts. It does not adapt during training.

### Problem
Users unfamiliar with deep learning must guess appropriate learning rates:
- **Too high** → Training diverges, loss explodes
- **Too low** → Training is painfully slow, may get stuck
- **Just right** → Requires expertise and experimentation

### Solution
A truly adaptive system that:
1. Starts with a reasonable default or user-specified LR
2. Monitors training loss over rolling windows
3. **Increases LR** when training is progressing well (learn faster)
4. **Decreases LR** when training stalls or becomes unstable (fine-tune)
5. Respects safety bounds, especially for MoE models

## Design

### Core Principle: Follow the Loss Gradient

```
Loss dropping steadily  →  Increase LR (we can learn faster!)
Loss plateauing         →  Decrease LR (need finer adjustments)
Loss spiking            →  Decrease LR significantly (stabilize)
Loss oscillating        →  Decrease LR (too aggressive)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AdaptiveLRScheduler                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Loss Monitor │───▶│   Decision   │───▶│  LR Adjuster │     │
│  │   (Window)   │    │    Engine    │    │   (Bounded)  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Rolling Avg  │    │  Thresholds  │    │ Safety Bounds│     │
│  │  Variance    │    │   Patience   │    │  Rate Limit  │     │
│  │   Trend      │    │   Cooldown   │    │  MoE Clamps  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Processing

#### 1. Rolling Loss Window
```python
# Collect losses over a window
window_size = 50  # steps
loss_window = deque(maxlen=window_size)

# Compute statistics
current_avg = mean(loss_window)
current_std = std(loss_window)
previous_avg = mean(previous_window)  # From last evaluation
```

#### 2. Derived Metrics
```python
# Improvement: How much did loss drop?
improvement = previous_avg - current_avg  # Positive = good

# Improvement rate: Normalized by previous loss
improvement_rate = improvement / previous_avg  # Percentage

# Stability: How consistent is training?
coefficient_of_variation = current_std / current_avg  # Lower = more stable

# Trend: Direction over the window (linear regression slope)
trend = compute_slope(loss_window)  # Negative = improving
```

### Decision Logic

```python
class LRDecision(Enum):
    INCREASE = "increase"   # Training going well, speed up
    DECREASE = "decrease"   # Training struggling, slow down
    HOLD = "hold"           # Maintain current LR
    
def make_decision(metrics: LossMetrics, config: AdaptiveConfig) -> LRDecision:
    # Safety: Don't adjust during cooldown
    if steps_since_last_adjustment < config.cooldown_steps:
        return LRDecision.HOLD
    
    # INCREASE: Consistent improvement + stable training
    if (metrics.improvement_rate > config.increase_threshold and
        metrics.cv < config.stability_threshold and
        metrics.trend < 0):  # Still improving
        return LRDecision.INCREASE
    
    # DECREASE (Spike): Sudden loss increase
    if metrics.current_avg > metrics.previous_avg * config.spike_threshold:
        return LRDecision.DECREASE
    
    # DECREASE (Plateau): No improvement for too long
    if (metrics.improvement_rate < config.plateau_threshold and
        patience_counter >= config.patience):
        return LRDecision.DECREASE
    
    # DECREASE (Unstable): High variance in loss
    if metrics.cv > config.instability_threshold:
        return LRDecision.DECREASE
    
    return LRDecision.HOLD
```

### LR Adjustment Strategy

#### Multiplicative Updates (Preferred)
```python
# Increase: Multiply by factor > 1
new_lr = current_lr * increase_factor  # e.g., 1.1 (+10%)

# Decrease: Multiply by factor < 1  
new_lr = current_lr * decrease_factor  # e.g., 0.8 (-20%)
```

#### Asymmetric Factors
Decreases should be **more aggressive** than increases to prevent runaway divergence:
```python
increase_factor = 1.1   # +10% when things are good
decrease_factor = 0.7   # -30% when things are bad (faster recovery)
spike_factor = 0.5      # -50% on loss spike (emergency brake)
```

### Safety Mechanisms

#### 1. Absolute Bounds
```python
lr_min = 1e-7   # Floor: never go below this
lr_max = 1e-2   # Ceiling: never go above this

new_lr = max(lr_min, min(lr_max, new_lr))
```

#### 2. MoE-Specific Bounds
MoE router networks are sensitive to LR changes:
```python
if config.use_moe:
    lr_min = max(lr_min, 1e-5)    # Higher floor for MoE
    lr_max = min(lr_max, 2e-3)    # Lower ceiling for MoE
    increase_factor = 1.05        # More conservative increases
    decrease_factor = 0.85        # Less aggressive decreases
```

#### 3. Rate Limiting
Prevent dramatic changes:
```python
max_change_per_step = 0.3  # Max 30% change in either direction
ratio = new_lr / current_lr
if ratio > 1 + max_change_per_step:
    new_lr = current_lr * (1 + max_change_per_step)
elif ratio < 1 - max_change_per_step:
    new_lr = current_lr * (1 - max_change_per_step)
```

#### 4. Cooldown Period
Don't adjust too frequently:
```python
cooldown_steps = 20  # Wait 20 steps after any adjustment
```

#### 5. Warmup Integration
Adaptive scheduling starts AFTER warmup completes:
```python
if step < warmup_steps:
    # Standard linear warmup
    lr = base_lr * (step / warmup_steps)
else:
    # Adaptive scheduling takes over
    lr = adaptive_scheduler.step(loss)
```

## Implementation

### Class Design

```python
# File: src/aios/cli/hrm_hf/adaptive_lr.py

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Any
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive learning rate scheduling."""
    
    # Core parameters (auto-calculated if not specified)
    initial_lr: float = 5e-5
    lr_min: float = 1e-7
    lr_max: float = 1e-2
    
    # Window and patience
    window_size: int = 50
    patience: int = 20  # Steps without improvement before decreasing
    cooldown_steps: int = 20  # Steps to wait after adjustment
    
    # Thresholds
    increase_threshold: float = 0.02   # 2% improvement to consider increasing
    plateau_threshold: float = 0.001   # <0.1% improvement = plateau
    spike_threshold: float = 1.2       # 20% loss increase = spike
    stability_threshold: float = 0.3   # CV > 0.3 = unstable
    instability_threshold: float = 0.5 # CV > 0.5 = very unstable
    
    # Adjustment factors
    increase_factor: float = 1.1   # +10%
    decrease_factor: float = 0.7   # -30%
    spike_factor: float = 0.5      # -50%
    
    # Safety
    max_change_per_step: float = 0.3  # Max 30% change
    
    # MoE-specific (auto-applied when use_moe=True)
    moe_lr_min: float = 1e-5
    moe_lr_max: float = 2e-3
    moe_increase_factor: float = 1.05
    moe_decrease_factor: float = 0.85


class LRDecision(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    SPIKE_DECREASE = "spike_decrease"


@dataclass
class LossMetrics:
    """Computed metrics from loss window."""
    current_avg: float
    previous_avg: float
    current_std: float
    improvement: float
    improvement_rate: float
    cv: float  # Coefficient of variation
    trend: float  # Slope of loss over window


class AdaptiveLRScheduler:
    """Loss-reactive learning rate scheduler.
    
    Monitors training loss and adjusts LR:
    - Increases when training is progressing well
    - Decreases when training stalls or becomes unstable
    
    Integrates with existing warmup phase.
    """
    
    def __init__(
        self,
        optimizer: Any,
        config: AdaptiveLRConfig,
        use_moe: bool = False,
        log_fn: Optional[Callable] = None,
    ):
        self.optimizer = optimizer
        self.config = self._apply_moe_overrides(config, use_moe)
        self.use_moe = use_moe
        self.log_fn = log_fn or (lambda x: None)
        
        # State
        self.current_lr = config.initial_lr
        self.loss_window: deque = deque(maxlen=config.window_size)
        self.previous_window_avg: Optional[float] = None
        self.steps_since_adjustment = 0
        self.patience_counter = 0
        self.total_steps = 0
        self.adjustments_made = 0
        
        # History for debugging
        self.lr_history: list[tuple[int, float, str]] = []
        
        # Set initial LR
        self._set_lr(self.current_lr)
        
        self.log_fn({
            "event": "adaptive_lr_init",
            "initial_lr": self.current_lr,
            "lr_min": self.config.lr_min,
            "lr_max": self.config.lr_max,
            "use_moe": use_moe,
            "window_size": self.config.window_size,
        })
    
    def _apply_moe_overrides(
        self, config: AdaptiveLRConfig, use_moe: bool
    ) -> AdaptiveLRConfig:
        """Apply MoE-specific conservative bounds."""
        if not use_moe:
            return config
        
        return AdaptiveLRConfig(
            initial_lr=config.initial_lr,
            lr_min=max(config.lr_min, config.moe_lr_min),
            lr_max=min(config.lr_max, config.moe_lr_max),
            window_size=config.window_size,
            patience=config.patience,
            cooldown_steps=config.cooldown_steps,
            increase_threshold=config.increase_threshold,
            plateau_threshold=config.plateau_threshold,
            spike_threshold=config.spike_threshold,
            stability_threshold=config.stability_threshold,
            instability_threshold=config.instability_threshold,
            increase_factor=config.moe_increase_factor,
            decrease_factor=config.moe_decrease_factor,
            spike_factor=config.spike_factor,
            max_change_per_step=config.max_change_per_step,
            moe_lr_min=config.moe_lr_min,
            moe_lr_max=config.moe_lr_max,
            moe_increase_factor=config.moe_increase_factor,
            moe_decrease_factor=config.moe_decrease_factor,
        )
    
    def step(self, loss: float) -> float:
        """Process a training step and potentially adjust LR.
        
        Args:
            loss: The loss value from this training step
            
        Returns:
            The current learning rate (may have been adjusted)
        """
        self.total_steps += 1
        self.steps_since_adjustment += 1
        self.loss_window.append(loss)
        
        # Need full window before making decisions
        if len(self.loss_window) < self.config.window_size:
            return self.current_lr
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Make decision
        decision = self._make_decision(metrics)
        
        # Apply decision
        if decision != LRDecision.HOLD:
            self._apply_decision(decision, metrics)
        
        # Update state for next window
        self.previous_window_avg = metrics.current_avg
        self.loss_window.clear()
        
        return self.current_lr
    
    def _compute_metrics(self) -> LossMetrics:
        """Compute loss statistics from current window."""
        losses = list(self.loss_window)
        n = len(losses)
        
        current_avg = sum(losses) / n
        variance = sum((x - current_avg) ** 2 for x in losses) / n
        current_std = math.sqrt(variance)
        
        # Coefficient of variation (normalized std)
        cv = current_std / current_avg if current_avg > 0 else 0
        
        # Improvement from previous window
        if self.previous_window_avg is not None:
            improvement = self.previous_window_avg - current_avg
            improvement_rate = improvement / self.previous_window_avg if self.previous_window_avg > 0 else 0
        else:
            improvement = 0
            improvement_rate = 0
        
        # Trend: simple linear regression slope
        trend = self._compute_trend(losses)
        
        return LossMetrics(
            current_avg=current_avg,
            previous_avg=self.previous_window_avg or current_avg,
            current_std=current_std,
            improvement=improvement,
            improvement_rate=improvement_rate,
            cv=cv,
            trend=trend,
        )
    
    def _compute_trend(self, losses: list[float]) -> float:
        """Compute slope of loss over window (negative = improving)."""
        n = len(losses)
        if n < 2:
            return 0
        
        # Simple linear regression
        x_mean = (n - 1) / 2
        y_mean = sum(losses) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(losses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _make_decision(self, metrics: LossMetrics) -> LRDecision:
        """Decide whether to adjust LR based on metrics."""
        cfg = self.config
        
        # Cooldown: don't adjust too frequently
        if self.steps_since_adjustment < cfg.cooldown_steps:
            return LRDecision.HOLD
        
        # SPIKE: Emergency decrease
        if self.previous_window_avg and metrics.current_avg > self.previous_window_avg * cfg.spike_threshold:
            self.patience_counter = 0
            return LRDecision.SPIKE_DECREASE
        
        # INSTABILITY: High variance, decrease
        if metrics.cv > cfg.instability_threshold:
            self.patience_counter = 0
            return LRDecision.DECREASE
        
        # INCREASE: Consistent improvement + stable
        if (metrics.improvement_rate > cfg.increase_threshold and
            metrics.cv < cfg.stability_threshold and
            metrics.trend < 0 and  # Still improving
            self.current_lr < cfg.lr_max):
            self.patience_counter = 0
            return LRDecision.INCREASE
        
        # PLATEAU: No improvement for too long
        if metrics.improvement_rate < cfg.plateau_threshold:
            self.patience_counter += 1
            if self.patience_counter >= cfg.patience and self.current_lr > cfg.lr_min:
                self.patience_counter = 0
                return LRDecision.DECREASE
        else:
            self.patience_counter = 0
        
        return LRDecision.HOLD
    
    def _apply_decision(self, decision: LRDecision, metrics: LossMetrics) -> None:
        """Apply LR adjustment based on decision."""
        cfg = self.config
        old_lr = self.current_lr
        
        if decision == LRDecision.INCREASE:
            new_lr = self.current_lr * cfg.increase_factor
            reason = f"improvement_rate={metrics.improvement_rate:.4f} > threshold"
        elif decision == LRDecision.DECREASE:
            new_lr = self.current_lr * cfg.decrease_factor
            reason = f"plateau/instability (cv={metrics.cv:.3f})"
        elif decision == LRDecision.SPIKE_DECREASE:
            new_lr = self.current_lr * cfg.spike_factor
            reason = f"loss_spike ({metrics.current_avg:.4f} vs {metrics.previous_avg:.4f})"
        else:
            return
        
        # Rate limiting
        ratio = new_lr / old_lr
        if ratio > 1 + cfg.max_change_per_step:
            new_lr = old_lr * (1 + cfg.max_change_per_step)
        elif ratio < 1 - cfg.max_change_per_step:
            new_lr = old_lr * (1 - cfg.max_change_per_step)
        
        # Bound enforcement
        new_lr = max(cfg.lr_min, min(cfg.lr_max, new_lr))
        
        # Apply
        if new_lr != old_lr:
            self._set_lr(new_lr)
            self.steps_since_adjustment = 0
            self.adjustments_made += 1
            
            self.lr_history.append((self.total_steps, new_lr, decision.value))
            
            self.log_fn({
                "event": "adaptive_lr_adjustment",
                "step": self.total_steps,
                "decision": decision.value,
                "old_lr": old_lr,
                "new_lr": new_lr,
                "change_pct": (new_lr - old_lr) / old_lr * 100,
                "reason": reason,
                "metrics": {
                    "loss_avg": metrics.current_avg,
                    "improvement_rate": metrics.improvement_rate,
                    "cv": metrics.cv,
                    "trend": metrics.trend,
                },
            })
    
    def _set_lr(self, lr: float) -> None:
        """Set learning rate on optimizer."""
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr
    
    def get_summary(self) -> dict:
        """Get summary statistics for logging."""
        return {
            "total_steps": self.total_steps,
            "adjustments_made": self.adjustments_made,
            "final_lr": self.current_lr,
            "lr_history_length": len(self.lr_history),
        }
```

### Integration with Training Loop

```python
# In train_epoch.py

# After warmup completes, create adaptive scheduler
if config.auto_adjust_lr and steps_done >= warmup_steps:
    if adaptive_scheduler is None:
        adaptive_scheduler = AdaptiveLRScheduler(
            optimizer=opt,
            config=AdaptiveLRConfig(initial_lr=base_lr),
            use_moe=config.use_moe,
            log_fn=write_jsonl,
        )
    
    # Call step() with current loss
    new_lr = adaptive_scheduler.step(loss.item())
```

### Warmup + Adaptive Phases

```
Phase 1: Linear Warmup (unchanged)
─────────────────────────────────
Step 0 → warmup_steps
LR: 0 → base_lr (linear ramp)

Phase 2: Adaptive Scheduling (NEW)
──────────────────────────────────
Step warmup_steps → end
LR: Dynamically adjusted based on loss

Example timeline (1000 steps, base_lr=0.0001):
┌────────────────────────────────────────────────────────────────┐
│ 0.0002 ─                        ╭─────╮                        │
│          ─                     ╱       ╲                       │
│ 0.0001 ─────────────╱─────────╱         ╲─────────             │
│        ╱                                   ╲                   │
│ 0.00005             ╱                        ╲─────────────    │
│      ╱                                                         │
│    ╱                                                           │
│  ╱    Warmup   │      Adaptive Phase                           │
│──────────────────────────────────────────────────────────────▶ │
│ 0     100    200    400    600    800    1000  steps           │
└────────────────────────────────────────────────────────────────┘
         │        │         │         │
         │        │         │         └─ Plateau detected → decrease
         │        │         └─ Loss improving → increase
         │        └─ Spike detected → emergency decrease
         └─ Warmup complete, adaptive starts
```

## Configuration

### Auto-Calculated Defaults

When user checks "Auto-adjust learning rate", all parameters are computed automatically:

| Parameter | Formula | Example |
|-----------|---------|---------|
| `initial_lr` | User's LR or 5e-5 | 0.00005 |
| `lr_min` | `initial_lr / 100` | 5e-7 |
| `lr_max` | `initial_lr * 20` | 0.001 |
| `window_size` | `min(50, steps // 20)` | 50 |
| `warmup_steps` | `steps * 0.1` | 100 (for 1000 steps) |

### MoE Automatic Adjustments

When `use_moe=True`, bounds are automatically tightened:

| Parameter | Standard | MoE |
|-----------|----------|-----|
| `lr_min` | 1e-7 | 1e-5 |
| `lr_max` | 1e-2 | 2e-3 |
| `increase_factor` | 1.1 | 1.05 |
| `decrease_factor` | 0.7 | 0.85 |

## User Experience

### Before (Current)
```
☑️ Auto-adjust learning rate
   → One-time clamp for MoE, then fixed LR
```

### After (This Feature)
```
☑️ Auto-adjust learning rate
   → Smart pre-flight clamp for MoE
   → Linear warmup phase
   → Loss-reactive adaptive phase:
     • Speeds up when learning is going well
     • Slows down when stuck or unstable
     • Emergency brake on loss spikes
   → All automatic, no tuning required
```

### Logging Output

Users will see LR adjustments in the training log:
```json
{"event": "adaptive_lr_adjustment", "step": 250, "decision": "increase", "old_lr": 0.0001, "new_lr": 0.00011, "change_pct": 10.0, "reason": "improvement_rate=0.0312 > threshold"}
{"event": "adaptive_lr_adjustment", "step": 500, "decision": "decrease", "old_lr": 0.00015, "new_lr": 0.000105, "change_pct": -30.0, "reason": "plateau/instability (cv=0.421)"}
{"event": "adaptive_lr_adjustment", "step": 750, "decision": "spike_decrease", "old_lr": 0.000105, "new_lr": 0.0000525, "change_pct": -50.0, "reason": "loss_spike (2.451 vs 1.892)"}
```

## Testing Strategy

### Unit Tests
1. `test_metrics_computation` - Verify loss statistics are correct
2. `test_decision_logic` - Test each decision path
3. `test_moe_bounds` - Verify MoE constraints are applied
4. `test_rate_limiting` - Ensure changes are bounded
5. `test_cooldown` - Verify cooldown prevents rapid changes

### Integration Tests
1. `test_warmup_to_adaptive_transition` - Seamless handoff
2. `test_with_real_training_loop` - End-to-end with actual model
3. `test_checkpoint_resume` - Scheduler state persists across restarts

### Scenarios to Validate
1. **Healthy training** - LR should gradually increase, then stabilize
2. **Stuck training** - LR should decrease after patience exhausted
3. **Diverging training** - LR should emergency decrease on spike
4. **MoE training** - Bounds should be tighter, changes more conservative

## Implementation Plan

### Phase 1: Core Module
- [ ] Create `src/aios/cli/hrm_hf/adaptive_lr.py`
- [ ] Implement `AdaptiveLRScheduler` class
- [ ] Implement `AdaptiveLRConfig` dataclass
- [ ] Add comprehensive logging

### Phase 2: Training Integration
- [ ] Modify `train_epoch.py` to accept scheduler
- [ ] Handle warmup → adaptive transition
- [ ] Pass scheduler through training loop chain

### Phase 3: Wiring
- [ ] Create scheduler in `train_actv1.py` when `auto_adjust_lr=True`
- [ ] Create scheduler in `parallel_training_v3.py` for multi-GPU
- [ ] Ensure scheduler state is per-GPU in parallel mode

### Phase 4: Checkpoint Support
- [ ] Save scheduler state in checkpoints
- [ ] Restore scheduler state on resume
- [ ] Handle missing scheduler state (backward compatibility)

### Phase 5: Testing & Validation
- [ ] Unit tests for scheduler logic
- [ ] Integration tests with training loop
- [ ] Manual validation with real training runs
- [ ] Performance profiling (ensure minimal overhead)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scheduler makes bad decisions | Training quality degrades | Conservative defaults, rate limiting, MoE bounds |
| Overhead slows training | Reduced throughput | Efficient windowed stats, only compute every window_size steps |
| State lost on crash | Resume behaves differently | Save scheduler state in checkpoint |
| Conflicts with DeepSpeed scheduler | Undefined behavior | Detect DeepSpeed scheduler, disable adaptive if present |

## Future Enhancements

### Gradient-Aware Adjustments
Monitor gradient norms in addition to loss:
- Exploding gradients → Immediate LR decrease
- Vanishing gradients → LR increase

### Learning Rate Range Test
Auto-discover good LR bounds at start of training:
- Run quick sweep from 1e-7 to 1e-1
- Find range where loss decreases
- Use as bounds for adaptive scheduling

### Per-Parameter-Group LR
Different LR schedules for different parts of model:
- Base model: Conservative
- Router (MoE): Very conservative
- LoRA adapters: More aggressive

## References

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) - Smith 2015
- [Super-Convergence: Very Fast Training of Neural Networks](https://arxiv.org/abs/1708.07120) - Smith & Topin 2017
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter 2017 (AdamW)
- Existing AI-OS `width_management.py` - Similar loss-reactive pattern for model width
