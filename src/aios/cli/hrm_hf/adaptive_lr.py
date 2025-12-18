"""Loss-reactive adaptive learning rate scheduling.

This module implements a lightweight, training-loss driven learning-rate
controller intended for HRM HF training loops.

Design goals:
- Safe defaults (bounded, rate-limited, cooldown)
- Low overhead (metrics computed every window)
- Minimal coupling (works with any optimizer-like object exposing param_groups)
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger(__name__)


def _load_adaptive_lr_overrides(path: str) -> dict[str, Any]:
    """Load adaptive LR overrides from a JSON/TOML/YAML file.

    Supported formats:
    - .json: plain JSON object
    - .toml: either top-level keys or [adaptive_lr] table
    - .yaml/.yml: either top-level keys or adaptive_lr: mapping (requires PyYAML)
    """
    p = os.path.expandvars(os.path.expanduser(str(path)))
    suffix = os.path.splitext(p)[1].lower()
    # Support double extensions like *.yaml.example / *.toml.example
    if suffix == ".example":
        try:
            from pathlib import Path

            suffixes = [s.lower() for s in Path(p).suffixes]
            if len(suffixes) >= 2:
                suffix = suffixes[-2]
        except Exception:
            pass

    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix == ".toml":
        import tomllib

        with open(p, "rb") as f:
            data = tomllib.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML adaptive-lr config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported adaptive-lr config format: {suffix}. Use .json, .toml, .yaml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("adaptive-lr config must parse to an object/dict")

    # Allow nesting under `adaptive_lr`
    if "adaptive_lr" in data and isinstance(data.get("adaptive_lr"), dict):
        data = data["adaptive_lr"]

    if not isinstance(data, dict):
        raise ValueError("adaptive-lr config must be a dict")

    return dict(data)


def _validate_adaptive_lr_config(cfg: "AdaptiveLRConfig") -> None:
    if cfg.window_size < 2:
        raise ValueError("window_size must be >= 2")
    if cfg.patience < 1:
        raise ValueError("patience must be >= 1")
    if cfg.cooldown_steps < 0:
        raise ValueError("cooldown_steps must be >= 0")
    if cfg.lr_min <= 0 or cfg.lr_max <= 0:
        raise ValueError("lr_min and lr_max must be > 0")
    if cfg.lr_min > cfg.lr_max:
        raise ValueError("lr_min must be <= lr_max")
    if cfg.initial_lr <= 0:
        raise ValueError("initial_lr must be > 0")
    if not (0.0 <= cfg.max_change_per_step <= 1.0):
        raise ValueError("max_change_per_step must be between 0 and 1")
    if cfg.increase_factor <= 0 or cfg.decrease_factor <= 0 or cfg.spike_factor <= 0:
        raise ValueError("increase_factor/decrease_factor/spike_factor must be > 0")
    if int(cfg.debug_level) < 0:
        raise ValueError("debug_level must be >= 0")
    if int(cfg.window_summary_every) < 1:
        raise ValueError("window_summary_every must be >= 1")

    mode = str(getattr(cfg, "mode", "balanced") or "balanced").strip().lower()
    if mode not in {"balanced", "conservative", "aggressive", "auto"}:
        raise ValueError("mode must be one of: balanced, conservative, aggressive, auto")
    if int(getattr(cfg, "auto_mode_cooldown_windows", 1) or 1) < 0:
        raise ValueError("auto_mode_cooldown_windows must be >= 0")
    if int(getattr(cfg, "auto_plateau_windows", 1) or 1) < 1:
        raise ValueError("auto_plateau_windows must be >= 1")
    if float(getattr(cfg, "auto_cv_conservative", 0.0) or 0.0) <= 0:
        raise ValueError("auto_cv_conservative must be > 0")


def build_adaptive_lr_config(
    *,
    base_lr: float,
    steps: Optional[int],
    use_moe: bool,
    config_path: Optional[str] = None,
    override_dict: Optional[dict[str, Any]] = None,
) -> "AdaptiveLRConfig":
    """Build an AdaptiveLRConfig from base defaults, optionally overridden by a config file."""
    cfg = AdaptiveLRConfig.from_base_lr(base_lr, steps=steps, use_moe=use_moe)

    overrides: dict[str, Any] = {}
    if config_path:
        overrides.update(_load_adaptive_lr_overrides(config_path))
    if override_dict:
        overrides.update({k: v for k, v in override_dict.items() if v is not None})

    if not overrides:
        _validate_adaptive_lr_config(cfg)
        return cfg

    allowed = set(AdaptiveLRConfig.__dataclass_fields__.keys())
    unknown = sorted(k for k in overrides.keys() if k not in allowed)
    if unknown:
        raise ValueError(f"Unknown adaptive-lr config keys: {unknown}")

    merged = {**cfg.__dict__, **overrides}
    cfg2 = AdaptiveLRConfig(**merged)
    _validate_adaptive_lr_config(cfg2)
    return cfg2


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class AdaptiveLRConfig:
    """Configuration for adaptive learning-rate scheduling."""

    initial_lr: float
    lr_min: float
    lr_max: float

    window_size: int = 16
    patience: int = 10
    cooldown_steps: int = 10

    increase_threshold: float = 0.02
    plateau_threshold: float = 0.001
    spike_threshold: float = 1.2
    stability_threshold: float = 0.3
    instability_threshold: float = 0.5

    increase_factor: float = 1.1
    decrease_factor: float = 0.7
    spike_factor: float = 0.5

    max_change_per_step: float = 0.3

    # MoE-specific conservative overrides
    moe_lr_min: float = 1e-5
    moe_lr_max: float = 2e-3
    moe_increase_factor: float = 1.05
    moe_decrease_factor: float = 0.85

    # Debug / analysis output
    # 0=off, 1=adjustments only, 2=periodic window summaries, 3=very verbose
    debug_level: int = 0
    emit_window_summary: bool = False
    window_summary_every: int = 5

    # Learning modes (preset multipliers on top of the base config)
    # - balanced: uses config values directly
    # - conservative: slower increases + earlier safety decreases
    # - aggressive: faster increases + less sensitive plateau decreases
    # - auto: switches between the above based on observed stability/plateaus
    mode: str = "balanced"  # balanced|conservative|aggressive|auto

    # Auto-mode switching knobs (only used when mode='auto')
    auto_mode_initial: str = "balanced"
    auto_mode_cooldown_windows: int = 2
    auto_plateau_windows: int = 3
    auto_cv_conservative: float = 0.6

    @staticmethod
    def from_base_lr(
        base_lr: float,
        *,
        steps: Optional[int] = None,
        use_moe: bool = False,
    ) -> "AdaptiveLRConfig":
        """Create a conservative config from a base learning rate.

        Notes:
        - bounds are relative to base_lr to reduce surprise
        - window_size is capped and can be scaled from steps (if provided)
        """
        base_lr = float(base_lr)
        if base_lr <= 0:
            raise ValueError("base_lr must be > 0")

        # Auto window sizing (keep stable, inexpensive)
        if steps is None:
            window_size = 50
        else:
            # roughly every ~5% of steps, capped
            window_size = int(max(10, min(50, max(10, int(steps) // 20))))

        cfg = AdaptiveLRConfig(
            initial_lr=base_lr,
            lr_min=max(base_lr / 100.0, 1e-9),
            lr_max=base_lr * 20.0,
            window_size=window_size,
        )

        if not use_moe:
            return cfg

        # Apply MoE conservative bounds and factors
        return AdaptiveLRConfig(
            initial_lr=cfg.initial_lr,
            lr_min=max(cfg.lr_min, cfg.moe_lr_min),
            lr_max=min(cfg.lr_max, cfg.moe_lr_max),
            window_size=cfg.window_size,
            patience=cfg.patience,
            cooldown_steps=cfg.cooldown_steps,
            increase_threshold=cfg.increase_threshold,
            plateau_threshold=cfg.plateau_threshold,
            spike_threshold=cfg.spike_threshold,
            stability_threshold=cfg.stability_threshold,
            instability_threshold=cfg.instability_threshold,
            increase_factor=cfg.moe_increase_factor,
            decrease_factor=cfg.moe_decrease_factor,
            spike_factor=cfg.spike_factor,
            max_change_per_step=cfg.max_change_per_step,
            moe_lr_min=cfg.moe_lr_min,
            moe_lr_max=cfg.moe_lr_max,
            moe_increase_factor=cfg.moe_increase_factor,
            moe_decrease_factor=cfg.moe_decrease_factor,
        )


class LRDecision(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    SPIKE_DECREASE = "spike_decrease"


@dataclass
class LossMetrics:
    current_avg: float
    previous_avg: float
    current_std: float
    improvement: float
    improvement_rate: float
    cv: float
    trend: float


class AdaptiveLRScheduler:
    """Loss-reactive learning rate scheduler.

    Call `observe(loss)` once per optimizer step (not per micro-batch) to allow
    the scheduler to update the optimizer learning rate.
    """

    def __init__(
        self,
        optimizer: Any,
        config: AdaptiveLRConfig,
        *,
        use_moe: bool = False,
        log_fn: Optional[Callable[[dict], None]] = None,
        state_path: Optional[str] = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.use_moe = bool(use_moe)
        self.log_fn = log_fn or (lambda _payload: None)

        # Learning mode state
        self._mode_requested = str(getattr(self.config, "mode", "balanced") or "balanced").strip().lower()
        if self._mode_requested == "auto":
            initial = str(getattr(self.config, "auto_mode_initial", "balanced") or "balanced").strip().lower()
            self._mode = initial if initial in {"balanced", "conservative", "aggressive"} else "balanced"
        else:
            self._mode = self._mode_requested
        self._mode_cooldown_remaining = 0
        self._low_improvement_windows_in_row = 0

        self.current_lr = float(config.initial_lr)
        self.loss_window: Deque[float] = deque(maxlen=int(config.window_size))
        self.previous_window_avg: Optional[float] = None
        self.steps_since_adjustment = 10**9
        self.patience_counter = 0
        self.total_observations = 0
        self.adjustments_made = 0
        self.lr_history: list[tuple[int, float, str]] = []

        self.window_index = 0
        self._state_path = state_path

        # Set initial LR
        self._set_lr(self.current_lr)

        self.log_fn(
            {
                "event": "adaptive_lr_init",
                "initial_lr": self.current_lr,
                "lr_min": self.config.lr_min,
                "lr_max": self.config.lr_max,
                "use_moe": self.use_moe,
                "window_size": self.config.window_size,
                "mode": self._mode_requested,
                "mode_active": self._mode,
            }
        )

        if int(self.config.debug_level) >= 1:
            logger.info(
                "AdaptiveLR enabled: initial_lr=%.6g lr_min=%.6g lr_max=%.6g window_size=%d patience=%d cooldown=%d",
                float(self.current_lr),
                float(self.config.lr_min),
                float(self.config.lr_max),
                int(self.config.window_size),
                int(self.config.patience),
                int(self.config.cooldown_steps),
            )
            if self._mode_requested != "balanced":
                logger.info(
                    "AdaptiveLR mode: requested=%s active=%s",
                    self._mode_requested,
                    self._mode,
                )

    def state_dict(self) -> dict:
        return {
            "version": 3,
            "current_lr": self.current_lr,
            "previous_window_avg": self.previous_window_avg,
            "steps_since_adjustment": self.steps_since_adjustment,
            "patience_counter": self.patience_counter,
            "total_observations": self.total_observations,
            "adjustments_made": self.adjustments_made,
            "loss_window": list(self.loss_window),
            "lr_history": list(self.lr_history),
            "window_index": int(self.window_index),
            "mode_requested": self._mode_requested,
            "mode_active": self._mode,
            "mode_cooldown_remaining": int(self._mode_cooldown_remaining),
            "low_improvement_windows_in_row": int(self._low_improvement_windows_in_row),
            "config": {
                **self.config.__dict__,
            },
            "use_moe": self.use_moe,
        }

    @classmethod
    def from_state_dict(
        cls,
        *,
        optimizer: Any,
        state: dict,
        log_fn: Optional[Callable[[dict], None]] = None,
        state_path: Optional[str] = None,
    ) -> "AdaptiveLRScheduler":
        cfg_dict = state.get("config") or {}
        cfg = AdaptiveLRConfig(**cfg_dict)
        inst = cls(
            optimizer=optimizer,
            config=cfg,
            use_moe=bool(state.get("use_moe", False)),
            log_fn=log_fn,
            state_path=state_path,
        )

        inst.current_lr = float(state.get("current_lr", inst.current_lr))
        inst.previous_window_avg = _safe_float(state.get("previous_window_avg"))
        inst.steps_since_adjustment = int(state.get("steps_since_adjustment", inst.steps_since_adjustment))
        inst.patience_counter = int(state.get("patience_counter", 0))
        inst.total_observations = int(state.get("total_observations", 0))
        inst.adjustments_made = int(state.get("adjustments_made", 0))
        inst.loss_window.clear()
        for v in state.get("loss_window", []) or []:
            fv = _safe_float(v)
            if fv is not None:
                inst.loss_window.append(fv)
        inst.lr_history = list(state.get("lr_history", []) or [])
        inst.window_index = int(state.get("window_index", 0) or 0)

        inst._mode_requested = str(state.get("mode_requested", inst._mode_requested) or inst._mode_requested).strip().lower()
        inst._mode = str(state.get("mode_active", inst._mode) or inst._mode).strip().lower()
        if inst._mode not in {"balanced", "conservative", "aggressive"}:
            inst._mode = "balanced"
        inst._mode_cooldown_remaining = int(state.get("mode_cooldown_remaining", 0) or 0)
        inst._low_improvement_windows_in_row = int(state.get("low_improvement_windows_in_row", 0) or 0)

        inst._set_lr(inst.current_lr)
        return inst

    def get_lr(self) -> float:
        return float(self.current_lr)

    def observe(self, loss: float) -> float:
        """Observe one loss value (ideally once per optimizer step)."""
        loss_f = float(loss)
        if not math.isfinite(loss_f):
            return self.current_lr

        self.total_observations += 1
        self.steps_since_adjustment += 1
        self.loss_window.append(loss_f)

        # --- Emergency Spike Detection (per-step, before window fills) ---
        # If we have at least 4 samples, check if current loss is dramatically
        # higher than recent average. This catches catastrophic spikes immediately.
        if len(self.loss_window) >= 4:
            recent = list(self.loss_window)[-4:-1]  # last 3 before current
            recent_avg = sum(recent) / len(recent)
            emergency_threshold = 1.5  # 50% spike = emergency
            if recent_avg > 0 and loss_f > recent_avg * emergency_threshold:
                # Emergency brake: immediate LR reduction
                old_lr = float(self.current_lr)
                emergency_factor = 0.5  # halve LR on emergency
                new_lr = max(float(self.config.lr_min), old_lr * emergency_factor)
                if new_lr < old_lr:
                    self._set_lr(new_lr)
                    self.steps_since_adjustment = 0
                    self.adjustments_made += 1
                    self.lr_history.append((self.total_observations, new_lr, "emergency_spike"))
                    spike_pct = ((loss_f - recent_avg) / recent_avg * 100.0)
                    print(
                        f"[AdaptiveLR] EMERGENCY step={int(self.total_observations)} "
                        f"loss={loss_f:.4f} vs recent_avg={recent_avg:.4f} (+{spike_pct:.1f}%) "
                        f"lr {old_lr:.6g}->{new_lr:.6g} (-50%)",
                        flush=True,
                    )
                    try:
                        self.log_fn({
                            "event": "adaptive_lr_emergency",
                            "step": int(self.total_observations),
                            "loss": loss_f,
                            "recent_avg": recent_avg,
                            "spike_pct": spike_pct,
                            "old_lr": old_lr,
                            "new_lr": new_lr,
                        })
                    except Exception:
                        pass
                    # Clear window to start fresh after emergency
                    self.loss_window.clear()
                    return self.current_lr

        if len(self.loss_window) < int(self.config.window_size):
            return self.current_lr

        metrics = self._compute_metrics()

        # Track low-improvement windows (used by auto mode switching).
        # This is intentionally independent of the decision logic, so that
        # mode switching can react even when we're still increasing LR.
        try:
            if float(metrics.improvement_rate) < float(self.config.plateau_threshold):
                self._low_improvement_windows_in_row += 1
            else:
                self._low_improvement_windows_in_row = 0
        except Exception:
            self._low_improvement_windows_in_row = 0

        decision = self._make_decision(metrics)
        self.window_index += 1

        # Auto-mode switching runs once per window.
        # Note: mode changes take effect on the *next* window (to avoid recomputing
        # the current decision mid-window).
        self._maybe_update_mode(metrics)

        self._maybe_emit_window_summary(metrics, decision)

        if decision != LRDecision.HOLD:
            self._apply_decision(decision, metrics)

        # Window roll
        self.previous_window_avg = metrics.current_avg
        self.loss_window.clear()
        self._maybe_persist_state()
        return self.current_lr

    def _maybe_emit_window_summary(self, metrics: LossMetrics, decision: LRDecision) -> None:
        cfg = self.config
        debug_level = int(getattr(cfg, "debug_level", 0) or 0)
        emit = bool(getattr(cfg, "emit_window_summary", False)) or debug_level >= 2
        every = int(getattr(cfg, "window_summary_every", 1) or 1)

        # ALWAYS print window check status to stdout for visibility
        try:
            prev_avg = float(metrics.previous_avg) if metrics.previous_avg else 0.0
            change_from_prev = ""
            if prev_avg > 0:
                delta_pct = ((metrics.current_avg - prev_avg) / prev_avg) * 100.0
                change_from_prev = f" delta={delta_pct:+.2f}%"
            print(
                f"[AdaptiveLR] window={int(self.window_index)} step={int(self.total_observations)} "
                f"decision={decision.value} lr={float(self.current_lr):.6g} "
                f"loss_avg={float(metrics.current_avg):.4f}{change_from_prev} "
                f"cv={float(metrics.cv):.3f} trend={float(metrics.trend):.4f}",
                flush=True,
            )
        except Exception:
            pass

        if not emit:
            return
        if every > 1 and (int(self.window_index) % every) != 0:
            return

        payload = {
            "event": "adaptive_lr_window_summary",
            "window_index": int(self.window_index),
            "step": int(self.total_observations),
            "decision": decision.value,
            "lr": float(self.current_lr),
            "metrics": {
                "loss_avg": float(metrics.current_avg),
                "loss_std": float(metrics.current_std),
                "improvement_rate": float(metrics.improvement_rate),
                "cv": float(metrics.cv),
                "trend": float(metrics.trend),
            },
        }

        try:
            self.log_fn(payload)
        except Exception:
            logger.debug("adaptive lr window summary log_fn failed", exc_info=True)

        if debug_level >= 2:
            logger.info(
                "AdaptiveLR window=%d step=%d decision=%s lr=%.6g loss_avg=%.6g std=%.6g cv=%.3f trend=%.6g imp_rate=%.4f",
                int(self.window_index),
                int(self.total_observations),
                decision.value,
                float(self.current_lr),
                float(metrics.current_avg),
                float(metrics.current_std),
                float(metrics.cv),
                float(metrics.trend),
                float(metrics.improvement_rate),
            )

    def _maybe_persist_state(self) -> None:
        if not self._state_path:
            return
        try:
            path = os.path.expandvars(os.path.expanduser(str(self._state_path)))
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state_dict(), f)
            os.replace(tmp, path)
        except Exception:
            logger.debug("Failed to persist adaptive lr state", exc_info=True)

    def _compute_metrics(self) -> LossMetrics:
        losses = list(self.loss_window)
        n = max(1, len(losses))
        current_avg = sum(losses) / n
        variance = sum((x - current_avg) ** 2 for x in losses) / n
        current_std = math.sqrt(variance)
        cv = (current_std / current_avg) if current_avg > 0 else 0.0

        prev = self.previous_window_avg if self.previous_window_avg is not None else current_avg
        improvement = prev - current_avg
        improvement_rate = (improvement / prev) if prev > 0 else 0.0
        trend = self._compute_trend(losses)

        return LossMetrics(
            current_avg=current_avg,
            previous_avg=prev,
            current_std=current_std,
            improvement=improvement,
            improvement_rate=improvement_rate,
            cv=cv,
            trend=trend,
        )

    @staticmethod
    def _compute_trend(losses: list[float]) -> float:
        n = len(losses)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(losses) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(losses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _make_decision(self, metrics: LossMetrics) -> LRDecision:
        cfg = self._effective_config()

        if self.steps_since_adjustment < int(cfg.cooldown_steps):
            return LRDecision.HOLD

        # Spike: sudden loss increase
        if metrics.current_avg > metrics.previous_avg * float(cfg.spike_threshold):
            self.patience_counter = 0
            return LRDecision.SPIKE_DECREASE

        # Unstable: too much variation
        if metrics.cv > float(cfg.instability_threshold):
            self.patience_counter = 0
            return LRDecision.DECREASE

        # Increase: stable and improving
        if (
            metrics.improvement_rate > float(cfg.increase_threshold)
            and metrics.cv < float(cfg.stability_threshold)
            and metrics.trend < 0.0
            and self.current_lr < float(cfg.lr_max)
        ):
            self.patience_counter = 0
            return LRDecision.INCREASE

        # Plateau: little improvement for long enough
        if metrics.improvement_rate < float(cfg.plateau_threshold):
            self.patience_counter += 1
            if self.patience_counter >= int(cfg.patience) and self.current_lr > float(cfg.lr_min):
                self.patience_counter = 0
                return LRDecision.DECREASE
        else:
            self.patience_counter = 0

        return LRDecision.HOLD

    def _apply_decision(self, decision: LRDecision, metrics: LossMetrics) -> None:
        cfg = self._effective_config()
        old_lr = float(self.current_lr)

        if decision == LRDecision.INCREASE:
            proposed = old_lr * float(cfg.increase_factor)
            reason = f"improvement_rate={metrics.improvement_rate:.4f}"
        elif decision == LRDecision.DECREASE:
            proposed = old_lr * float(cfg.decrease_factor)
            reason = f"plateau/instability cv={metrics.cv:.3f}"
        elif decision == LRDecision.SPIKE_DECREASE:
            proposed = old_lr * float(cfg.spike_factor)
            reason = f"loss_spike {metrics.current_avg:.4f}>{metrics.previous_avg:.4f}"
        else:
            return

        # Rate limiting
        ratio = proposed / old_lr if old_lr > 0 else 1.0
        max_change = float(cfg.max_change_per_step)
        if ratio > 1.0 + max_change:
            proposed = old_lr * (1.0 + max_change)
        elif ratio < 1.0 - max_change:
            proposed = old_lr * (1.0 - max_change)

        # Bounds
        new_lr = _clamp(proposed, float(cfg.lr_min), float(cfg.lr_max))
        if new_lr == old_lr:
            return

        self._set_lr(new_lr)
        self.steps_since_adjustment = 0
        self.adjustments_made += 1
        self.lr_history.append((self.total_observations, new_lr, decision.value))

        try:
            self.log_fn(
                {
                    "event": "adaptive_lr_adjustment",
                    "step": int(self.total_observations),
                    "window_index": int(self.window_index),
                    "decision": decision.value,
                    "old_lr": old_lr,
                    "new_lr": new_lr,
                    "change_pct": ((new_lr - old_lr) / old_lr * 100.0) if old_lr > 0 else None,
                    "reason": reason,
                    "metrics": {
                        "loss_avg": metrics.current_avg,
                        "improvement_rate": metrics.improvement_rate,
                        "cv": metrics.cv,
                        "trend": metrics.trend,
                    },
                }
            )
        except Exception:
            logger.debug("adaptive lr log_fn failed", exc_info=True)

        # Ensure users see LR changes even when only watching stdout.
        try:
            change_pct = ((new_lr - old_lr) / old_lr * 100.0) if old_lr > 0 else 0.0
            print(
                f"[AdaptiveLR] step={int(self.total_observations)} window={int(self.window_index)} "
                f"decision={decision.value} lr {old_lr:.6g}->{new_lr:.6g} ({change_pct:+.2f}%) reason={reason}",
                flush=True,
            )
        except Exception:
            pass

        if int(getattr(self.config, "debug_level", 0) or 0) >= 1:
            logger.info(
                "AdaptiveLR adjust step=%d decision=%s lr %.6g->%.6g (%+.2f%%) reason=%s",
                int(self.total_observations),
                decision.value,
                float(old_lr),
                float(new_lr),
                ((float(new_lr) - float(old_lr)) / float(old_lr) * 100.0) if float(old_lr) > 0 else 0.0,
                reason,
            )

    def _set_lr(self, lr: float) -> None:
        self.current_lr = float(lr)
        if self.optimizer is None:
            return
        try:
            for pg in getattr(self.optimizer, "param_groups", []) or []:
                pg["lr"] = float(lr)
        except Exception:
            logger.debug("Failed to set lr on optimizer", exc_info=True)

    def _effective_config(self) -> AdaptiveLRConfig:
        """Return an effective config after applying the active learning mode preset."""
        base = self.config
        mode = (self._mode or "balanced").strip().lower()
        if mode == "balanced":
            return base

        # Start from base values and apply mode multipliers.
        # Goal: keep behavior intuitive while preserving user-set hard bounds.
        merged = dict(base.__dict__)

        if mode == "conservative":
            merged["increase_factor"] = float(base.increase_factor) * 0.75
            merged["decrease_factor"] = float(base.decrease_factor) * 0.90
            merged["spike_factor"] = float(base.spike_factor) * 0.85
            merged["max_change_per_step"] = _clamp(float(base.max_change_per_step) * 0.75, 0.0, 1.0)
            merged["increase_threshold"] = float(base.increase_threshold) * 1.25
            merged["plateau_threshold"] = float(base.plateau_threshold) * 1.10
            merged["stability_threshold"] = float(base.stability_threshold) * 0.90
            merged["instability_threshold"] = float(base.instability_threshold) * 0.90
            merged["patience"] = int(base.patience) + 2
        elif mode == "aggressive":
            merged["increase_factor"] = float(base.increase_factor) * 1.20
            merged["decrease_factor"] = float(base.decrease_factor) * 1.05
            merged["spike_factor"] = float(base.spike_factor)
            merged["max_change_per_step"] = _clamp(float(base.max_change_per_step) * 1.20, 0.0, 1.0)
            merged["increase_threshold"] = float(base.increase_threshold) * 0.85
            merged["plateau_threshold"] = float(base.plateau_threshold) * 0.85
            merged["stability_threshold"] = float(base.stability_threshold) * 1.10
            merged["instability_threshold"] = float(base.instability_threshold) * 1.10
            merged["patience"] = max(1, int(base.patience) + 2)

        # Keep within sane ranges
        merged["increase_factor"] = max(1e-12, float(merged["increase_factor"]))
        merged["decrease_factor"] = max(1e-12, float(merged["decrease_factor"]))
        merged["spike_factor"] = max(1e-12, float(merged["spike_factor"]))
        merged["increase_threshold"] = max(0.0, float(merged["increase_threshold"]))
        merged["plateau_threshold"] = max(0.0, float(merged["plateau_threshold"]))

        return AdaptiveLRConfig(**merged)

    def _maybe_update_mode(self, metrics: LossMetrics) -> None:
        """Auto-switch between modes to adapt aggressiveness.

        Strategy (simple, stable):
        - If unstable (high CV) or spike-like behavior, switch to conservative.
        - If plateauing for several windows *and* stable, switch to aggressive.
        - Otherwise, fall back to balanced.
        """
        if self._mode_requested != "auto":
            return

        if self._mode_cooldown_remaining > 0:
            self._mode_cooldown_remaining -= 1
            return

        base = self.config
        cv_conservative = float(getattr(base, "auto_cv_conservative", 0.6) or 0.6)
        plateau_windows = int(getattr(base, "auto_plateau_windows", 3) or 3)
        cooldown_windows = int(getattr(base, "auto_mode_cooldown_windows", 2) or 2)

        target = "balanced"
        reason = ""

        # Safety-first: instability or spike-like jump -> conservative
        if metrics.cv >= cv_conservative or (metrics.current_avg > metrics.previous_avg * float(base.spike_threshold)):
            target = "conservative"
            reason = f"unstable cv={metrics.cv:.3f}"
        # Plateau (tracked elsewhere) + stable -> aggressive
        elif self._low_improvement_windows_in_row >= plateau_windows and metrics.cv < float(base.stability_threshold):
            target = "aggressive"
            reason = f"low_improvement_windows={int(self._low_improvement_windows_in_row)}"
        else:
            target = "balanced"
            reason = "default"

        if target == self._mode:
            return

        old = self._mode
        self._mode = target
        self._mode_cooldown_remaining = max(0, cooldown_windows)
        self._low_improvement_windows_in_row = 0

        payload = {
            "event": "adaptive_lr_mode_changed",
            "from": old,
            "to": target,
            "reason": reason,
            "window_index": int(self.window_index),
            "step": int(self.total_observations),
            "lr": float(self.current_lr),
            "metrics": {
                "loss_avg": float(metrics.current_avg),
                "loss_std": float(metrics.current_std),
                "improvement_rate": float(metrics.improvement_rate),
                "cv": float(metrics.cv),
                "trend": float(metrics.trend),
            },
        }
        try:
            self.log_fn(payload)
        except Exception:
            logger.debug("adaptive lr mode change log_fn failed", exc_info=True)

        if int(getattr(self.config, "debug_level", 0) or 0) >= 1:
            logger.info("AdaptiveLR mode change: %s -> %s (%s)", old, target, reason)
