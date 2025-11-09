"""Training step methods and batch utilities for Trainer."""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from ..replay import ReplayBuffer


class TrainingMixin:
    """Mixin providing training step methods for Trainer."""

    def _batch_from_replay(
        self, rb: ReplayBuffer, batch: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a training batch from replay buffer."""
        if len(rb) == 0:
            # synthesize random transitions if buffer empty
            x = np.random.normal(size=(batch, self.cfg.input_dim)).astype(np.float32)
            y = np.sum(x[:, :2], axis=1, keepdims=True).astype(np.float32)
            return x, y
        trans = rb.sample(batch)
        # map states (Any) into fixed-size features; for now, use simple featurizer
        feats = []
        targets = []
        for _s, a, r, _s2, _done in trans:
            # If state is already a vector of correct dim, use it; otherwise synthesize
            if isinstance(_s, np.ndarray) and _s.dtype in (np.float32, np.float64) and _s.ndim == 1 and _s.shape[0] == int(self.cfg.input_dim):
                vec = _s.astype(np.float32)
            else:
                vec = np.zeros((self.cfg.input_dim,), dtype=np.float32)
                # cheap features: encode action id in first cell, reward in second
                try:
                    vec[0] = float(a) if isinstance(a, (int, float)) else 0.0
                except Exception:
                    vec[0] = 0.0
                try:
                    vec[1] = float(r)
                except Exception:
                    vec[1] = 0.0
            feats.append(vec)
            targets.append([float(r)])
        x = np.stack(feats, axis=0)
        y = np.array(targets, dtype=np.float32)
        return x, y

    def _compute_cost(self, x: np.ndarray) -> float:
        """Cheap synthetic cost signal based on feature magnitudes.
        Uses abs of first feature (encoded action id) and reward proxy (second feature).
        """
        # avoid NaNs if dims < 2
        c = 0.0
        if x.shape[1] >= 1:
            c += float(np.mean(np.abs(x[:, 0])))
        if x.shape[1] >= 2:
            c += 0.5 * float(np.mean(np.maximum(0.0, x[:, 1])))
        return self.cfg.cost_coef * c

    def train_step(self, rb: ReplayBuffer) -> float:
        """Run one training step. Returns loss."""
        batch = max(1, self.cfg.batch_size)
        x, y = self._batch_from_replay(rb, batch)
        if self.torch_available:  # pragma: no cover - optional path
            t = self.torch
            self.mlp_t.train()
            xb = t.from_numpy(x).to(self.device)
            yb = t.from_numpy(y).to(self.device)
            self.opt.zero_grad()
            if getattr(self, "_use_amp", False):
                # Prefer torch.autocast (new API), fallback to torch.cuda.amp.autocast
                try:
                    try:
                        cm = t.autocast(device_type="cuda")  # type: ignore[attr-defined]
                    except Exception:
                        cm = t.autocast("cuda")  # type: ignore[attr-defined]
                except Exception:
                    cm = t.cuda.amp.autocast()
                with cm:
                    yp = self.mlp_t(xb)
                    loss = self.loss_fn(yp, yb)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                loss_value = float(loss.item())
            else:
                yp = self.mlp_t(xb)
                loss = self.loss_fn(yp, yb)
                loss.backward()
                self.opt.step()
                loss_value = float(loss.item())
            self._maybe_adjust_width(loss_value)
            return loss_value
        else:
            loss_value = self.model_np.step(x, y, self.cfg.lr)
            self._maybe_adjust_width(loss_value)
            return loss_value

    def train_step_with_cost(self, rb: ReplayBuffer) -> tuple[float, float]:
        """Run one step and return (loss, cost)."""
        batch = max(1, self.cfg.batch_size)
        x, y = self._batch_from_replay(rb, batch)
        if self.torch_available:  # pragma: no cover - optional path
            t = self.torch
            self.mlp_t.train()
            xb = t.from_numpy(x).to(self.device)
            yb = t.from_numpy(y).to(self.device)
            self.opt.zero_grad()
            if getattr(self, "_use_amp", False):
                try:
                    try:
                        cm = t.autocast(device_type="cuda")  # type: ignore[attr-defined]
                    except Exception:
                        cm = t.autocast("cuda")  # type: ignore[attr-defined]
                except Exception:
                    cm = t.cuda.amp.autocast()
                with cm:
                    yp = self.mlp_t(xb)
                    loss_t = self.loss_fn(yp, yb)
                self.scaler.scale(loss_t).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                loss = float(loss_t.item())
            else:
                yp = self.mlp_t(xb)
                loss_t = self.loss_fn(yp, yb)
                loss_t.backward()
                self.opt.step()
                loss = float(loss_t.item())
            self._maybe_adjust_width(loss)
        else:
            loss = self.model_np.step(x, y, self.cfg.lr)
            self._maybe_adjust_width(loss)
        cost = self._compute_cost(x)
        self.total_cost += cost
        return loss, cost

    def train(self, rb: ReplayBuffer, steps: Optional[int] = None) -> float:
        """Run training for specified number of steps. Returns final loss."""
        n = steps or self.cfg.max_steps
        loss = 0.0
        for _ in range(n):
            loss = self.train_step(rb)
        return loss

    def train_with_budgets(self, rb: ReplayBuffer, steps: Optional[int] = None) -> dict:
        """Budget-aware training loop. Returns summary dict.

        Returns keys: loss, total_cost, over_budget(bool), steps_run
        """
        n = steps or self.cfg.max_steps
        self.total_cost = 0.0
        last_loss = 0.0
        over = False
        ran = 0
        for i in range(n):
            last_loss, c = self.train_step_with_cost(rb)
            ran = i + 1
            if self.total_cost > self.cfg.cost_budget:
                over = True
                break
        return {
            "loss": float(last_loss),
            "total_cost": float(self.total_cost),
            "over_budget": over,
            "steps_run": ran,
        }
