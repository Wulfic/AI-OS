"""NumpyMLPBrain - Thin wrapper around Trainer as a brain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking only (avoid import cycles at runtime)
    from aios.core.train import Trainer, TrainConfig  # pragma: no cover - typing only


@dataclass
class NumpyMLPBrain:
    """Thin wrapper around aios.core.train.Trainer as a 'brain'.

    This provides a uniform interface and allows dynamic instantiation with different widths.
    """

    name: str
    modalities: List[str]
    cfg: "TrainConfig"
    _trainer: Optional["Trainer"] = None

    def _trainer_ready(self) -> "Trainer":
        if self._trainer is None:
            from aios.core.train import Trainer

            self._trainer = Trainer(self.cfg)
        return self._trainer

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        tr = self._trainer_ready()
        # Ensure minimal dims for our featurizer expectations
        if getattr(tr.cfg, "input_dim", 0) < 2:
            tr.cfg.input_dim = 2
            # rebuild to apply new input dim
            from aios.core.train import NumpyMLP  # type: ignore
            tr.model_np = NumpyMLP(tr.cfg.input_dim, tr.cfg.hidden, tr.cfg.output_dim)
        if getattr(tr.cfg, "output_dim", 0) < 1:
            tr.cfg.output_dim = 1
            from aios.core.train import NumpyMLP  # type: ignore
            tr.model_np = NumpyMLP(tr.cfg.input_dim, tr.cfg.hidden, tr.cfg.output_dim)
        # Minimal forward: do one training step on synthetic replay for now
        from aios.core.replay import ReplayBuffer
        rb = ReplayBuffer(capacity=max(64, int(self.cfg.batch_size) * 2))
        for i in range(16):
            rb.push([0], i % 3, float(i % 5), [0], False)
        loss = tr.train(rb, steps=min(5, int(self.cfg.max_steps)))
        return {"ok": True, "loss": float(loss)}

    def size_bytes(self) -> int:
        # Conservative FP32 param size estimate for the tiny MLP
        in_d = int(self.cfg.input_dim)
        out_d = int(self.cfg.output_dim)
        h = int(self.cfg.hidden)
        params = in_d * h + h + h * out_d + out_d
        return int(params * 4)
