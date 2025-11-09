"""Sleep cycle and consolidation methods for Trainer."""

from __future__ import annotations

from ..replay import ReplayBuffer


class SleepCycleMixin:
    """Mixin providing sleep cycle functionality for Trainer."""

    def sleep_cycle(self, rb: ReplayBuffer) -> None:
        """Perform a light 'sleep' phase: synaptic downscaling + optional consolidation on replay.

        - Downscale all weights slightly toward zero (synaptic homeostasis-inspired)
        - Optionally run a few consolidation steps using the same replay buffer
        """
        # numpy downscale
        s = float(max(0.0, self.cfg.sleep_downscale))
        if s > 0.0:
            self.model_np.W1 *= (1.0 - s)
            self.model_np.b1 *= (1.0 - s)
            self.model_np.W2 *= (1.0 - s)
            self.model_np.b2 *= (1.0 - s)
        # torch downscale
        if self.torch_available:  # pragma: no cover - optional
            t = self.torch
            try:
                with t.no_grad():
                    for p in self._torch_params():
                        p.mul_(1.0 - s)
                # small empty_cache to release fragmentation
                if self.device.type == "cuda":
                    t.cuda.empty_cache()
            except Exception:
                pass
        # consolidation steps (reuse normal train_step to keep code simple)
        k = max(0, int(self.cfg.sleep_consolidation_steps))
        for _ in range(k):
            try:
                self.train_step(rb)
            except Exception:
                break
