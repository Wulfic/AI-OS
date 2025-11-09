from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import random


@dataclass
class ArmStats:
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0  # failures + 1

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        # Thompson sample from Beta(alpha, beta)
        return random.betavariate(self.alpha, self.beta)


class ThompsonBandit:
    """Simple Bernoulli Thompson Sampling bandit over named arms.

    Each arm is keyed by a string. We track Beta priors and sample to pick.
    """

    def __init__(self) -> None:
        self._arms: Dict[str, ArmStats] = {}

    def ensure_arms(self, names: Iterable[str]) -> None:
        for n in names:
            self._arms.setdefault(n, ArmStats())

    def update(self, name: str, success: bool) -> None:
        self.ensure_arms([name])
        self._arms[name].update(success)

    def choose(self, candidates: Iterable[str]) -> Optional[str]:
        cands = list(candidates)
        if not cands:
            return None
        self.ensure_arms(cands)
        # Pick the arm with highest Thompson sample
        scored = [(self._arms[n].sample(), n) for n in cands]
        scored.sort(reverse=True)
        return scored[0][1]
