from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple
import random


Transition = Tuple[Any, Any, float, Any, bool]


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self) -> None:
        self._buf: List[Transition] = []
        self._idx: int = 0

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, s: Any, a: Any, r: float, s2: Any, done: bool) -> None:
        t = (s, a, r, s2, done)
        if len(self._buf) < self.capacity:
            self._buf.append(t)
        else:
            self._buf[self._idx] = t
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        n = min(batch_size, len(self._buf))
        return random.sample(self._buf, n)
