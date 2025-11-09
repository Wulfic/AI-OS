from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from aios.policies.bandit import ThompsonBandit


class Operator(Protocol):
    name: str

    async def run(self, context: Dict[str, Any]) -> bool:  # returns success
        ...


@dataclass
class SimpleOperator:
    name: str
    func: Callable[[Dict[str, Any]], Any]

    async def run(self, context: Dict[str, Any]) -> bool:
        try:
            out = self.func(context)
            return bool(out)
        except Exception:
            return False


class OperatorRegistry:
    def __init__(self) -> None:
        self._ops: Dict[str, Operator] = {}

    def register(self, op: Operator) -> None:
        self._ops[op.name] = op

    def get(self, name: str) -> Optional[Operator]:
        return self._ops.get(name)

    def names(self) -> List[str]:
        return list(self._ops.keys())


class Manager:
    """Minimal HRM manager selecting among operators via Thompson bandit."""

    def __init__(
        self, registry: OperatorRegistry, recorder: Optional["Recorder"] = None
    ) -> None:
        self.registry = registry
        self.bandit = ThompsonBandit()
        self.recorder = recorder

    async def act(
        self, context: Dict[str, Any], candidates: Optional[List[str]] = None
    ) -> Optional[str]:
        names = candidates or self.registry.names()
        pick = self.bandit.choose(names)
        if pick is None:
            return None
        op = self.registry.get(pick)
        if op is None:
            return None
        success = await op.run(context)
        self.bandit.update(pick, success)
        if self.recorder is not None:
            try:
                await self.recorder.record(pick, success)
            except Exception:
                pass
        return pick


class Recorder:
    async def record(
        self, operator_name: str, success: bool
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class AsyncOperator:
    name: str
    async_func: Callable[[Dict[str, Any]], Any]

    async def run(self, context: Dict[str, Any]) -> bool:
        try:
            out = await self.async_func(context)
            return bool(out)
        except Exception:
            return False


async def build_default_manager_with_recorder(conn=None) -> Manager:
    """Helper to build Manager with default registry and optional SqliteRecorder."""
    from .operators_builtin import build_default_registry

    reg = build_default_registry()
    recorder = None
    if conn is not None:
        try:
            from aios.memory.recorder import SqliteRecorder

            recorder = SqliteRecorder(conn)
        except Exception:
            recorder = None
    return Manager(registry=reg, recorder=recorder)
