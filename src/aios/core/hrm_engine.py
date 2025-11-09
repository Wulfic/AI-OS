from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aios.core.hrm import Manager, OperatorRegistry, build_default_registry


def _get_mode_from_config(cfg: Optional[dict]) -> str:
    # Vendor mode removed; always operate in builtin mode.
    return "builtin"


@dataclass
class HRMEngine:
    """Unified HRM facade (builtin only)."""

    config: Optional[dict] = None

    def __post_init__(self) -> None:
        self.mode = _get_mode_from_config(self.config)
        self._mgr: Optional[Manager] = None
        self._reg: Optional[OperatorRegistry] = None

    # --- Introspection ---
    def info(self) -> Dict[str, Any]:
        reg = self._registry()
        return {
            "mode": "builtin",
            "operators": reg.names(),
            "policy": "ThompsonBandit",
        }

    # --- Builtin path ---
    def _registry(self) -> OperatorRegistry:
        if self._reg is None:
            self._reg = build_default_registry()
        return self._reg

    def _manager(self) -> Manager:
        if self._mgr is None:
            self._mgr = Manager(self._registry())
        return self._mgr

    async def act(self, context: Dict[str, Any], candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select and run an operator from the builtin registry."""
        pick = await self._manager().act(context, candidates)
        return {"mode": "builtin", "picked": pick}

    # --- Legacy vendor stubs (no-op) ---
    def setup(self, force: bool = False) -> Dict[str, Any]:
        return {"ok": True, "mode": "builtin", "note": "vendor removed"}

    def pretrain(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"started": False, "error": "vendor pretrain removed; use 'aios train'"}

    def evaluate(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        return {"returncode": None, "error": "vendor evaluate removed; use core HRM ops"}
