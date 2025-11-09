from __future__ import annotations

import sqlite3
from typing import Optional

from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.memory.store import load_budgets, load_budget_usage, update_budget_usage


def install(
    name: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    simulate: bool = True,
    cost: float = 1.0,
) -> bool:
    """Simulate installing a package with pkg_ops budget enforcement."""
    return _pkg_op("install", name, cfg=cfg, conn=conn, simulate=simulate, cost=cost)


def remove(
    name: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    simulate: bool = True,
    cost: float = 1.0,
) -> bool:
    """Simulate removing a package with pkg_ops budget enforcement."""
    return _pkg_op("remove", name, cfg=cfg, conn=conn, simulate=simulate, cost=cost)


def _pkg_op(
    action: str,
    name: str,
    *,
    cfg: Optional[dict],
    conn: Optional[sqlite3.Connection],
    simulate: bool,
    cost: float,
) -> bool:
    tier = (cfg or {}).get("risk_tier", "conservative")
    limits = defaults_for_risk_tier(tier)
    used = {}
    if conn is not None:
        try:
            limits.update(load_budgets(conn))
            used = load_budget_usage(conn)
        except Exception:
            pass
    sb = SafetyBudget(limits=limits)
    for d, v in used.items():
        sb.usage[d] = float(v)
    if not sb.allow("pkg_ops", cost):
        raise PermissionError("Budget exceeded for pkg_ops")

    ok = True  # simulate success

    if ok and conn is not None:
        try:
            update_budget_usage(conn, "pkg_ops", cost)
        except Exception:
            pass
    return ok
