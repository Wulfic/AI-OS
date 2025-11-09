from __future__ import annotations

import sqlite3
from typing import Callable, Optional, TypeVar

from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.memory.store import load_budgets, load_budget_usage, update_budget_usage

T = TypeVar("T")


def run_privileged(
    fn: Callable[[], T],
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    cost: float = 1.0,
) -> T:
    """Execute a privileged function while enforcing `privileged_calls` budget.

    The provided function is expected to perform a privileged operation (or proxy call).
    This wrapper enforces budget before running, and records usage on success.
    """
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
    if not sb.allow("privileged_calls", cost):
        raise PermissionError("Budget exceeded for privileged_calls")

    result = fn()
    if conn is not None:
        try:
            update_budget_usage(conn, "privileged_calls", cost)
        except Exception:
            pass
    return result
