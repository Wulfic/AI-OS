from __future__ import annotations

import sqlite3
from typing import Optional

from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.memory.store import load_budgets, load_budget_usage, update_budget_usage


def restart_service(
    name: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    simulate: bool = True,
    cost: float = 1.0,
) -> bool:
    """Simulate a service restart with budget enforcement.

    This is a portable stub that enforces budgets and records usage.
    On Linux, a future non-simulated path would shell out to systemctl or root-helper.
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
    if not sb.allow("service_changes", cost):
        raise PermissionError("Budget exceeded for service_changes")

    # simulate success
    ok = True

    # record usage if success
    if ok and conn is not None:
        try:
            update_budget_usage(conn, "service_changes", cost)
        except Exception:
            pass
    return ok
