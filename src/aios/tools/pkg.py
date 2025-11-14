from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.memory.store import load_budgets, load_budget_usage, update_budget_usage

logger = logging.getLogger(__name__)


def install(
    name: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    simulate: bool = True,
    cost: float = 1.0,
) -> bool:
    """Simulate installing a package with pkg_ops budget enforcement."""
    logger.info(f"Package install requested: {name} (simulate={simulate}, cost={cost})")
    result = _pkg_op("install", name, cfg=cfg, conn=conn, simulate=simulate, cost=cost)
    if result:
        logger.info(f"Package install {'simulated' if simulate else 'completed'}: {name}")
    else:
        logger.error(f"Package install failed: {name}")
    return result


def remove(
    name: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    simulate: bool = True,
    cost: float = 1.0,
) -> bool:
    """Simulate removing a package with pkg_ops budget enforcement."""
    logger.info(f"Package removal requested: {name} (simulate={simulate}, cost={cost})")
    result = _pkg_op("remove", name, cfg=cfg, conn=conn, simulate=simulate, cost=cost)
    if result:
        logger.info(f"Package removal {'simulated' if simulate else 'completed'}: {name}")
    else:
        logger.error(f"Package removal failed: {name}")
    return result


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
    logger.debug(f"Package operation: {action} {name} (tier={tier}, cost={cost})")
    limits = defaults_for_risk_tier(tier)
    used = {}
    if conn is not None:
        try:
            limits.update(load_budgets(conn))
            used = load_budget_usage(conn)
            logger.debug(f"Budget limits loaded from database")
        except Exception as e:
            logger.warning(f"Failed to load budgets from database: {e}")
    sb = SafetyBudget(limits=limits)
    for d, v in used.items():
        sb.usage[d] = float(v)
    if not sb.allow("pkg_ops", cost):
        logger.error(f"Budget exceeded for pkg_ops (requested={cost}, available={sb.limits.get('pkg_ops', 0)})")
        raise PermissionError("Budget exceeded for pkg_ops")

    ok = True  # simulate success

    if ok and conn is not None:
        try:
            update_budget_usage(conn, "pkg_ops", cost)
            logger.debug(f"Budget usage updated: pkg_ops +{cost}")
        except Exception as e:
            logger.warning(f"Failed to update budget usage: {e}")
    return ok
