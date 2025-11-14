from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.core.guards import enforce_write_allowed
from aios.memory.store import update_budget_usage, load_budgets, load_budget_usage

logger = logging.getLogger(__name__)


def write_text(
    path: str | Path,
    data: str,
    *,
    cfg: Optional[dict] = None,
    conn: Optional[sqlite3.Connection] = None,
    cost: float = 1.0,
) -> None:
    """Write text to a file with guard and budget enforcement.

    - Enforces write guard allow/deny rules from cfg.guards
    - Enforces SafetyBudget for domain 'file_writes' using limits from DB overrides or tier defaults
    - Persists budget usage in DB if conn provided
    """
    logger.info(f"Writing file: {path}")
    p = Path(path).expanduser()
    
    try:
        enforce_write_allowed(p, cfg or {})
    except PermissionError as e:
        logger.error(f"Write guard denied file write to {path}: {e}")
        raise

    # Budget enforcement (deny if over budget)
    tier = (cfg or {}).get("risk_tier", "conservative")
    limits = defaults_for_risk_tier(tier)
    used = {}
    if conn is not None:
        try:
            persisted = load_budgets(conn)
            limits.update(persisted)
            used = load_budget_usage(conn)
        except Exception as e:
            logger.warning(f"Failed to load budgets from database: {e}")
    sb = SafetyBudget(limits=limits)
    # incorporate used so allow() reflects remaining budget
    for d, v in used.items():
        sb.usage[d] = float(v)
    if not sb.allow("file_writes", cost):
        logger.error(f"Budget exceeded for file_writes: {path}")
        raise PermissionError("Budget exceeded for file_writes")

    # Perform write
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")
        logger.info(f"File written successfully: {path} ({len(data)} bytes)")
    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}")
        raise

    # Record usage if possible
    if conn is not None:
        try:
            update_budget_usage(conn, "file_writes", cost)
        except Exception as e:
            logger.warning(f"Failed to update budget usage: {e}")
