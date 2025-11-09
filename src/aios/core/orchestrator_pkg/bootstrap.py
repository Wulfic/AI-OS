from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from aios.memory.store import (
    get_db,
    init_db,
    load_budgets,
    save_budgets,
    load_budget_usage,
)
from aios.core.budgets import SafetyBudget, defaults_for_risk_tier


def open_db(db_path: str | None):
    conn = get_db(Path(db_path).expanduser() if db_path else None)
    init_db(conn)
    return conn


def compute_limits_and_usage(config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    tier = config.get("risk_tier", "conservative")
    try:
        conn = get_db()
        init_db(conn)
        persisted = load_budgets(conn)
        used = load_budget_usage(conn)
    except Exception:
        persisted = {}
        used = {}
    finally:
        try:
            conn.close()  # type: ignore[has-type]
        except Exception:
            pass
    cfg_limits = {}
    try:
        raw_cfg_limits = config.get("budgets", {}) or {}
        cfg_limits = {str(k): float(v) for k, v in raw_cfg_limits.items()}
    except Exception:
        cfg_limits = {}
    # precedence: defaults < config overrides < persisted overrides
    limits = {**defaults_for_risk_tier(tier), **cfg_limits, **persisted}
    return limits, used


def ensure_budgets_in_db(conn, *, config: Dict[str, Any]) -> None:
    tier = config.get("risk_tier", "conservative")
    existing = load_budgets(conn)
    if not existing:
        cfg_limits = {}
        try:
            raw_cfg_limits = config.get("budgets", {}) or {}
            cfg_limits = {str(k): float(v) for k, v in raw_cfg_limits.items()}
        except Exception:
            cfg_limits = {}
        merged = {**defaults_for_risk_tier(tier), **cfg_limits}
        save_budgets(conn, merged)
