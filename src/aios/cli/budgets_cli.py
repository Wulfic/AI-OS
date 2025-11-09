from __future__ import annotations

from typing import Optional

import typer

from aios.cli.utils import load_config
from aios.core.budgets import SafetyBudget, defaults_for_risk_tier
from aios.memory.store import get_db, init_db, load_budgets, load_budget_usage, save_budgets


def budgets_show():
    """Show current budget limits and usage summary."""
    cfg = load_config(None)
    tier = cfg.get("risk_tier", "conservative")
    conn = get_db()
    init_db(conn)
    try:
        persisted = load_budgets(conn)
        usage = load_budget_usage(conn)
        limits = {**defaults_for_risk_tier(tier), **persisted}
        sb = SafetyBudget(limits=limits, usage=usage)
        print({"risk_tier": tier, "budgets": sb.summary()})
    finally:
        conn.close()


def budgets_set(
    domain: str = typer.Argument(..., help="Budget domain (e.g., file_writes, service_changes, pkg_ops, privileged_calls)"),
    limit: float = typer.Argument(..., help="New budget limit (use a large number for 'infinite')"),
):
    """Set/override a budget limit for a domain and persist it."""
    conn = get_db()
    init_db(conn)
    try:
        save_budgets(conn, {domain: float(limit)})
        print({"updated": {domain: float(limit)}})
    finally:
        conn.close()


def budgets_reset_tier():
    """Reset budgets to defaults for current risk_tier from config."""
    cfg = load_config(None)
    tier = cfg.get("risk_tier", "conservative")
    conn = get_db()
    init_db(conn)
    try:
        save_budgets(conn, defaults_for_risk_tier(tier))
        print({"reset_to": tier})
    finally:
        conn.close()


def budgets_usage():
    """Show current budget usage per domain from SQLite."""
    conn = get_db()
    init_db(conn)
    try:
        usage = load_budget_usage(conn)
        print({"usage": usage})
    finally:
        conn.close()


def register(app: typer.Typer) -> None:
    app.command("budgets-show")(budgets_show)
    app.command("budgets-set")(budgets_set)
    app.command("budgets-reset-tier")(budgets_reset_tier)
    app.command("budgets-usage")(budgets_usage)
