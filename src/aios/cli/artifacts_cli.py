from __future__ import annotations

from typing import Optional

import typer

from aios.memory.store import get_db, init_db, list_artifacts, get_artifact_by_id


def artifacts_list(
    kind: Optional[str] = typer.Option(None, "--kind", help="Filter by artifact kind"),
    limit: int = typer.Option(20, "--limit", help="Max items to list"),
    unit: Optional[str] = typer.Option(None, "--unit", help="Filter by unit field in data"),
    label: Optional[str] = typer.Option(None, "--label", help="Filter by label"),
    compact: bool = typer.Option(False, "--compact", help="Compact output (only id, kind, label, created_ts)"),
):
    conn = get_db()
    init_db(conn)
    try:
        rows = list_artifacts(conn, kind=kind, limit=limit)
        def _match(it: dict) -> bool:
            if label is not None and str(it.get("label") or "") != label:
                return False
            if unit is not None:
                data = it.get("data") or {}
                if str(data.get("unit") or "") != unit:
                    return False
            return True
        rows = [it for it in rows if _match(it)]
        if compact:
            items = [
                {"artifact_id": it.get("artifact_id"), "kind": it.get("kind"), "label": it.get("label"), "created_ts": it.get("created_ts")}
                for it in rows
            ]
        else:
            items = rows
        print({"count": len(items), "items": items})
    finally:
        conn.close()


def artifacts_show_latest(
    kind: str = typer.Option("journal_summary", "--kind", help="Artifact kind to match"),
    unit: Optional[str] = typer.Option(None, "--unit", help="Filter by unit name in data"),
    label: Optional[str] = typer.Option(None, "--label", help="Filter by label"),
):
    conn = get_db()
    init_db(conn)
    try:
        items = list_artifacts(conn, kind=kind, limit=200)
        def _match(it: dict) -> bool:
            if label is not None and str(it.get("label") or "") != label:
                return False
            if unit is not None:
                data = it.get("data") or {}
                if str(data.get("unit") or "") != unit:
                    return False
            return True
        sel = next((it for it in items if _match(it)), None)
        if sel is None:
            print({"found": False, "reason": "no match"})
        else:
            print({"found": True, "item": sel})
    finally:
        conn.close()


def artifacts_show(artifact_id: int = typer.Argument(..., help="Artifact ID")):
    conn = get_db()
    init_db(conn)
    try:
        item = get_artifact_by_id(conn, int(artifact_id))
        if item is None:
            print({"found": False, "reason": "not found", "artifact_id": int(artifact_id)})
        else:
            print({"found": True, "item": item})
    finally:
        conn.close()


def register(app: typer.Typer) -> None:
    app.command("artifacts-list")(artifacts_list)
    app.command("artifacts-show-latest")(artifacts_show_latest)
    app.command("artifacts-show")(artifacts_show)
