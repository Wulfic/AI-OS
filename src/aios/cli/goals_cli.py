from __future__ import annotations

import json as _json

import typer

from aios.memory.store import get_db, init_db
from aios.core.directives import (
    add_directive, 
    list_directives, 
    deactivate_directive, 
    deactivate_directives_like,
    link_directive_to_expert,
    unlink_directive_from_expert,
    get_directives_for_expert,
    get_experts_for_directive
)


def goals_add(
    text: str = typer.Argument(..., help="Plain-English directive that biases training focus, e.g., 'focus on service reliability'"),
    expert_id: str = typer.Option(None, "--expert-id", help="Expert ID to link this goal to"),
):
    conn = get_db()
    init_db(conn)
    try:
        directive_id = add_directive(conn, text, expert_id=expert_id)
        result = {"added": directive_id, "text": text}
        if expert_id:
            result["expert_id"] = expert_id
        print(result)
    finally:
        conn.close()


def goals_list():
    conn = get_db()
    init_db(conn)
    try:
        ds = list_directives(conn, active_only=True)
        items = []
        for d in ds:
            item = {
                "id": d.directive_id,
                "text": d.text,
                "created_ts": d.created_ts,
                "active": bool(d.active),
                "protected": False,  # No protected goals anymore
            }
            if d.expert_id:
                item["expert_id"] = d.expert_id
            items.append(item)
        print(_json.dumps({"count": len(ds), "directives": [d.text for d in ds], "items": items}, ensure_ascii=False))
    finally:
        conn.close()


def goals_remove(directive_id: int = typer.Argument(..., help="Directive ID to deactivate (soft-delete)")):
    conn = get_db()
    init_db(conn)
    try:
        ok = deactivate_directive(conn, int(directive_id))
        print(_json.dumps({"removed": ok, "id": int(directive_id)}, ensure_ascii=False))
    finally:
        conn.close()


def goals_remove_like(text_substr: str = typer.Argument(..., help="Case-insensitive substring to match in directive text")):
    conn = get_db()
    init_db(conn)
    try:
        n = deactivate_directives_like(conn, text_substr)
        print(_json.dumps({"removed_count": int(n), "match": text_substr}, ensure_ascii=False))
    finally:
        conn.close()


def goals_link_expert(
    directive_id: int = typer.Argument(..., help="Directive ID to link to expert"),
    expert_id: str = typer.Argument(..., help="Expert ID to link"),
):
    """Link a goal/directive to a specific expert."""
    conn = get_db()
    init_db(conn)
    try:
        success = link_directive_to_expert(conn, directive_id, expert_id)
        print(_json.dumps({"success": success, "directive_id": directive_id, "expert_id": expert_id}, ensure_ascii=False))
    finally:
        conn.close()


def goals_unlink_expert(directive_id: int = typer.Argument(..., help="Directive ID to unlink from expert")):
    """Remove expert association from a goal/directive."""
    conn = get_db()
    init_db(conn)
    try:
        success = unlink_directive_from_expert(conn, directive_id)
        print(_json.dumps({"success": success, "directive_id": directive_id}, ensure_ascii=False))
    finally:
        conn.close()


def goals_list_for_expert(expert_id: str = typer.Argument(..., help="Expert ID to get goals for")):
    """List all goals linked to a specific expert."""
    conn = get_db()
    init_db(conn)
    try:
        ds = get_directives_for_expert(conn, expert_id, active_only=True)
        items = []
        for d in ds:
            items.append({
                "id": d.directive_id,
                "text": d.text,
                "created_ts": d.created_ts,
                "active": bool(d.active),
                "expert_id": d.expert_id,
            })
        print(_json.dumps({"count": len(ds), "expert_id": expert_id, "directives": [d.text for d in ds], "items": items}, ensure_ascii=False))
    finally:
        conn.close()


def goals_get_expert(directive_id: int = typer.Argument(..., help="Directive ID to get expert for")):
    """Get the expert ID linked to a specific goal/directive."""
    conn = get_db()
    init_db(conn)
    try:
        expert_id = get_experts_for_directive(conn, directive_id)
        print(_json.dumps({"directive_id": directive_id, "expert_id": expert_id}, ensure_ascii=False))
    finally:
        conn.close()


def register(app: typer.Typer) -> None:
    app.command("goals-add")(goals_add)
    app.command("goals-list")(goals_list)
    app.command("goals-remove")(goals_remove)
    app.command("goals-remove-like")(goals_remove_like)
    app.command("goals-link-expert")(goals_link_expert)
    app.command("goals-unlink-expert")(goals_unlink_expert)
    app.command("goals-list-for-expert")(goals_list_for_expert)
    app.command("goals-get-expert")(goals_get_expert)
