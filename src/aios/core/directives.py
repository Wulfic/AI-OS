from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Directive:
    directive_id: int
    text: str
    created_ts: str
    active: bool
    expert_id: Optional[str] = None  # Link to expert (Dynamic Subbrains)


def _ensure_expert_id_column(conn: sqlite3.Connection) -> None:
    """Ensure the expert_id column exists in directives table (migration)."""
    try:
        # Check if column exists
        cursor = conn.execute("PRAGMA table_info(directives)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "expert_id" not in columns:
            # Add column if it doesn't exist
            conn.execute("ALTER TABLE directives ADD COLUMN expert_id TEXT")
            conn.commit()
    except Exception:
        # Column might already exist or table might not exist yet
        pass


def add_directive(conn: sqlite3.Connection, text: str, expert_id: Optional[str] = None) -> int:
    """Add a new directive (goal).
    
    Args:
        conn: Database connection
        text: Directive text
        expert_id: Optional expert ID to link this goal to
    
    Returns:
        The ID of the newly created directive
    """
    clean = text.strip()
    if not clean:
        raise ValueError("Directive text cannot be empty")
    
    # Ensure expert_id column exists (migration)
    _ensure_expert_id_column(conn)
    
    cur = conn.execute(
        "INSERT INTO directives(text, active, expert_id) VALUES (?, 1, ?)",
        (clean, expert_id),
    )
    conn.commit()
    last_id = (
        cur.lastrowid
        if cur.lastrowid is not None
        else conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    )
    return int(last_id)


def list_directives(
    conn: sqlite3.Connection, active_only: bool = True, expert_id: Optional[str] = None
) -> List[Directive]:
    """List directives (goals).
    
    Args:
        conn: Database connection
        active_only: If True, only return active directives
        expert_id: If provided, only return directives linked to this expert
    
    Returns:
        List of Directive objects
    """
    # Ensure expert_id column exists (migration)
    _ensure_expert_id_column(conn)
    
    q = "SELECT directive_id, text, created_ts, active, expert_id FROM directives WHERE 1=1"
    params = []
    
    if active_only:
        q += " AND active=1"
    
    if expert_id is not None:
        q += " AND expert_id=?"
        params.append(expert_id)
    
    q += " ORDER BY directive_id DESC"
    
    rows = conn.execute(q, params).fetchall()
    return [Directive(r[0], r[1], r[2], bool(r[3]), r[4]) for r in rows]


def deactivate_directive(conn: sqlite3.Connection, directive_id: int) -> bool:
    """Mark a directive as inactive (soft-delete).

    Returns True if a row was updated.
    """
    cur = conn.execute(
        "UPDATE directives SET active=0 WHERE directive_id=?",
        (int(directive_id),),
    )
    conn.commit()
    return cur.rowcount > 0


def deactivate_directives_like(conn: sqlite3.Connection, text_substr: str) -> int:
    """Deactivate directives whose text contains the given substring (case-insensitive).

    Returns the number of rows updated.
    """
    pat = f"%{text_substr.strip()}%"
    cur = conn.execute(
        "UPDATE directives SET active=0 WHERE text LIKE ?",
        (pat,),
    )
    conn.commit()
    return cur.rowcount or 0


def link_directive_to_expert(conn: sqlite3.Connection, directive_id: int, expert_id: str) -> bool:
    """Link a directive (goal) to an expert.
    
    Args:
        conn: Database connection
        directive_id: ID of the directive to link
        expert_id: ID of the expert to link to
    
    Returns:
        True if successful, False otherwise
    """
    _ensure_expert_id_column(conn)
    
    try:
        cur = conn.execute(
            "UPDATE directives SET expert_id=? WHERE directive_id=?",
            (expert_id, int(directive_id)),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        return False


def unlink_directive_from_expert(conn: sqlite3.Connection, directive_id: int) -> bool:
    """Remove expert association from a directive.
    
    Args:
        conn: Database connection
        directive_id: ID of the directive to unlink
    
    Returns:
        True if successful, False otherwise
    """
    _ensure_expert_id_column(conn)
    
    try:
        cur = conn.execute(
            "UPDATE directives SET expert_id=NULL WHERE directive_id=?",
            (int(directive_id),),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        return False


def get_directives_for_expert(conn: sqlite3.Connection, expert_id: str, active_only: bool = True) -> List[Directive]:
    """Get all directives linked to a specific expert.
    
    Args:
        conn: Database connection
        expert_id: ID of the expert
        active_only: If True, only return active directives
    
    Returns:
        List of Directive objects
    """
    return list_directives(conn, active_only=active_only, expert_id=expert_id)


def get_experts_for_directive(conn: sqlite3.Connection, directive_id: int) -> Optional[str]:
    """Get the expert ID linked to a directive.
    
    Args:
        conn: Database connection
        directive_id: ID of the directive
    
    Returns:
        Expert ID if linked, None otherwise
    """
    _ensure_expert_id_column(conn)
    
    try:
        row = conn.execute(
            "SELECT expert_id FROM directives WHERE directive_id=?",
            (int(directive_id),),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None
