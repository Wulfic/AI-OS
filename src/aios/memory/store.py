from __future__ import annotations
import sqlite3
import threading
from pathlib import Path
import os
from typing import Optional, Iterator
from contextlib import contextmanager


def _default_db_path() -> Path:
    # Priority: explicit override via env, then XDG_DATA_HOME, then HOME/USERPROFILE
    env_db = os.environ.get("AIOS_DB_PATH")
    if env_db:
        return Path(env_db)
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        base = Path(xdg) / "aios"
        return base / "aios.db"
    # Fallbacks: prefer HOME, else USERPROFILE (Windows), else Path.home()
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        return Path(home) / ".local" / "share" / "aios" / "aios.db"
    return Path.home() / ".local" / "share" / "aios" / "aios.db"


def ensure_dirs(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    ensure_dirs(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db_connection(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Context manager for database connections with automatic cleanup.
    
    Usage:
        with db_connection() as conn:
            # Use connection
            conn.execute(...)
        # Automatically closed and committed here
    
    On exception, automatically rolls back. On success, commits.
    """
    conn = None
    try:
        conn = get_db(db_path)
        yield conn
        conn.commit()  # Commit on success
    except Exception:
        if conn:
            try:
                conn.rollback()  # Rollback on error
            except Exception:
                pass
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


class ConnectionPool:
    """Thread-safe connection pool for SQLite.
    
    Reuses connections to reduce overhead and prevent resource exhaustion.
    """
    
    def __init__(self, db_path: Optional[Path] = None, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._in_use: set[sqlite3.Connection] = set()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        with self._lock:
            # Reuse existing connection if available
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(conn)
                return conn
            
            # Create new connection if under limit
            if len(self._in_use) < self.max_connections:
                conn = get_db(self.db_path)
                self._in_use.add(conn)
                return conn
            
            # Pool exhausted
            raise RuntimeError(
                f"Connection pool exhausted (max {self.max_connections} connections). "
                f"This may indicate a connection leak - ensure connections are properly returned."
            )
    
    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.append(conn)
    
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for pooled connections."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self.return_connection(conn)
    
    def close_all(self) -> None:
        """Close all connections in pool (call on shutdown)."""
        with self._lock:
            for conn in self._pool + list(self._in_use):
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()
            self._in_use.clear()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts TEXT,
            end_ts TEXT,
            goal TEXT,
            success INTEGER,
            reward REAL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            title TEXT,
            fetched_ts TEXT,
            text TEXT,
            hash TEXT,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS operators (
            op_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            yaml TEXT,
            created_ts TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS operators_name_version ON operators(name, version);
        CREATE TABLE IF NOT EXISTS operator_usage (
            op_id INTEGER NOT NULL,
            context_hash TEXT,
            success INTEGER DEFAULT 0,
            trials INTEGER DEFAULT 0,
            lat_sum_ms REAL DEFAULT 0,
            PRIMARY KEY (op_id, context_hash),
            FOREIGN KEY (op_id) REFERENCES operators(op_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS directives (
            directive_id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS budgets (
            domain TEXT PRIMARY KEY,
            limit_value REAL,
            used REAL NOT NULL DEFAULT 0,
            updated_ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
        -- Generic artifacts storage for structured operator outputs (e.g., journal summaries)
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            label TEXT,
            data_json TEXT NOT NULL,
            created_ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
        CREATE INDEX IF NOT EXISTS artifacts_kind_created ON artifacts(kind, created_ts DESC);
        """
    )
    conn.commit()


def load_budgets(conn: sqlite3.Connection) -> dict[str, float]:
    cur = conn.execute("SELECT domain, limit_value FROM budgets")
    out: dict[str, float] = {}
    for domain, limit in cur.fetchall():
        try:
            out[str(domain)] = float(limit) if limit is not None else float("inf")
        except Exception:
            continue
    return out


def save_budgets(conn: sqlite3.Connection, limits: dict[str, float]) -> None:
    for domain, limit in limits.items():
        conn.execute(
            """
            INSERT INTO budgets (domain, limit_value, used)
            VALUES (?, ?, COALESCE((SELECT used FROM budgets WHERE domain = ?), 0))
            ON CONFLICT(domain) DO UPDATE SET limit_value=excluded.limit_value, updated_ts=strftime('%Y-%m-%dT%H:%M:%SZ','now')
            """,
            (domain, float(limit), domain),
        )
    conn.commit()


def update_budget_usage(conn: sqlite3.Connection, domain: str, delta: float) -> None:
    conn.execute(
        """
    INSERT INTO budgets (domain, limit_value, used)
    VALUES (?, NULL, ?)
        ON CONFLICT(domain) DO UPDATE SET used=budgets.used + excluded.used, updated_ts=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        """,
        (domain, float(delta)),
    )
    conn.commit()


def load_budget_usage(conn: sqlite3.Connection) -> dict[str, float]:
    cur = conn.execute("SELECT domain, used FROM budgets")
    out: dict[str, float] = {}
    for domain, used in cur.fetchall():
        try:
            out[str(domain)] = float(used)
        except Exception:
            continue
    return out


# Artifacts helpers
def save_artifact(conn: sqlite3.Connection, kind: str, label: str | None, data: dict) -> int:
    """Persist a structured artifact as JSON and return its ID.

    kind: a short category string, e.g., "journal_summary".
    label: optional label/key, e.g., operator-provided output label.
    data: JSON-serializable dict payload.
    """
    try:
        import json

        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        # As a last resort, stringify
        payload = str(data)
    cur = conn.execute(
        "INSERT INTO artifacts (kind, label, data_json) VALUES (?, ?, ?)",
        (str(kind), str(label) if label is not None else None, payload),
    )
    conn.commit()
    rid = cur.lastrowid
    return int(rid) if rid is not None else 0


def list_artifacts(
    conn: sqlite3.Connection, kind: str | None = None, limit: int = 50
) -> list[dict]:
    """List recent artifacts, optionally filtered by kind."""
    q = "SELECT artifact_id, kind, label, data_json, created_ts FROM artifacts"
    args: list = []
    if kind:
        q += " WHERE kind = ?"
        args.append(str(kind))
    q += " ORDER BY created_ts DESC, artifact_id DESC LIMIT ?"
    args.append(int(limit))
    rows = conn.execute(q, args).fetchall()
    out: list[dict] = []
    import json

    for artifact_id, k, label, data_json, ts in rows:
        try:
            data = json.loads(data_json)
        except Exception:
            data = {"raw": data_json}
        out.append(
            {
                "artifact_id": int(artifact_id),
                "kind": str(k),
                "label": label if label is None else str(label),
                "data": data,
                "created_ts": str(ts),
            }
        )
    return out


def get_artifact_by_id(conn: sqlite3.Connection, artifact_id: int) -> dict | None:
    """Return a single artifact by ID or None if not found."""
    row = conn.execute(
        "SELECT artifact_id, kind, label, data_json, created_ts FROM artifacts WHERE artifact_id = ?",
        (int(artifact_id),),
    ).fetchone()
    if not row:
        return None
    import json

    aid, k, label, data_json, ts = row
    try:
        data = json.loads(data_json)
    except Exception:
        data = {"raw": data_json}
    return {
        "artifact_id": int(aid),
        "kind": str(k),
        "label": label if label is None else str(label),
        "data": data,
        "created_ts": str(ts),
    }
