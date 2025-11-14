from __future__ import annotations
import sqlite3
import threading
import time
from pathlib import Path
import os
import logging
from typing import Optional, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


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
    if not db_path.parent.exists():
        logger.debug(f"Creating database directory: {db_path.parent}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database directory created: {db_path.parent}")
    else:
        logger.debug(f"Database directory already exists: {db_path.parent}")


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    ensure_dirs(path)
    logger.debug(f"Opening database connection: {path}")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    logger.debug(f"Enabled WAL mode for database: {path}")
    conn.execute("PRAGMA foreign_keys=ON;")
    logger.debug(f"Enabled foreign key constraints for database: {path}")
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
        conn.commit()
        logger.debug("Database transaction committed successfully")
    except Exception as e:
        if conn:
            try:
                conn.rollback()
                logger.warning(f"Database transaction rolled back due to error: {e}")
            except Exception as rollback_err:
                logger.error(f"Failed to rollback transaction: {rollback_err}")
                pass
        raise
    finally:
        if conn:
            try:
                conn.close()
                logger.debug("Database connection closed")
            except Exception as close_err:
                logger.warning(f"Error closing database connection: {close_err}")
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
        # Connection pool statistics
        self._stats_reused = 0
        self._stats_created = 0
        self._stats_exhausted = 0
        logger.debug(f"Connection pool initialized: max_connections={max_connections}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        logger.debug("Acquiring connection pool lock")
        with self._lock:
            logger.debug("Connection pool lock acquired")
            # Reuse existing connection if available
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(conn)
                self._stats_reused += 1
                logger.debug(f"Reused pooled connection (in use: {len(self._in_use)}/{self.max_connections}, reuse rate: {self._get_reuse_rate():.1f}%)")
                logger.debug("Connection pool lock released")
                return conn
            
            # Create new connection if under limit
            if len(self._in_use) < self.max_connections:
                conn = get_db(self.db_path)
                self._in_use.add(conn)
                self._stats_created += 1
                logger.debug(f"Created new pooled connection (in use: {len(self._in_use)}/{self.max_connections}, total created: {self._stats_created})")
                logger.debug("Connection pool lock released")
                return conn
            
            # Pool exhausted
            self._stats_exhausted += 1
            logger.warning(f"Connection pool exhausted ({self.max_connections} connections in use, exhausted {self._stats_exhausted} times)")
            logger.debug("Connection pool lock released")
            raise RuntimeError(
                f"Connection pool exhausted (max {self.max_connections} connections). "
                f"This may indicate a connection leak - ensure connections are properly returned."
            )
    
    def _get_reuse_rate(self) -> float:
        """Calculate connection reuse percentage."""
        total_gets = self._stats_reused + self._stats_created
        if total_gets == 0:
            return 0.0
        return (self._stats_reused / total_gets) * 100
    
    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        logger.debug("Acquiring connection pool lock")
        with self._lock:
            logger.debug("Connection pool lock acquired")
            if conn in self._in_use:
                self._in_use.remove(conn)
                self._pool.append(conn)
                logger.debug(f"Returned connection to pool (available: {len(self._pool)}, in use: {len(self._in_use)})")
            else:
                logger.warning("Attempted to return connection that was not in use")
            logger.debug("Connection pool lock released")
    
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for pooled connections."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
            logger.debug("Pooled connection transaction committed")
        except Exception as e:
            try:
                conn.rollback()
                logger.warning(f"Pooled connection transaction rolled back: {e}")
            except Exception as rollback_err:
                logger.error(f"Failed to rollback pooled transaction: {rollback_err}")
                pass
            raise
        finally:
            self.return_connection(conn)
    
    def close_all(self) -> None:
        """Close all connections in pool (call on shutdown)."""
        logger.debug("Acquiring connection pool lock for close_all")
        with self._lock:
            logger.debug("Connection pool lock acquired")
            total_conns = len(self._pool) + len(self._in_use)
            logger.info(f"Closing all database connections (pool: {len(self._pool)}, in use: {len(self._in_use)})")
            
            # Log pool efficiency stats before closing
            if self._stats_created > 0 or self._stats_reused > 0:
                reuse_rate = self._get_reuse_rate()
                logger.info(f"Connection pool stats: created={self._stats_created}, reused={self._stats_reused}, "
                          f"exhausted={self._stats_exhausted}, reuse_rate={reuse_rate:.1f}%")
            
            closed_count = 0
            for conn in self._pool + list(self._in_use):
                try:
                    conn.close()
                    closed_count += 1
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            self._pool.clear()
            self._in_use.clear()
            logger.debug(f"Closed {closed_count}/{total_conns} database connection(s)")
            
            if closed_count < total_conns:
                logger.warning(f"Failed to close {total_conns - closed_count} connection(s)")
            logger.debug("Connection pool lock released")


def init_db(conn: sqlite3.Connection) -> None:
    logger.info("Initializing database schema")
    start_time = time.time()
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
    elapsed_ms = (time.time() - start_time) * 1000
    logger.debug(f"Database schema initialized successfully in {elapsed_ms:.2f}ms")
    
    # Log database file size if available
    try:
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        if db_path and os.path.exists(db_path):
            size_bytes = os.path.getsize(db_path)
            size_mb = size_bytes / (1024 * 1024)
            logger.debug(f"Database file size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    except Exception as e:
        logger.debug(f"Could not determine database file size: {e}")


def load_budgets(conn: sqlite3.Connection) -> dict[str, float]:
    logger.debug("Loading budget limits from database")
    start_time = time.time()
    cur = conn.execute("SELECT domain, limit_value FROM budgets")
    out: dict[str, float] = {}
    for domain, limit in cur.fetchall():
        try:
            out[str(domain)] = float(limit) if limit is not None else float("inf")
        except Exception as e:
            logger.warning(f"Invalid budget limit for domain '{domain}': {e}")
            continue
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Loaded {len(out)} budget limit(s) in {elapsed_ms:.2f}ms")
    return out


def save_budgets(conn: sqlite3.Connection, limits: dict[str, float]) -> None:
    logger.debug(f"Saving {len(limits)} budget limit(s) to database")
    start_time = time.time()
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
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"Saved {len(limits)} budget limit(s) in {elapsed_ms:.2f}ms")
    
    # Log if any budgets are approaching limits
    try:
        cur = conn.execute("SELECT domain, limit_value, used FROM budgets WHERE limit_value IS NOT NULL AND used > 0")
        for domain, limit, used in cur.fetchall():
            if limit and used:
                pct = (used / limit) * 100
                if pct >= 90:
                    logger.warning(f"Budget '{domain}' at {pct:.1f}% of limit ({used:.2f}/{limit:.2f})")
                elif pct >= 75:
                    logger.info(f"Budget '{domain}' at {pct:.1f}% of limit ({used:.2f}/{limit:.2f})")
    except Exception as e:
        logger.debug(f"Could not check budget thresholds: {e}")


def update_budget_usage(conn: sqlite3.Connection, domain: str, delta: float) -> None:
    logger.debug(f"Updating budget usage for domain '{domain}': delta={delta:.2f}")
    conn.execute(
        """
    INSERT INTO budgets (domain, limit_value, used)
    VALUES (?, NULL, ?)
        ON CONFLICT(domain) DO UPDATE SET used=budgets.used + excluded.used, updated_ts=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        """,
        (domain, float(delta)),
    )
    conn.commit()
    logger.info(f"Budget usage updated for domain '{domain}': delta={delta:.2f}")


def load_budget_usage(conn: sqlite3.Connection) -> dict[str, float]:
    logger.debug("Loading budget usage from database")
    cur = conn.execute("SELECT domain, used FROM budgets")
    out: dict[str, float] = {}
    for domain, used in cur.fetchall():
        try:
            out[str(domain)] = float(used)
        except Exception as e:
            logger.warning(f"Invalid budget usage value for domain '{domain}': {e}")
            continue
    logger.info(f"Loaded budget usage for {len(out)} domain(s)")
    return out


# Artifacts helpers
def save_artifact(conn: sqlite3.Connection, kind: str, label: str | None, data: dict) -> int:
    """Persist a structured artifact as JSON and return its ID.

    kind: a short category string, e.g., "journal_summary".
    label: optional label/key, e.g., operator-provided output label.
    data: JSON-serializable dict payload.
    """
    logger.debug(f"Saving artifact: kind='{kind}', label='{label}'")
    try:
        import json

        payload = json.dumps(data, ensure_ascii=False)
    except Exception as e:
        # As a last resort, stringify
        logger.warning(f"Failed to JSON-encode artifact data: {e}, using string representation")
        payload = str(data)
    cur = conn.execute(
        "INSERT INTO artifacts (kind, label, data_json) VALUES (?, ?, ?)",
        (str(kind), str(label) if label is not None else None, payload),
    )
    conn.commit()
    rid = cur.lastrowid
    artifact_id = int(rid) if rid is not None else 0
    logger.info(f"Saved artifact: id={artifact_id}, kind='{kind}', size={len(payload)} bytes")
    return artifact_id


def list_artifacts(
    conn: sqlite3.Connection, kind: str | None = None, limit: int = 50
) -> list[dict]:
    """List recent artifacts, optionally filtered by kind."""
    logger.debug(f"Listing artifacts: kind={kind if kind else 'all'}, limit={limit}")
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
        except Exception as e:
            logger.warning(f"Failed to parse JSON for artifact {artifact_id}: {e}")
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
    logger.info(f"Retrieved {len(out)} artifact(s) (kind={kind if kind else 'all'})")
    return out


def get_artifact_by_id(conn: sqlite3.Connection, artifact_id: int) -> dict | None:
    """Return a single artifact by ID or None if not found."""
    logger.debug(f"Fetching artifact by ID: {artifact_id}")
    row = conn.execute(
        "SELECT artifact_id, kind, label, data_json, created_ts FROM artifacts WHERE artifact_id = ?",
        (int(artifact_id),),
    ).fetchone()
    if not row:
        logger.info(f"Artifact not found: {artifact_id}")
        return None
    import json

    aid, k, label, data_json, ts = row
    try:
        data = json.loads(data_json)
    except Exception as e:
        logger.warning(f"Failed to parse JSON for artifact {artifact_id}: {e}")
        data = {"raw": data_json}
    
    result = {
        "artifact_id": int(aid),
        "kind": str(k),
        "label": label if label is None else str(label),
        "data": data,
        "created_ts": str(ts),
    }
    logger.info(f"Retrieved artifact {artifact_id}: kind='{k}', size={len(data_json)} bytes")
    return result
