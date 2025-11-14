from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from aios.core.hrm import Recorder

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SqliteRecorder(Recorder):
    conn: sqlite3.Connection

    async def record(self, operator_name: str, success: bool) -> None:
        logger.debug(f"Async recording operation: operator='{operator_name}', success={success}")
        cur = self.conn.cursor()
        # Ensure operator exists
        cur.execute(
            "SELECT op_id FROM operators WHERE name = ? ORDER BY version DESC LIMIT 1",
            (operator_name,),
        )
        row = cur.fetchone()
        if row:
            op_id = row[0]
            logger.debug(f"Found existing operator '{operator_name}' (op_id={op_id})")
        else:
            cur.execute(
                "INSERT INTO operators (name, version, yaml, created_ts) VALUES (?, 1, NULL, ?)",
                (operator_name, _utc_now()),
            )
            op_id = cur.lastrowid
            logger.debug(f"Created new operator '{operator_name}' (op_id={op_id})")
        ctx_hash = hashlib.sha1(b"default").hexdigest()
        # Upsert usage
        cur.execute(
            """
            INSERT INTO operator_usage (op_id, context_hash, success, trials, lat_sum_ms)
            VALUES (?, ?, ?, 1, 0)
            ON CONFLICT(op_id, context_hash) DO UPDATE SET
                success = operator_usage.success + ?,
                trials = operator_usage.trials + 1
            """,
            (op_id, ctx_hash, 1 if success else 0, 1 if success else 0),
        )
        self.conn.commit()
        logger.info(f"Recorded operator usage: operator='{operator_name}', success={success}")
