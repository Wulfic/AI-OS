from __future__ import annotations

import time
from typing import Optional

from rich import print

from aios.memory.store import get_db, init_db, list_artifacts


def training_watch(
    *,
    label: Optional[str] = None,
    interval: float = 2.0,
    count: int = 0,
) -> None:
    iterations = 0
    while True:
        conn = get_db()
        init_db(conn)
        try:
            items = list_artifacts(conn, kind="training_metrics", limit=50)
            if label is not None:
                items = [it for it in items if str(it.get("label") or "") == label]
            latest = items[0] if items else None
            if not latest:
                print({"found": False, "reason": "no training_metrics yet", "label": label})
            else:
                data = latest.get("data") or {}
                losses = data.get("losses") or []
                last_loss = float(losses[-1]) if losses else None
                out = {
                    "found": True,
                    "label": latest.get("label"),
                    "steps": data.get("steps"),
                    "last_loss": last_loss,
                    "created_ts": latest.get("created_ts"),
                }
                ec = data.get("english_corpus")
                if isinstance(ec, dict):
                    out["english"] = {
                        "count": ec.get("count"),
                        "avg_words": ec.get("avg_words"),
                        "avg_flesch": ec.get("avg_flesch"),
                    }
                et = data.get("english_task")
                if isinstance(et, dict):
                    out["task"] = {"spec": et.get("spec")}
                ea = data.get("english_adherence")
                if isinstance(ea, dict):
                    out["adherence"] = {"rate": ea.get("rate"), "count": ea.get("count")}
                print(out)
        finally:
            conn.close()
        iterations += 1
        if count and iterations >= count:
            break
        time.sleep(max(0.05, float(interval)))
