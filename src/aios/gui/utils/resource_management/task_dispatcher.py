"""Shared task dispatch helpers for GUI background work."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Callable, Deque, Dict, Optional

from . import get_worker_pool

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BACKLOG = 512
_WARN_BACKLOG_RATIO = 0.75

_metrics_lock = threading.Lock()
_recent_tasks: Deque[dict] = deque(maxlen=200)
_task_totals = {
    "submitted": 0,
    "completed": 0,
    "rejected": 0,
}


def _record_task(label: str, duration: float, backlog_at_submit: int) -> None:
    with _metrics_lock:
        _task_totals["completed"] += 1
        _recent_tasks.appendleft(
            {
                "label": label,
                "duration_sec": round(duration, 4),
                "backlog_at_submit": backlog_at_submit,
                "timestamp": time.time(),
            }
        )


def snapshot_metrics() -> Dict[str, object]:
    """Return a shallow copy of dispatcher metrics for diagnostics."""
    with _metrics_lock:
        return {
            "totals": dict(_task_totals),
            "recent": list(_recent_tasks)[:25],
        }


def submit_background(
    label: str,
    func: Callable[..., object],
    *args,
    pool: Optional[object] = None,
    max_backlog: int = _DEFAULT_MAX_BACKLOG,
    warn_ratio: float = _WARN_BACKLOG_RATIO,
    **kwargs,
) -> Future:
    """Submit *func* to the shared worker pool with backlog guards.

    Args:
        label: Short identifier for logging/metrics.
        func: Callable executed in background.
        *args: Positional arguments forwarded to *func*.
        pool: Optional explicit AsyncWorkerPool; defaults to global pool.
        max_backlog: Upper bound on pending tasks before rejecting the call.
        warn_ratio: Ratio of backlog/workers that triggers a warning log.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Future representing the background execution.

    Raises:
        RuntimeError: If no worker pool is available or backlog limits exceeded.
    """

    worker_pool = pool or get_worker_pool()
    if worker_pool is None:
        with _metrics_lock:
            _task_totals["rejected"] += 1
        raise RuntimeError(f"submit_background('{label}') called without available worker pool")

    backlog = getattr(worker_pool, "pending_tasks", 0)
    max_workers = max(getattr(worker_pool, "max_workers", 1), 1)
    load_ratio = backlog / max_workers

    if backlog >= max_backlog:
        with _metrics_lock:
            _task_totals["rejected"] += 1
        message = (
            f"Refusing to queue task '{label}' (backlog={backlog}, max_backlog={max_backlog})"
        )
        logger.error(message)
        raise RuntimeError(message)

    if load_ratio >= warn_ratio:
        logger.warning(
            "Worker pool backlog warning (label=%s backlog=%s max_workers=%s ratio=%.2f)",
            label,
            backlog,
            max_workers,
            load_ratio,
        )

    with _metrics_lock:
        _task_totals["submitted"] += 1

    submit_time = time.perf_counter()

    def _wrapped() -> object:
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.perf_counter() - submit_time
            _record_task(label, duration, backlog)

    try:
        func_name = getattr(func, "__name__", type(func).__name__)
        _wrapped.__name__ = f"{func_name}[{label}]"
        _wrapped.__qualname__ = _wrapped.__name__
    except Exception:
        pass
    setattr(_wrapped, "_aios_task_label", label)

    future: Future = worker_pool.submit(_wrapped)
    setattr(future, "_aios_task_label", label)
    setattr(future, "_aios_submitted_at", submit_time)
    return future


__all__ = ["submit_background", "snapshot_metrics"]
