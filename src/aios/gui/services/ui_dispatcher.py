"""Thread-safe dispatcher for scheduling Tkinter UI updates.

This module provides a small utility class that allows background threads to
schedule callbacks on the Tkinter main thread without risking "main thread is
not in main loop" errors. Callbacks submitted via ``dispatch`` are executed in
FIFO order on the Tk event loop using ``root.after``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TkUiDispatcher:
    """Coordinate UI work onto the Tkinter main thread.

    The dispatcher owns a queue that background threads can push callables into.
    A repeating ``after`` callback drains the queue on the Tk thread. If a task
    is dispatched while already on the UI thread, it executes immediately to
    keep ordering predictable and latency low.
    """

    def __init__(self, root: Any, poll_interval_ms: int = 8, queue_maxsize: int | None = None) -> None:
        self._root = root
        self._poll_interval_ms = max(1, poll_interval_ms)
        self._queue: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = queue.Queue(maxsize=queue_maxsize or 0)
        self._after_id: Any | None = None
        self._running = False
        self._main_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        # Track dropped tasks to avoid log spam during shutdown
        self._dropped_count = 0
        self._last_drop_log_time = 0.0
        logger.info(
            "TkUiDispatcher initializing (poll_interval_ms=%s, queue_maxsize=%s, main_thread_id=%s)",
            self._poll_interval_ms,
            queue_maxsize,
            self._main_thread_id,
        )
        self.start()
        logger.info("TkUiDispatcher started")

    def start(self) -> None:
        """Begin draining the queue on the Tk loop."""
        schedule = False
        with self._lock:
            if self._running:
                return
            self._running = True
            schedule = True
        if schedule:
            logger.debug("Scheduling initial UI dispatcher poll")
            self._schedule_poll()

    def stop(self) -> None:
        """Stop scheduling new drains and cancel any pending ``after`` call."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            after_id, self._after_id = self._after_id, None
            # Reset drop counter for next run
            dropped = self._dropped_count
            self._dropped_count = 0
        if after_id is not None:
            try:
                self._root.after_cancel(after_id)
            except Exception:
                pass
        logger.debug("UI dispatcher stopped (had %d tasks dropped during shutdown)", dropped)

    def is_ui_thread(self) -> bool:
        """Return ``True`` when executing on the Tk main thread."""
        return threading.get_ident() == self._main_thread_id

    def dispatch(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run ``func`` on the UI thread as soon as possible."""
        if not callable(func):
            raise TypeError("func must be callable")

        if self.is_ui_thread():
            self._execute(func, *args, **kwargs)
            return

        if not self._running:
            # Rate-limit dropped task messages to avoid log spam during shutdown
            self._dropped_count += 1
            now = time.time()
            if now - self._last_drop_log_time >= 1.0:  # Log at most once per second
                if self._dropped_count == 1:
                    logger.debug("UI dispatcher stopped; dropping task %r", getattr(func, "__name__", func))
                else:
                    logger.debug("UI dispatcher stopped; dropped %d tasks (latest: %r)", 
                               self._dropped_count, getattr(func, "__name__", func))
                self._last_drop_log_time = now
            return

        try:
            self._queue.put_nowait((func, args, kwargs))
        except queue.Full:
            logger.warning("UI dispatcher queue is full; dropping task %r", getattr(func, "__name__", func))
            return

    def flush(self) -> None:
        """Synchronously execute all pending callbacks if on the UI thread."""
        if not self.is_ui_thread():
            return
        while True:
            try:
                func, args, kwargs = self._queue.get_nowait()
            except queue.Empty:
                break
            self._execute(func, *args, **kwargs)

    def _schedule_poll(self) -> None:
        with self._lock:
            if not self._running or self._after_id is not None:
                return
            try:
                self._after_id = self._root.after(self._poll_interval_ms, self._poll)
            except Exception as exc:
                # ``after`` can raise if the root window is being torn down. Avoid
                # spamming the log and simply drop the scheduling request.
                logger.debug("Failed to schedule UI dispatcher poll: %s", exc)
                self._after_id = None

    def _poll(self) -> None:
        with self._lock:
            self._after_id = None
            if not self._running:
                return

        while True:
            try:
                func, args, kwargs = self._queue.get_nowait()
            except queue.Empty:
                break
            self._execute(func, *args, **kwargs)

        self._schedule_poll()

    def _execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except Exception:
            logger.exception("UI dispatcher task failed")