from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar

import httpx

_T = TypeVar("_T")

_logger = logging.getLogger("aios.diagnostics")
_async_logger = logging.getLogger("aios.diagnostics.asyncio")

_HF_DATASET_HOSTS = {"datasets-server.huggingface.co"}
_COMMON_SERVER_STATUS_CODES = {500, 501, 502, 503, 504}


def _env_ms(name: str, fallback_ms: float) -> float:
    try:
        raw = os.environ.get(name)
        if raw is None:
            return fallback_ms
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return fallback_ms


_DEFAULT_SLOW_CALLBACK_MS = _env_ms("AIOS_DIAG_SLOW_CALLBACK_MS", 100.0)
_DEFAULT_TASK_WARN_MS = _env_ms("AIOS_DIAG_TASK_WARN_MS", 1000.0)
_DEFAULT_BLOCKING_WARN_MS = _env_ms("AIOS_DIAG_BLOCKING_WARN_MS", 750.0)


class _DiagnosticsPolicy(asyncio.AbstractEventLoopPolicy):
    def __init__(
        self,
        base_policy: asyncio.AbstractEventLoopPolicy,
        slow_callback_threshold: float,
        task_duration_threshold: Optional[float],
    ) -> None:
        self._base = base_policy
        self._slow_threshold = slow_callback_threshold
        self._task_threshold = task_duration_threshold

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._base.get_event_loop()
        self._configure_loop(loop)
        return loop

    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        self._base.set_event_loop(loop)
        if loop is not None:
            self._configure_loop(loop)

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._base.new_event_loop()
        self._configure_loop(loop)
        return loop

    def get_child_watcher(self):  # type: ignore[override]
        getter = getattr(self._base, "get_child_watcher", None)
        return getter() if getter is not None else None

    def set_child_watcher(self, watcher) -> None:  # type: ignore[override]
        setter = getattr(self._base, "set_child_watcher", None)
        if setter is not None:
            setter(watcher)

    def _configure_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if getattr(loop, "_aios_diag_configured", False):
            return
        try:
            loop.set_debug(True)
        except Exception:
            pass
        if self._slow_threshold is not None:
            try:
                loop.slow_callback_duration = self._slow_threshold
            except Exception:
                pass
        if self._task_threshold is not None:
            existing = loop.get_task_factory()
            factory = getattr(loop, "_aios_diag_task_factory", None)
            if factory is None:
                factory = _DiagnosticsTaskFactory(existing, self._task_threshold)
                loop.set_task_factory(factory)
                loop._aios_diag_task_factory = factory  # type: ignore[attr-defined]
            else:
                factory.update_threshold(self._task_threshold)
        handler = getattr(loop, "_aios_diag_exception_handler", None)
        if handler is None:
            loop.set_exception_handler(_diagnostic_exception_handler)
            loop._aios_diag_exception_handler = True  # type: ignore[attr-defined]
        loop._aios_diag_configured = True  # type: ignore[attr-defined]


class _DiagnosticsTaskFactory:
    def __init__(
        self,
        base_factory: Optional[Callable[..., asyncio.Task[Any]]],
        warn_after: float,
    ) -> None:
        self._base = base_factory
        self._warn_after = warn_after

    def update_threshold(self, warn_after: float) -> None:
        self._warn_after = warn_after

    def __call__(self, loop: asyncio.AbstractEventLoop, coro, *args, **kwargs) -> asyncio.Task[Any]:
        start = time.perf_counter()
        if self._base is None:
            task = asyncio.Task(coro, loop=loop)  # type: ignore[arg-type]
        else:
            task = self._base(loop, coro, *args, **kwargs)
        _attach_task_probes(task, start, self._warn_after)
        return task


def _attach_task_probes(task: asyncio.Task[Any], start: float, warn_after: float) -> None:
    def _on_done(done: asyncio.Task[Any]) -> None:
        duration = time.perf_counter() - start
        cancelled = False
        try:
            exc = done.exception()
        except asyncio.CancelledError:
            exc = None
            cancelled = True
        name = _describe_task(done)
        if exc is not None:
            host = _extract_request_host(exc)
            if isinstance(exc, httpx.TimeoutException) and host in _HF_DATASET_HOSTS:
                _async_logger.info(
                    "Task %s timed out contacting %s after %.3fs",
                    name,
                    host or "unknown host",
                    duration,
                )
                return
            if _should_downgrade_exception(exc):
                status = None
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        status = exc.response.status_code
                    except Exception:
                        status = None
                _async_logger.info(
                    "Task %s received expected HTTP failure %s from %s after %.3fs",
                    name,
                    status if status is not None else "unknown",
                    host or "unknown host",
                    duration,
                )
            else:
                _async_logger.error(
                    "Task %s failed after %.3fs",
                    name,
                    duration,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            return
        if cancelled:
            if duration >= warn_after:
                _async_logger.info("Task %s cancelled after %.3fs", name, duration)
            return
        if duration >= warn_after:
            _async_logger.warning("Task %s ran for %.3fs", name, duration)
        else:
            _async_logger.debug("Task %s completed in %.3fs", name, duration)

    task.add_done_callback(_on_done)


def _describe_task(task: asyncio.Task[Any]) -> str:
    try:
        name = task.get_name()
    except Exception:
        name = None
    if name:
        return name
    try:
        coro = task.get_coro()
        return getattr(coro, "__qualname__", repr(coro))
    except Exception:
        return repr(task)


def _should_downgrade_exception(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    response = exc.response
    request = exc.request
    if response is None or request is None:
        return False
    status = response.status_code
    if status not in _COMMON_SERVER_STATUS_CODES:
        return False
    host = _extract_request_host(exc)
    return host in _HF_DATASET_HOSTS


def _extract_request_host(exc: BaseException) -> Optional[str]:
    request = getattr(exc, "request", None)
    if request is None:
        return None
    try:
        return request.url.host
    except Exception:
        return None


def _diagnostic_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    message = context.get("message") or "Asyncio exception in event loop"
    exc = context.get("exception")
    if exc:
        _async_logger.error("%s", message, exc_info=(type(exc), exc, exc.__traceback__))
    else:
        _async_logger.error("%s: %s", message, context)


def enable_asyncio_diagnostics(
    *,
    slow_callback_threshold: Optional[float] = None,
    task_duration_warning: Optional[float] = None,
) -> None:
    if os.environ.get("AIOS_DIAGNOSTICS_DISABLED"):
        _logger.info("AIOS diagnostics disabled via environment variable")
        return

    if slow_callback_threshold is None:
        slow_callback_threshold = _DEFAULT_SLOW_CALLBACK_MS / 1000.0
    if task_duration_warning is None:
        task_duration_warning = _DEFAULT_TASK_WARN_MS / 1000.0

    policy = asyncio.get_event_loop_policy()
    if isinstance(policy, _DiagnosticsPolicy):
        policy._slow_threshold = slow_callback_threshold
        policy._task_threshold = task_duration_warning
        return

    wrapped = _DiagnosticsPolicy(policy, slow_callback_threshold, task_duration_warning)
    asyncio.set_event_loop_policy(wrapped)
    _logger.debug(
        "Asyncio diagnostics enabled (slow_callback=%.3fs, task_warning=%.3fs)",
        slow_callback_threshold,
        task_duration_warning,
    )


def timed_blocking_call(
    label: str,
    func: Callable[..., _T],
    *args: Any,
    warn_after: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> _T:
    active_logger = logger or _logger
    threshold = warn_after if warn_after is not None else (_DEFAULT_BLOCKING_WARN_MS / 1000.0)
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        duration = time.perf_counter() - start
        active_logger.error(
            "Blocking call %s failed after %.3fs", label, duration, exc_info=(type(exc), exc, exc.__traceback__)
        )
        raise
    duration = time.perf_counter() - start
    if threshold and duration >= threshold:
        active_logger.warning("Blocking call %s took %.3fs", label, duration)
    else:
        active_logger.debug("Blocking call %s completed in %.3fs", label, duration)
    return result
