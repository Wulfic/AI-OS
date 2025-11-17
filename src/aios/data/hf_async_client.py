"""Async HTTP client helpers for Hugging Face API access."""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

import httpx

from aios.gui.utils.resource_management import get_async_loop

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504, 520, 521, 522}
_MAX_ATTEMPTS = 4
_BASE_DELAY = 0.6
_MAX_DELAY = 6.0
_NON_RETRY_ERROR_CODES = {
    "ConfigNamesError",
}

_DEFAULT_OVERALL_TIMEOUT = 12.0


class _AsyncHttpClient:
    """Singleton async HTTP client with shared connection pool."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = threading.Lock()
        self._failure_cache: dict[str, tuple[int, float]] = {}
        self._failure_lock = threading.Lock()

    @staticmethod
    def _extract_error_code(response: httpx.Response) -> Optional[str]:
        header_code = response.headers.get("x-error-code")
        if header_code:
            return header_code
        try:
            payload = response.json()
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload.get("error_code") or payload.get("errorCode")
        return None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    timeout = httpx.Timeout(5.0, read=5.0, connect=3.0)
                    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
                    headers = {"User-Agent": "AI-OS Dataset Client"}
                    self._client = httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers)
        return self._client

    def _should_defer(self, url: str) -> bool:
        now = time.monotonic()
        with self._failure_lock:
            entry = self._failure_cache.get(url)
            if not entry:
                return False
            attempts, retry_at = entry
            if retry_at > now:
                logger.debug("Skipping %s; backoff in effect after %d failures", url, attempts)
                return True
            del self._failure_cache[url]
            return False

    def _record_failure(self, url: str, attempts: int) -> None:
        backoff = min(_BASE_DELAY * (2 ** max(0, attempts - 1)), _MAX_DELAY)
        cooldown = min(backoff * 10.0, 600.0)
        with self._failure_lock:
            self._failure_cache[url] = (attempts, time.monotonic() + cooldown)

    def _record_success(self, url: str) -> None:
        with self._failure_lock:
            if url in self._failure_cache:
                del self._failure_cache[url]

    async def fetch_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
    ) -> Dict[str, Any]:
        if self._should_defer(url):
            request = httpx.Request("GET", url)
            raise httpx.RequestError("Hugging Face request suppressed by backoff", request=request)

        client = await self._ensure_client()
        attempts = 0
        attempt_limit = max(1, max_attempts or _MAX_ATTEMPTS)
        last_exc: Exception | None = None

        while attempts < attempt_limit:
            attempts += 1
            try:
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()
                self._record_success(url)
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                last_exc = exc
                should_retry = status in _TRANSIENT_STATUS and attempts < attempt_limit

                if should_retry:
                    error_code = self._extract_error_code(exc.response)
                    if error_code and error_code in _NON_RETRY_ERROR_CODES:
                        self._record_failure(url, attempts)
                        logger.debug(
                            "HTTP %s for %s with non-retry error code %s; aborting retries",
                            status,
                            url,
                            error_code,
                        )
                        raise

                if not should_retry:
                    if status >= 500 or status == 429:
                        self._record_failure(url, attempts)
                    raise

                retry_after = exc.response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = _BASE_DELAY * (2 ** (attempts - 1))
                else:
                    delay = _BASE_DELAY * (2 ** (attempts - 1))
                delay = min(delay, _MAX_DELAY)
                delay += random.uniform(0.0, 0.25)
                logger.debug(
                    "HTTP %s for %s (attempt %d/%d); retrying in %.2fs",
                    status,
                    url,
                    attempts,
                    _MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempts >= attempt_limit:
                    self._record_failure(url, attempts)
                    raise
                delay = min(_BASE_DELAY * (2 ** (attempts - 1)), _MAX_DELAY)
                delay += random.uniform(0.0, 0.25)
                logger.debug(
                    "HTTP transport issue for %s (attempt %d/%d); retrying in %.2fs",
                    url,
                    attempts,
                    attempt_limit,
                    delay,
                )
                await asyncio.sleep(delay)

        if last_exc is not None:
            self._record_failure(url, attempts)
            raise last_exc
        raise RuntimeError(f"Failed to fetch {url}; no attempts executed")

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None


_client = _AsyncHttpClient()


def fetch_json_sync(
    url: str,
    *,
    timeout: float | None = None,
    max_attempts: int | None = None,
) -> Dict[str, Any]:
    """Fetch JSON from *url* using the shared async HTTP client.

    The coroutine is executed on the global async event loop. Raises RuntimeError
    if the async loop is not running.
    """
    loop = get_async_loop()
    if loop is None:
        raise RuntimeError("Async event loop is not initialized")
    future = loop.run_coroutine(_client.fetch_json(url, timeout=timeout, max_attempts=max_attempts))

    per_attempt = timeout if timeout is not None else 5.0
    attempt_limit = max(1, max_attempts or _MAX_ATTEMPTS)
    extra_slack = max(1.0, per_attempt * 0.4)
    computed = per_attempt * attempt_limit + extra_slack
    cap = max(_DEFAULT_OVERALL_TIMEOUT, per_attempt + extra_slack)
    overall_timeout = max(per_attempt + extra_slack, min(computed, cap))

    try:
        return future.result(timeout=overall_timeout)
    except FuturesTimeoutError as exc:
        if future.cancel():
            logger.debug(
                "Cancelled Hugging Face request wait after %.1fs: %s",
                overall_timeout,
                url,
            )
        request = httpx.Request("GET", url)
        raise httpx.TimeoutException(
            f"Blocking wait exceeded {overall_timeout:.1f}s",
            request=request,
        ) from exc


async def fetch_json(
    url: str,
    *,
    timeout: float | None = None,
    max_attempts: int | None = None,
) -> Dict[str, Any]:
    """Async variant for direct coroutine usage."""
    return await _client.fetch_json(url, timeout=timeout, max_attempts=max_attempts)


async def shutdown_client() -> None:
    await _client.close()


__all__ = ["fetch_json_sync", "fetch_json", "shutdown_client"]
