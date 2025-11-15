"""Lightweight analytics helper utilities.

This module centralizes telemetry/analytics emission so callers can log
structured events without depending on a specific backend. Events are always
written to a local JSONL file for diagnostics. When Google Analytics (GA4)
credentials are present (``AIOS_GA_MEASUREMENT_ID`` and ``AIOS_GA_API_SECRET``)
*and* analytics have not been opted out via environment flag, the event is also
forwarded to GA using the Measurement Protocol.

Opt-out flags honoured (case-insensitive truthy values like ``1``/``true``):
- ``AIOS_ANALYTICS_OPTOUT``
- ``AIOS_TELEMETRY_OPTOUT``
- ``AIOS_DISABLE_ANALYTICS``

The implementation is intentionally best-effort; failures are logged at DEBUG
level so analytics never interfere with the primary UX.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency safeguard
    import httpx
except Exception:  # pragma: no cover - degrade gracefully if httpx missing
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_GA_ENDPOINT = "https://www.google-analytics.com/mp/collect"
_MEASUREMENT_ID = os.environ.get("AIOS_GA_MEASUREMENT_ID")
_API_SECRET = os.environ.get("AIOS_GA_API_SECRET")
_OPTOUT_ENV_KEYS = (
    "AIOS_ANALYTICS_OPTOUT",
    "AIOS_TELEMETRY_OPTOUT",
    "AIOS_DISABLE_ANALYTICS",
)

_ANALYTICS_DIR = Path("artifacts/diagnostics")
_EVENTS_PATH = _ANALYTICS_DIR / "analytics_events.jsonl"
_CLIENT_ID_PATH = _ANALYTICS_DIR / "analytics_client_id"

_client_id_lock = threading.Lock()
_client_id_cache: str | None = None


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().lower()
    except Exception:
        return False
    return text in {"1", "true", "yes", "on"}


def _analytics_opted_out() -> bool:
    return any(_is_truthy(os.environ.get(key)) for key in _OPTOUT_ENV_KEYS)


def analytics_enabled() -> bool:
    """Return True when analytics should be forwarded to external services."""

    if _analytics_opted_out():
        return False
    if not _MEASUREMENT_ID or not _API_SECRET:
        return False
    return True


def _ensure_analytics_dir() -> None:
    try:
        _ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Failed to ensure analytics directory exists", exc_info=True)


def _get_client_id() -> str:
    global _client_id_cache
    with _client_id_lock:
        if _client_id_cache:
            return _client_id_cache
        try:
            _ensure_analytics_dir()
            if _CLIENT_ID_PATH.exists():
                cached = _CLIENT_ID_PATH.read_text(encoding="utf-8").strip()
                if cached:
                    _client_id_cache = cached
                    return cached
        except Exception:
            logger.debug("Failed to read analytics client identifier", exc_info=True)

        generated = uuid.uuid4().hex
        try:
            _ensure_analytics_dir()
            _CLIENT_ID_PATH.write_text(generated, encoding="ascii")
        except Exception:
            logger.debug("Failed to persist analytics client identifier", exc_info=True)
        _client_id_cache = generated
        return generated


def _normalise_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    if not params:
        return normalised
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            normalised[key] = value
        else:
            try:
                normalised[key] = str(value)
            except Exception:
                normalised[key] = repr(value)
    return normalised


def _write_local_event(event: dict[str, Any]) -> None:
    try:
        _ensure_analytics_dir()
        with _EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception:
        logger.debug("Failed to persist analytics event locally", exc_info=True)


def _dispatch_ga(event_name: str, params: Mapping[str, Any]) -> None:
    if not analytics_enabled() or httpx is None:
        return

    payload = {
        "client_id": _get_client_id(),
        "non_personalized_ads": True,
        "events": [
            {
                "name": event_name,
                "params": {
                    **params,
                    "engagement_time_msec": params.get("engagement_time_msec", 1),
                },
            }
        ],
    }

    def _send() -> None:
        try:
            url = f"{_GA_ENDPOINT}?measurement_id={_MEASUREMENT_ID}&api_secret={_API_SECRET}"
            response = httpx.post(url, json=payload, timeout=2.0)
            if response.status_code >= 400:
                logger.debug(
                    "Analytics event rejected by GA (%s): %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception:
            logger.debug("Failed to transmit analytics event", exc_info=True)

    threading.Thread(target=_send, name="analytics-dispatch", daemon=True).start()


def emit_analytics_event(
    event_name: str,
    params: Mapping[str, Any] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Record an analytics event.

    Parameters
    ----------
    event_name:
        Canonical event identifier (e.g. ``"eval.device_selection"``).
    params:
        Key/value pairs describing the event. Values are coerced to str unless they
        are numeric.
    context:
        Optional additional context stored locally but not forwarded to GA.
    """

    timestamp = time.time()
    payload = {
        "event": event_name,
        "params": _normalise_params(params),
        "timestamp": timestamp,
    }
    if context:
        payload["context"] = _normalise_params(context)

    _write_local_event(payload)

    try:
        _dispatch_ga(event_name, payload["params"])
    except Exception:
        logger.debug("Analytics dispatch failed", exc_info=True)