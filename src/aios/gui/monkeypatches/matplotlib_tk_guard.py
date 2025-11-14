"""Monkeypatch Matplotlib's Tk backend to guard against malformed MouseWheel events.

On some Windows configurations, Tkinter can deliver <MouseWheel> events where
event.widget is a string, causing matplotlib.backends._backend_tk to raise:
    AttributeError: 'str' object has no attribute 'winfo_containing'

This patch wraps the backend's scroll handler(s) to ignore such events safely.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency and environment specific
    import matplotlib  # type: ignore
    # Ensure Tk backend is set (no-op if already set elsewhere)
    try:
        matplotlib.use('TkAgg')  # type: ignore[attr-defined]
        logger.debug("Matplotlib Tk backend set to TkAgg")
    except Exception as e:
        logger.debug(f"Failed to set matplotlib backend: {e}")
    try:
        # Import the private backend module
        import matplotlib.backends._backend_tk as _mbtk  # type: ignore
        logger.debug("Matplotlib Tk backend module imported successfully")
    except Exception as e:
        logger.warning(f"Failed to import matplotlib Tk backend: {e}")
        _mbtk = None  # type: ignore
except Exception as e:  # pragma: no cover
    logger.debug(f"Matplotlib not available: {e}")
    matplotlib = None  # type: ignore
    _mbtk = None  # type: ignore


def _install_guard() -> None:
    if _mbtk is None:
        logger.debug("Matplotlib Tk backend not available, skipping guard installation")
        return

    logger.info("Installing matplotlib Tk MouseWheel event guard")
    
    # Patch module-level function scroll_event_windows if present
    try:
        if hasattr(_mbtk, 'scroll_event_windows'):
            _orig = _mbtk.scroll_event_windows  # type: ignore[attr-defined]

            def _safe_scroll_event_windows(event: Any) -> Any:  # type: ignore[override]
                try:
                    w = getattr(event, 'widget', None)
                    if (w is None) or isinstance(w, str) or (not hasattr(w, 'winfo_containing')):
                        logger.debug("Blocked malformed MouseWheel event (module-level)")
                        return None
                except Exception as e:
                    logger.debug(f"MouseWheel event validation failed: {e}")
                    return None
                try:
                    return _orig(event)
                except Exception as e:
                    logger.warning(f"MouseWheel event handler failed: {e}")
                    return None

            try:
                _mbtk.scroll_event_windows = _safe_scroll_event_windows  # type: ignore[attr-defined]
                logger.debug("Module-level scroll_event_windows patched")
            except Exception as e:
                logger.error(f"Failed to patch module-level scroll_event_windows: {e}")
    except Exception as e:
        logger.error(f"Failed to install module-level guard: {e}", exc_info=True)

    # Patch method on classes that implement scroll_event_windows
    patched_classes = []
    for attr_name in (
        'FigureCanvasTkAgg', 'FigureCanvasTk', 'FigureCanvasTkAggBase',
        'FigureCanvasBase', 'NavigationToolbar2Tk',
    ):
        try:
            cls = getattr(_mbtk, attr_name, None)
            if cls is None:
                continue
            meth = getattr(cls, 'scroll_event_windows', None)
            if meth is None:
                continue
            _orig_meth = meth

            def _guarded(self: Any, event: Any) -> Any:  # type: ignore[misc]
                try:
                    w = getattr(event, 'widget', None)
                    if (w is None) or isinstance(w, str) or (not hasattr(w, 'winfo_containing')):
                        logger.debug(f"Blocked malformed MouseWheel event in {type(self).__name__}")
                        return None
                except Exception as e:
                    logger.debug(f"MouseWheel event validation failed in {type(self).__name__}: {e}")
                    return None
                try:
                    return _orig_meth(self, event)  # type: ignore[misc]
                except Exception as e:
                    logger.warning(f"MouseWheel event handler failed in {type(self).__name__}: {e}")
                    return None

            try:
                setattr(cls, 'scroll_event_windows', _guarded)
                patched_classes.append(attr_name)
            except Exception as e:
                logger.error(f"Failed to patch {attr_name}.scroll_event_windows: {e}")
        except Exception as e:
            logger.error(f"Failed to process class {attr_name}: {e}", exc_info=True)
    
    if patched_classes:
        logger.info(f"Patched classes: {', '.join(patched_classes)}")
    else:
        logger.debug("No classes with scroll_event_windows found to patch")


# Apply immediately on import
try:
    _install_guard()
except Exception as e:
    logger.error(f"Failed to install matplotlib Tk guard: {e}", exc_info=True)
