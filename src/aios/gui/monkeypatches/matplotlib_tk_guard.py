"""Monkeypatch Matplotlib's Tk backend to guard against malformed MouseWheel events.

On some Windows configurations, Tkinter can deliver <MouseWheel> events where
event.widget is a string, causing matplotlib.backends._backend_tk to raise:
    AttributeError: 'str' object has no attribute 'winfo_containing'

This patch wraps the backend's scroll handler(s) to ignore such events safely.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency and environment specific
    import matplotlib  # type: ignore
    # Ensure Tk backend is set (no-op if already set elsewhere)
    try:
        matplotlib.use('TkAgg')  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        # Import the private backend module
        import matplotlib.backends._backend_tk as _mbtk  # type: ignore
    except Exception:
        _mbtk = None  # type: ignore
except Exception:  # pragma: no cover
    matplotlib = None  # type: ignore
    _mbtk = None  # type: ignore


def _install_guard() -> None:
    if _mbtk is None:
        return

    # Patch module-level function scroll_event_windows if present
    try:
        if hasattr(_mbtk, 'scroll_event_windows'):
            _orig = _mbtk.scroll_event_windows  # type: ignore[attr-defined]

            def _safe_scroll_event_windows(event: Any) -> Any:  # type: ignore[override]
                try:
                    w = getattr(event, 'widget', None)
                    if (w is None) or isinstance(w, str) or (not hasattr(w, 'winfo_containing')):
                        return None
                except Exception:
                    return None
                try:
                    return _orig(event)
                except Exception:
                    return None

            try:
                _mbtk.scroll_event_windows = _safe_scroll_event_windows  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass

    # Patch method on classes that implement scroll_event_windows
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
                        return None
                except Exception:
                    return None
                try:
                    return _orig_meth(self, event)  # type: ignore[misc]
                except Exception:
                    return None

            try:
                setattr(cls, 'scroll_event_windows', _guarded)
            except Exception:
                pass
        except Exception:
            pass


# Apply immediately on import
try:
    _install_guard()
except Exception:
    pass
