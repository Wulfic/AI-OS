"""Lightweight Tkinter tooltip helper used across GUI components.

Usage:
    from .tooltips import add_tooltip
    add_tooltip(widget, "Helpful explanation shown on hover")

Fails safely (no-op) if Tk is not available (e.g. headless test env).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)  # type: ignore


class _Tooltip:
    def __init__(self, widget: "tk.Misc", text: str, *, delay_ms: int = 500, wrap: int = 288) -> None:  # type: ignore[name-defined]
        self.widget = widget
        self.text = text.strip() if text else ""
        self.delay_ms = max(50, int(delay_ms))
        self.wrap = max(40, int(wrap))
        self._after_id: Optional[str] = None
        self._tip: Optional["tk.Toplevel"] = None  # type: ignore[name-defined]
        try:
            self.widget.bind("<Enter>", self._schedule_show, add="+")
            self.widget.bind("<Leave>", self._hide, add="+")
            self.widget.bind("<ButtonPress>", self._hide, add="+")
        except Exception:
            pass

    def _schedule_show(self, _evt=None):
        if not self.text:
            return
        try:
            if self._after_id:
                self.widget.after_cancel(self._after_id)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)  # type: ignore[attr-defined]
        except Exception:
            pass

    def _show(self):
        try:
            if self._tip or not self.text:
                return
            
            # Don't log tooltip shows - it creates noise in debug panel
            # especially when hovering over the debug panel itself!
            
            tip = tk.Toplevel(self.widget)
            self._tip = tip
            try:
                tip.wm_overrideredirect(True)
            except Exception:
                pass
            try:
                x = self.widget.winfo_rootx() + 12
                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            except Exception:
                x = y = 0
            try:
                tip.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass
            lbl = tk.Label(
                tip,
                text=self.text,
                justify="left",
                relief="solid",
                borderwidth=1,
                padx=6,
                pady=4,
                background="#ffffe0",
                wraplength=self.wrap,
            )
            lbl.pack()
        except Exception:
            self._tip = None

    def _hide(self, _evt=None):
        try:
            if self._after_id:
                self.widget.after_cancel(self._after_id)  # type: ignore[attr-defined]
        except Exception:
            pass
        self._after_id = None
        try:
            if self._tip is not None:
                # Don't log tooltip hides - creates noise
                self._tip.destroy()
        except Exception:
            pass
        self._tip = None


def add_tooltip(widget: Any, text: str, *, delay_ms: int = 500, wrap: int = 288) -> Any:
    if tk is None:
        return None
    if not widget or not text:
        return None
    try:
        tip = _Tooltip(widget, text, delay_ms=delay_ms, wrap=wrap)
        setattr(widget, "_aios_tooltip", tip)  # prevent GC
        return tip
    except Exception:
        return None

__all__ = ["add_tooltip"]
