"""Lightweight Tkinter tooltip helper used across GUI components.

Usage:
    from .tooltips import add_tooltip
    add_tooltip(widget, "Helpful explanation shown on hover")

Fails safely (no-op) if Tk is not available (e.g. headless test env).
"""
from __future__ import annotations

from typing import Any, Optional, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)  # type: ignore


class _Tooltip:
    def __init__(self, widget: "tk.Misc", text: str, *, delay_ms: int = 500, wrap: int = 400) -> None:  # type: ignore[name-defined]
        self.widget = widget
        self.text = text.strip() if text else ""
        self.delay_ms = max(50, int(delay_ms))
        # Increased default wrap from 288 to 400 for better text display
        self.wrap = max(100, int(wrap))
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
            
            # Improved label formatting to fix BUG-020: Tooltip Text Truncated
            # - Increased padding for better readability
            # - Better font settings
            # - Proper text wrapping
            lbl = tk.Label(
                tip,
                text=self.text,
                justify="left",
                relief="solid",
                borderwidth=1,
                padx=8,  # Increased from 6 for better spacing
                pady=6,  # Increased from 4 for better spacing
                background="#ffffe0",
                foreground="#000000",  # Explicit text color
                font=("TkDefaultFont", 9),  # Explicit font for consistency
                wraplength=self.wrap,
                anchor="w",  # Left-align text
            )
            lbl.pack(fill="both", expand=True)
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
                self._tip.destroy()
        except Exception:
            pass
        self._tip = None


def add_tooltip(widget: Any, text: str, *, delay_ms: int = 500, wrap: int = 400) -> Any:
    """Add a tooltip to a widget with improved text wrapping.
    
    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text to display
        delay_ms: Delay in milliseconds before showing tooltip (default: 500)
        wrap: Maximum width in pixels for text wrapping (default: 400, increased from 288)
    
    Returns:
        The tooltip object, or None if creation failed
    
    Note: Increased default wrap from 288 to 400 pixels to fix BUG-020 (truncated tooltips)
    """
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
