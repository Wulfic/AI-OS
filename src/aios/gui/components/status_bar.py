from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
from typing import Any, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    ttk = cast(Any, None)

try:  # pragma: no cover
    from .tooltips import add_tooltip
except Exception:
    def add_tooltip(*args, **kwargs):
        pass


class StatusBar:
    def __init__(self, parent: Any) -> None:
        if tk is None:
            raise RuntimeError("Tkinter not available")
        frame = ttk.Frame(parent)
        # Keep status bar slim and theme-aware to avoid bright banding on dark themes
        frame.pack(fill="x", side="bottom")
        try:
            sep = ttk.Separator(frame, orient="horizontal")
            sep.pack(fill="x", side="top")
        except Exception:
            pass

        # Use ttk widgets with a dedicated style so we inherit the active theme colors
        try:
            style = ttk.Style()
            style.configure("StatusBar.TFrame", padding=(8, 4))
            style.configure("StatusBar.TLabel", padding=(4, 0))
        except Exception:
            pass

        inner = ttk.Frame(frame, style="StatusBar.TFrame")
        inner.pack(fill="x", padx=6, pady=(2, 4))

        self.var = safe_variables.StringVar(value="Ready")
        try:
            _font = ("Segoe UI", 10)
        except Exception:
            _font = None  # type: ignore[assignment]

        lbl_kwargs = {"textvariable": self.var, "anchor": "w", "style": "StatusBar.TLabel"}
        if _font:
            lbl_kwargs["font"] = _font  # type: ignore[index]
        lbl = ttk.Label(inner, **lbl_kwargs)
        lbl.pack(fill="x")
        add_tooltip(lbl, "Application status and notifications. Shows current operations and system messages.")

    def set(self, text: str) -> None:
        try:
            logger.debug(f"Status updated: {text}")
            self.var.set(text)
        except Exception:
            pass

    def get_var(self):
        return self.var
