from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
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
        # Visual separation and sunken border to ensure visibility
        frame.pack(fill="x", side="bottom")
        try:
            sep = ttk.Separator(frame, orient="horizontal")
            sep.pack(fill="x", side="top")
        except Exception:
            pass
        # Use a nested frame with a sunken border for the bar content
        inner = tk.Frame(frame, bd=1, relief="sunken", bg="#f0f0f0")  # type: ignore[misc]
        # Add more internal padding so the bar is thicker and easier to read
        inner.pack(fill="x", padx=0, pady=2, ipady=8)
        self.var = safe_variables.StringVar(value="Ready")
        # Explicit label background helps on some Windows themes
        try:
            # Slightly larger, readable font for system status (Windows-friendly)
            _font = ("Segoe UI", 10)
        except Exception:
            _font = None  # type: ignore[assignment]
        lbl_kwargs = {"textvariable": self.var, "anchor": "w", "bg": "#f0f0f0"}  # type: ignore[var-annotated]
        if _font:
            lbl_kwargs["font"] = _font  # type: ignore[index]
        lbl = tk.Label(inner, **lbl_kwargs)  # type: ignore[misc]
        lbl.pack(fill="x", padx=10, pady=6)
        add_tooltip(lbl, "Application status and notifications. Shows current operations and system messages.")

    def set(self, text: str) -> None:
        try:
            logger.debug(f"Status updated: {text}")
            self.var.set(text)
        except Exception:
            pass

    def get_var(self):
        return self.var
