"""Import guards and constants for resources panel.

This module handles optional dependencies (tkinter, psutil, matplotlib)
with graceful fallbacks when libraries are not available.
"""

from __future__ import annotations

from typing import Any, cast

# Tkinter imports with fallback
try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

# Import safe variable wrappers
from ...utils import safe_variables

# psutil imports with fallback
try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

# Matplotlib imports with fallback
try:  # pragma: no cover - optional dependency
    import matplotlib  # type: ignore
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure  # type: ignore
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore
    import matplotlib.dates as mdates  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Figure = None  # type: ignore
    FigureCanvasTkAgg = None  # type: ignore
    mdates = None  # type: ignore
    MATPLOTLIB_AVAILABLE = False


__all__ = [
    "tk",
    "ttk",
    "safe_variables",
    "psutil",
    "Figure",
    "FigureCanvasTkAgg",
    "mdates",
    "MATPLOTLIB_AVAILABLE",
]
