from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
from typing import Any, Callable, List, cast

logger = logging.getLogger(__name__)

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)


class OutputPanel:
    """Encapsulates the scrolling output area with an optional summary toggle.

    Public API:
    - write(text): replace entire content and reset raw buffer
    - append(text): append and keep buffer within max size
    - refresh(): re-render based on summary state
    - get_text(): current text
    - on_copy(): copy content to clipboard
    - summary_var: BoolVar you can read or set
    """

    def __init__(self, parent: "tk.Misc", *, max_buffer: int = 5000, height_lines: int = 10, show_summary_toggle: bool = True) -> None:  # type: ignore[name-defined]
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self._raw: List[str] = []
        self._max = int(max_buffer)

        frame = ttk.Frame(parent)
        # Keep compact so other controls remain visible
        frame.pack(fill="x", expand=False)

        # top bar
        topbar = ttk.Frame(frame)
        topbar.pack(fill="x")
        self.summary_var = safe_variables.BooleanVar(value=False)
        summary_chk = None  # default when toggle disabled
        if show_summary_toggle:
            summary_chk = ttk.Checkbutton(topbar, text="Summary view", variable=self.summary_var, command=self.refresh)
            summary_chk.pack(side="left")
        copy_btn = ttk.Button(topbar, text="Copy Output", command=self.on_copy)
        copy_btn.pack(side="right")

        # body
        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False)
        # Compact text area; scrollbar allows full browsing
        self.text = tk.Text(body, wrap="word", height=int(height_lines))
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        self.text.pack(side="left", fill="x", expand=False)
        vsb.pack(side="right", fill="y")
        # Tooltips
        try:  # pragma: no cover - UI affordance
            from .tooltips import add_tooltip
            if show_summary_toggle and summary_chk is not None:
                try:
                    add_tooltip(summary_chk, "Toggle a filtered summary of notable lines (errors, loss, checkpoints).")
                except Exception:
                    pass
            add_tooltip(copy_btn, "Copy the full (or summary) output text to clipboard.")
            add_tooltip(self.text, "Live output log. Scroll to review earlier lines. Height reduced to show status below.")
        except Exception:
            pass

    # --- public ops ---
    def write(self, text: str) -> None:
        # If clearing (empty text), log the action
        if not text:
            logger.info("User action: Cleared output panel")
        self._raw = []
        if text:
            self._raw.extend(str(text).splitlines())
        self.refresh()

    def append(self, text: str) -> None:
        if not text:
            return
        lines = str(text).splitlines()
        if not lines:
            return
        if not text.endswith("\n"):
            text += "\n"
        self._raw.extend(lines)
        if len(self._raw) > self._max:
            self._raw = self._raw[-self._max :]
        if self.summary_var.get():
            self._render_summary()
        else:
            self.text.insert(tk.END, text)
            self._scroll_if_at_bottom()

    def refresh(self) -> None:
        if self.summary_var.get():
            self._render_summary()
        else:
            self.text.delete("1.0", tk.END)
            self.text.insert("1.0", "\n".join(self._raw))
            self._scroll_if_at_bottom()

    def get_text(self) -> str:
        try:
            return self.text.get("1.0", tk.END)
        except Exception:
            return ""

    def on_copy(self) -> None:
        try:
            text = self.get_text()
            char_count = len(text)
            w = self.text.winfo_toplevel()
            w.clipboard_clear()
            w.clipboard_append(text)
            logger.info(f"User action: Copied output to clipboard ({char_count} characters)")
            self.append("[ui] Output copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy output to clipboard: {e}")
            self.append(f"[ui] copy failed: {e}")

    # --- internals ---
    def _scroll_if_at_bottom(self) -> None:
        """Only scroll to bottom if user is already viewing the bottom.
        
        This prevents auto-scrolling from interrupting users who have scrolled up
        to view earlier log entries.
        """
        try:
            # Get the current view position (0.0 = top, 1.0 = bottom)
            yview = self.text.yview()
            # If the bottom of the view is at or very close to the end (within ~2 lines),
            # then auto-scroll. Otherwise, respect the user's scroll position.
            if yview[1] >= 0.95:  # Within ~5% of the bottom
                self.text.see(tk.END)
        except Exception:
            # Fallback to always scrolling if we can't check position
            self.text.see(tk.END)
    
    def _render_summary(self) -> None:
        import json as _json

        keys = (
            "error",
            "exception",
            "fail",
            "warn",
            "step ",
            "loss",
            "checkpoint",
            "returncode",
            "cuda",
            "directml",
            "mps",
            "gpu",
            "%",
            "dataset",
            "hybrid",
        )
        picked: list[str] = []
        for ln in self._raw[-2000:]:
            low = ln.lower()
            if any(k in low for k in keys):
                s = ln
                if low.strip().startswith("{") and low.strip().endswith("}"):
                    try:
                        d = _json.loads(ln)
                        sel = {}
                        for k2 in ["cmd", "world_size", "returncode", "steps", "path", "label", "device", "tag"]:
                            if k2 in d:
                                sel[k2] = d[k2]
                        s = _json.dumps(sel) if sel else _json.dumps(d)
                    except Exception:
                        pass
                picked.append(s)
        if not picked:
            picked = self._raw[-200:]
        view = "\n".join(picked[-1000:])
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", view)
        self._scroll_if_at_bottom()
