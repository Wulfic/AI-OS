from __future__ import annotations

# Import safe variable wrappers
from ..utils import safe_variables

import logging
from typing import Any, Callable, Iterable, List, cast

try:
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

logger = logging.getLogger(__name__)


class GoalsPanel:
    """Active goals view and quick-add bar.

    Args:
        parent: container
        on_add: callback receiving directive string
        on_list: callback returning list[str] of current goals
    """

    def __init__(self, parent: "tk.Misc", on_add: Callable[[str], None], on_list: Callable[[], Iterable[str]], on_remove: Callable[[int], None] | None = None) -> None:  # type: ignore[name-defined]
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self._on_add = on_add
        self._on_list = on_list
        self._on_remove = on_remove

        frame = ttk.LabelFrame(parent, text="Active Goals")
        # Fill the tab since this is the only element on the page
        frame.pack(fill="both", expand=True)

        # top
        top = ttk.Frame(frame)
        top.pack(fill="x")
        refresh_btn = ttk.Button(top, text="Refresh", command=self.refresh)
        refresh_btn.pack(side="left")
        self.count_var = safe_variables.StringVar(value="0")
        ttk.Label(top, textvariable=self.count_var).pack(side="left", padx=(8, 0))
        remove_btn = None
        if self._on_remove is not None:
            remove_btn = ttk.Button(top, text="Remove Selected", command=self._remove_selected)
            remove_btn.pack(side="right")

        # list
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        # Larger list area and multi-select enabled
        self.list = tk.Listbox(body, height=18, selectmode="extended")
        self.list.pack(side="left", fill="both", expand=True)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.list.yview)
        self.list.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        # add bar
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(4, 0))
        dir_lbl = ttk.Label(bar, text="Directive:")
        dir_lbl.pack(side="left")
        # Empty default - user should set custom goal
        self.text_var = safe_variables.StringVar(value="")
        dir_entry = ttk.Entry(bar, textvariable=self.text_var)
        dir_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        add_btn = ttk.Button(bar, text="Add Goal", command=self._add)
        add_btn.pack(side="left")

        # Tooltips (ignore failures quietly)
        try:  # pragma: no cover - UI enhancement only
            from .tooltips import add_tooltip
            add_tooltip(refresh_btn, "Reload the current active goals list from the system state.")
            add_tooltip(self.list, "Active goals (multi-select enabled). Protected [primary] goals can't be removed.")
            add_tooltip(dir_lbl, "Label for directive/goal input field.")
            add_tooltip(dir_entry, "Enter a new directive/goal. Press Add Goal to persist.")
            add_tooltip(add_btn, "Add the entered directive to the active goals list.")
            add_tooltip(self.count_var, "Number of active goals currently loaded.")  # benign if variable unsupported
            if self._on_remove is not None:
                add_tooltip(remove_btn, "Remove the selected goals (excluding protected [primary] goals).")
        except Exception:
            pass

    def refresh(self) -> None:
        logger.debug("Refreshing goals list")
        items = list(self._on_list() or [])
        try:
            self.list.delete(0, tk.END)
            for it in items:
                self.list.insert(tk.END, str(it))
            self.count_var.set(f"{len(items)} active")
            logger.info(f"Goals list refreshed: {len(items)} active goals")
        except Exception as e:
            logger.error(f"Failed to refresh goals list: {e}")
            pass

    def _add(self) -> None:
        txt = (self.text_var.get() or "").strip()
        if not txt:
            logger.debug("Add goal attempted with empty text, ignoring")
            return
        logger.info(f"Adding new goal: {txt}")
        try:
            self._on_add(txt)
            # Clear entry after adding
            self.text_var.set("")
            self.refresh()
            logger.info(f"Successfully added goal: {txt}")
        except Exception as e:
            logger.error(f"Failed to add goal '{txt}': {e}")
            pass

    def _remove_selected(self) -> None:
        if self._on_remove is None:
            return
        try:
            sel = self.list.curselection()
            if not sel:
                logger.debug("Remove selected attempted with no selection")
                return
            logger.info(f"Removing {len(sel)} selected goal(s)")
            import re as _re
            removed = 0
            skipped_primary = 0
            # Work on a copy to avoid index shifts
            for idx in list(sel):
                raw = self.list.get(idx)
                if "[primary]" in str(raw):
                    skipped_primary += 1
                    logger.debug(f"Skipping protected [primary] goal: {raw}")
                    continue
                m = _re.match(r"^\s*\d+\)\s*#(\d+)\b|^\s*#(\d+)\b", str(raw).strip())
                did = None
                if m:
                    did = m.group(1) or m.group(2)
                if did is None:
                    logger.warning(f"Could not extract goal ID from: {raw}")
                    continue
                try:
                    self._on_remove(int(did))
                    removed += 1
                    logger.debug(f"Removed goal #{did}: {raw}")
                except Exception as e:
                    logger.error(f"Failed to remove goal #{did}: {e}")
                    continue
            # Refresh after batch operations
            self.refresh()
            logger.info(f"Goal removal complete: {removed} removed, {skipped_primary} skipped (protected)")
            if skipped_primary:
                try:
                    import tkinter.messagebox as mb  # type: ignore
                    mb.showinfo("Protected", f"Skipped {skipped_primary} protected [primary] goal(s).")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to remove selected goals: {e}")
            pass
