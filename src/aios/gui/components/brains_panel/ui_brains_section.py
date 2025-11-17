"""Build brains section UI (table and management controls).

Creates the brain registry table and all management buttons/entries.
"""

from __future__ import annotations

# Import safe variable wrappers
from ...utils import safe_variables

from typing import Any, cast
import sys

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)


def build_brains_section(parent: Any, panel: Any) -> None:
    """Build complete brains section with table and controls.
    
    Args:
        parent: Parent Tk widget
        panel: BrainsPanel instance
    """
    if tk is None or ttk is None:
        return
    
    brains_frame = ttk.LabelFrame(parent, text="Brain Models Registry", padding=4)
    expand_brains = not sys.platform.startswith("win")
    brains_frame.pack(fill="both", expand=expand_brains, pady=(0, 8))
    
    # Build table
    _build_brains_table(brains_frame, panel)
    
    # Build controls
    _build_brains_controls(brains_frame, panel)


def _build_brains_table(parent: Any, panel: Any) -> None:
    """Build brains table/tree widget.
    
    Args:
        parent: Parent frame
        panel: BrainsPanel instance
    """
    cols = ("name", "size_mb", "params_m", "pinned", "master", "child", "training_steps", "last_used")
    tree_height = 8
    if sys.platform.startswith("win"):
        tree_height = 6
    panel.tree = ttk.Treeview(parent, columns=cols, show="headings", height=tree_height)
    
    panel.tree.heading("name", text="Name")
    panel.tree.heading("size_mb", text="Size (MB)")
    panel.tree.heading("params_m", text="Params (M)")
    panel.tree.heading("pinned", text="Pinned")
    panel.tree.heading("master", text="Master")
    panel.tree.heading("child", text="Parent/Children")
    panel.tree.heading("training_steps", text="Training Steps")
    panel.tree.heading("last_used", text="Last Used")
    
    panel.tree.column("name", width=160, anchor="w")
    panel.tree.column("size_mb", width=80, anchor="e")
    panel.tree.column("params_m", width=90, anchor="e")
    panel.tree.column("pinned", width=60, anchor="center")
    panel.tree.column("master", width=60, anchor="center")
    panel.tree.column("child", width=160, anchor="w")
    panel.tree.column("training_steps", width=100, anchor="e")
    panel.tree.column("last_used", width=140, anchor="e")
    
    panel.tree.pack(fill="both", expand=True, pady=(0, 6))
    
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(
            panel.tree,
            "List of brain models. Select a row then use actions below to manage.",
            wrap=78,
        )
    except Exception:
        pass


def _build_brains_controls(parent: Any, panel: Any) -> None:
    """Build brain management controls (buttons and entries).
    
    Args:
        parent: Parent frame
        panel: BrainsPanel instance
    """
    # Consolidated management row (all brain controls in one row)
    mgmt_row = ttk.Frame(parent)
    mgmt_row.pack(fill="x", pady=(6, 0))
    
    # Pin/Unpin/Delete buttons
    btn_pin = ttk.Button(mgmt_row, text="Pin", command=panel._on_pin)
    btn_pin.pack(side="left")
    
    btn_unpin = ttk.Button(mgmt_row, text="Unpin", command=panel._on_unpin)
    btn_unpin.pack(side="left", padx=(6, 0))
    
    btn_delete = ttk.Button(mgmt_row, text="Delete", command=panel._on_delete)
    btn_delete.pack(side="left", padx=(6, 0))
    
    btn_details = ttk.Button(mgmt_row, text="Details", command=panel._on_details)
    btn_details.pack(side="left", padx=(6, 0))
    
    # Rename controls
    lbl_rename = ttk.Label(mgmt_row, text="Rename:")
    lbl_rename.pack(side="left", padx=(12, 2))
    panel.rename_var = safe_variables.StringVar(value="")
    entry_rename = ttk.Entry(mgmt_row, textvariable=panel.rename_var, width=15)
    entry_rename.pack(side="left")
    btn_rename = ttk.Button(mgmt_row, text="Apply", command=panel._on_rename)
    btn_rename.pack(side="left", padx=(4, 0))
    
    # Master controls
    btn_set_master = ttk.Button(mgmt_row, text="Set Master", command=lambda: panel._on_set_master(True))
    btn_set_master.pack(side="left", padx=(12, 0))
    btn_unset_master = ttk.Button(mgmt_row, text="Unset Master", command=lambda: panel._on_set_master(False))
    btn_unset_master.pack(side="left", padx=(6, 0))
    
    # Parent controls
    lbl_parent = ttk.Label(mgmt_row, text="Set Parent:")
    lbl_parent.pack(side="left", padx=(12, 2))
    panel.parent_var = safe_variables.StringVar(value="")
    entry_parent = ttk.Entry(mgmt_row, textvariable=panel.parent_var, width=15)
    entry_parent.pack(side="left")
    btn_set_parent = ttk.Button(mgmt_row, text="Apply", command=panel._on_set_parent)
    btn_set_parent.pack(side="left", padx=(4, 0))
    btn_clear_parent = ttk.Button(mgmt_row, text="Clear Parent", command=panel._on_clear_parent)
    btn_clear_parent.pack(side="left", padx=(6, 0))
    
    # Tooltips
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(btn_pin, "Mark selected brain as pinned (protect from implicit pruning).")
        add_tooltip(btn_unpin, "Remove pin from selected brain (unless it is a master).")
        add_tooltip(btn_delete, "Delete selected brain file (confirmation required, master brains blocked).")
        add_tooltip(btn_details, "Show detailed information about the selected brain (parameters, size, goals, MoE config, etc.).")
        add_tooltip(lbl_rename, "Enter a new name then click Apply to rename the selected brain.")
        add_tooltip(entry_rename, "Type the new name for the selected brain.")
        add_tooltip(btn_rename, "Rename selected brain to the text provided.")
        add_tooltip(btn_set_master, "Designate selected brain as a master (cannot be deleted or unpinned).")
        add_tooltip(btn_unset_master, "Remove master designation from selected brain.")
        add_tooltip(lbl_parent, "Specify a parent brain for hierarchical lineage tracking.")
        add_tooltip(entry_parent, "Name of parent brain to assign to the selected brain.")
        add_tooltip(btn_set_parent, "Associate selected brain with the given parent name.")
        add_tooltip(btn_clear_parent, "Remove any parent association from the selected brain.")
    except Exception:
        pass
