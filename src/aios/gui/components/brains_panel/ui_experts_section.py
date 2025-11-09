"""Build experts (subbrains) section UI (table and management controls).

Creates the dynamic experts table and all management buttons/entries.
"""

from __future__ import annotations

from typing import Any, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)


def build_experts_section(parent: Any, panel: Any) -> None:
    """Build complete experts section with table and controls.
    
    Args:
        parent: Parent Tk widget
        panel: BrainsPanel instance
    """
    if tk is None or ttk is None:
        return
    
    experts_frame = ttk.LabelFrame(parent, text="Dynamic Experts (Subbrains) - [WIP - Feature Under Development]", padding=4)
    experts_frame.pack(fill="both", expand=True, pady=(0, 8))
    
    # Build table
    _build_experts_table(experts_frame, panel)
    
    # Build controls
    _build_experts_controls(experts_frame, panel)


def _build_experts_table(parent: Any, panel: Any) -> None:
    """Build experts table/tree widget.
    
    Args:
        parent: Parent frame
        panel: BrainsPanel instance
    """
    expert_cols = ("name", "category", "status", "activations", "avg_weight", "goals", "hierarchy")
    panel.experts_tree = ttk.Treeview(parent, columns=expert_cols, show="headings", height=8)
    
    panel.experts_tree.heading("name", text="Name")
    panel.experts_tree.heading("category", text="Category")
    panel.experts_tree.heading("status", text="Status")
    panel.experts_tree.heading("activations", text="Activations")
    panel.experts_tree.heading("avg_weight", text="Avg Weight")
    panel.experts_tree.heading("goals", text="Goals")
    panel.experts_tree.heading("hierarchy", text="Parent/Children")
    
    panel.experts_tree.column("name", width=150, anchor="w")
    panel.experts_tree.column("category", width=100, anchor="w")
    panel.experts_tree.column("status", width=100, anchor="center")
    panel.experts_tree.column("activations", width=90, anchor="e")
    panel.experts_tree.column("avg_weight", width=90, anchor="e")
    panel.experts_tree.column("goals", width=60, anchor="center")
    panel.experts_tree.column("hierarchy", width=150, anchor="w")
    
    panel.experts_tree.pack(fill="both", expand=True, pady=(0, 6))
    
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(
            panel.experts_tree,
            "⚠️ WIP: Dynamic experts (subbrains) feature is under development. "
            "Management operations (create, delete, activate, etc.) are not yet implemented and will show placeholder messages. "
            "Status: Active (routing enabled), Frozen (weights locked), Inactive (not in use). "
            "Select an expert to view/manage its goals.",
            wrap=80
        )
    except Exception:
        pass
    
    # Bind BOTH trees' selection events to refresh goals (single binding point)
    panel.tree.bind("<<TreeviewSelect>>", panel._on_tree_select)
    panel.experts_tree.bind("<<TreeviewSelect>>", panel._on_tree_select)


def _build_experts_controls(parent: Any, panel: Any) -> None:
    """Build expert management controls (buttons and entries).
    
    Args:
        parent: Parent frame
        panel: BrainsPanel instance
    """
    # Expert management controls (consolidated into single row)
    expert_mgmt = ttk.Frame(parent)
    expert_mgmt.pack(fill="x", pady=(0, 6))
    
    btn_create_expert = ttk.Button(expert_mgmt, text="Create Expert", command=panel._on_create_expert)
    btn_create_expert.pack(side="left")
    
    btn_delete_expert = ttk.Button(expert_mgmt, text="Delete Selected", command=panel._on_delete_expert)
    btn_delete_expert.pack(side="left", padx=(6, 0))
    
    btn_activate_expert = ttk.Button(expert_mgmt, text="Activate", 
                                     command=lambda: panel._on_set_expert_status("active"))
    btn_activate_expert.pack(side="left", padx=(12, 0))
    
    btn_deactivate_expert = ttk.Button(expert_mgmt, text="Deactivate", 
                                       command=lambda: panel._on_set_expert_status("inactive"))
    btn_deactivate_expert.pack(side="left", padx=(6, 0))
    
    btn_freeze_expert = ttk.Button(expert_mgmt, text="Freeze", 
                                   command=lambda: panel._on_set_expert_status("freeze"))
    btn_freeze_expert.pack(side="left", padx=(6, 0))
    
    btn_unfreeze_expert = ttk.Button(expert_mgmt, text="Unfreeze", 
                                     command=lambda: panel._on_set_expert_status("unfreeze"))
    btn_unfreeze_expert.pack(side="left", padx=(6, 0))
    
    ttk.Label(expert_mgmt, text="Set Parent:").pack(side="left", padx=(12, 2))
    panel.expert_parent_var = tk.StringVar(value="")
    entry_expert_parent = ttk.Entry(expert_mgmt, textvariable=panel.expert_parent_var, width=15)
    entry_expert_parent.pack(side="left")
    
    btn_set_expert_parent = ttk.Button(expert_mgmt, text="Apply", command=panel._on_set_expert_parent)
    btn_set_expert_parent.pack(side="left", padx=(4, 0))
    
    btn_clear_expert_parent = ttk.Button(expert_mgmt, text="Clear Parent", command=panel._on_clear_expert_parent)
    btn_clear_expert_parent.pack(side="left", padx=(6, 0))
    
    # Tooltips
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(btn_create_expert, "[WIP - Not Implemented] Create a new expert with custom name, category, and description")
        add_tooltip(btn_delete_expert, "[WIP - Not Implemented] Delete the selected expert (confirmation required)")
        add_tooltip(btn_activate_expert, "[WIP - Not Implemented] Enable routing to this expert")
        add_tooltip(btn_deactivate_expert, "[WIP - Not Implemented] Disable routing to this expert")
        add_tooltip(btn_freeze_expert, "[WIP - Not Implemented] Freeze expert weights (prevent training)")
        add_tooltip(btn_unfreeze_expert, "[WIP - Not Implemented] Unfreeze expert weights (allow training)")
        add_tooltip(entry_expert_parent, "[WIP - Not Implemented] Expert ID to set as parent for hierarchical relationships")
        add_tooltip(btn_set_expert_parent, "[WIP - Not Implemented] Link selected expert to specified parent")
        add_tooltip(btn_clear_expert_parent, "[WIP - Not Implemented] Remove parent relationship from selected expert")
    except Exception:
        pass
