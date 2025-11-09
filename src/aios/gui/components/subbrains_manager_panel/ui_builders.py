"""UI builder functions for the Subbrains Manager Panel.

Provides functions to create UI sections:
- Summary statistics row
- Expert tree view
- Management controls
- Goals management section
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk
    from tkinter import ttk


def create_summary_section(
    parent: Any,
    total_experts_var: "tk.StringVar",
    active_experts_var: "tk.StringVar",
    frozen_experts_var: "tk.StringVar",
    total_activations_var: "tk.StringVar",
    refresh_callback: Any,
) -> None:
    """Create summary statistics row.
    
    Args:
        parent: Parent widget
        total_experts_var: StringVar for total experts count
        active_experts_var: StringVar for active experts count
        frozen_experts_var: StringVar for frozen experts count
        total_activations_var: StringVar for total activations count
        refresh_callback: Callback for refresh button
    """
    try:
        from tkinter import ttk
    except Exception:
        return
    
    summary = ttk.Frame(parent)
    summary.pack(fill="x", pady=(0, 8))
    
    # Total experts
    ttk.Label(summary, text="Total Experts:").pack(side="left")
    ttk.Label(summary, textvariable=total_experts_var, width=6).pack(side="left")
    
    # Active experts
    ttk.Label(summary, text="Active:").pack(side="left", padx=(12, 0))
    ttk.Label(summary, textvariable=active_experts_var, width=6).pack(side="left")
    
    # Frozen experts
    ttk.Label(summary, text="Frozen:").pack(side="left", padx=(12, 0))
    ttk.Label(summary, textvariable=frozen_experts_var, width=6).pack(side="left")
    
    # Total activations
    ttk.Label(summary, text="Total Activations:").pack(side="left", padx=(12, 0))
    ttk.Label(summary, textvariable=total_activations_var, width=10).pack(side="left")
    
    # Refresh button
    btn_refresh = ttk.Button(summary, text="Refresh", command=refresh_callback)
    btn_refresh.pack(side="right")
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(btn_refresh, "Reload expert registry from disk and update display")
    except Exception:
        pass


def create_experts_tree(parent: Any, on_select_callback: Any) -> "ttk.Treeview":
    """Create expert tree view with hierarchy and metrics.
    
    Args:
        parent: Parent widget
        on_select_callback: Callback for tree selection events
        
    Returns:
        The created Treeview widget
    """
    try:
        from tkinter import ttk
    except Exception:
        return None  # type: ignore[return-value]
    
    tree_frame = ttk.LabelFrame(parent, text="Experts", padding=4)
    tree_frame.pack(fill="both", expand=True, pady=(0, 8))
    
    # Columns: name, category, status, activations, avg_weight, goals, parent/children
    cols = ("name", "category", "status", "activations", "avg_weight", "goals", "hierarchy")
    experts_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
    
    experts_tree.heading("name", text="Name")
    experts_tree.heading("category", text="Category")
    experts_tree.heading("status", text="Status")
    experts_tree.heading("activations", text="Activations")
    experts_tree.heading("avg_weight", text="Avg Weight")
    experts_tree.heading("goals", text="Goals")
    experts_tree.heading("hierarchy", text="Parent/Children")
    
    experts_tree.column("name", width=150, anchor="w")
    experts_tree.column("category", width=100, anchor="w")
    experts_tree.column("status", width=100, anchor="center")
    experts_tree.column("activations", width=90, anchor="e")
    experts_tree.column("avg_weight", width=90, anchor="e")
    experts_tree.column("goals", width=60, anchor="center")
    experts_tree.column("hierarchy", width=150, anchor="w")
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=experts_tree.yview)
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=experts_tree.xview)
    experts_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    
    # Pack
    experts_tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    
    tree_frame.grid_rowconfigure(0, weight=1)
    tree_frame.grid_columnconfigure(0, weight=1)
    
    # Bind selection event
    experts_tree.bind("<<TreeviewSelect>>", on_select_callback)
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(
            experts_tree,
            "Expert registry. Status: Active (routing enabled), Frozen (weights locked), Inactive (not in use). "
            "Select an expert to view/manage its goals.",
            wrap=80
        )
    except Exception:
        pass
    
    return experts_tree


def create_management_controls(
    parent: Any,
    parent_expert_var: "tk.StringVar",
    on_create_expert: Any,
    on_delete_expert: Any,
    on_set_status: Any,
    on_set_parent: Any,
    on_clear_parent: Any,
) -> None:
    """Create expert management controls.
    
    Args:
        parent: Parent widget
        parent_expert_var: StringVar for parent expert ID entry
        on_create_expert: Callback for create button
        on_delete_expert: Callback for delete button
        on_set_status: Callback for status buttons (takes action string)
        on_set_parent: Callback for set parent button
        on_clear_parent: Callback for clear parent button
    """
    try:
        from tkinter import ttk
    except Exception:
        return
    
    mgmt_frame = ttk.Frame(parent)
    mgmt_frame.pack(fill="x", pady=(0, 8))
    
    # Row 1: Create, Delete, Status controls
    row1 = ttk.Frame(mgmt_frame)
    row1.pack(fill="x", pady=(0, 4))
    
    btn_create = ttk.Button(row1, text="Create Expert", command=on_create_expert)
    btn_create.pack(side="left")
    
    btn_delete = ttk.Button(row1, text="Delete Selected", command=on_delete_expert)
    btn_delete.pack(side="left", padx=(6, 0))
    
    btn_activate = ttk.Button(row1, text="Activate", command=lambda: on_set_status("active"))
    btn_activate.pack(side="left", padx=(12, 0))
    
    btn_deactivate = ttk.Button(row1, text="Deactivate", command=lambda: on_set_status("inactive"))
    btn_deactivate.pack(side="left", padx=(6, 0))
    
    btn_freeze = ttk.Button(row1, text="Freeze", command=lambda: on_set_status("freeze"))
    btn_freeze.pack(side="left", padx=(6, 0))
    
    btn_unfreeze = ttk.Button(row1, text="Unfreeze", command=lambda: on_set_status("unfreeze"))
    btn_unfreeze.pack(side="left", padx=(6, 0))
    
    # Row 2: Parent/child management
    row2 = ttk.Frame(mgmt_frame)
    row2.pack(fill="x")
    
    ttk.Label(row2, text="Set Parent:").pack(side="left")
    entry_parent = ttk.Entry(row2, textvariable=parent_expert_var, width=20)
    entry_parent.pack(side="left", padx=(4, 0))
    
    btn_set_parent = ttk.Button(row2, text="Apply", command=on_set_parent)
    btn_set_parent.pack(side="left", padx=(4, 0))
    
    btn_clear_parent = ttk.Button(row2, text="Clear Parent", command=on_clear_parent)
    btn_clear_parent.pack(side="left", padx=(6, 0))
    
    # Tooltips
    try:
        from ..tooltips import add_tooltip
        add_tooltip(btn_create, "Create a new expert with custom name, category, and description")
        add_tooltip(btn_delete, "Delete the selected expert (confirmation required)")
        add_tooltip(btn_activate, "Enable routing to this expert")
        add_tooltip(btn_deactivate, "Disable routing to this expert")
        add_tooltip(btn_freeze, "Freeze expert weights (prevent training)")
        add_tooltip(btn_unfreeze, "Unfreeze expert weights (allow training)")
        add_tooltip(entry_parent, "Expert ID to set as parent for hierarchical relationships")
        add_tooltip(btn_set_parent, "Link selected expert to specified parent")
        add_tooltip(btn_clear_parent, "Remove parent relationship from selected expert")
    except Exception:
        pass


def create_goals_section(
    parent: Any,
    link_goal_var: "tk.StringVar",
    goals_count_var: "tk.StringVar",
    on_goal_add: Any,
    on_goal_remove: Any,
    on_goals_list: Any,
    link_goal_callback: Any,
    unlink_goals_callback: Any,
) -> tuple["tk.Listbox | None", Any]:
    """Create goals management section for selected expert.
    
    Args:
        parent: Parent widget
        link_goal_var: StringVar for goal entry field
        goals_count_var: StringVar for goals count display
        on_goal_add: Goal add callback (if None, section not created)
        on_goal_remove: Goal remove callback
        on_goals_list: Goals list callback
        link_goal_callback: Callback for link button
        unlink_goals_callback: Callback for unlink button
        
    Returns:
        Tuple of (goals_listbox, goal_entry_widget)
    """
    if on_goals_list is None:
        return None, None  # Goals integration not available
    
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return None, None
    
    goals_frame = ttk.LabelFrame(parent, text="Goals for Selected Expert", padding=8)
    goals_frame.pack(fill="both", expand=True)
    
    # Info label
    info = ttk.Label(
        goals_frame,
        text="Link goals to experts to control routing bias. When a goal is active, "
             "its linked expert receives higher routing priority.",
        foreground="gray",
        wraplength=600
    )
    info.pack(anchor="w", pady=(0, 6))
    
    # Goals list with scrollbar
    list_container = ttk.Frame(goals_frame)
    list_container.pack(fill="both", expand=True, pady=(0, 6))
    
    goals_list = tk.Listbox(list_container, height=6, selectmode="extended")
    goals_list.pack(side="left", fill="both", expand=True)
    
    scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=goals_list.yview)
    goals_list.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    
    # Goals control bar
    controls = ttk.Frame(goals_frame)
    controls.pack(fill="x", pady=(0, 6))
    
    if on_goal_remove is not None:
        btn_unlink = ttk.Button(controls, text="Unlink Selected", command=unlink_goals_callback)
        btn_unlink.pack(side="left")
        try:
            from ..tooltips import add_tooltip
            add_tooltip(btn_unlink, "Remove goal-expert binding for selected goals")
        except Exception:
            pass
    
    ttk.Label(controls, textvariable=goals_count_var).pack(side="right")
    
    # Link goal bar
    link_bar = ttk.Frame(goals_frame)
    link_bar.pack(fill="x")
    
    ttk.Label(link_bar, text="Link Goal:").pack(side="left")
    
    goal_entry = ttk.Entry(link_bar, textvariable=link_goal_var)
    goal_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
    
    if on_goal_add is not None:
        btn_link = ttk.Button(link_bar, text="Link", command=link_goal_callback)
        btn_link.pack(side="left")
        
        # Bind Return key
        goal_entry.bind("<Return>", lambda e: link_goal_callback())
        
        try:
            from ..tooltips import add_tooltip
            add_tooltip(goal_entry, "Enter goal ID or text to link to the selected expert")
            add_tooltip(btn_link, "Create goal-expert binding. Expert will receive routing bias when goal is active.")
        except Exception:
            pass
    
    return goals_list, goal_entry
