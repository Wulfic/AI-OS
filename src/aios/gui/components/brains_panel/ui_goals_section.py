"""Build goals section UI for managing brain/expert goals.

Creates the goals listbox, add/remove controls (conditional on callbacks).
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


def build_goals_section(parent: Any, panel: Any) -> None:
    """Build goals section if goal callbacks are provided.
    
    Only builds the section if panel has goal management callbacks.
    
    Args:
        parent: Parent Tk widget
        panel: BrainsPanel instance
    """
    if tk is None or ttk is None:
        return
    
    if panel._on_goals_list is None:
        return
    
    # Add debounce tracking for goals to prevent rapid duplicate adds
    panel._last_goal_add_time = 0.0
    panel._last_goal_text = ""
    
    is_windows = sys.platform.startswith("win")
    goals_frame = ttk.LabelFrame(parent, text="Goals for Selected Brain / Expert", padding=6 if is_windows else 8)
    goals_frame.pack(fill="both", expand=True, pady=(8 if is_windows else 12, 0))
    
    # Info label
    info_label = ttk.Label(
        goals_frame,
        text="Goals help guide training focus. Select a brain or expert above to manage its goals.",
        foreground="gray"
    )
    info_label.pack(anchor="w", pady=(0, 6))
    
    # Goals list with scrollbar
    list_container = ttk.Frame(goals_frame)
    list_container.pack(fill="both", expand=True, pady=(0, 6))
    
    list_height = 4 if is_windows else 6
    panel.goals_list = tk.Listbox(list_container, height=list_height, selectmode="extended")
    panel.goals_list.pack(side="left", fill="both", expand=True)
    
    goals_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=panel.goals_list.yview)
    panel.goals_list.configure(yscrollcommand=goals_scrollbar.set)
    goals_scrollbar.pack(side="right", fill="y")
    
    # Goals control bar (remove + count)
    goals_controls = ttk.Frame(goals_frame)
    goals_controls.pack(fill="x", pady=(0, 6))
    
    if panel._on_goal_remove is not None:
        btn_remove_goal = ttk.Button(goals_controls, text="Remove Selected", command=panel._remove_selected_goals)
        btn_remove_goal.pack(side="left")
        panel.goal_remove_button = btn_remove_goal
        try:  # pragma: no cover
            from ..tooltips import add_tooltip
            add_tooltip(btn_remove_goal, "Remove selected goals (excluding protected [primary] goals).")
        except Exception:
            pass
    
    panel.goals_count_var = safe_variables.StringVar(value="0 goals")
    ttk.Label(goals_controls, textvariable=panel.goals_count_var).pack(side="right")
    
    # Add goal bar
    add_goal_bar = ttk.Frame(goals_frame)
    add_goal_bar.pack(fill="x")
    
    ttk.Label(add_goal_bar, text="Add Goal:").pack(side="left")
    
    # Empty default - user should set custom goal
    panel.goal_text_var = safe_variables.StringVar(value="")
    goal_entry = ttk.Entry(add_goal_bar, textvariable=panel.goal_text_var)
    goal_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
    panel.goal_text_entry = goal_entry
    
    if panel._on_goal_add is not None:
        btn_add_goal = ttk.Button(add_goal_bar, text="Add", command=panel._add_goal)
        btn_add_goal.pack(side="left")
        panel.goal_add_button = btn_add_goal
        # Bind Return key to add goal for convenience
        goal_entry.bind("<Return>", lambda e: panel._add_goal())
        try:  # pragma: no cover
            from ..tooltips import add_tooltip
            add_tooltip(goal_entry, "Enter a new goal/directive for the selected brain. Press Enter or click Add.")
            add_tooltip(btn_add_goal, "Add this goal to the selected brain's training directives.")
        except Exception:
            pass
    
    # Note: TreeviewSelect binding is done in ui_experts_section.py to avoid duplicate bindings
    # Both trees trigger the same _on_tree_select handler
    
    try:  # pragma: no cover
        from ..tooltips import add_tooltip
        add_tooltip(panel.goals_list, "Goals for the currently selected brain. Multi-select enabled. Protected [primary] goals cannot be removed.")
        add_tooltip(info_label, "Goals are brain-specific training directives that guide what the brain learns.")
    except Exception:
        pass
