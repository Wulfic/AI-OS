"""Main Subbrains Manager Panel component.

Provides a Tkinter panel for managing dynamic subbrains (experts) with:
- Viewing expert registry with hierarchy
- Creating, deleting, freezing experts
- Binding goals to experts
- Viewing performance metrics and routing stats
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, cast

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)

from .data_manager import (
    get_selected_expert_id,
    load_and_refresh_experts,
    load_goals_for_expert,
)
from .event_handlers import (
    handle_clear_parent,
    handle_create_expert,
    handle_delete_expert,
    handle_link_goal,
    handle_set_parent,
    handle_set_status,
    handle_unlink_goals,
)
from .ui_builders import (
    create_experts_tree,
    create_goals_section,
    create_management_controls,
    create_summary_section,
)


class SubbrainsManagerPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Panel for managing dynamic subbrains (experts) in the system."""

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Optional[Callable[[str], None]] = None,
        on_goal_add: Optional[Callable[[str, str], None]] = None,
        on_goals_list: Optional[Callable[[str], list[str]]] = None,
        on_goal_remove: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Initialize the subbrains manager panel.
        
        Args:
            parent: Parent Tkinter widget
            run_cli: Callback to run CLI commands
            append_out: Optional callback for debug output
            on_goal_add: Optional callback for adding goals
            on_goals_list: Optional callback for listing goals
            on_goal_remove: Optional callback for removing goals
        """
        super().__init__(parent, text="Dynamic Subbrains Manager - [WIP - Feature Under Development]")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._on_goal_add = on_goal_add
        self._on_goals_list = on_goals_list
        self._on_goal_remove = on_goal_remove
        
        # Detect project root
        self._project_root = self._detect_project_root()
        self._registry_path = os.path.join(
            self._project_root, "artifacts", "models", "expert_registry.json"
        )
        
        # Create UI state variables
        self.total_experts_var = tk.StringVar(value="0")
        self.active_experts_var = tk.StringVar(value="0")
        self.frozen_experts_var = tk.StringVar(value="0")
        self.total_activations_var = tk.StringVar(value="0")
        self.parent_expert_var = tk.StringVar(value="")
        self.link_goal_var = tk.StringVar(value="")
        self.goals_count_var = tk.StringVar(value="No expert selected")
        
        # Create UI sections
        create_summary_section(
            self,
            self.total_experts_var,
            self.active_experts_var,
            self.frozen_experts_var,
            self.total_activations_var,
            self.refresh,
        )
        
        self.experts_tree = create_experts_tree(self, self._on_expert_select)
        
        create_management_controls(
            self,
            self.parent_expert_var,
            self._on_create_expert,
            self._on_delete_expert,
            self._on_set_status,
            self._on_set_parent,
            self._on_clear_parent,
        )
        
        self.goals_list, _ = create_goals_section(
            self,
            self.link_goal_var,
            self.goals_count_var,
            self._on_goal_add,
            self._on_goal_remove,
            self._on_goals_list,
            self._link_goal_to_expert,
            self._unlink_selected_goals,
        )
        
        # Initial refresh
        self.refresh()
    
    @staticmethod
    def _detect_project_root() -> str:
        """Detect the project root directory.
        
        Returns:
            Path to project root (directory containing pyproject.toml)
        """
        try:
            cur = os.path.abspath(os.getcwd())
            for _ in range(8):
                if os.path.exists(os.path.join(cur, "pyproject.toml")):
                    return cur
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
            return os.path.abspath(os.getcwd())
        except Exception:
            return os.path.abspath(os.getcwd())

    def refresh(self) -> None:
        """Refresh expert registry data from disk."""
        load_and_refresh_experts(
            self._registry_path,
            self.experts_tree,
            self.total_experts_var,
            self.active_experts_var,
            self.frozen_experts_var,
            self.total_activations_var,
            self._append_out,
        )

    def _selected_expert_id(self) -> Optional[str]:
        """Get the expert_id of the currently selected expert."""
        return get_selected_expert_id(self.experts_tree)

    def _on_expert_select(self, event=None) -> None:
        """Handle expert selection - refresh goals for selected expert."""
        if not hasattr(self, 'goals_list') or self.goals_list is None:
            return
        
        expert_id = self._selected_expert_id()
        if not expert_id:
            # No expert selected
            try:
                import tkinter as tk
                self.goals_list.delete(0, tk.END)
            except Exception:
                pass
            self.goals_count_var.set("No expert selected")
            return
        
        load_goals_for_expert(
            expert_id,
            self._registry_path,
            self.goals_list,
            self.goals_count_var,
            self._append_out,
        )

    def _on_create_expert(self) -> None:
        """Handle create expert button."""
        handle_create_expert(self._append_out)

    def _on_delete_expert(self) -> None:
        """Handle delete expert button."""
        handle_delete_expert(self._selected_expert_id(), self._append_out)

    def _on_set_status(self, action: str) -> None:
        """Handle status change buttons."""
        handle_set_status(self._selected_expert_id(), action, self._append_out)

    def _on_set_parent(self) -> None:
        """Handle set parent button."""
        handle_set_parent(
            self._selected_expert_id(),
            self.parent_expert_var.get().strip(),
            self._append_out,
        )

    def _on_clear_parent(self) -> None:
        """Handle clear parent button."""
        handle_clear_parent(self._selected_expert_id(), self._append_out)

    def _link_goal_to_expert(self) -> None:
        """Handle link goal button."""
        handle_link_goal(
            self._selected_expert_id(),
            self.link_goal_var.get().strip(),
            self.link_goal_var,
            self._append_out,
            self._on_expert_select,
        )

    def _unlink_selected_goals(self) -> None:
        """Handle unlink goals button."""
        handle_unlink_goals(
            self._selected_expert_id(),
            self.goals_list,
            self._append_out,
            self._on_expert_select,
        )
