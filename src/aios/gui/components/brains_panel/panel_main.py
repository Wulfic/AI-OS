"""Main BrainsPanel class - orchestrates all brains and experts management UI.

The panel class delegates to builder functions and operation handlers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    ttk = cast(Any, None)


class BrainsPanel(ttk.LabelFrame):  # type: ignore[misc]
    """Unified panel displaying brains registry and dynamic experts (subbrains) management."""

    def __init__(
        self,
        parent: Any,
        *,
        run_cli: Callable[[list[str]], str],
        append_out: Optional[Callable[[str], None]] = None,
        on_goal_add: Optional[Callable[[str, str], None]] = None,
        on_goals_list: Optional[Callable[[str], list[str]]] = None,
        on_goal_remove: Optional[Callable[[int], None]] = None,
        worker_pool: Any = None,
    ) -> None:
        """Initialize brains panel.
        
        Args:
            parent: Parent Tk widget
            run_cli: Callback to run CLI commands (takes list of args, returns output string)
            append_out: Optional callback to append output messages
            on_goal_add: Optional callback to add a goal (brain_name, goal_text)
            on_goals_list: Optional callback to list goals for a brain (returns list of strings)
            on_goal_remove: Optional callback to remove a goal by ID
            worker_pool: Optional AsyncWorkerPool for background operations
        """
        super().__init__(parent, text="Brains & Experts (Subbrains) Management")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        self.pack(fill="both", expand=True, padx=8, pady=8)

        # Store callbacks
        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._on_goal_add = on_goal_add
        self._on_goals_list = on_goals_list
        self._on_goal_remove = on_goal_remove
        self._worker_pool = worker_pool
        
        # Detect project root and build paths
        from .helpers import find_project_root
        
        self._project_root = find_project_root()
        self._store_dir = os.path.join(self._project_root, "artifacts", "brains")
        self._registry_path = os.path.join(
            self._project_root, "artifacts", "models", "expert_registry.json"
        )

        # Build UI sections
        from .ui_summary import build_summary_row
        from .ui_brains_section import build_brains_section
        from .ui_experts_section import build_experts_section
        from .ui_goals_section import build_goals_section
        
        build_summary_row(self, self)
        build_brains_section(self, self)
        build_experts_section(self, self)
        build_goals_section(self, self)

    def refresh(self, force: bool = False) -> None:
        """Fetch stats via CLI and populate both brains and experts tables/summaries.
        
        This method runs asynchronously via worker pool to prevent blocking the GUI.
        
        Args:
            force: If True, bypass throttling and force refresh
        """
        from .data_loading import refresh_brains_data, refresh_experts_data
        
        # Throttle refreshes - only refresh if >5 seconds since last refresh (unless forced)
        if not force:
            if not hasattr(self, '_last_refresh_time'):
                self._last_refresh_time = 0.0
            
            current_time = time.time()
            if current_time - self._last_refresh_time < 5.0:
                # Too soon - skip refresh
                return
            
            self._last_refresh_time = current_time
        
        # Set loading indicator
        if not hasattr(self, '_refresh_in_progress'):
            self._refresh_in_progress = False
        
        if self._refresh_in_progress:
            # Already refreshing, skip
            return
        
        self._refresh_in_progress = True
        
        # Update status to show loading
        if hasattr(self, 'status_var'):
            self.status_var.set("Loading...")
        
        def _do_refresh():
            """Background refresh operation."""
            try:
                names = refresh_brains_data(self)
                refresh_experts_data(self)
                
                total_experts = int(self.total_experts_var.get() or "0")
                
                # Schedule UI update on main thread
                def _update_ui():
                    self._append_out(f"[brains] Loaded {len(names)} brain(s) and {total_experts} expert(s)")
                    if hasattr(self, 'status_var'):
                        self.status_var.set("")  # Clear loading status
                    self._refresh_in_progress = False
                
                try:
                    self.after(0, _update_ui)
                except Exception:
                    self._refresh_in_progress = False
                    if hasattr(self, 'status_var'):
                        try:
                            self.status_var.set("")
                        except Exception:
                            pass
            except Exception as e:
                # Schedule error handling on main thread
                def _handle_error():
                    self._append_out(f"[brains] Refresh failed: {e}")
                    if hasattr(self, 'status_var'):
                        self.status_var.set("Error")
                    self._refresh_in_progress = False
                
                try:
                    self.after(0, _handle_error)
                except Exception:
                    self._refresh_in_progress = False
                    if hasattr(self, 'status_var'):
                        try:
                            self.status_var.set("Error")
                        except Exception:
                            pass
        
        # Submit to worker pool if available
        if self._worker_pool:
            self._worker_pool.submit(_do_refresh)
        else:
            # Fallback to direct execution
            import threading
            threading.Thread(target=_do_refresh, daemon=True).start()

    # === BRAIN OPERATION HANDLERS ===
    
    def _on_pin(self) -> None:
        """Pin selected brain."""
        from .brain_operations import pin_brain
        pin_brain(self)

    def _on_unpin(self) -> None:
        """Unpin selected brain."""
        from .brain_operations import unpin_brain
        unpin_brain(self)

    def _on_delete(self) -> None:
        """Delete selected brain with confirmation."""
        from .brain_operations import delete_brain
        delete_brain(self)

    def _on_details(self) -> None:
        """Show detailed information dialog for selected brain."""
        from .brain_operations import show_brain_details
        show_brain_details(self)

    def _on_rename(self) -> None:
        """Rename selected brain."""
        from .brain_operations import rename_brain
        rename_brain(self)

    def _on_set_master(self, enabled: bool) -> None:
        """Set or unset master status."""
        from .brain_operations import set_master_status
        set_master_status(self, enabled)

    def _on_set_parent(self) -> None:
        """Set parent brain."""
        from .brain_operations import set_parent_brain
        set_parent_brain(self)

    def _on_clear_parent(self) -> None:
        """Clear parent brain."""
        from .brain_operations import clear_parent_brain
        clear_parent_brain(self)

    # === EXPERT OPERATION HANDLERS ===

    def _on_create_expert(self) -> None:
        """Create a new expert."""
        from .expert_operations import create_expert
        create_expert(self)

    def _on_delete_expert(self) -> None:
        """Delete selected expert."""
        from .expert_operations import delete_expert
        delete_expert(self)

    def _on_set_expert_status(self, action: str) -> None:
        """Set expert status."""
        from .expert_operations import set_expert_status
        set_expert_status(self, action)

    def _on_set_expert_parent(self) -> None:
        """Set parent expert."""
        from .expert_operations import set_expert_parent
        set_expert_parent(self)

    def _on_clear_expert_parent(self) -> None:
        """Clear parent expert."""
        from .expert_operations import clear_expert_parent
        clear_expert_parent(self)

    # === GOALS OPERATION HANDLERS ===

    def _on_tree_select(self, event: Any = None) -> None:
        """Handle tree selection change - refresh goals for newly selected brain or expert."""
        from .goals_operations import refresh_goals
        refresh_goals(self)

    def _add_goal(self) -> None:
        """Add a new goal for the currently selected brain or expert."""
        from .goals_operations import add_goal
        add_goal(self)

    def _remove_selected_goals(self) -> None:
        """Remove selected goals from the currently selected brain or expert."""
        from .goals_operations import remove_selected_goals
        remove_selected_goals(self)
