"""Subbrains Manager Panel for dynamic expert management.

This panel provides a visual interface for:
- Viewing expert registry with hierarchy
- Creating, deleting, freezing experts
- Binding goals to experts
- Viewing performance metrics and routing stats
"""

from __future__ import annotations

from typing import Any, Callable, cast, Optional, Dict, List
import os
import json

try:  # pragma: no cover - environment dependent
    import tkinter as tk  # type: ignore
    from tkinter import ttk  # type: ignore
    from tkinter import messagebox, simpledialog  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    tk = cast(Any, None)
    ttk = cast(Any, None)
    messagebox = cast(Any, None)
    simpledialog = cast(Any, None)


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
        super().__init__(parent, text="Dynamic Subbrains Manager")
        if tk is None:
            raise RuntimeError("Tkinter not available")
        
        self.pack(fill="both", expand=True, padx=8, pady=8)

        self._run_cli = run_cli
        self._append_out = append_out or (lambda s: None)
        self._on_goal_add = on_goal_add
        self._on_goals_list = on_goals_list
        self._on_goal_remove = on_goal_remove
        
        # Detect project root
        def _project_root() -> str:
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

        self._project_root = _project_root()
        self._registry_path = os.path.join(
            self._project_root, "artifacts", "models", "expert_registry.json"
        )
        
        # Create UI sections
        self._create_summary_section()
        self._create_experts_tree()
        self._create_management_controls()
        self._create_goals_section()
        
        # Initial refresh
        self.refresh()

    def _create_summary_section(self):
        """Create summary statistics row."""
        summary = ttk.Frame(self)
        summary.pack(fill="x", pady=(0, 8))
        
        # Total experts
        ttk.Label(summary, text="Total Experts:").pack(side="left")
        self.total_experts_var = tk.StringVar(value="0")
        ttk.Label(summary, textvariable=self.total_experts_var, width=6).pack(side="left")
        
        # Active experts
        ttk.Label(summary, text="Active:").pack(side="left", padx=(12, 0))
        self.active_experts_var = tk.StringVar(value="0")
        ttk.Label(summary, textvariable=self.active_experts_var, width=6).pack(side="left")
        
        # Frozen experts
        ttk.Label(summary, text="Frozen:").pack(side="left", padx=(12, 0))
        self.frozen_experts_var = tk.StringVar(value="0")
        ttk.Label(summary, textvariable=self.frozen_experts_var, width=6).pack(side="left")
        
        # Total activations
        ttk.Label(summary, text="Total Activations:").pack(side="left", padx=(12, 0))
        self.total_activations_var = tk.StringVar(value="0")
        ttk.Label(summary, textvariable=self.total_activations_var, width=10).pack(side="left")
        
        # Refresh button
        btn_refresh = ttk.Button(summary, text="Refresh", command=self.refresh)
        btn_refresh.pack(side="right")
        
        # Tooltips
        try:
            from .tooltips import add_tooltip
            add_tooltip(btn_refresh, "Reload expert registry from disk and update display")
        except Exception:
            pass

    def _create_experts_tree(self):
        """Create expert tree view with hierarchy and metrics."""
        tree_frame = ttk.LabelFrame(self, text="Experts", padding=4)
        tree_frame.pack(fill="both", expand=True, pady=(0, 8))
        
        # Columns: name, category, status, activations, avg_weight, goals, parent/children
        cols = ("name", "category", "status", "activations", "avg_weight", "goals", "hierarchy")
        self.experts_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
        
        self.experts_tree.heading("name", text="Name")
        self.experts_tree.heading("category", text="Category")
        self.experts_tree.heading("status", text="Status")
        self.experts_tree.heading("activations", text="Activations")
        self.experts_tree.heading("avg_weight", text="Avg Weight")
        self.experts_tree.heading("goals", text="Goals")
        self.experts_tree.heading("hierarchy", text="Parent/Children")
        
        self.experts_tree.column("name", width=150, anchor="w")
        self.experts_tree.column("category", width=100, anchor="w")
        self.experts_tree.column("status", width=100, anchor="center")
        self.experts_tree.column("activations", width=90, anchor="e")
        self.experts_tree.column("avg_weight", width=90, anchor="e")
        self.experts_tree.column("goals", width=60, anchor="center")
        self.experts_tree.column("hierarchy", width=150, anchor="w")
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.experts_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.experts_tree.xview)
        self.experts_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack
        self.experts_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.experts_tree.bind("<<TreeviewSelect>>", self._on_expert_select)
        
        # Tooltips
        try:
            from .tooltips import add_tooltip
            add_tooltip(
                self.experts_tree,
                "Expert registry. Status: Active (routing enabled), Frozen (weights locked), Inactive (not in use). "
                "Select an expert to view/manage its goals.",
                wrap=80
            )
        except Exception:
            pass

    def _create_management_controls(self):
        """Create expert management controls."""
        mgmt_frame = ttk.Frame(self)
        mgmt_frame.pack(fill="x", pady=(0, 8))
        
        # Row 1: Create, Delete, Status controls
        row1 = ttk.Frame(mgmt_frame)
        row1.pack(fill="x", pady=(0, 4))
        
        btn_create = ttk.Button(row1, text="Create Expert", command=self._on_create_expert)
        btn_create.pack(side="left")
        
        btn_delete = ttk.Button(row1, text="Delete Selected", command=self._on_delete_expert)
        btn_delete.pack(side="left", padx=(6, 0))
        
        btn_activate = ttk.Button(row1, text="Activate", command=lambda: self._on_set_status("active"))
        btn_activate.pack(side="left", padx=(12, 0))
        
        btn_deactivate = ttk.Button(row1, text="Deactivate", command=lambda: self._on_set_status("inactive"))
        btn_deactivate.pack(side="left", padx=(6, 0))
        
        btn_freeze = ttk.Button(row1, text="Freeze", command=lambda: self._on_set_status("freeze"))
        btn_freeze.pack(side="left", padx=(6, 0))
        
        btn_unfreeze = ttk.Button(row1, text="Unfreeze", command=lambda: self._on_set_status("unfreeze"))
        btn_unfreeze.pack(side="left", padx=(6, 0))
        
        # Row 2: Parent/child management
        row2 = ttk.Frame(mgmt_frame)
        row2.pack(fill="x")
        
        ttk.Label(row2, text="Set Parent:").pack(side="left")
        self.parent_expert_var = tk.StringVar(value="")
        entry_parent = ttk.Entry(row2, textvariable=self.parent_expert_var, width=20)
        entry_parent.pack(side="left", padx=(4, 0))
        
        btn_set_parent = ttk.Button(row2, text="Apply", command=self._on_set_parent)
        btn_set_parent.pack(side="left", padx=(4, 0))
        
        btn_clear_parent = ttk.Button(row2, text="Clear Parent", command=self._on_clear_parent)
        btn_clear_parent.pack(side="left", padx=(6, 0))
        
        # Tooltips
        try:
            from .tooltips import add_tooltip
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

    def _create_goals_section(self):
        """Create goals management section for selected expert."""
        if self._on_goals_list is None:
            return  # Goals integration not available
        
        goals_frame = ttk.LabelFrame(self, text="Goals for Selected Expert", padding=8)
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
        
        self.goals_list = tk.Listbox(list_container, height=6, selectmode="extended")
        self.goals_list.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.goals_list.yview)
        self.goals_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Goals control bar
        controls = ttk.Frame(goals_frame)
        controls.pack(fill="x", pady=(0, 6))
        
        if self._on_goal_remove is not None:
            btn_unlink = ttk.Button(controls, text="Unlink Selected", command=self._unlink_selected_goals)
            btn_unlink.pack(side="left")
            try:
                from .tooltips import add_tooltip
                add_tooltip(btn_unlink, "Remove goal-expert binding for selected goals")
            except Exception:
                pass
        
        self.goals_count_var = tk.StringVar(value="No expert selected")
        ttk.Label(controls, textvariable=self.goals_count_var).pack(side="right")
        
        # Link goal bar
        link_bar = ttk.Frame(goals_frame)
        link_bar.pack(fill="x")
        
        ttk.Label(link_bar, text="Link Goal:").pack(side="left")
        
        self.link_goal_var = tk.StringVar(value="")
        goal_entry = ttk.Entry(link_bar, textvariable=self.link_goal_var)
        goal_entry.pack(side="left", fill="x", expand=True, padx=(4, 8))
        
        if self._on_goal_add is not None:
            btn_link = ttk.Button(link_bar, text="Link", command=self._link_goal_to_expert)
            btn_link.pack(side="left")
            
            # Bind Return key
            goal_entry.bind("<Return>", lambda e: self._link_goal_to_expert())
            
            try:
                from .tooltips import add_tooltip
                add_tooltip(goal_entry, "Enter goal ID or text to link to the selected expert")
                add_tooltip(btn_link, "Create goal-expert binding. Expert will receive routing bias when goal is active.")
            except Exception:
                pass

    def refresh(self):
        """Refresh expert registry data from disk."""
        try:
            # Load expert registry
            if os.path.exists(self._registry_path):
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
                
                experts = registry_data.get("experts", [])
            else:
                experts = []
                self._append_out(f"[Subbrains] No registry found at {self._registry_path}")
            
            # Update summary stats
            total = len(experts)
            active = sum(1 for e in experts if e.get("is_active", False))
            frozen = sum(1 for e in experts if e.get("is_frozen", False))
            total_acts = sum(e.get("total_activations", 0) for e in experts)
            
            self.total_experts_var.set(str(total))
            self.active_experts_var.set(str(active))
            self.frozen_experts_var.set(str(frozen))
            self.total_activations_var.set(str(total_acts))
            
            # Update tree
            for item in self.experts_tree.get_children():
                self.experts_tree.delete(item)
            
            for expert in experts:
                try:
                    name = expert.get("name", "Unnamed")
                    category = expert.get("category", "")
                    
                    # Status
                    is_active = expert.get("is_active", False)
                    is_frozen = expert.get("is_frozen", False)
                    if is_frozen:
                        status = "Frozen"
                    elif is_active:
                        status = "Active"
                    else:
                        status = "Inactive"
                    
                    # Metrics
                    activations = expert.get("total_activations", 0)
                    avg_weight = expert.get("avg_routing_weight", 0.0)
                    
                    # Goals
                    goals = expert.get("goals", [])
                    goals_display = f"{len(goals)} linked" if goals else "—"
                    
                    # Hierarchy
                    parent = expert.get("parent_expert_id") or ""
                    children = expert.get("child_expert_ids", [])
                    hierarchy = ""
                    if parent:
                        hierarchy = f"p:{parent[:8]}"
                    if children:
                        hierarchy += ("; " if hierarchy else "") + f"{len(children)} child(ren)"
                    if not hierarchy:
                        hierarchy = "—"
                    
                    values = (
                        name,
                        category,
                        status,
                        str(activations),
                        f"{avg_weight:.3f}",
                        goals_display,
                        hierarchy
                    )
                    
                    # Store expert_id as item tag for later retrieval
                    expert_id = expert.get("expert_id", "")
                    item = self.experts_tree.insert("", "end", values=values, tags=(expert_id,))
                    
                except Exception as e:
                    self._append_out(f"[Subbrains] Error displaying expert: {e}")
                    continue
            
            self._append_out(f"[Subbrains] Loaded {total} expert(s)")
            
        except Exception as e:
            self._append_out(f"[Subbrains] Error refreshing: {e}")
            import traceback
            self._append_out(traceback.format_exc())

    def _selected_expert_id(self) -> Optional[str]:
        """Get the expert_id of the currently selected expert."""
        sel = self.experts_tree.selection()
        if not sel:
            return None
        
        try:
            tags = self.experts_tree.item(sel[0]).get("tags", ())
            if tags:
                return tags[0]
        except Exception:
            pass
        
        return None

    def _on_expert_select(self, event=None):
        """Handle expert selection - refresh goals for selected expert."""
        if not hasattr(self, 'goals_list'):
            return
        
        expert_id = self._selected_expert_id()
        if not expert_id:
            # No expert selected
            self.goals_list.delete(0, tk.END)
            self.goals_count_var.set("No expert selected")
            return
        
        # TODO: Load goals linked to this expert
        # For now, show placeholder
        self.goals_list.delete(0, tk.END)
        self.goals_count_var.set(f"Expert: {expert_id[:8]}...")
        
        # Load goals from registry
        try:
            if os.path.exists(self._registry_path):
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    registry_data = json.load(f)
                
                experts = registry_data.get("experts", [])
                for expert in experts:
                    if expert.get("expert_id") == expert_id:
                        goals = expert.get("goals", [])
                        for goal_id in goals:
                            self.goals_list.insert(tk.END, f"Goal #{goal_id}")
                        
                        count = len(goals)
                        self.goals_count_var.set(f"{count} goal{'s' if count != 1 else ''} linked")
                        break
        except Exception as e:
            self._append_out(f"[Subbrains] Error loading goals: {e}")

    def _on_create_expert(self):
        """Create a new expert via CLI."""
        if tk is None or simpledialog is None:
            return
        
        # Get expert details from user
        name = simpledialog.askstring("Create Expert", "Expert name:")
        if not name:
            return
        
        category = simpledialog.askstring("Create Expert", "Category (e.g., Programming, Math):") or "General"
        description = simpledialog.askstring("Create Expert", "Description:") or ""
        
        # TODO: Add CLI command to create expert
        # For now, show what would be created
        self._append_out(
            f"[Subbrains] Create expert: name='{name}', category='{category}', description='{description}'\n"
            f"[Subbrains] CLI command needed: not yet implemented"
        )
        
        messagebox.showinfo("Create Expert", "Expert creation CLI command not yet implemented.\n"
                                             "This will be added in the next phase.")

    def _on_delete_expert(self):
        """Delete selected expert."""
        expert_id = self._selected_expert_id()
        if not expert_id:
            messagebox.showwarning("Delete Expert", "Please select an expert first.")
            return
        
        if messagebox is None:
            return
        
        # Confirm
        ok = messagebox.askyesno("Delete Expert", f"Are you sure you want to delete expert {expert_id[:8]}...?")
        if not ok:
            return
        
        # TODO: Add CLI command to delete expert
        self._append_out(f"[Subbrains] Delete expert: {expert_id}\n"
                        f"[Subbrains] CLI command needed: not yet implemented")
        
        messagebox.showinfo("Delete Expert", "Expert deletion CLI command not yet implemented.\n"
                                            "This will be added in the next phase.")

    def _on_set_status(self, action: str):
        """Set expert status (activate, deactivate, freeze, unfreeze)."""
        expert_id = self._selected_expert_id()
        if not expert_id:
            if messagebox is not None:
                messagebox.showwarning("Set Status", "Please select an expert first.")
            return
        
        # TODO: Add CLI command to set expert status
        self._append_out(f"[Subbrains] Set status: expert={expert_id[:8]}..., action={action}\n"
                        f"[Subbrains] CLI command needed: not yet implemented")
        
        if messagebox is not None:
            messagebox.showinfo("Set Status", f"Expert {action} CLI command not yet implemented.\n"
                                             "This will be added in the next phase.")

    def _on_set_parent(self):
        """Set parent expert for selected expert."""
        expert_id = self._selected_expert_id()
        parent_id = self.parent_expert_var.get().strip()
        
        if not expert_id:
            if messagebox is not None:
                messagebox.showwarning("Set Parent", "Please select an expert first.")
            return
        
        if not parent_id:
            if messagebox is not None:
                messagebox.showwarning("Set Parent", "Please enter a parent expert ID.")
            return
        
        # TODO: Add CLI command to set parent
        self._append_out(f"[Subbrains] Set parent: expert={expert_id[:8]}..., parent={parent_id}\n"
                        f"[Subbrains] CLI command needed: not yet implemented")
        
        if messagebox is not None:
            messagebox.showinfo("Set Parent", "Parent assignment CLI command not yet implemented.\n"
                                             "This will be added in the next phase.")

    def _on_clear_parent(self):
        """Clear parent expert from selected expert."""
        expert_id = self._selected_expert_id()
        if not expert_id:
            if messagebox is not None:
                messagebox.showwarning("Clear Parent", "Please select an expert first.")
            return
        
        # TODO: Add CLI command to clear parent
        self._append_out(f"[Subbrains] Clear parent: expert={expert_id[:8]}...\n"
                        f"[Subbrains] CLI command needed: not yet implemented")
        
        if messagebox is not None:
            messagebox.showinfo("Clear Parent", "Clear parent CLI command not yet implemented.\n"
                                               "This will be added in the next phase.")

    def _link_goal_to_expert(self):
        """Link a goal to the selected expert."""
        expert_id = self._selected_expert_id()
        goal_text = self.link_goal_var.get().strip()
        
        if not expert_id:
            if messagebox is not None:
                messagebox.showwarning("Link Goal", "Please select an expert first.")
            return
        
        if not goal_text:
            return
        
        # TODO: Use goals-link-expert CLI command
        self._append_out(f"[Subbrains] Link goal to expert: expert={expert_id[:8]}..., goal='{goal_text}'\n"
                        f"[Subbrains] Using CLI: goals-link-expert <goal_id> {expert_id}")
        
        # Clear entry
        self.link_goal_var.set("")
        
        # Refresh goals list
        self._on_expert_select()

    def _unlink_selected_goals(self):
        """Unlink selected goals from expert."""
        expert_id = self._selected_expert_id()
        if not expert_id or not hasattr(self, 'goals_list'):
            return
        
        sel = self.goals_list.curselection()
        if not sel:
            return
        
        # TODO: Extract goal IDs and use goals-unlink-expert CLI command
        self._append_out(f"[Subbrains] Unlink goals from expert {expert_id[:8]}...\n"
                        f"[Subbrains] CLI command needed: goals-unlink-expert <goal_id> {expert_id}")
        
        # Refresh
        self._on_expert_select()
