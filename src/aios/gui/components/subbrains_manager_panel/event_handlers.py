"""Event handlers for the Subbrains Manager Panel.

Provides event handling functions for expert management actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    pass

try:  # pragma: no cover - environment dependent
    from tkinter import messagebox, simpledialog  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    messagebox = cast(Any, None)
    simpledialog = cast(Any, None)


def handle_create_expert(append_out_callback: Any) -> None:
    """Handle creating a new expert.
    
    Args:
        append_out_callback: Callback for logging output
    """
    if simpledialog is None or messagebox is None:
        return
    
    # Get expert details from user
    name = simpledialog.askstring("Create Expert", "Expert name:")
    if not name:
        return
    
    category = simpledialog.askstring("Create Expert", "Category (e.g., Programming, Math):") or "General"
    description = simpledialog.askstring("Create Expert", "Description:") or ""
    
    # NOTE: Expert management CLI commands are planned for next phase
    # Will use: aios experts create --name <name> --category <category>
    # For now, show what would be created
    append_out_callback(
        f"[Subbrains] Create expert: name='{name}', category='{category}', description='{description}'\n"
        f"[Subbrains] CLI command needed: not yet implemented"
    )
    
    messagebox.showinfo("Create Expert", "Expert creation CLI command not yet implemented.\n"
                                         "This will be added in the next phase.")


def handle_delete_expert(expert_id: Optional[str], append_out_callback: Any) -> None:
    """Handle deleting an expert.
    
    Args:
        expert_id: ID of expert to delete
        append_out_callback: Callback for logging output
    """
    if not expert_id:
        if messagebox is not None:
            messagebox.showwarning("Delete Expert", "Please select an expert first.")
        return
    
    if messagebox is None:
        return
    
    # Confirm
    ok = messagebox.askyesno("Delete Expert", f"Are you sure you want to delete expert {expert_id[:8]}...?")
    if not ok:
        return
    
    # NOTE: Expert deletion CLI command planned for next phase
    # Will use: aios experts delete <expert_id>
    append_out_callback(f"[Subbrains] Delete expert: {expert_id}\n"
                       f"[Subbrains] CLI command needed: not yet implemented")
    
    messagebox.showinfo("Delete Expert", "Expert deletion CLI command not yet implemented.\n"
                                        "This will be added in the next phase.")


def handle_set_status(expert_id: Optional[str], action: str, append_out_callback: Any) -> None:
    """Handle setting expert status.
    
    Args:
        expert_id: ID of expert to modify
        action: Status action ('active', 'inactive', 'freeze', 'unfreeze')
        append_out_callback: Callback for logging output
    """
    if not expert_id:
        if messagebox is not None:
            messagebox.showwarning("Set Status", "Please select an expert first.")
        return
    
    # NOTE: Expert status CLI commands planned for next phase
    # Will use: aios experts set-status <expert_id> <status>
    append_out_callback(f"[Subbrains] Set status: expert={expert_id[:8]}..., action={action}\n"
                       f"[Subbrains] CLI command needed: not yet implemented")
    
    if messagebox is not None:
        messagebox.showinfo("Set Status", f"Expert {action} CLI command not yet implemented.\n"
                                         "This will be added in the next phase.")


def handle_set_parent(expert_id: Optional[str], parent_id: str, append_out_callback: Any) -> None:
    """Handle setting parent expert.
    
    Args:
        expert_id: ID of expert to modify
        parent_id: ID of parent expert
        append_out_callback: Callback for logging output
    """
    if not expert_id:
        if messagebox is not None:
            messagebox.showwarning("Set Parent", "Please select an expert first.")
        return
    
    if not parent_id:
        if messagebox is not None:
            messagebox.showwarning("Set Parent", "Please enter a parent expert ID.")
        return
    
    # NOTE: Expert parent management CLI command planned for next phase
    # Will use: aios experts set-parent <expert_id> <parent_id>
    append_out_callback(f"[Subbrains] Set parent: expert={expert_id[:8]}..., parent={parent_id}\n"
                       f"[Subbrains] CLI command needed: not yet implemented")
    
    if messagebox is not None:
        messagebox.showinfo("Set Parent", "Parent assignment CLI command not yet implemented.\n"
                                         "This will be added in the next phase.")


def handle_clear_parent(expert_id: Optional[str], append_out_callback: Any) -> None:
    """Handle clearing parent expert.
    
    Args:
        expert_id: ID of expert to modify
        append_out_callback: Callback for logging output
    """
    if not expert_id:
        if messagebox is not None:
            messagebox.showwarning("Clear Parent", "Please select an expert first.")
        return
    
    # NOTE: Clear parent CLI command planned for next phase
    # Will use: aios experts clear-parent <expert_id>
    append_out_callback(f"[Subbrains] Clear parent: expert={expert_id[:8]}...\n"
                       f"[Subbrains] CLI command needed: not yet implemented")
    
    if messagebox is not None:
        messagebox.showinfo("Clear Parent", "Clear parent CLI command not yet implemented.\n"
                                           "This will be added in the next phase.")


def handle_link_goal(
    expert_id: Optional[str],
    goal_text: str,
    link_goal_var: Any,
    append_out_callback: Any,
    on_expert_select_callback: Any,
) -> None:
    """Handle linking a goal to an expert with async refresh.
    
    Args:
        expert_id: ID of expert to link goal to
        goal_text: Goal text/ID to link
        link_goal_var: StringVar to clear after linking
        append_out_callback: Callback for logging output
        on_expert_select_callback: Callback to refresh goals list
    """
    import threading
    
    if not expert_id:
        if messagebox is not None:
            messagebox.showwarning("Link Goal", "Please select an expert first.")
        return
    
    if not goal_text:
        return
    
    append_out_callback(f"[Subbrains] Linking goal to expert {expert_id[:8]}...")
    
    def run_link():
        # NOTE: Goal-expert linking CLI command planned for next phase
        # Will use: aios goals link-expert <goal_id> <expert_id>
        append_out_callback(f"[Subbrains] Link goal to expert: expert={expert_id[:8]}..., goal='{goal_text}'\n"
                           f"[Subbrains] Using CLI: goals-link-expert <goal_id> {expert_id}")
        
        # Clear entry and refresh on main thread
        def update_ui():
            link_goal_var.set("")
            on_expert_select_callback()
        
        # Schedule UI update on main thread
        try:
            import tkinter as tk
            tk._default_root.after(0, update_ui)  # type: ignore
        except Exception:
            update_ui()  # Fallback to direct call
    
    threading.Thread(target=run_link, daemon=True).start()


def handle_unlink_goals(
    expert_id: Optional[str],
    goals_list: Any,
    append_out_callback: Any,
    on_expert_select_callback: Any,
) -> None:
    """Handle unlinking selected goals from expert with async refresh.
    
    Args:
        expert_id: ID of expert to unlink goals from
        goals_list: Listbox widget with goal selection
        append_out_callback: Callback for logging output
        on_expert_select_callback: Callback to refresh goals list
    """
    import threading
    
    if not expert_id or goals_list is None:
        return
    
    sel = goals_list.curselection()
    if not sel:
        return
    
    append_out_callback(f"[Subbrains] Unlinking goals from expert {expert_id[:8]}...")
    
    def run_unlink():
        # NOTE: Goal-expert unlinking CLI command planned for next phase
        # Will use: aios goals unlink-expert <goal_id> <expert_id>
        append_out_callback(f"[Subbrains] Unlink goals from expert {expert_id[:8]}...\n"
                           f"[Subbrains] CLI command needed: goals-unlink-expert <goal_id> {expert_id}")
        
        # Refresh on main thread
        try:
            import tkinter as tk
            tk._default_root.after(0, on_expert_select_callback)  # type: ignore
        except Exception:
            on_expert_select_callback()  # Fallback to direct call
    
    threading.Thread(target=run_unlink, daemon=True).start()
