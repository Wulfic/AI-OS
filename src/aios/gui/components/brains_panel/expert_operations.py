"""Expert operation handlers (create, delete, status, parent).

All expert action methods extracted as functions.
"""

from __future__ import annotations

from typing import Any, Optional, cast

try:  # pragma: no cover
    from tkinter import messagebox, simpledialog  # type: ignore
except Exception:  # pragma: no cover
    messagebox = cast(Any, None)
    simpledialog = cast(Any, None)


def get_selected_expert_id(panel: Any) -> Optional[str]:
    """Get the expert_id of the currently selected expert.
    
    Args:
        panel: BrainsPanel instance
        
    Returns:
        Expert ID string, or None if no selection
    """
    sel = panel.experts_tree.selection()
    if not sel:
        return None
    
    try:
        tags = panel.experts_tree.item(sel[0]).get("tags", ())
        if tags:
            return tags[0]
    except Exception:
        pass
    
    return None


def create_expert(panel: Any) -> None:
    """Create a new expert via CLI (not yet implemented).
    
    Args:
        panel: BrainsPanel instance
    """
    if simpledialog is None or messagebox is None:
        return
    
    # Get expert details from user
    name = simpledialog.askstring("Create Expert", "Expert name:")
    if not name:
        return
    
    category = simpledialog.askstring("Create Expert", "Category (e.g., Programming, Math):") or "General"
    description = simpledialog.askstring("Create Expert", "Description:") or ""
    
    # Expert creation via GUI - CLI integration pending
    panel._append_out(
        f"[experts] Create expert: name='{name}', category='{category}', description='{description}'\n"
        f"[experts] Note: CLI integration for expert creation is in development"
    )
    
    messagebox.showinfo("Create Expert", "Expert creation via GUI is under development.\n"
                                         "Use CLI commands for expert management.")

def delete_expert(panel: Any) -> None:
    """Delete selected expert (not yet implemented).
    
    Args:
        panel: BrainsPanel instance
    """
    if messagebox is None:
        return
    
    expert_id = get_selected_expert_id(panel)
    if not expert_id:
        messagebox.showwarning("Delete Expert", "Please select an expert first.")
        return
    
    # Confirm
    ok = messagebox.askyesno("Delete Expert", f"Are you sure you want to delete expert {expert_id[:8]}...?")
    if not ok:
        return
    
    # Expert deletion via GUI - CLI integration pending
    panel._append_out(f"[experts] Delete expert: {expert_id}\n"
                     f"[experts] Note: CLI integration for expert deletion is in development")
    
    messagebox.showinfo("Delete Expert", "Expert deletion via GUI is under development.\n"
                                        "Use CLI commands for expert management.")

def set_expert_status(panel: Any, action: str) -> None:
    """Set expert status (activate, deactivate, freeze, unfreeze).
    
    Not yet implemented - placeholder for future CLI commands.
    
    Args:
        panel: BrainsPanel instance
        action: Status action string ('active', 'inactive', 'freeze', 'unfreeze')
    """
    if messagebox is None:
        return
    
    expert_id = get_selected_expert_id(panel)
    if not expert_id:
        messagebox.showwarning("Set Status", "Please select an expert first.")
        return
    
    # Expert status management via GUI - CLI integration pending
    panel._append_out(f"[experts] Set status: expert={expert_id[:8]}..., action={action}\n"
                     f"[experts] Note: CLI integration for expert status is in development")
    
    messagebox.showinfo("Set Status", f"Expert {action} via GUI is under development.\n"
                                     "Use CLI commands for expert management.")

def set_expert_parent(panel: Any) -> None:
    """Set parent expert for selected expert (not yet implemented).
    
    Args:
        panel: BrainsPanel instance
    """
    if messagebox is None:
        return
    
    expert_id = get_selected_expert_id(panel)
    parent_id = panel.expert_parent_var.get().strip()
    
    if not expert_id:
        messagebox.showwarning("Set Parent", "Please select an expert first.")
        return
    
    if not parent_id:
        messagebox.showwarning("Set Parent", "Please enter a parent expert ID.")
        return
    
    # Expert parent assignment via GUI - CLI integration pending
    panel._append_out(f"[experts] Set parent: expert={expert_id[:8]}..., parent={parent_id}\n"
                     f"[experts] Note: CLI integration for parent assignment is in development")
    
    messagebox.showinfo("Set Parent", "Parent assignment via GUI is under development.\n"
                                     "Use CLI commands for expert management.")

def clear_expert_parent(panel: Any) -> None:
    """Clear parent expert from selected expert (not yet implemented).
    
    Args:
        panel: BrainsPanel instance
    """
    if messagebox is None:
        return
    
    expert_id = get_selected_expert_id(panel)
    if not expert_id:
        messagebox.showwarning("Clear Parent", "Please select an expert first.")
        return
    
    # Expert parent clearing via GUI - CLI integration pending
    panel._append_out(f"[experts] Clear parent: expert={expert_id[:8]}...\n"
                     f"[experts] Note: CLI integration for parent clearing is in development")
    
    messagebox.showinfo("Clear Parent", "Clear parent via GUI is under development.\n"
                                       "Use CLI commands for expert management.")
