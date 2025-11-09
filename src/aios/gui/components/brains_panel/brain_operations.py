"""Brain operation handlers (pin, delete, rename, master, parent, details).

All brain action methods extracted as functions for maintainability.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

try:  # pragma: no cover
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    messagebox = cast(Any, None)


def pin_brain(panel: Any) -> None:
    """Pin selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        panel._append_out("No brain selected")
        return
    
    def run_pin():
        try:
            out = panel._run_cli(["brains", "pin", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error pinning brain: {e}"))
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_pin, daemon=True).start()
    panel._append_out(f"Pinning '{name}'...")


def unpin_brain(panel: Any) -> None:
    """Unpin selected brain with async refresh (checks for master status first).
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    # If selected is a master, block unpin at UI level as well
    try:
        is_master = get_selected_tree_value(panel.tree, 4) == "yes"
        if is_master:
            panel._append_out('{"ok": false, "error": "cannot unpin master brain"}')
            return
    except Exception:
        pass
    
    def run_unpin():
        try:
            out = panel._run_cli(["brains", "unpin", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error unpinning brain: {e}"))
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_unpin, daemon=True).start()
    panel._append_out(f"Unpinning '{name}'...")


def delete_brain(panel: Any) -> None:
    """Delete selected brain with confirmation and async refresh (blocks master brains).
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value, parse_cli_dict
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    # Prevent deleting master brains at UI level
    try:
        is_master = get_selected_tree_value(panel.tree, 4) == "yes"
        if is_master:
            if messagebox is not None:
                messagebox.showerror("Delete Brain", "Cannot delete a master brain. Unset master first.")
            return
    except Exception:
        pass
    
    # Ask for confirmation
    if messagebox is not None:
        ok = messagebox.askyesno("Delete Brain", 
                                f"Are you sure you want to delete '{name}'? This cannot be undone.")
        if not ok:
            return
    
    panel._append_out(f"Deleting '{name}'...")
    
    def run_delete():
        try:
            out = panel._run_cli(["brains", "delete", name, "--store-dir", panel._store_dir])
            
            # Parse and show result on main thread
            def show_result():
                panel._append_out(out)
                try:
                    res = parse_cli_dict(out)
                    if messagebox is not None:
                        if bool(res.get("ok")):
                            messagebox.showinfo("Delete Brain", f"Deleted '{name}'.")
                        else:
                            error_msg = res.get("error", "Unknown error")
                            messagebox.showerror("Delete Brain", f"Failed to delete '{name}': {error_msg}")
                            panel._append_out(f"[delete] Error: {error_msg}")
                except Exception as e:
                    if messagebox is not None:
                        messagebox.showerror("Delete Brain", f"Failed to parse response: {e}")
                    panel._append_out(f"[delete] Parse error: {e}")
            
            panel.after(0, show_result)
            
        except Exception as e:
            def show_error():
                if messagebox is not None:
                    messagebox.showerror("Delete Brain", f"Error during deletion: {e}")
                panel._append_out(f"[delete] Exception: {e}")
            
            panel.after(0, show_error)
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_delete, daemon=True).start()


def show_brain_details(panel: Any) -> None:
    """Show detailed information dialog for selected brain.
    
    Args:
        panel: BrainsPanel instance
    """
    from .brain_details_builder import build_brain_details_text
    from .brain_details_dialog import show_brain_details_dialog
    from .helpers import get_selected_tree_value, parse_cli_dict
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        if messagebox is not None:
            messagebox.showwarning("No Selection", "Please select a brain first.")
        return
    
    try:
        details_text = build_brain_details_text(
            brain_name=name,
            store_dir=panel._store_dir,
            run_cli=panel._run_cli,
            parse_cli_dict=parse_cli_dict,
            goals_list_callback=panel._on_goals_list
        )
        
        show_brain_details_dialog(panel, name, details_text)
        
    except Exception as e:
        if messagebox is not None:
            messagebox.showerror("Error", f"Failed to load brain details: {e}")
        panel._append_out(f"[details] Error: {e}")


def rename_brain(panel: Any) -> None:
    """Rename selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    new = (panel.rename_var.get() or "").strip()
    if not name or not new:
        return
    
    panel._append_out(f"Renaming '{name}' to '{new}'...")
    
    def run_rename():
        try:
            out = panel._run_cli(["brains", "rename", name, new, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error renaming: {e}"))
        finally:
            # Refresh in background and clear rename field
            panel.after(0, lambda: panel.rename_var.set(""))
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_rename, daemon=True).start()


def set_master_status(panel: Any, enabled: bool) -> None:
    """Set or unset master status for selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
        enabled: True to set as master, False to unset
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        panel._append_out("No brain selected")
        return
    
    action = "Setting" if enabled else "Unsetting"
    panel._append_out(f"{action} master for '{name}'...")
    
    def run_set_master():
        try:
            flag = "--enabled" if enabled else "--disabled"
            out = panel._run_cli(["brains", "set-master", name, flag, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error setting master: {e}"))
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_set_master, daemon=True).start()


def set_parent_brain(panel: Any) -> None:
    """Set parent brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    parent = (panel.parent_var.get() or "").strip()
    if not name or not parent:
        return
    
    panel._append_out(f"Setting parent of '{name}' to '{parent}'...")
    
    def run_set_parent():
        try:
            out = panel._run_cli(["brains", "set-parent", name, parent, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error setting parent: {e}"))
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_set_parent, daemon=True).start()


def clear_parent_brain(panel: Any) -> None:
    """Clear parent brain relationship with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    import threading
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    panel._append_out(f"Clearing parent of '{name}'...")
    
    def run_clear_parent():
        try:
            out = panel._run_cli(["brains", "set-parent", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
        except Exception as e:
            panel.after(0, lambda: panel._append_out(f"Error clearing parent: {e}"))
        finally:
            # Refresh in background
            panel.after(0, panel.refresh)
    
    threading.Thread(target=run_clear_parent, daemon=True).start()
