"""Brain operation handlers (pin, delete, rename, master, parent, details).

All brain action methods extracted as functions for maintainability.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, cast

from ...services.brain_registry_service import get_brain_stats, invalidate_brain_cache

try:  # pragma: no cover
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    messagebox = cast(Any, None)

logger = logging.getLogger(__name__)


def _submit_brain_task(panel: Any, work: Callable[[], None], name: str) -> None:
    """Submit a brain operation to the shared worker pool or a fallback thread."""
    pool = getattr(panel, "_worker_pool", None)
    if pool is not None:
        try:
            pool.submit(work)
            logger.debug("Submitted brain task '%s' to worker pool", name)
            return
        except Exception as exc:
            logger.debug("Worker pool submission failed for '%s': %s", name, exc)

    threading.Thread(target=work, daemon=True, name=name).start()


def pin_brain(panel: Any) -> None:
    """Pin selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        panel._append_out("No brain selected")
        return
    
    logger.info(f"User action: Pinning brain '{name}'")
    
    def run_pin():
        try:
            logger.debug(f"Executing pin command for brain '{name}' in store: {panel._store_dir}")
            out = panel._run_cli(["brains", "pin", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully pinned brain '{name}'")
        except Exception as e:
            error_context = f"Failed to pin brain '{name}'"
            
            # Add contextual suggestions based on error type
            if "permission" in str(e).lower() or "access" in str(e).lower():
                suggestion = "Check file permissions in the brains store directory"
            elif "not found" in str(e).lower() or "does not exist" in str(e).lower():
                suggestion = "Verify the brain exists and the store directory is correct"
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                suggestion = "Check available disk space"
            else:
                suggestion = "Try refreshing the brains list and ensuring the brain exists"
            
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error pinning brain: {e}\nSuggestion: {suggestion}"))
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_pin, name="BrainPin")
    panel._append_out(f"Pinning '{name}'...")


def unpin_brain(panel: Any) -> None:
    """Unpin selected brain with async refresh (checks for master status first).
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    # If selected is a master, block unpin at UI level as well
    try:
        is_master = get_selected_tree_value(panel.tree, 4) == "yes"
        if is_master:
            logger.warning(f"User attempted to unpin master brain '{name}' - operation blocked")
            panel._append_out('{"ok": false, "error": "cannot unpin master brain"}')
            return
    except Exception:
        pass
    
    logger.info(f"User action: Unpinning brain '{name}'")
    
    def run_unpin():
        try:
            logger.debug(f"Executing unpin command for brain '{name}'")
            out = panel._run_cli(["brains", "unpin", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully unpinned brain '{name}'")
        except Exception as e:
            error_context = f"Failed to unpin brain '{name}'"
            
            # Add contextual suggestions
            if "master" in str(e).lower():
                suggestion = "Cannot unpin a master brain. Unset master status first using 'Set Master' button"
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                suggestion = "Check file permissions in the brains store directory"
            elif "not found" in str(e).lower():
                suggestion = "Brain may have been deleted. Try refreshing the brains list"
            else:
                suggestion = "Verify the brain is currently pinned and try again"
            
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error unpinning brain: {e}\nSuggestion: {suggestion}"))
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_unpin, name="BrainUnpin")
    panel._append_out(f"Unpinning '{name}'...")


def delete_brain(panel: Any) -> None:
    """Delete selected brain with confirmation and async refresh (blocks master brains).
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value, parse_cli_dict
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    # Prevent deleting master brains at UI level
    try:
        is_master = get_selected_tree_value(panel.tree, 4) == "yes"
        if is_master:
            logger.warning(f"User attempted to delete master brain '{name}' - operation blocked")
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
            logger.info(f"User cancelled deletion of brain '{name}'")
            return
    
    logger.info(f"User action: Deleting brain '{name}'")
    panel._append_out(f"Deleting '{name}'...")
    
    def run_delete():
        try:
            logger.debug(f"Executing delete command for brain '{name}'")
            out = panel._run_cli(["brains", "delete", name, "--store-dir", panel._store_dir])
            
            # Parse and show result on main thread
            def show_result():
                panel._append_out(out)
                try:
                    res = parse_cli_dict(out)
                    if bool(res.get("ok")):
                        logger.info(f"Successfully deleted brain '{name}'")
                        if messagebox is not None:
                            messagebox.showinfo("Delete Brain", f"Deleted '{name}'.")
                    else:
                        error_msg = res.get("error", "Unknown error")
                        logger.error(f"Failed to delete brain '{name}': {error_msg}")
                        if messagebox is not None:
                            messagebox.showerror("Delete Brain", f"Failed to delete '{name}': {error_msg}")
                        panel._append_out(f"[delete] Error: {error_msg}")
                except Exception as e:
                    logger.error(f"Failed to parse delete response for brain '{name}': {e}", exc_info=True)
                    if messagebox is not None:
                        messagebox.showerror("Delete Brain", f"Failed to parse response: {e}")
                    panel._append_out(f"[delete] Parse error: {e}")
            
            panel.after(0, show_result)
            
        except Exception as e:
            error_context = f"Exception during deletion of brain '{name}'"
            
            # Provide helpful context based on error type
            if "permission" in str(e).lower() or "access" in str(e).lower():
                suggestion = "Check file permissions. Brain files may be locked by another process"
            elif "in use" in str(e).lower() or "locked" in str(e).lower():
                suggestion = "Brain may be loaded in chat. Unload it first, then try deleting"
            elif "disk" in str(e).lower() or "i/o" in str(e).lower():
                suggestion = "Disk I/O error. Check disk health and available space"
            else:
                suggestion = "Ensure the brain is not loaded in chat and try again"
            
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            def show_error():
                if messagebox is not None:
                    messagebox.showerror("Delete Brain", f"Error during deletion: {e}\n\nSuggestion: {suggestion}")
                panel._append_out(f"[delete] Exception: {e}\nSuggestion: {suggestion}")
            
            panel.after(0, show_error)
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_delete, name="BrainDelete")


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
    if getattr(panel, "_brain_details_loading", False):
        panel._append_out("Brain details are already loading. Please wait...")
        return

    setattr(panel, "_brain_details_loading", True)
    if hasattr(panel, "status_var"):
        try:
            panel.status_var.set("Loading details...")
        except Exception:
            pass

    try:
        panel._append_out(f"[details] Loading '{name}'...")
    except Exception:
        pass

    def _finish_status() -> None:
        setattr(panel, "_brain_details_loading", False)
        if hasattr(panel, "status_var"):
            try:
                panel.status_var.set("")
            except Exception:
                pass

    def _notify_error(err: Exception, suggestion: str) -> None:
        error_context = f"Failed to load brain details for '{name}'"
        logger.error(f"{error_context}: {err}. Suggestion: {suggestion}", exc_info=True)
        if messagebox is not None:
            messagebox.showerror("Error", f"{error_context}: {err}\n\nSuggestion: {suggestion}")
        panel._append_out(f"[details] Error: {err}\nSuggestion: {suggestion}")

    def _worker() -> None:
        stats_data: dict[str, Any] = {}
        cached = getattr(panel, "_brain_stats_cache", None)
        if isinstance(cached, dict) and cached:
            stats_data = cached
        else:
            try:
                stats_data = get_brain_stats(panel._store_dir)
            except Exception as stats_err:
                logger.debug("Fast stats read failed, falling back to CLI: %s", stats_err, exc_info=True)
                stats_data = {}

        try:
            details_text = build_brain_details_text(
                brain_name=name,
                store_dir=panel._store_dir,
                run_cli=panel._run_cli,
                parse_cli_dict=parse_cli_dict,
                goals_list_callback=panel._on_goals_list,
                brain_stats_data=stats_data
            )
        except Exception as build_err:
            def _handle_build_error() -> None:
                _finish_status()
                err_str = str(build_err).lower()
                if "not found" in err_str or "does not exist" in err_str:
                    suggestion = "Brain may have been deleted. Try refreshing the brains list"
                elif "permission" in err_str:
                    suggestion = "Check file permissions for the brain's metadata files"
                elif "corrupt" in err_str or "invalid" in err_str:
                    suggestion = "Brain metadata may be corrupted. Consider re-creating the brain"
                else:
                    suggestion = "Try refreshing the brains list and selecting the brain again"
                _notify_error(build_err, suggestion)

            try:
                panel.after(0, _handle_build_error)
            except Exception:
                _handle_build_error()
            return

        def _show_dialog() -> None:
            _finish_status()
            show_brain_details_dialog(panel, name, details_text)

        try:
            panel.after(0, _show_dialog)
        except Exception:
            _show_dialog()

    _submit_brain_task(panel, _worker, name="BrainDetails")


def rename_brain(panel: Any) -> None:
    """Rename selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    new = (panel.rename_var.get() or "").strip()
    if not name or not new:
        return
    
    logger.info(f"User action: Renaming brain '{name}' to '{new}'")
    panel._append_out(f"Renaming '{name}' to '{new}'...")
    
    def run_rename():
        try:
            logger.debug(f"Executing rename command: '{name}' -> '{new}'")
            out = panel._run_cli(["brains", "rename", name, new, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully renamed brain '{name}' to '{new}'")
        except Exception as e:
            error_context = f"Failed to rename brain '{name}' to '{new}'"
            
            # Provide contextual suggestions
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                suggestion = f"A brain named '{new}' already exists. Choose a different name"
            elif "invalid" in str(e).lower() or "illegal" in str(e).lower():
                suggestion = "Brain name contains invalid characters. Use alphanumeric characters, hyphens, and underscores"
            elif "permission" in str(e).lower():
                suggestion = "Check file permissions in the brains store directory"
            elif "not found" in str(e).lower():
                suggestion = "Source brain no longer exists. Try refreshing the brains list"
            else:
                suggestion = "Ensure the new name is unique and contains only valid characters"
            
            logger.error(f"{error_context}: {e}. Suggestion: {suggestion}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error renaming: {e}\nSuggestion: {suggestion}"))
        finally:
            # Refresh in background and clear rename field
            panel.after(0, lambda: panel.rename_var.set(""))
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_rename, name="BrainRename")


def set_master_status(panel: Any, enabled: bool) -> None:
    """Set or unset master status for selected brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
        enabled: True to set as master, False to unset
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        panel._append_out("No brain selected")
        return
    
    action = "Setting" if enabled else "Unsetting"
    logger.info(f"User action: {action} master status for brain '{name}'")
    panel._append_out(f"{action} master for '{name}'...")
    
    def run_set_master():
        try:
            flag = "--enabled" if enabled else "--disabled"
            logger.debug(f"Executing set-master command for '{name}' with flag {flag}")
            out = panel._run_cli(["brains", "set-master", name, flag, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully {action.lower()} master status for brain '{name}'")
        except Exception as e:
            logger.error(f"Failed to {action.lower()} master for brain '{name}': {e}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error setting master: {e}"))
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_set_master, name="BrainSetMaster")


def set_parent_brain(panel: Any) -> None:
    """Set parent brain with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    parent = (panel.parent_var.get() or "").strip()
    if not name or not parent:
        return
    
    logger.info(f"User action: Setting parent of brain '{name}' to '{parent}'")
    panel._append_out(f"Setting parent of '{name}' to '{parent}'...")
    
    def run_set_parent():
        try:
            logger.debug(f"Executing set-parent command for '{name}' -> '{parent}'")
            out = panel._run_cli(["brains", "set-parent", name, parent, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully set parent of '{name}' to '{parent}'")
        except Exception as e:
            logger.error(f"Failed to set parent for brain '{name}': {e}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error setting parent: {e}"))
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_set_parent, name="BrainSetParent")


def clear_parent_brain(panel: Any) -> None:
    """Clear parent brain relationship with async refresh.
    
    Args:
        panel: BrainsPanel instance
    """
    from .helpers import get_selected_tree_value
    
    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        return
    
    logger.info(f"User action: Clearing parent of brain '{name}'")
    panel._append_out(f"Clearing parent of '{name}'...")
    
    def run_clear_parent():
        try:
            logger.debug(f"Executing clear-parent command for '{name}'")
            out = panel._run_cli(["brains", "set-parent", name, "--store-dir", panel._store_dir])
            panel.after(0, lambda: panel._append_out(out))
            logger.info(f"Successfully cleared parent of brain '{name}'")
        except Exception as e:
            logger.error(f"Failed to clear parent for brain '{name}': {e}", exc_info=True)
            panel.after(0, lambda: panel._append_out(f"Error clearing parent: {e}"))
        finally:
            # Refresh in background
            invalidate_brain_cache(panel._store_dir)
            panel.after(0, panel.refresh)
    
    _submit_brain_task(panel, run_clear_parent, name="BrainClearParent")
