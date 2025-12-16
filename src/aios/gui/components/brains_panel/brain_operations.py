"""Brain operation handlers (pin, delete, rename, master, parent, details).

All brain action methods extracted as functions for maintainability.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime, timezone
from typing import Any, Callable, Optional, cast

from ...services.brain_registry_service import get_brain_stats, invalidate_brain_cache


def _update_transfer_status(panel: Any, text: str) -> None:
    """Schedule a transfer progress update on the Tk event loop."""
    if not hasattr(panel, "transfer_progress_var"):
        return

    previous = getattr(panel, "_last_transfer_status", None)
    if text == previous:
        return
    panel._last_transfer_status = text
    logger.debug("Transfer status -> %s", text)

    def _apply() -> None:
        try:
            panel.transfer_progress_var.set(text)
        except Exception:
            pass

    try:
        panel.after(0, _apply)
    except Exception:
        _apply()


def _clear_transfer_status(panel: Any, delay_ms: int = 2000) -> None:
    """Clear the transfer progress indicator after ``delay_ms`` milliseconds."""
    if not hasattr(panel, "transfer_progress_var"):
        return

    logger.debug("Scheduling transfer status clear in %sms", delay_ms)

    def _clear() -> None:
        try:
            panel.transfer_progress_var.set("")
        except Exception:
            pass
        finally:
            if hasattr(panel, "_last_transfer_status"):
                panel._last_transfer_status = ""
            logger.debug("Transfer status cleared")


def _collect_existing_brain_names(base_dir: str) -> set[str]:
    names: set[str] = set()
    if not os.path.isdir(base_dir):
        return names

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if os.path.isfile(entry_path):
            stem, ext = os.path.splitext(entry)
            if ext.lower() in {".json", ".npz", ".bin", ".pt", ".safetensors"} and stem not in {"pinned", "masters"}:
                names.add(stem)

    actv_base = os.path.join(base_dir, "actv1")
    if os.path.isdir(actv_base):
        for entry in os.listdir(actv_base):
            if os.path.isdir(os.path.join(actv_base, entry)):
                names.add(entry)

    return names


def _split_numeric_suffix(name: str) -> tuple[str, Optional[int]]:
    if "_" in name:
        base, suffix = name.rsplit("_", 1)
        if suffix.isdigit():
            return base, int(suffix)
    return name, None


def _ensure_unique_brain_name(base_dir: str, desired: str) -> str:
    existing = _collect_existing_brain_names(base_dir)
    if desired not in existing and not _brain_exists_in_store(base_dir, desired):
        return desired

    base, suffix = _split_numeric_suffix(desired)
    existing.add(desired)
    max_suffix = suffix if suffix is not None else 0
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")

    for name in existing:
        match = pattern.match(name)
        if match:
            max_suffix = max(max_suffix, int(match.group(1)))
        elif name == base and suffix is None:
            max_suffix = max(max_suffix, 0)

    next_idx = max_suffix + 1 if max_suffix is not None else 1
    candidate = f"{base}_{next_idx}"
    while candidate in existing:
        next_idx += 1
        candidate = f"{base}_{next_idx}"
    return candidate


def _brain_exists_in_store(base_dir: str, name: str) -> bool:
    if not os.path.isdir(base_dir):
        return False
    actv_dir = os.path.join(base_dir, "actv1", name)
    if os.path.isdir(actv_dir):
        return True
    for path in glob.glob(os.path.join(base_dir, f"{name}.*")):
        if os.path.exists(path):
            return True
    return False


def _remap_relative_path(rel: str, old: str, new: str) -> str:
    if old == new:
        return rel

    parts = rel.split("/")
    prefix = old
    for idx, part in enumerate(parts):
        if part == prefix:
            parts[idx] = new
            continue
        if part.startswith(prefix):
            remainder = part[len(prefix):]
            if not remainder or remainder[0] in {".", "_", "-", " ", "("} or remainder[0].isdigit():
                parts[idx] = f"{new}{remainder}"
    return "/".join(parts)


def _replace_name_in_json(data: Any, old: str, new: str) -> bool:
    changed = False
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if isinstance(value, str) and value == old:
                data[key] = new
                changed = True
            else:
                if _replace_name_in_json(value, old, new):
                    changed = True
    elif isinstance(data, list):
        for idx, item in enumerate(list(data)):
            if isinstance(item, str) and item == old:
                data[idx] = new
                changed = True
            else:
                if _replace_name_in_json(item, old, new):
                    changed = True
    return changed


def _rewrite_json_name_fields(path: str, old: str, new: str) -> None:
    if old == new or not path.lower().endswith(".json"):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    if _replace_name_in_json(data, old, new):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug("Updated embedded name references in %s", path)
        except Exception:
            logger.debug("Failed to rewrite name references in %s", path, exc_info=True)

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
                            # Show success message
                            warning = res.get("warning")
                            if warning:
                                messagebox.showinfo("Delete Brain", f"Deleted '{name}' with warnings:\n{warning}")
                            else:
                                messagebox.showinfo("Delete Brain", f"Deleted '{name}'.")
                    else:
                        error_msg = res.get("error", "Unknown error")
                        logger.error(f"Failed to delete brain '{name}': {error_msg}")
                        if messagebox is not None:
                            messagebox.showerror("Delete Brain", f"Failed to delete '{name}': {error_msg}")
                        panel._append_out(f"[delete] Error: {error_msg}")
                except Exception as e:
                    # Parse error - but check if deletion actually succeeded by checking if brain files are gone
                    logger.warning(f"Failed to parse delete response for brain '{name}': {e}, but deletion may have succeeded", exc_info=True)
                    # Don't show error messagebox - just log the parse error
                    # The brain refresh will show if it's actually gone or not
                    panel._append_out(f"[delete] Response parse error (brain may still be deleted): {e}")
            
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


def export_brain(panel: Any) -> None:
    """Export the selected brain bundle to a portable ZIP archive."""
    from .helpers import get_selected_tree_value

    try:  # pragma: no cover - Tk availability varies per platform
        from tkinter import filedialog, messagebox  # type: ignore
    except Exception:  # pragma: no cover
        filedialog = cast(Any, None)
        messagebox = cast(Any, None)

    name = get_selected_tree_value(panel.tree, 0)
    if not name:
        if messagebox is not None:
            messagebox.showwarning("Export Brain", "Select a brain to export first.")
        panel._append_out("[brains] Select a brain before exporting")
        return

    if filedialog is None:
        panel._append_out("[brains] Export requires a graphical file dialog")
        return

    store_dir = getattr(panel, "_store_dir", None)
    if not store_dir:
        if messagebox is not None:
            messagebox.showerror("Export Brain", "Brains store directory is not configured.")
        panel._append_out("[brains] Store directory not configured")
        return

    base_dir = os.path.abspath(store_dir)
    if not os.path.isdir(base_dir):
        if messagebox is not None:
            messagebox.showerror("Export Brain", f"Store directory not found: {base_dir}")
        panel._append_out(f"[brains] Store directory not found: {base_dir}")
        return

    files: list[str] = []
    seen: set[str] = set()

    def _add_file(path: str) -> None:
        if os.path.isfile(path):
            rel = os.path.relpath(path, base_dir)
            rel_norm = rel.replace("\\", "/")
            if rel_norm not in seen:
                seen.add(rel_norm)
                files.append(rel_norm)

    # Include any top-level files matching the brain name
    for path in glob.glob(os.path.join(base_dir, f"{name}.*")):
        _add_file(path)
    candidate_no_ext = os.path.join(base_dir, name)
    _add_file(candidate_no_ext)

    # Include bundled directories (ACTv1 and others)
    bundle_dirs = [
        os.path.join(base_dir, "actv1", name),
        os.path.join(base_dir, name),
    ]
    for bundle in bundle_dirs:
        if os.path.isdir(bundle):
            for root, _dirs, filenames in os.walk(bundle):
                for filename in filenames:
                    _add_file(os.path.join(root, filename))

    if not files:
        if messagebox is not None:
            messagebox.showerror("Export Brain", f"No files found for brain '{name}'.")
        panel._append_out(f"[brains] Export aborted: no files found for '{name}'")
        return

    dest_path = filedialog.asksaveasfilename(
        title="Export Brain Archive",
        initialfile=f"{name}.zip",
        defaultextension=".zip",
        filetypes=[("Brain archives", "*.zip"), ("All files", "*.*")],
    )
    if not dest_path:
        return

    dest_path = os.path.abspath(dest_path)
    if not dest_path.lower().endswith(".zip"):
        dest_path += ".zip"

    dest_dir = os.path.dirname(dest_path)
    if dest_dir and not os.path.isdir(dest_dir):
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as exc:  # pragma: no cover - OS dependent
            if messagebox is not None:
                messagebox.showerror("Export Brain", f"Cannot create destination folder: {exc}")
            panel._append_out(f"[brains] Export aborted: {exc}")
            return

    logger.info("Exporting brain '%s' to %s", name, dest_path)
    files_to_package = sorted(files)

    _update_transfer_status(panel, "Export 0%")

    def _run_export() -> None:
        try:
            manifest = {
                "format_version": 1,
                "name": name,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "paths": files_to_package,
                "compression": "zip",
            }
            # Use ZIP for compatibility across platforms.
            zip_kwargs: dict[str, Any] = {"mode": "w", "compression": zipfile.ZIP_DEFLATED}
            if hasattr(zipfile.ZipFile, "compresslevel"):
                zip_kwargs["compresslevel"] = 9
            total_bytes = 0
            for rel in files_to_package:
                src = os.path.join(base_dir, rel)
                if os.path.isfile(src):
                    total_bytes += os.path.getsize(src)
            total_bytes = max(total_bytes, 1)
            processed_bytes = 0
            last_pct = -1

            with zipfile.ZipFile(dest_path, **zip_kwargs) as zf:
                for rel in files_to_package:
                    src = os.path.join(base_dir, rel)
                    if not os.path.isfile(src):
                        raise FileNotFoundError(f"Missing file during export: {rel}")
                    zf.write(src, arcname=rel)
                    try:
                        size = os.path.getsize(src)
                    except OSError:
                        size = 0
                    processed_bytes += max(size, 0)
                    pct = int((processed_bytes * 100) / total_bytes)
                    if pct > last_pct:
                        last_pct = pct
                        _update_transfer_status(panel, f"Export {pct}%")
                zf.writestr(
                    "brain_manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                )
            _update_transfer_status(panel, "Export 100%")

            def _on_done() -> None:
                panel._append_out(f"[brains] Exported '{name}' to {dest_path}")
                if messagebox is not None:
                    messagebox.showinfo("Export Brain", f"Exported '{name}' to:\n{dest_path}")
                _clear_transfer_status(panel)

            panel.after(0, _on_done)
        except Exception as exc:  # pragma: no cover - error path
            logger.error("Failed to export brain '%s': %s", name, exc, exc_info=True)
            _update_transfer_status(panel, "Export failed")
            _clear_transfer_status(panel, delay_ms=4000)

            def _on_error() -> None:
                if messagebox is not None:
                    messagebox.showerror("Export Brain", f"Failed to export '{name}': {exc}")
                panel._append_out(f"[brains] Export failed for '{name}': {exc}")

            try:
                panel.after(0, _on_error)
            except Exception:
                _on_error()

    _submit_brain_task(panel, _run_export, name="BrainExport")
    panel._append_out(f"[brains] Exporting '{name}'...")


def import_brain(panel: Any) -> None:
    """Import a previously exported brain archive."""
    try:  # pragma: no cover - Tk availability varies per platform
        from tkinter import filedialog, messagebox  # type: ignore
    except Exception:  # pragma: no cover
        filedialog = cast(Any, None)
        messagebox = cast(Any, None)

    if filedialog is None:
        panel._append_out("[brains] Import requires a graphical file dialog")
        return

    src_path = filedialog.askopenfilename(
        title="Import Brain Archive",
        filetypes=[("Brain archives", "*.zip"), ("All files", "*.*")],
    )
    if not src_path:
        return

    src_path = os.path.abspath(src_path)
    logger.info("Importing brain archive from %s", src_path)

    try:
        with zipfile.ZipFile(src_path, "r") as zf:
            try:
                with zf.open("brain_manifest.json") as manifest_file:
                    manifest = json.load(manifest_file)
            except KeyError as exc:
                raise ValueError("Archive missing brain_manifest.json") from exc
    except (zipfile.BadZipFile, ValueError, json.JSONDecodeError) as exc:
        if messagebox is not None:
            messagebox.showerror("Import Brain", f"Invalid archive: {exc}")
        panel._append_out(f"[brains] Import aborted: invalid archive ({exc})")
        return

    exported_name = str(manifest.get("name") or "").strip()
    if not exported_name:
        if messagebox is not None:
            messagebox.showerror("Import Brain", "Archive does not declare a brain name.")
        panel._append_out("[brains] Import aborted: missing brain name in manifest")
        return

    raw_paths = manifest.get("paths") or []
    if not isinstance(raw_paths, list) or not raw_paths:
        if messagebox is not None:
            messagebox.showerror("Import Brain", "Archive manifest does not include any files.")
        panel._append_out("[brains] Import aborted: archive has no files")
        return

    store_dir = getattr(panel, "_store_dir", None)
    if not store_dir:
        if messagebox is not None:
            messagebox.showerror("Import Brain", "Brains store directory is not configured.")
        panel._append_out("[brains] Store directory not configured")
        return

    base_dir = os.path.abspath(store_dir)
    os.makedirs(base_dir, exist_ok=True)

    sanitized_paths: list[str] = []
    for rel in raw_paths:
        if not isinstance(rel, str):
            continue
        norm = os.path.normpath(rel).replace("\\", "/")
        if norm.startswith("../") or norm.startswith("..\\") or norm.startswith("..") or os.path.isabs(norm):
            if messagebox is not None:
                messagebox.showerror("Import Brain", f"Archive contains unsafe path: {rel}")
            panel._append_out(f"[brains] Import aborted: unsafe path {rel}")
            return
        if norm and norm != ".":
            sanitized_paths.append(norm)

    if not sanitized_paths:
        if messagebox is not None:
            messagebox.showerror("Import Brain", "Archive has no transferable files.")
        panel._append_out("[brains] Import aborted: nothing to import")
        return

    target_name = _ensure_unique_brain_name(base_dir, exported_name)
    if _brain_exists_in_store(base_dir, exported_name) and target_name != exported_name:
        logger.info(
            "Name collision detected for imported brain '%s'; will import as '%s'",
            exported_name,
            target_name,
        )
        panel._append_out(
            f"[brains] '{exported_name}' exists. Importing as '{target_name}'"
        )
    else:
        logger.info("Importing brain archive '%s' into store %s", exported_name, base_dir)

    path_remap = {rel: _remap_relative_path(rel, exported_name, target_name) for rel in sanitized_paths}

    remapped_values = list(path_remap.values())
    if len(remapped_values) != len(set(remapped_values)):
        if messagebox is not None:
            messagebox.showerror(
                "Import Brain",
                "Archive contains duplicate files after renaming. Import aborted.",
            )
        panel._append_out("[brains] Import aborted: duplicate paths after renaming")
        return

    panel._append_out(f"[brains] Importing '{target_name}'...")
    _update_transfer_status(panel, "Import 0%")

    def _run_import() -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="aios_brain_import_") as tmpdir:
                with zipfile.ZipFile(src_path, "r") as zf:
                    zf.extractall(tmpdir)

                manifest_path = os.path.join(tmpdir, "brain_manifest.json")
                if os.path.exists(manifest_path):
                    os.remove(manifest_path)

                total_bytes = 0
                for rel in sanitized_paths:
                    src = os.path.join(tmpdir, rel)
                    if os.path.exists(src) and os.path.isfile(src):
                        total_bytes += os.path.getsize(src)
                total_bytes = max(total_bytes, 1)
                processed_bytes = 0
                last_pct = -1

                for rel in sanitized_paths:
                    src = os.path.join(tmpdir, rel)
                    if not os.path.exists(src):
                        raise FileNotFoundError(f"Archive missing expected file: {rel}")
                    dst_rel = path_remap[rel]
                    dst = os.path.join(base_dir, dst_rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            raise FileExistsError(f"Destination already exists: {dst_rel}")
                        shutil.copytree(src, dst)
                        for root, _dirs, filenames in os.walk(dst):
                            for filename in filenames:
                                _rewrite_json_name_fields(
                                    os.path.join(root, filename),
                                    exported_name,
                                    target_name,
                                )
                    else:
                        if os.path.exists(dst):
                            raise FileExistsError(f"Destination already exists: {dst_rel}")
                        shutil.copy2(src, dst)
                        _rewrite_json_name_fields(dst, exported_name, target_name)
                        try:
                            size = os.path.getsize(src)
                        except OSError:
                            size = 0
                        processed_bytes += max(size, 0)
                        pct = int((processed_bytes * 100) / total_bytes)
                        if pct > last_pct:
                            last_pct = pct
                            _update_transfer_status(panel, f"Import {pct}%")

            invalidate_brain_cache(panel._store_dir)
            _update_transfer_status(panel, "Import 100%")

            def _on_done() -> None:
                panel.refresh(force=True)
                if target_name != exported_name:
                    msg = f"[brains] Imported '{target_name}' (renamed from '{exported_name}')"
                    mb_text = f"Imported '{target_name}' (renamed from '{exported_name}')."
                else:
                    msg = f"[brains] Imported '{target_name}' successfully"
                    mb_text = f"Imported '{target_name}'."
                panel._append_out(msg)
                if messagebox is not None:
                    messagebox.showinfo("Import Brain", mb_text)
                _clear_transfer_status(panel)

            panel.after(0, _on_done)
        except Exception as exc:  # pragma: no cover - error path
            logger.error("Failed to import brain from %s: %s", src_path, exc, exc_info=True)
            _update_transfer_status(panel, "Import failed")
            _clear_transfer_status(panel, delay_ms=4000)

            def _on_error() -> None:
                if messagebox is not None:
                    messagebox.showerror("Import Brain", f"Failed to import '{exported_name}': {exc}")
                panel._append_out(f"[brains] Import failed: {exc}")

            try:
                panel.after(0, _on_error)
            except Exception:
                _on_error()

    _submit_brain_task(panel, _run_import, name="BrainImport")
