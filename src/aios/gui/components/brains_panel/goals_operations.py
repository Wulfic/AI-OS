"""Goals management operations (add, remove, refresh).

All goals-related action methods with debouncing logic.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    messagebox = cast(Any, None)

logger = logging.getLogger(__name__)


def _set_goals_controls_enabled(panel: Any, enabled: bool) -> None:
    """Enable or disable goal management controls."""
    state = "normal" if enabled else "disabled"
    for attr in ("goal_add_button", "goal_remove_button"):
        widget = getattr(panel, attr, None)
        if widget is not None:
            try:
                widget.config(state=state)
            except Exception:
                pass
    entry = getattr(panel, "goal_text_entry", None)
    if entry is not None:
        try:
            entry.config(state=state)
        except Exception:
            pass
    if hasattr(panel, "goals_list") and panel.goals_list is not None:
        try:
            panel.goals_list.config(state=state)
        except Exception:
            pass


def _enter_goals_busy(panel: Any) -> None:
    count = getattr(panel, "_goals_busy_count", 0) + 1
    panel._goals_busy_count = count
    if count == 1:
        _set_goals_controls_enabled(panel, False)


def _exit_goals_busy(panel: Any) -> None:
    count = max(0, getattr(panel, "_goals_busy_count", 1) - 1)
    panel._goals_busy_count = count
    if count == 0:
        _set_goals_controls_enabled(panel, True)


def refresh_goals(panel: Any) -> None:
    """Refresh goals list for the currently selected brain or expert.
    
    Includes debouncing to prevent excessive CLI calls during rapid selection changes.
    Implements caching to avoid redundant data fetches.
    
    Args:
        panel: BrainsPanel instance
    """
    from .expert_operations import get_selected_expert_id
    from .helpers import get_selected_tree_value
    
    if not hasattr(panel, 'goals_list') or panel._on_goals_list is None:
        return
    
    # Initialize tracking variables
    if not hasattr(panel, '_last_goals_refresh_time'):
        panel._last_goals_refresh_time = 0.0
    if not hasattr(panel, '_goals_cache'):
        panel._goals_cache = {}  # {name: (goals_list, timestamp)}
    if not hasattr(panel, '_cache_ttl'):
        panel._cache_ttl = 30.0  # Cache for 30 seconds
    
    # Debounce: prevent excessive refreshes within 500ms (increased from 200ms)
    current_time = time.time()
    if current_time - panel._last_goals_refresh_time < 0.5:
        # Schedule a delayed refresh instead
        if hasattr(panel, '_goals_refresh_timer'):
            try:
                panel.after_cancel(panel._goals_refresh_timer)
            except Exception:
                pass
        panel._goals_refresh_timer = panel.after(500, lambda: _do_refresh_goals(panel))
        return
    
    _do_refresh_goals(panel)


def _do_refresh_goals(panel: Any) -> None:
    """Internal function to actually refresh goals (called by refresh_goals after debounce).
    
    Uses caching to avoid redundant CLI calls. Cache is invalidated on add/remove operations.
    Uses threading for non-cached fetches to avoid UI freezing.
    
    Args:
        panel: BrainsPanel instance
    """
    from .expert_operations import get_selected_expert_id
    from .helpers import get_selected_tree_value
    import threading
    
    if not hasattr(panel, 'goals_list') or panel._on_goals_list is None:
        return
    
    current_time = time.time()
    panel._last_goals_refresh_time = current_time
    
    # Check which tree has selection - use cached approach to avoid redundant queries
    brain_sel = panel.tree.selection()
    expert_sel = panel.experts_tree.selection()
    
    selected_name = None
    selected_type = None
    
    # Determine selection based on which tree has active selection
    if brain_sel and not expert_sel:
        # Brain selected
        brain_name = get_selected_tree_value(panel.tree, 0)
        if brain_name:
            selected_name = brain_name
            selected_type = "brain"
    elif expert_sel and not brain_sel:
        # Expert selected
        expert_id = get_selected_expert_id(panel)
        if expert_id:
            selected_name = expert_id
            selected_type = "expert"
    elif brain_sel and expert_sel:
        # Both selected - prefer brain
        brain_name = get_selected_tree_value(panel.tree, 0)
        if brain_name:
            selected_name = brain_name
            selected_type = "brain"
    
    # Handle no selection
    if not selected_name:
        try:
            panel.goals_list.delete(0, tk.END)
            panel.goals_count_var.set("No brain or expert selected")
        except Exception:
            pass
        return
    
    # Type assertion for selected_type (we know it's not None here)
    type_str = str(selected_type or "brain")
    
    try:
        # Check cache first
        cache_key = f"{selected_type}:{selected_name}"
        cached = panel._goals_cache.get(cache_key)
        
        # Use cache if valid (within TTL)
        if cached and (current_time - cached[1]) < panel._cache_ttl:
            items = cached[0]
            _update_goals_ui(panel, items, type_str, cache_hit=True)
            return

        try:
            panel.goals_list.delete(0, tk.END)
            panel.goals_count_var.set("Loading goals…")
        except Exception:
            pass

        result = None
        try:
            result = panel._on_goals_list(selected_name)
        except Exception as exc:
            panel._append_out(f"[goals] Error queueing goals refresh: {exc}")
            return

        if isinstance(result, list):
            try:
                fetch_time = time.time()
                panel._goals_cache[cache_key] = (result, fetch_time)
            except Exception:
                pass
            _update_goals_ui(panel, result, type_str, cache_hit=False)
            return

        def _finalize(items: list[Any], *, refresh_time: float) -> None:
            try:
                panel._goals_cache[cache_key] = (items, refresh_time)
                if len(panel._goals_cache) > 50:
                    sorted_cache = sorted(panel._goals_cache.items(), key=lambda x: x[1][1])
                    panel._goals_cache = dict(sorted_cache[-50:])
            except Exception:
                pass
            _update_goals_ui(panel, items, type_str, cache_hit=False)

        def _handle_future(fut_result: list[Any]) -> None:
            _finalize(fut_result, refresh_time=time.time())

        def _handle_error(exc: Exception) -> None:
            panel._append_out(f"[goals] Error fetching goals: {exc}")
            try:
                panel.goals_count_var.set("Goal load failed")
            except Exception:
                pass

        if hasattr(result, "add_done_callback"):
            def _on_complete(fut):
                try:
                    items = fut.result()  # type: ignore[assignment]
                    if not isinstance(items, list):
                        items = []
                except Exception as exc:  # pragma: no cover - defensive
                    panel.after(0, lambda: _handle_error(exc))
                    return
                panel.after(0, lambda: _handle_future(items))

            result.add_done_callback(_on_complete)  # type: ignore[call-arg]
        else:
            import threading

            def _worker() -> None:
                try:
                    items = list(result or [])  # type: ignore[arg-type]
                except Exception as exc:
                    panel.after(0, lambda: _handle_error(exc))
                    return
                panel.after(0, lambda: _handle_future(items))

            threading.Thread(target=_worker, daemon=True).start()
            
    except Exception as e:
        panel._append_out(f"[goals] Error refreshing goals: {e}")


def _update_goals_ui(panel: Any, items: list, selected_type: str, cache_hit: bool = False) -> None:
    """Update the goals list UI with fetched items.
    
    Args:
        panel: BrainsPanel instance
        items: List of goal items to display
        selected_type: "brain" or "expert"
        cache_hit: Whether this was served from cache
    """
    try:
        panel.goals_list.delete(0, tk.END)
        for item in items:
            panel.goals_list.insert(tk.END, str(item))
        
        # Update count with type indicator and cache hint
        count = len(items)
        type_label = "Brain" if selected_type == "brain" else "Expert"
        cache_hint = " [cached]" if cache_hit else ""
        panel.goals_count_var.set(f"{count} goal{'s' if count != 1 else ''} ({type_label}){cache_hint}")
    except Exception as e:
        panel._append_out(f"[goals] Error updating goals UI: {e}")


def add_goal(panel: Any) -> None:
    """Add a new goal for the currently selected brain or expert.
    
    Includes debouncing to prevent duplicate additions.
    
    Args:
        panel: BrainsPanel instance
    """
    from .expert_operations import get_selected_expert_id
    from .helpers import get_selected_tree_value
    
    if panel._on_goal_add is None:
        panel._append_out("[goals] Goal add callback not available")
        return
    
    # Check which tree has selection
    brain_name = get_selected_tree_value(panel.tree, 0)
    expert_id = get_selected_expert_id(panel)
    
    selected_name = None
    selected_type = None
    
    if brain_name and not expert_id:
        selected_name = brain_name
        selected_type = "brain"
    elif expert_id and not brain_name:
        selected_name = expert_id
        selected_type = "expert"
    elif not brain_name and not expert_id:
        if messagebox is not None:
            messagebox.showwarning("No Selection", "Please select a brain or expert first.")
        panel._append_out("[goals] No brain or expert selected")
        return
    else:
        # Both selected? Prefer brain
        selected_name = brain_name
        selected_type = "brain"
    
    goal_text = (panel.goal_text_var.get() or "").strip()
    if not goal_text:
        panel._append_out("[goals] Goal text is empty")
        return
    
    # Debounce: prevent adding the same goal twice within 2 seconds
    current_time = time.time()
    if (current_time - panel._last_goal_add_time < 2.0 and 
        goal_text == panel._last_goal_text):
        logger.debug(f"Debounced duplicate goal add: '{goal_text}' (within 2s)")
        panel._append_out("[goals] Skipping duplicate goal add (debounced)")
        return
    
    try:
        panel._last_goal_add_time = current_time
        panel._last_goal_text = goal_text

        if selected_name is None:
            logger.warning("Cannot add goal - selected_name is None")
            panel._append_out("[goals] Selected name is None, cannot add goal")
            return

        logger.info(f"User action: Adding goal to {selected_type} '{selected_name}': {goal_text[:100]}")
        panel._append_out(f"[goals] Adding goal to {selected_type} '{selected_name}': {goal_text}")

        cache_key = f"{selected_type}:{selected_name}"
        _enter_goals_busy(panel)
        try:
            panel.goals_count_var.set("Adding goal…")
        except Exception:
            pass

        try:
            result = panel._on_goal_add(selected_name, goal_text)
        except Exception as exc:
            _exit_goals_busy(panel)
            import traceback
            logger.error(f"Failed to queue goal add for {selected_type} '{selected_name}': {exc}", exc_info=True)
            panel._append_out(f"[goals] Error adding goal: {exc}")
            panel._append_out(f"[goals] Traceback: {traceback.format_exc()}")
            return

        def _on_success() -> None:
            logger.info(f"Successfully added goal to {selected_type} '{selected_name}'")
            panel._append_out("[goals] Goal added successfully")
            try:
                panel.goal_text_var.set("")
            except Exception:
                pass
            if hasattr(panel, '_goals_cache'):
                try:
                    panel._goals_cache.pop(cache_key, None)
                except Exception:
                    pass
            refresh_goals(panel)
            _exit_goals_busy(panel)

        def _on_error(exc: Exception) -> None:
            import traceback
            logger.error(f"Failed to add goal to {selected_type} '{selected_name}': {exc}", exc_info=True)
            panel._append_out(f"[goals] Error adding goal: {exc}")
            panel._append_out(f"[goals] Traceback: {traceback.format_exc()}")
            try:
                panel.goals_count_var.set("Goal add failed")
            except Exception:
                pass
            _exit_goals_busy(panel)

        if isinstance(result, list):
            _on_success()
            return

        if hasattr(result, "add_done_callback"):
            def _callback(fut):
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    panel.after(0, lambda: _on_error(exc))
                    return
                panel.after(0, _on_success)

            result.add_done_callback(_callback)  # type: ignore[call-arg]
        else:
            # Treat return as immediate success
            _on_success()
    except Exception as e:
        import traceback
        logger.error(f"Failed to add goal to {selected_type} '{selected_name}': {e}", exc_info=True)
        panel._append_out(f"[goals] Error adding goal: {e}")
        panel._append_out(f"[goals] Traceback: {traceback.format_exc()}")
        _exit_goals_busy(panel)


def remove_selected_goals(panel: Any) -> None:
    """Remove selected goals from the currently selected brain or expert.
    
    Skips protected [primary] goals.
    
    Args:
        panel: BrainsPanel instance
    """
    from .expert_operations import get_selected_expert_id
    from .helpers import get_selected_tree_value
    
    if panel._on_goal_remove is None or not hasattr(panel, 'goals_list'):
        return
    
    # Check which tree has selection
    brain_name = get_selected_tree_value(panel.tree, 0)
    expert_id = get_selected_expert_id(panel)
    
    if not brain_name and not expert_id:
        return
    
    try:
        sel = panel.goals_list.curselection()
        if not sel:
            return

        selected_name = brain_name or expert_id
        selected_type = "brain" if brain_name else "expert"

        logger.info(f"User action: Removing {len(sel)} goal(s) from {selected_type} '{selected_name}'")

        rows = []
        skipped_primary = 0
        for idx in list(sel):
            raw = panel.goals_list.get(idx)
            if "[primary]" in str(raw):
                skipped_primary += 1
                logger.debug(f"Skipped protected [primary] goal: {raw}")
                continue
            m = re.match(r"^\s*\d+\)\s*#(\d+)\b|^\s*#(\d+)\b", str(raw).strip())
            goal_id = None
            if m:
                goal_id = m.group(1) or m.group(2)
            if goal_id is None:
                logger.warning(f"Could not extract goal ID from: {raw}")
                continue
            rows.append(int(goal_id))

        if not rows:
            if skipped_primary and messagebox is not None:
                messagebox.showinfo("Protected Goals", f"Skipped {skipped_primary} protected [primary] goal(s).")
            return

        _enter_goals_busy(panel)
        try:
            panel.goals_count_var.set("Removing goals…")
        except Exception:
            pass

        removed = 0
        failures: list[tuple[int, Exception]] = []
        pending = len(rows)

        cache_key = f"{selected_type}:{selected_name}" if selected_name else None

        def _finalize() -> None:
            nonlocal removed
            if cache_key and hasattr(panel, '_goals_cache'):
                try:
                    panel._goals_cache.pop(cache_key, None)
                except Exception:
                    pass
            refresh_goals(panel)
            summary = f"Removed {removed} goal(s)" if removed else "No goals removed"
            if failures:
                panel._append_out(f"[goals] {summary}; {len(failures)} failed")
            else:
                panel._append_out(f"[goals] {summary}")
            if skipped_primary and messagebox is not None:
                messagebox.showinfo("Protected Goals", f"Skipped {skipped_primary} protected [primary] goal(s).")
            _exit_goals_busy(panel)

        def _mark_success() -> None:
            nonlocal removed, pending
            removed += 1
            pending -= 1
            if pending <= 0:
                _finalize()

        def _mark_failure(goal_id: int, exc: Exception) -> None:
            nonlocal pending
            failures.append((goal_id, exc))
            panel._append_out(f"[goals] Failed to remove goal #{goal_id}: {exc}")
            pending -= 1
            if pending <= 0:
                _finalize()

        for goal_id in rows:
            try:
                logger.debug(f"Removing goal ID {goal_id}")
                result = panel._on_goal_remove(goal_id)
            except Exception as exc:
                panel.after(0, lambda gid=goal_id, err=exc: _mark_failure(gid, err))
                continue

            if hasattr(result, "add_done_callback"):
                def _callback(fut, gid=goal_id):
                    try:
                        fut.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        panel.after(0, lambda: _mark_failure(gid, exc))
                        return
                    panel.after(0, _mark_success)

                result.add_done_callback(_callback)  # type: ignore[call-arg]
            else:
                panel.after(0, _mark_success)

        # If callbacks never run (e.g., no futures returned), ensure finalization occurs
        if pending == 0:
            panel.after(0, _finalize)
    except Exception as e:
        logger.error(f"Error removing goals: {e}", exc_info=True)
        panel._append_out(f"[goals] Error removing goals: {e}")
        _exit_goals_busy(panel)
