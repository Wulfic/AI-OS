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

from .expert_operations import get_selected_expert_id
from .helpers import get_selected_tree_value

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
    
    if not selected_name:
        cached_target = getattr(panel, "_current_goal_target", None)
        if isinstance(cached_target, tuple) and len(cached_target) == 2:
            cached_type, cached_name = cached_target
            if cached_name:
                selected_type = str(cached_type or "brain")
                selected_name = cached_name
                try:
                    if selected_type == "brain" and hasattr(panel, "tree") and panel.tree is not None:
                        for item in panel.tree.get_children():
                            values = panel.tree.item(item).get("values", [])
                            if values and str(values[0]) == cached_name:
                                panel.tree.selection_set(item)
                                panel.tree.focus(item)
                                try:
                                    panel.tree.see(item)
                                except Exception:
                                    pass
                                break
                    elif selected_type == "expert" and hasattr(panel, "experts_tree") and panel.experts_tree is not None:
                        for item in panel.experts_tree.get_children():
                            tags = panel.experts_tree.item(item).get("tags", ())
                            if tags and str(tags[0]) == cached_name:
                                panel.experts_tree.selection_set(item)
                                panel.experts_tree.focus(item)
                                try:
                                    panel.experts_tree.see(item)
                                except Exception:
                                    pass
                                break
                except Exception:
                    pass
                try:
                    logger.debug(
                        "refresh_goals: reused cached goal target %s:%s", selected_type, selected_name
                    )
                except Exception:
                    pass

    # Handle no selection
    if not selected_name:
        try:
            logger.debug(
                "refresh_goals: no active selection (brain_sel=%s expert_sel=%s brain_focus=%s expert_focus=%s cached=%s)",
                brain_sel,
                expert_sel,
                panel.tree.focus() if hasattr(panel, "tree") else None,
                panel.experts_tree.focus() if hasattr(panel, "experts_tree") else None,
                getattr(panel, "_current_goal_target", None),
            )
        except Exception:
            logger.debug("refresh_goals: no active selection (debug introspection failed)")
        try:
            panel._current_goal_target = None
        except Exception:
            pass
        try:
            panel.goals_list.delete(0, tk.END)
            panel.goals_count_var.set("No brain or expert selected")
        except Exception:
            pass
        return
    
    # Type assertion for selected_type (we know it's not None here)
    type_str = str(selected_type or "brain")
    try:
        panel._current_goal_target = (type_str, selected_name)
    except Exception:
        pass
    
    try:
        # Check cache first
        cache_key = f"{selected_type}:{selected_name}"
        cached = panel._goals_cache.get(cache_key)
        
        # Use cache if valid (within TTL)
        if cached and (current_time - cached[1]) < panel._cache_ttl:
            items = cached[0]
            _update_goals_ui(panel, items, type_str, selected_name, cache_hit=True)
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
            _update_goals_ui(panel, result, type_str, selected_name, cache_hit=False)
            return

        def _finalize(items: list[Any], *, refresh_time: float) -> None:
            try:
                panel._goals_cache[cache_key] = (items, refresh_time)
                if len(panel._goals_cache) > 50:
                    sorted_cache = sorted(panel._goals_cache.items(), key=lambda x: x[1][1])
                    panel._goals_cache = dict(sorted_cache[-50:])
            except Exception:
                pass
            _update_goals_ui(panel, items, type_str, selected_name, cache_hit=False)

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
                    panel.after(0, lambda err=exc: _handle_error(err))
                    return
                panel.after(0, lambda: _handle_future(items))

            result.add_done_callback(_on_complete)  # type: ignore[call-arg]
        else:
            import threading

            def _worker() -> None:
                try:
                    items = list(result or [])  # type: ignore[arg-type]
                except Exception as exc:
                    panel.after(0, lambda err=exc: _handle_error(err))
                    return
                panel.after(0, lambda: _handle_future(items))

            threading.Thread(target=_worker, daemon=True).start()
            
    except Exception as e:
        panel._append_out(f"[goals] Error refreshing goals: {e}")


def _format_goal_display(goal_id: int | None, text: str, *, expert_id: str | None,
                          selected_type: str, selected_name: str, protected: bool) -> str:
    """Create a concise display string for the goals list."""
    prefix = f"#{goal_id} " if goal_id is not None else ""
    suffix_parts: list[str] = []

    if expert_id:
        if expert_id != selected_name:
            suffix_parts.append(f"@{expert_id}")
    elif selected_type == "brain":
        suffix_parts.append("(global)")

    if protected:
        suffix_parts.append("[protected]")

    suffix = f" {' '.join(suffix_parts)}" if suffix_parts else ""
    return f"{prefix}{text}{suffix}".strip()


def _normalize_goal_items(
    items: list[Any],
    *,
    selected_type: str,
    selected_name: str,
) -> list[dict[str, Any]]:
    """Normalize raw goal payloads into display-friendly dictionaries."""
    normalized: list[dict[str, Any]] = []
    target = (selected_name or "").strip()
    target_lower = target.lower()

    for raw in items:
        entry: dict[str, Any]
        goal_id: int | None = None
        text = ""
        expert_id: str | None = None
        protected = False

        if isinstance(raw, dict):
            candidate_id = raw.get("id")
            try:
                goal_id = int(candidate_id) if candidate_id is not None else None
            except (TypeError, ValueError):
                goal_id = None

            text = str(raw.get("text", "")).strip()
            expert_val = raw.get("expert_id")
            expert_id = str(expert_val).strip() if expert_val is not None else None
            protected = bool(raw.get("protected"))
        else:
            payload = str(raw)
            text = payload.strip()
            match = re.search(r"#(\d+)", payload)
            if match:
                try:
                    goal_id = int(match.group(1))
                except Exception:
                    goal_id = None
            protected = "[primary]" in payload.lower()

        if not text:
            text = "<empty goal>"

        expert_lower = expert_id.lower() if isinstance(expert_id, str) else ""
        include = True
        if selected_type == "expert":
            include = bool(expert_id) and expert_lower == target_lower
        elif selected_type == "brain":
            if expert_id and expert_lower != target_lower:
                include = False

        if not include:
            continue

        display = _format_goal_display(
            goal_id,
            text,
            expert_id=expert_id,
            selected_type=selected_type,
            selected_name=target,
            protected=protected,
        )

        entry = {
            "id": goal_id,
            "text": text,
            "expert_id": expert_id,
            "protected": protected,
            "display": display,
        }
        normalized.append(entry)

    return normalized


def _update_goals_ui(
    panel: Any,
    items: list[Any],
    selected_type: str,
    selected_name: str,
    cache_hit: bool = False,
) -> None:
    """Update the goals list UI with fetched items."""
    try:
        try:
            panel._goal_index_map = {}
        except Exception:
            pass

        normalized = _normalize_goal_items(
            items,
            selected_type=selected_type,
            selected_name=selected_name,
        )

        panel.goals_list.delete(0, tk.END)
        for idx, entry in enumerate(normalized):
            panel.goals_list.insert(tk.END, entry["display"])
            panel._goal_index_map[idx] = entry

        count = len(normalized)
        type_label = "Brain" if selected_type == "brain" else "Expert"
        cache_hint = " [cached]" if cache_hit else ""
        panel.goals_count_var.set(f"{count} goal{'s' if count != 1 else ''} ({type_label}){cache_hint}")

        try:
            panel._goal_entries = normalized
        except Exception:
            pass
    except Exception as e:
        panel._append_out(f"[goals] Error updating goals UI: {e}")


def add_goal(panel: Any) -> None:
    """Add a new goal for the currently selected brain or expert.
    
    Includes debouncing to prevent duplicate additions.
    
    Args:
        panel: BrainsPanel instance
    """
    if panel._on_goal_add is None:
        panel._append_out("[goals] Goal add callback not available")
        return
    
    logger.info("add_goal invoked for current panel state")

    tree_sel = ()
    tree_focus = None
    experts_sel = ()
    experts_focus = None
    focus_widget = None

    try:
        if hasattr(panel, "tree") and panel.tree is not None:
            try:
                tree_sel = panel.tree.selection()
            except Exception:
                tree_sel = ()
            try:
                tree_focus = panel.tree.focus()
            except Exception:
                tree_focus = None
        if hasattr(panel, "experts_tree") and panel.experts_tree is not None:
            try:
                experts_sel = panel.experts_tree.selection()
            except Exception:
                experts_sel = ()
            try:
                experts_focus = panel.experts_tree.focus()
            except Exception:
                experts_focus = None
        if hasattr(panel, "focus_get"):
            try:
                focus_widget = panel.focus_get()
            except Exception:
                focus_widget = None
        logger.debug(
            "add_goal entry: tree_sel=%s tree_focus=%s expert_sel=%s expert_focus=%s current_target=%s focus_widget=%s",
            tree_sel,
            tree_focus,
            experts_sel,
            experts_focus,
            getattr(panel, "_current_goal_target", None),
            focus_widget,
        )
    except Exception:
        logger.debug("add_goal entry: failed to introspect UI state")

    # Check which tree has selection
    brain_name = get_selected_tree_value(panel.tree, 0)
    expert_id = get_selected_expert_id(panel)

    # Fallback to focused rows if selection metadata is missing
    if not brain_name:
        try:
            focus_id = panel.tree.focus()
            if focus_id:
                focus_values = panel.tree.item(focus_id).get("values", [])
                if focus_values:
                    brain_name = str(focus_values[0]) or None
                    try:
                        panel.tree.selection_set(focus_id)
                    except Exception:
                        pass
                    logger.debug("add_goal fallback: using tree focus %s -> %s", focus_id, brain_name)
        except Exception:
            pass

    if not expert_id:
        try:
            focus_id = panel.experts_tree.focus()
            if focus_id:
                focus_tags = panel.experts_tree.item(focus_id).get("tags", ())
                if focus_tags:
                    expert_id = str(focus_tags[0])
                    try:
                        panel.experts_tree.selection_set(focus_id)
                    except Exception:
                        pass
                    logger.debug("add_goal fallback: using experts focus %s -> %s", focus_id, expert_id)
        except Exception:
            pass

    if not brain_name and not expert_id:
        target = getattr(panel, "_current_goal_target", None)
        if isinstance(target, tuple) and len(target) == 2:
            target_type, target_name = target
            if target_type == "brain":
                brain_name = target_name
            elif target_type == "expert":
                expert_id = target_name
            logger.debug("add_goal fallback: using cached target %s", target)
    
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
        logger.warning(
            "add_goal aborted: no selection (tree_sel=%s tree_focus=%s expert_sel=%s expert_focus=%s cached=%s focus_widget=%s)",
            tree_sel,
            tree_focus,
            experts_sel,
            experts_focus,
            getattr(panel, "_current_goal_target", None),
            focus_widget,
        )
        panel._append_out("[goals] No brain or expert selected")
        return
    else:
        # Both selected? Prefer brain
        selected_name = brain_name
        selected_type = "brain"
    
    if selected_name is not None and selected_type is not None:
        try:
            panel._current_goal_target = (selected_type, selected_name)
        except Exception:
            pass

    logger.debug(
        "add_goal resolved target: type=%s name=%s", selected_type, selected_name
    )

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
                    panel.after(0, lambda err=exc: _on_error(err))
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
    
    Skips protected goals.
    
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
        target = getattr(panel, "_current_goal_target", None)
        if isinstance(target, tuple) and len(target) == 2:
            target_type, target_name = target
            if target_type == "brain":
                brain_name = target_name
            elif target_type == "expert":
                expert_id = target_name
    
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
        index_map = getattr(panel, "_goal_index_map", {}) if hasattr(panel, "_goal_index_map") else {}
        for idx in list(sel):
            entry = index_map.get(idx)
            if entry is None:
                raw = panel.goals_list.get(idx)
                logger.warning(f"Goal metadata missing for list row: {raw}")
                continue
            if entry.get("protected"):
                skipped_primary += 1
                logger.debug(f"Skipped protected goal: {entry}")
                continue
            goal_id = entry.get("id")
            if goal_id is None:
                raw = panel.goals_list.get(idx)
                logger.warning(f"Goal ID missing for row: {raw}")
                continue
            try:
                rows.append(int(goal_id))
            except (TypeError, ValueError):
                logger.warning(f"Invalid goal ID value: {goal_id}")

        if not rows:
            if skipped_primary and messagebox is not None:
                messagebox.showinfo("Protected Goals", f"Skipped {skipped_primary} protected goal(s).")
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
                messagebox.showinfo("Protected Goals", f"Skipped {skipped_primary} protected goal(s).")
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
                    current_goal_id = gid
                    try:
                        fut.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        panel.after(0, lambda err=exc, goal=current_goal_id: _mark_failure(goal, err))
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
