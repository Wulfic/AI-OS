"""Goals management operations (add, remove, refresh).

All goals-related action methods with debouncing logic.
"""

from __future__ import annotations

import re
import time
from typing import Any, cast

try:  # pragma: no cover
    import tkinter as tk  # type: ignore
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover
    tk = cast(Any, None)
    messagebox = cast(Any, None)


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
        else:
            # Show loading state immediately
            panel.goals_list.delete(0, tk.END)
            panel.goals_count_var.set("Loading goals...")
            
            # Fetch from CLI in background thread to avoid UI freeze
            def fetch_goals():
                try:
                    items = list(panel._on_goals_list(selected_name) or [])
                    fetch_time = time.time()
                    panel._goals_cache[cache_key] = (items, fetch_time)
                    
                    # Cleanup old cache entries (keep only last 50)
                    if len(panel._goals_cache) > 50:
                        sorted_cache = sorted(panel._goals_cache.items(), key=lambda x: x[1][1])
                        panel._goals_cache = dict(sorted_cache[-50:])
                    
                    # Update UI on main thread
                    panel.after(0, lambda: _update_goals_ui(panel, items, type_str, cache_hit=False))
                except Exception as e:
                    panel.after(0, lambda: panel._append_out(f"[goals] Error fetching goals: {e}"))
            
            thread = threading.Thread(target=fetch_goals, daemon=True)
            thread.start()
            
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
        panel._append_out("[goals] Skipping duplicate goal add (debounced)")
        return
    
    try:
        panel._last_goal_add_time = current_time
        panel._last_goal_text = goal_text
        
        if selected_name is None:
            panel._append_out("[goals] Selected name is None, cannot add goal")
            return
        
        panel._append_out(f"[goals] Adding goal to {selected_type} '{selected_name}': {goal_text}")
        panel._on_goal_add(selected_name, goal_text)
        panel._append_out("[goals] Goal added successfully")
        
        # Clear entry after adding
        panel.goal_text_var.set("")
        
        # Invalidate cache for this item
        if hasattr(panel, '_goals_cache'):
            cache_key = f"{selected_type}:{selected_name}"
            panel._goals_cache.pop(cache_key, None)
        
        # Refresh goals list
        refresh_goals(panel)
    except Exception as e:
        import traceback
        panel._append_out(f"[goals] Error adding goal: {e}")
        panel._append_out(f"[goals] Traceback: {traceback.format_exc()}")


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
        
        removed = 0
        skipped_primary = 0
        
        # Work on a copy to avoid index shifts
        for idx in list(sel):
            raw = panel.goals_list.get(idx)
            if "[primary]" in str(raw):
                skipped_primary += 1
                continue
            
            # Extract goal ID from format: "1) #42 • Goal text"
            m = re.match(r"^\s*\d+\)\s*#(\d+)\b|^\s*#(\d+)\b", str(raw).strip())
            goal_id = None
            if m:
                goal_id = m.group(1) or m.group(2)
            
            if goal_id is None:
                continue
            
            try:
                panel._on_goal_remove(int(goal_id))
                removed += 1
            except Exception:
                continue
        
        # Invalidate cache for the selected item
        if hasattr(panel, '_goals_cache'):
            selected_name = brain_name or expert_id
            selected_type = "brain" if brain_name else "expert"
            if selected_name:
                cache_key = f"{selected_type}:{selected_name}"
                panel._goals_cache.pop(cache_key, None)
        
        # Refresh after batch operations
        refresh_goals(panel)
        
        if skipped_primary and messagebox is not None:
            messagebox.showinfo("Protected Goals", f"Skipped {skipped_primary} protected [primary] goal(s).")
    except Exception as e:
        panel._append_out(f"[goals] Error removing goals: {e}")
