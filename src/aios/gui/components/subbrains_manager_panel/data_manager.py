"""Data management utilities for the Subbrains Manager Panel.

Provides functions for:
- Loading expert registry data
- Updating tree view with experts
- Extracting selected expert information
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tkinter import ttk

logger = logging.getLogger(__name__)


def get_selected_expert_id(experts_tree: "ttk.Treeview") -> Optional[str]:
    """Get the expert_id of the currently selected expert.
    
    Args:
        experts_tree: The Treeview widget
        
    Returns:
        Expert ID string if an expert is selected, None otherwise
    """
    sel = experts_tree.selection()
    if not sel:
        return None
    
    try:
        tags = experts_tree.item(sel[0]).get("tags", ())
        if tags:
            return tags[0]
    except Exception:
        pass
    
    return None


def load_and_refresh_experts(
    registry_path: str,
    experts_tree: "ttk.Treeview",
    total_experts_var: Any,
    active_experts_var: Any,
    frozen_experts_var: Any,
    total_activations_var: Any,
    append_out_callback: Any,
) -> None:
    """Load expert registry from disk and refresh UI.
    
    Args:
        registry_path: Path to expert_registry.json
        experts_tree: Treeview widget to populate
        total_experts_var: StringVar for total count
        active_experts_var: StringVar for active count
        frozen_experts_var: StringVar for frozen count
        total_activations_var: StringVar for activations count
        append_out_callback: Callback for logging output
    """
    try:
        logger.info(f"Loading experts from registry: {registry_path}")
        # Load expert registry
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
            
            experts = registry_data.get("experts", [])
            logger.info(f"Found {len(experts)} expert(s) in registry")
            logger.debug(f"Expert IDs: {[e.get('expert_id', 'Unknown')[:8] for e in experts]}")
        else:
            experts = []
            logger.warning(f"No registry found at {registry_path}")
            append_out_callback(f"[Subbrains] No registry found at {registry_path}")
        
        # Update summary stats
        total = len(experts)
        active = sum(1 for e in experts if e.get("is_active", False))
        frozen = sum(1 for e in experts if e.get("is_frozen", False))
        total_acts = sum(e.get("total_activations", 0) for e in experts)
        
        logger.debug(f"Expert stats - Total: {total}, Active: {active}, Frozen: {frozen}, Activations: {total_acts}")
        
        total_experts_var.set(str(total))
        active_experts_var.set(str(active))
        frozen_experts_var.set(str(frozen))
        total_activations_var.set(str(total_acts))
        
        # Update tree
        for item in experts_tree.get_children():
            experts_tree.delete(item)
        
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
                experts_tree.insert("", "end", values=values, tags=(expert_id,))
                
            except Exception as e:
                logger.error(f"Error displaying expert {expert.get('expert_id', 'Unknown')}: {e}")
                append_out_callback(f"[Subbrains] Error displaying expert: {e}")
                continue
        
        append_out_callback(f"[Subbrains] Loaded {total} expert(s)")
        
    except Exception as e:
        logger.error(f"Error refreshing experts from {registry_path}: {e}", exc_info=True)
        append_out_callback(f"[Subbrains] Error refreshing: {e}")
        import traceback
        append_out_callback(traceback.format_exc())


def load_goals_for_expert(
    expert_id: str,
    registry_path: str,
    goals_list: Any,
    goals_count_var: Any,
    append_out_callback: Any,
) -> None:
    """Load goals linked to a specific expert.
    
    Args:
        expert_id: The expert's ID
        registry_path: Path to expert_registry.json
        goals_list: Listbox widget to populate with goals
        goals_count_var: StringVar for goals count display
        append_out_callback: Callback for logging output
    """
    try:
        import tkinter as tk
    except Exception:
        return
    
    logger.info(f"Loading goals for expert: {expert_id[:8]}...")
    
    goals_list.delete(0, tk.END)
    goals_count_var.set(f"Expert: {expert_id[:8]}...")
    
    # Load goals from registry
    try:
        if os.path.exists(registry_path):
            with open(registry_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
            
            experts = registry_data.get("experts", [])
            for expert in experts:
                if expert.get("expert_id") == expert_id:
                    goals = expert.get("goals", [])
                    logger.info(f"Found {len(goals)} goal(s) for expert {expert_id[:8]}")
                    logger.debug(f"Goal IDs: {goals}")
                    for goal_id in goals:
                        goals_list.insert(tk.END, f"Goal #{goal_id}")
                    
                    count = len(goals)
                    goals_count_var.set(f"{count} goal{'s' if count != 1 else ''} linked")
                    break
    except Exception as e:
        logger.error(f"Error loading goals for expert {expert_id[:8]}: {e}")
        append_out_callback(f"[Subbrains] Error loading goals: {e}")
