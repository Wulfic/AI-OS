"""Data loading and refresh operations for brains and experts.

Fetches data from CLI and populates tree widgets.
"""

from __future__ import annotations

import json
import os
from typing import Any


def refresh_brains_data(panel: Any) -> list[str]:
    """Fetch and display brains data in tree widget.
    
    Args:
        panel: BrainsPanel instance
        
    Returns:
        List of brain names loaded
    """
    from .helpers import is_temporary_brain, load_training_steps, parse_cli_dict, scan_actv1_bundles
    
    names = []
    
    try:
        raw = panel._run_cli(["brains", "stats", "--store-dir", panel._store_dir]) or "{}"
        data = parse_cli_dict(raw)
        brains = data.get("brains") if isinstance(data, dict) else {}
        if not isinstance(brains, dict):
            brains = {}
        
        # Merge in ACTV1 bundle directories (same source as Select Student dialog)
        try:
            actv1_base = os.path.join(panel._project_root, "artifacts", "brains", "actv1")
            actv1_brains = scan_actv1_bundles(actv1_base)
            for entry, info in actv1_brains.items():
                if entry not in brains:
                    brains[entry] = info
        except Exception:
            pass
        
        # Filter out temporary/internal brains
        brains = {k: v for k, v in brains.items() if not is_temporary_brain(k)}
        
        # Summary
        names = sorted(list(brains.keys()))
        panel.brain_count_var.set(str(len(names)))
        used_bytes = int(data.get("used_bytes", 0) or 0)
        total_mb = float(used_bytes) / (1024.0 * 1024.0)
        total_params_m = float(used_bytes) / 4.0 / 1_000_000.0 if used_bytes > 0 else 0.0
        panel.total_mb_var.set(f"{total_mb:.2f}")
        panel.total_params_m_var.set(f"{total_params_m:.3f}")
        
        # Table
        for item in panel.tree.get_children():
            panel.tree.delete(item)
        
        for n in names:
            try:
                info = brains.get(n) or {}
                size_b = int(info.get("size_bytes", 0) or 0)
                size_mb = float(size_b) / (1024.0 * 1024.0)
                params_m = float(size_b) / 4.0 / 1_000_000.0 if size_b > 0 else 0.0
                pinned = bool(info.get("pinned", False))
                master = bool(info.get("master", False))
                parent = info.get("parent") or ""
                children = []
                try:
                    ch = info.get("children")
                    if isinstance(ch, list):
                        children = [str(c) for c in ch]
                except Exception:
                    children = []
                
                rel = f"p:{parent}" if parent else ""
                if children:
                    rel = (rel + ("; " if rel else "")) + "kids:" + ",".join(children[:5])
                    if len(children) > 5:
                        rel += "…"
                
                # Get training steps (with metrics.jsonl fallback)
                training_steps = int(info.get("training_steps", 0) or 0)
                brain_metadata = None
                if training_steps == 0:
                    try:
                        # Try actv1 bundle directory first
                        actv1_brain_path = os.path.join(panel._store_dir, "actv1", n)
                        if os.path.isdir(actv1_brain_path):
                            meta_path = os.path.join(actv1_brain_path, "brain.json")
                            if os.path.exists(meta_path):
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    brain_metadata = json.load(f)
                                training_steps = load_training_steps(actv1_brain_path, brain_metadata)
                        else:
                            # Fallback: legacy direct path
                            brain_path = os.path.join(panel._store_dir, n)
                            if os.path.isdir(brain_path):
                                meta_path = os.path.join(brain_path, "brain.json")
                                if os.path.exists(meta_path):
                                    with open(meta_path, "r", encoding="utf-8") as f:
                                        brain_metadata = json.load(f)
                                    training_steps = load_training_steps(brain_path, brain_metadata)
                    except Exception:
                        pass
                
                # Calculate params from architecture if file size is too small or metadata available
                params_m = float(size_b) / 4.0 / 1_000_000.0 if size_b > 0 else 0.0
                if params_m < 0.01 and brain_metadata is not None:
                    # Try to calculate from architecture
                    from .helpers import calculate_params_from_metadata
                    calculated_params = calculate_params_from_metadata(brain_metadata)
                    if calculated_params > 0:
                        params_m = calculated_params / 1_000_000.0
                        # Also update size estimate based on params (4 bytes per param for float32)
                        size_b = int(calculated_params * 4)
                        size_mb = float(size_b) / (1024.0 * 1024.0)
                
                last_used = info.get("last_used") or ""
                values_tuple = (
                    n,
                    f"{size_mb:.2f}",
                    f"{params_m:.3f}",
                    ("yes" if pinned else ""),
                    ("yes" if master else ""),
                    rel,
                    f"{training_steps:,}",  # Format with thousands separator
                    str(last_used),
                )
                panel.tree.insert("", "end", values=values_tuple)
            except Exception as e:
                panel._append_out(f"[brains] Error processing brain {n}: {e}")
                continue
    except Exception as e:
        # Report error instead of silently failing
        panel._append_out(f"[brains] Refresh failed: {e}")
        import traceback
        panel._append_out(traceback.format_exc())
    
    return names


def refresh_experts_data(panel: Any) -> None:
    """Fetch and display experts data in tree widget.
    
    Args:
        panel: BrainsPanel instance
    """
    try:
        # Load expert registry
        if os.path.exists(panel._registry_path):
            with open(panel._registry_path, "r", encoding="utf-8") as f:
                registry_data = json.load(f)
            
            experts = registry_data.get("experts", [])
        else:
            experts = []
        
        # Update summary stats
        total = len(experts)
        active = sum(1 for e in experts if e.get("is_active", False))
        frozen = sum(1 for e in experts if e.get("is_frozen", False))
        total_acts = sum(e.get("total_activations", 0) for e in experts)
        
        panel.total_experts_var.set(str(total))
        panel.active_experts_var.set(str(active))
        panel.frozen_experts_var.set(str(frozen))
        panel.total_activations_var.set(str(total_acts))
        
        # Update experts tree
        for item in panel.experts_tree.get_children():
            panel.experts_tree.delete(item)
        
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
                panel.experts_tree.insert("", "end", values=values, tags=(expert_id,))
                
            except Exception as e:
                panel._append_out(f"[experts] Error displaying expert: {e}")
                continue
        
    except Exception as e:
        panel._append_out(f"[experts] Refresh failed: {e}")
        import traceback
        panel._append_out(traceback.format_exc())
