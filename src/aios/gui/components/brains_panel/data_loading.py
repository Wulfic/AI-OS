"""Data loading and refresh operations for brains and experts.

Fetches data from CLI and populates tree widgets.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from typing import Any

from ...services.brain_registry_service import get_brain_stats

try:  # pragma: no cover - import guard for early bootstrap
    from aios.system import paths as system_paths
except Exception:  # pragma: no cover
    system_paths = None


logger = logging.getLogger(__name__)


def _run_on_main_thread(panel: Any, callback) -> None:
    """Execute callback on Tk main thread, scheduling via `after` if needed."""
    try:
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            panel.after(0, callback)
    except Exception as schedule_err:
        message = str(schedule_err)
        if "main thread is not in main loop" in message:
            try:
                pending = panel._startup_callbacks  # type: ignore[attr-defined]
            except AttributeError:
                pending = []
                panel._startup_callbacks = pending  # type: ignore[attr-defined]
            pending.append(callback)
            logger.debug("Queued brains panel UI callback until main loop starts")
        else:
            logger.warning("Failed to schedule brains panel UI update: %s", schedule_err)

def _detect_actv1_base(panel: Any) -> str:
    candidates: list[str] = []
    store_dir = getattr(panel, "_store_dir", None)
    if store_dir:
        candidates.append(os.path.join(store_dir, "actv1"))
    project_root = getattr(panel, "_project_root", None)
    if project_root:
        candidates.append(os.path.join(project_root, "artifacts", "brains", "actv1"))
    if system_paths is not None:
        candidates.append(str(system_paths.get_brain_family_dir("actv1")))

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return candidates[0] if candidates else ""


def refresh_brains_data(panel: Any) -> list[str]:
    """Fetch brains data and update the UI safely."""
    from .helpers import (
        is_temporary_brain,
        load_training_steps,
        scan_actv1_bundles,
    )

    names: list[str] = []

    try:
        data = get_brain_stats(panel._store_dir)
        brains = data.get("brains") if isinstance(data, dict) else {}
        if not isinstance(brains, dict):
            brains = {}

        try:
            actv1_base = _detect_actv1_base(panel)
            actv1_brains = scan_actv1_bundles(actv1_base)
            for entry, info in actv1_brains.items():
                if entry not in brains:
                    brains[entry] = info
        except Exception:
            pass

        brains = {k: v for k, v in brains.items() if not is_temporary_brain(k)}

        names = sorted(brains.keys())
        used_bytes = int(data.get("used_bytes", 0) or 0)
        total_mb = float(used_bytes) / (1024.0 * 1024.0)
        total_params_m = float(used_bytes) / 4.0 / 1_000_000.0 if used_bytes > 0 else 0.0

        table_rows: list[tuple[str, ...]] = []
        error_messages: list[str] = []

        for brain_name in names:
            try:
                info = brains.get(brain_name) or {}
                size_b = int(info.get("size_bytes", 0) or 0)
                size_mb = float(size_b) / (1024.0 * 1024.0)
                params_m = float(size_b) / 4.0 / 1_000_000.0 if size_b > 0 else 0.0
                pinned = bool(info.get("pinned", False))
                master = bool(info.get("master", False))
                parent = info.get("parent") or ""
                children: list[str] = []
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
                        rel += "..."

                training_steps = int(info.get("training_steps", 0) or 0)
                brain_metadata = None
                if training_steps == 0:
                    try:
                        actv1_brain_path = os.path.join(panel._store_dir, "actv1", brain_name)
                        if os.path.isdir(actv1_brain_path):
                            meta_path = os.path.join(actv1_brain_path, "brain.json")
                            if os.path.exists(meta_path):
                                with open(meta_path, "r", encoding="utf-8") as brain_file:
                                    brain_metadata = json.load(brain_file)
                                training_steps = load_training_steps(actv1_brain_path, brain_metadata)
                        else:
                            brain_path = os.path.join(panel._store_dir, brain_name)
                            if os.path.isdir(brain_path):
                                meta_path = os.path.join(brain_path, "brain.json")
                                if os.path.exists(meta_path):
                                    with open(meta_path, "r", encoding="utf-8") as brain_file:
                                        brain_metadata = json.load(brain_file)
                                    training_steps = load_training_steps(brain_path, brain_metadata)
                    except Exception:
                        pass

                if params_m < 0.01 and brain_metadata is not None:
                    from .helpers import calculate_params_from_metadata

                    calculated_params = calculate_params_from_metadata(brain_metadata)
                    if calculated_params > 0:
                        params_m = calculated_params / 1_000_000.0
                        size_b = int(calculated_params * 4)
                        size_mb = float(size_b) / (1024.0 * 1024.0)

                last_used = info.get("last_used") or ""
                values_tuple = (
                    brain_name,
                    f"{size_mb:.2f}",
                    f"{params_m:.3f}",
                    "yes" if pinned else "",
                    "yes" if master else "",
                    rel,
                    f"{training_steps:,}",
                    str(last_used),
                )
                table_rows.append(values_tuple)
            except Exception as brain_err:
                error_messages.append(f"[brains] Error processing brain {brain_name}: {brain_err}")

        def apply_success() -> None:
            try:
                try:
                    panel._brain_stats_cache = data  # type: ignore[attr-defined]
                except Exception:
                    pass

                panel.brain_count_var.set(str(len(names)))
                panel.total_mb_var.set(f"{total_mb:.2f}")
                panel.total_params_m_var.set(f"{total_params_m:.3f}")

                for item in panel.tree.get_children():
                    panel.tree.delete(item)
                for row in table_rows:
                    panel.tree.insert("", "end", values=row)

                for message in error_messages:
                    panel._append_out(message)

                # Ensure a brain row stays selected so goal operations keep context.
                try:
                    cached_target = getattr(panel, "_current_goal_target", None)
                except Exception:
                    cached_target = None

                target_name: str | None = None
                if isinstance(cached_target, tuple) and len(cached_target) == 2:
                    target_type, name = cached_target
                    if target_type == "brain" and isinstance(name, str) and name:
                        target_name = name

                selected_item = None
                if target_name:
                    for item in panel.tree.get_children():
                        values = panel.tree.item(item).get("values", [])
                        if values and str(values[0]) == target_name:
                            selected_item = item
                            break

                if selected_item is None:
                    children = panel.tree.get_children()
                    if children:
                        selected_item = children[0]

                if selected_item is not None:
                    try:
                        panel.tree.selection_set(selected_item)
                        panel.tree.focus(selected_item)
                        panel.tree.see(selected_item)
                    except Exception:
                        pass
                    try:
                        panel._on_tree_select()
                    except Exception:
                        pass
            except Exception as ui_err:
                logger.warning("Failed to apply brains data to UI: %s", ui_err)

        _run_on_main_thread(panel, apply_success)

    except Exception as err:
        tb = traceback.format_exc()
        error_message = str(err)

        def apply_error() -> None:
            try:
                panel._append_out(f"[brains] Refresh failed: {error_message}")
                panel._append_out(tb)
                try:
                    panel._brain_stats_cache = {}  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as ui_err:
                logger.warning("Failed to report brains refresh error: %s", ui_err)

        _run_on_main_thread(panel, apply_error)

    return names


def refresh_experts_data(panel: Any) -> None:
    """Fetch and display experts data in tree widget."""
    try:
        if os.path.exists(panel._registry_path):
            with open(panel._registry_path, "r", encoding="utf-8") as registry_file:
                registry_data = json.load(registry_file)
            experts = registry_data.get("experts", [])
        else:
            experts = []

        total = len(experts)
        active = sum(1 for expert in experts if expert.get("is_active", False))
        frozen = sum(1 for expert in experts if expert.get("is_frozen", False))
        total_activations = sum(expert.get("total_activations", 0) for expert in experts)

        rows: list[tuple[tuple[str, ...], str]] = []
        error_messages: list[str] = []

        for expert in experts:
            try:
                name = expert.get("name", "Unnamed")
                category = expert.get("category", "")

                is_active = expert.get("is_active", False)
                is_frozen = expert.get("is_frozen", False)
                if is_frozen:
                    status = "Frozen"
                elif is_active:
                    status = "Active"
                else:
                    status = "Inactive"

                activations = expert.get("total_activations", 0)
                avg_weight = expert.get("avg_routing_weight", 0.0)

                goals = expert.get("goals", [])
                goals_display = f"{len(goals)} linked" if goals else "--"

                parent = expert.get("parent_expert_id") or ""
                children = expert.get("child_expert_ids", [])
                hierarchy = ""
                if parent:
                    hierarchy = f"p:{parent[:8]}"
                if children:
                    hierarchy += ("; " if hierarchy else "") + f"{len(children)} child(ren)"
                if not hierarchy:
                    hierarchy = "--"

                values = (
                    name,
                    category,
                    status,
                    str(activations),
                    f"{avg_weight:.3f}",
                    goals_display,
                    hierarchy,
                )

                expert_id = expert.get("expert_id", "")
                rows.append((values, expert_id))
            except Exception as expert_err:
                error_messages.append(f"[experts] Error displaying expert: {expert_err}")

        def apply_success() -> None:
            try:
                panel.total_experts_var.set(str(total))
                panel.active_experts_var.set(str(active))
                panel.frozen_experts_var.set(str(frozen))
                panel.total_activations_var.set(str(total_activations))

                for item in panel.experts_tree.get_children():
                    panel.experts_tree.delete(item)
                for values, expert_id in rows:
                    if expert_id:
                        panel.experts_tree.insert("", "end", values=values, tags=(expert_id,))
                    else:
                        panel.experts_tree.insert("", "end", values=values)

                for message in error_messages:
                    panel._append_out(message)
            except Exception as ui_err:
                logger.warning("Failed to apply experts data to UI: %s", ui_err)

        _run_on_main_thread(panel, apply_success)

    except Exception as err:
        tb = traceback.format_exc()
        error_message = str(err)

        def apply_error() -> None:
            try:
                panel._append_out(f"[experts] Refresh failed: {error_message}")
                panel._append_out(tb)
            except Exception as ui_err:
                logger.warning("Failed to report experts refresh error: %s", ui_err)

        _run_on_main_thread(panel, apply_error)
