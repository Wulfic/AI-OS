"""Tree management for benchmark selection."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .benchmark_data import BENCHMARKS

if TYPE_CHECKING:
    from .panel_main import EvaluationPanel


def populate_benchmark_tree(panel: "EvaluationPanel") -> None:
    """Populate the benchmark tree with available benchmarks.
    
    Args:
        panel: The evaluation panel instance
    """
    panel.bench_tree.delete(*panel.bench_tree.get_children())
    panel._tree_items = {}
    
    for category, benchmarks in BENCHMARKS.items():
        # Add category
        cat_id = panel.bench_tree.insert("", "end", text="☐", values=(category, "", ""))
        panel._tree_items[cat_id] = {"type": "category", "name": category, "checked": False}
        
        # Add benchmarks
        for bench_name, bench_desc in benchmarks:
            item_id = panel.bench_tree.insert(
                cat_id, "end",
                text="☐",
                values=("", bench_name, bench_desc)
            )
            panel._tree_items[item_id] = {
                "type": "benchmark",
                "name": bench_name,
                "category": category,
                "checked": False
            }


def on_benchmark_click(panel: "EvaluationPanel", event: Any) -> None:
    """Handle benchmark selection toggle.
    
    Args:
        panel: The evaluation panel instance
        event: The click event
    """
    region = panel.bench_tree.identify_region(event.x, event.y)
    if region != "tree":
        return
        
    item_id = panel.bench_tree.identify_row(event.y)
    if not item_id or item_id not in panel._tree_items:
        return
        
    item_info = panel._tree_items[item_id]
    
    # Toggle checked state
    item_info["checked"] = not item_info["checked"]
    check_mark = "☑" if item_info["checked"] else "☐"
    panel.bench_tree.item(item_id, text=check_mark)
    
    # If category, toggle all children
    if item_info["type"] == "category":
        for child_id in panel.bench_tree.get_children(item_id):
            if child_id in panel._tree_items:
                panel._tree_items[child_id]["checked"] = item_info["checked"]
                panel.bench_tree.item(child_id, text=check_mark)
    
    update_selected_benchmarks(panel)


def update_selected_benchmarks(panel: "EvaluationPanel") -> None:
    """Update the selected benchmarks list and info label.
    
    Args:
        panel: The evaluation panel instance
    """
    selected = []
    for item_id, item_info in panel._tree_items.items():
        if item_info["type"] == "benchmark" and item_info["checked"]:
            selected.append(item_info["name"])
    
    panel.selected_benchmarks_var.set(",".join(selected))
    panel.selection_label.config(
        text=f"Selected: {len(selected)} benchmark{'s' if len(selected) != 1 else ''}"
    )
    panel._schedule_save()


def select_preset(panel: "EvaluationPanel", preset: str) -> None:
    """Select a benchmark preset.
    
    Args:
        panel: The evaluation panel instance
        preset: The preset name (Language, Coding, Math, Science, All, Custom)
    """
    # Uncheck all first
    for item_id, item_info in panel._tree_items.items():
        item_info["checked"] = False
        panel.bench_tree.item(item_id, text="☐")
    
    if preset == "All":
        # Check all benchmarks
        for item_id, item_info in panel._tree_items.items():
            if item_info["type"] == "benchmark":
                item_info["checked"] = True
                panel.bench_tree.item(item_id, text="☑")
    elif preset == "Custom":
        # Do nothing, let user select manually
        pass
    elif preset in BENCHMARKS:
        # Check benchmarks in this category
        for item_id, item_info in panel._tree_items.items():
            if item_info["type"] == "benchmark" and item_info.get("category") == preset:
                item_info["checked"] = True
                panel.bench_tree.item(item_id, text="☑")
    
    # Update category checkmarks
    for item_id, item_info in panel._tree_items.items():
        if item_info["type"] == "category":
            children = panel.bench_tree.get_children(item_id)
            if children:
                all_checked = all(
                    panel._tree_items[child]["checked"]
                    for child in children
                    if child in panel._tree_items
                )
                item_info["checked"] = all_checked
                panel.bench_tree.item(item_id, text="☑" if all_checked else "☐")
    
    update_selected_benchmarks(panel)
