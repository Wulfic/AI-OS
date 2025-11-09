"""Event handlers for tool permissions management."""

from __future__ import annotations

from typing import Any, Dict, cast

try:  # pragma: no cover - environment dependent
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    messagebox = cast(Any, None)


def handle_tool_toggle(tools_tree: Any, tools_loader: Any, tools_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Toggle enabled status of selected tool.
    
    Args:
        tools_tree: Treeview widget with tool selection
        tools_loader: Callback to load current tools
        tools_saver: Callback to save tools
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    selection = tools_tree.selection()
    if not selection:
        return
    
    item = selection[0]
    # Skip if it's a category header
    if not tools_tree.parent(item):  # Root level item (category)
        return
    
    values = list(tools_tree.item(item, "values"))
    tool_name = values[1]
    current_status = values[0]
    
    # Toggle status
    new_status = "Disabled" if current_status == "✅ Enabled" else "Enabled"
    enabled = (new_status == "Enabled")
    
    # Update in config
    tools = tools_loader()
    if tool_name in tools:
        tools[tool_name]["enabled"] = enabled
        tools_saver(tools)
        refresh_callback()
        log_callback(f"[Tools] {new_status}: {tool_name}")


def handle_enable_all_tools(category_filter: str, tools_loader: Any, tools_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Enable all tools in current filter.
    
    Args:
        category_filter: Current category filter value
        tools_loader: Callback to load current tools
        tools_saver: Callback to save tools
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    if messagebox and messagebox.askyesno("Confirm", "Enable all visible tools?"):
        _set_all_tools_status(True, category_filter, tools_loader, tools_saver, refresh_callback, log_callback)


def handle_disable_all_tools(category_filter: str, tools_loader: Any, tools_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Disable all tools in current filter.
    
    Args:
        category_filter: Current category filter value
        tools_loader: Callback to load current tools
        tools_saver: Callback to save tools
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    if messagebox and messagebox.askyesno("Confirm", "Disable all visible tools?"):
        _set_all_tools_status(False, category_filter, tools_loader, tools_saver, refresh_callback, log_callback)


def handle_reset_tools(default_tools_getter: Any, tools_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Reset tool permissions to defaults.
    
    Args:
        default_tools_getter: Callback to get default tools
        tools_saver: Callback to save tools
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    if messagebox and messagebox.askyesno("Confirm", "Reset all tool permissions to defaults?\nThis will re-enable all tools."):
        # Get default tools and reset all to enabled
        tools = default_tools_getter()
        for tool_name in tools:
            tools[tool_name]["enabled"] = True
        
        tools_saver(tools)
        refresh_callback()
        log_callback("[Tools] Reset to default permissions")


def _set_all_tools_status(enabled: bool, category_filter: str, tools_loader: Any, tools_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Set enabled status for all filtered tools.
    
    Args:
        enabled: Whether to enable or disable
        category_filter: Current category filter value
        tools_loader: Callback to load current tools
        tools_saver: Callback to save tools
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    tools = tools_loader()
    
    count = 0
    for tool_name, tool_data in tools.items():
        if category_filter == "All" or tool_data.get("category") == category_filter:
            tools[tool_name]["enabled"] = enabled
            count += 1
    
    tools_saver(tools)
    refresh_callback()
    status = "enabled" if enabled else "disabled"
    log_callback(f"[Tools] {status.capitalize()} {count} tools")
