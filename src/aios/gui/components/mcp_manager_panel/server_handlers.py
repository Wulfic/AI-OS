"""Event handlers for MCP server management."""

from __future__ import annotations

from typing import Any, Optional, cast

try:  # pragma: no cover - environment dependent
    from tkinter import messagebox  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    messagebox = cast(Any, None)


def handle_add_server(parent: Any, dialog_class: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Handle adding a new MCP server.
    
    Args:
        parent: Parent widget for dialog
        dialog_class: MCPServerDialog class
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    dialog = dialog_class(parent, title="Add MCP Server")
    parent.wait_window(dialog)
    
    if dialog.result:
        servers = servers_loader()
        servers.append(dialog.result)
        servers_saver(servers)
        refresh_callback()
        log_callback(f"[MCP] Added server: {dialog.result['name']}")


def handle_edit_server(parent: Any, servers_tree: Any, dialog_class: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Handle editing selected server.
    
    Args:
        parent: Parent widget for dialog
        servers_tree: Treeview widget with server selection
        dialog_class: MCPServerDialog class
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    selection = servers_tree.selection()
    if not selection:
        if messagebox:
            messagebox.showwarning("No Selection", "Please select a server to edit")
        return
    
    item = selection[0]
    values = servers_tree.item(item, "values")
    name = values[0]
    
    # Find server in config
    servers = servers_loader()
    server = next((s for s in servers if s["name"] == name), None)
    
    if server:
        dialog = dialog_class(parent, title="Edit MCP Server", initial_data=server)
        parent.wait_window(dialog)
        
        if dialog.result:
            # Update server
            idx = servers.index(server)
            servers[idx] = dialog.result
            servers_saver(servers)
            refresh_callback()
            log_callback(f"[MCP] Updated server: {dialog.result['name']}")


def handle_delete_server(servers_tree: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Handle deleting selected server.
    
    Args:
        servers_tree: Treeview widget with server selection
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    selection = servers_tree.selection()
    if not selection:
        if messagebox:
            messagebox.showwarning("No Selection", "Please select a server to delete")
        return
    
    item = selection[0]
    values = servers_tree.item(item, "values")
    name = values[0]
    
    if messagebox and messagebox.askyesno("Confirm Delete", f"Delete server '{name}'?"):
        servers = servers_loader()
        servers = [s for s in servers if s["name"] != name]
        servers_saver(servers)
        refresh_callback()
        log_callback(f"[MCP] Deleted server: {name}")


def handle_test_server(servers_tree: Any, log_callback: Any) -> None:
    """Handle testing server connection.
    
    Args:
        servers_tree: Treeview widget with server selection
        log_callback: Callback for logging
    """
    selection = servers_tree.selection()
    if not selection:
        if messagebox:
            messagebox.showwarning("No Selection", "Please select a server to test")
        return
    
    item = selection[0]
    values = servers_tree.item(item, "values")
    name = values[0]
    
    # Connection testing placeholder - full implementation pending
    if messagebox:
        messagebox.showinfo(
            "Test Connection",
            f"Testing connection to '{name}'...\n\n"
            "Connection testing will verify authentication, API availability, and tool discovery."
        )
    log_callback(f"[MCP] Connection test initiated for: {name}")

def handle_enable_server(servers_tree: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Enable selected server.
    
    Args:
        servers_tree: Treeview widget with server selection
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    _set_server_status(True, servers_tree, servers_loader, servers_saver, refresh_callback, log_callback)


def handle_disable_server(servers_tree: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Disable selected server.
    
    Args:
        servers_tree: Treeview widget with server selection
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    _set_server_status(False, servers_tree, servers_loader, servers_saver, refresh_callback, log_callback)


def _set_server_status(enabled: bool, servers_tree: Any, servers_loader: Any, servers_saver: Any, refresh_callback: Any, log_callback: Any) -> None:
    """Set enabled status for selected server.
    
    Args:
        enabled: Whether to enable or disable
        servers_tree: Treeview widget with server selection
        servers_loader: Callback to load current servers
        servers_saver: Callback to save servers
        refresh_callback: Callback to refresh UI
        log_callback: Callback for logging
    """
    selection = servers_tree.selection()
    if not selection:
        if messagebox:
            messagebox.showwarning("No Selection", "Please select a server")
        return
    
    item = selection[0]
    values = servers_tree.item(item, "values")
    name = values[0]
    
    servers = servers_loader()
    server = next((s for s in servers if s["name"] == name), None)
    
    if server:
        server["enabled"] = enabled
        idx = servers.index(server)
        servers[idx] = server
        servers_saver(servers)
        refresh_callback()
        status = "enabled" if enabled else "disabled"
        log_callback(f"[MCP] Server {status}: {name}")
