"""UI update functions for MCP Manager Panel."""

from __future__ import annotations

from typing import Any, Dict, List


def populate_servers_tree(servers_tree: Any, servers: List[Dict[str, Any]]) -> None:
    """Populate MCP servers tree view.
    
    Args:
        servers_tree: Treeview widget to populate
        servers: List of server configuration dictionaries
    """
    # Clear existing
    for item in servers_tree.get_children():
        servers_tree.delete(item)
    
    # Load servers
    for server in servers:
        status = "🟢 Enabled" if server.get("enabled", False) else "⚫ Disabled"
        values = (
            server.get("name", ""),
            server.get("type", "stdio"),
            server.get("command", "") + " " + " ".join(server.get("args", [])),
            status,
            server.get("tools_count", 0),
            server.get("description", ""),
        )
        servers_tree.insert("", "end", values=values)


def populate_tools_tree(tools_tree: Any, tools: Dict[str, Dict[str, Any]], category_filter: str) -> None:
    """Populate tools tree view with category grouping.
    
    Args:
        tools_tree: Treeview widget to populate
        tools: Dictionary mapping tool names to their configuration
        category_filter: Current category filter value
    """
    # Clear existing
    for item in tools_tree.get_children():
        tools_tree.delete(item)
    
    # Group by category
    categories: Dict[str, List[tuple]] = {}
    for tool_name, tool_data in tools.items():
        category = tool_data.get("category", "Other")
        
        if category_filter != "All" and category != category_filter:
            continue
        
        if category not in categories:
            categories[category] = []
        
        enabled = tool_data.get("enabled", True)
        status = "✅ Enabled" if enabled else "⏸️ Disabled"
        risk = tool_data.get("risk", "Low")
        
        # Color code risk
        risk_color = {
            "Low": "green",
            "Medium": "orange",
            "High": "red"
        }.get(risk, "black")
        
        categories[category].append((
            tool_name,
            (
                status,
                tool_name,
                category,
                tool_data.get("description", ""),
                risk,
                tool_data.get("usage_count", 0),
            ),
            risk_color
        ))
    
    # Insert by category
    for category in sorted(categories.keys()):
        # Category header
        cat_id = tools_tree.insert("", "end", text=f"📁 {category}", values=("", "", "", "", "", ""))
        tools_tree.item(cat_id, tags=("category",))
        
        # Tools in category
        for tool_name, values, risk_color in sorted(categories[category], key=lambda x: x[0]):
            item_id = tools_tree.insert(cat_id, "end", text="", values=values)
            if risk_color != "green":
                tools_tree.item(item_id, tags=(f"risk_{risk_color}",))
    
    # Configure tag colors
    tools_tree.tag_configure("category", font=("TkDefaultFont", 10, "bold"))
    tools_tree.tag_configure("risk_orange", foreground="orange")
    tools_tree.tag_configure("risk_red", foreground="red")
    
    # Expand all categories
    for item in tools_tree.get_children():
        tools_tree.item(item, open=True)


def update_summary(
    servers: List[Dict[str, Any]],
    tools: Dict[str, Dict[str, Any]],
    servers_count_var: Any,
    servers_active_var: Any,
    tools_enabled_var: Any,
) -> None:
    """Update header summary statistics.
    
    Args:
        servers: List of server configurations
        tools: Dictionary of tool configurations
        servers_count_var: StringVar for total servers count
        servers_active_var: StringVar for active servers count
        tools_enabled_var: StringVar for enabled tools count
    """
    total_servers = len(servers)
    active_servers = sum(1 for s in servers if s.get("enabled", False))
    enabled_tools = sum(1 for t in tools.values() if t.get("enabled", True))
    
    servers_count_var.set(str(total_servers))
    servers_active_var.set(str(active_servers))
    tools_enabled_var.set(f"{enabled_tools}/{len(tools)}")


def get_enabled_tools(tools: Dict[str, Dict[str, Any]]) -> List[str]:
    """Get list of enabled tool names.
    
    Args:
        tools: Dictionary of tool configurations
        
    Returns:
        List of enabled tool names
    """
    return [name for name, data in tools.items() if data.get("enabled", True)]


def get_active_servers(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get list of active MCP servers.
    
    Args:
        servers: List of server configurations
        
    Returns:
        List of active server configurations
    """
    return [s for s in servers if s.get("enabled", False)]
