"""UI builders for MCP Manager Panel components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk
    from tkinter import ttk


def create_header(
    parent: Any,
    servers_count_var: "tk.StringVar",
    servers_active_var: "tk.StringVar",
    tools_enabled_var: "tk.StringVar",
    refresh_callback: Any,
) -> None:
    """Create header with quick stats.
    
    Args:
        parent: Parent widget
        servers_count_var: StringVar for total servers count
        servers_active_var: StringVar for active servers count
        tools_enabled_var: StringVar for enabled tools count
        refresh_callback: Callback for refresh button
    """
    try:
        from tkinter import ttk
        from ..tooltips import add_tooltip
    except Exception:
        return
    
    header = ttk.Frame(parent)
    header.pack(fill="x", pady=(0, 8))
    
    # MCP servers status
    ttk.Label(header, text="MCP Servers:").pack(side="left")
    servers_count_label = ttk.Label(header, textvariable=servers_count_var, width=6)
    servers_count_label.pack(side="left")
    add_tooltip(servers_count_label, "Total number of configured MCP servers")
    
    # Active servers
    ttk.Label(header, text="Active:").pack(side="left", padx=(12, 0))
    servers_active_label = ttk.Label(header, textvariable=servers_active_var, width=6, foreground="green")
    servers_active_label.pack(side="left")
    add_tooltip(servers_active_label, "Number of currently enabled servers")
    
    # Enabled tools
    ttk.Label(header, text="Tools Enabled:").pack(side="left", padx=(12, 0))
    tools_enabled_label = ttk.Label(header, textvariable=tools_enabled_var, width=6)
    tools_enabled_label.pack(side="left")
    add_tooltip(tools_enabled_label, "Number of enabled tools available for use")
    
    # Refresh button
    refresh_btn = ttk.Button(header, text="🔄 Refresh", command=refresh_callback)
    refresh_btn.pack(side="right")
    add_tooltip(refresh_btn, "Reload MCP server and tool configurations from disk")


def create_notebook(parent: Any, servers_section_callback: Any, tools_section_callback: Any) -> tuple[Any, Any, Any]:
    """Create tabbed interface for servers and tools.
    
    Args:
        parent: Parent widget
        servers_section_callback: Callback to create servers section
        tools_section_callback: Callback to create tools section
        
    Returns:
        Tuple of (notebook, servers_tab, tools_tab)
    """
    try:
        from tkinter import ttk
    except Exception:
        return None, None, None
    
    notebook = ttk.Notebook(parent)
    notebook.pack(fill="both", expand=True)
    
    # MCP Servers tab
    servers_tab = ttk.Frame(notebook)
    notebook.add(servers_tab, text="MCP Servers")
    servers_section_callback(servers_tab)
    
    # Tool Permissions tab
    tools_tab = ttk.Frame(notebook)
    notebook.add(tools_tab, text="Tool Permissions")
    tools_section_callback(tools_tab)
    
    return notebook, servers_tab, tools_tab


def create_servers_section(
    parent: Any,
    on_add: Any,
    on_edit: Any,
    on_delete: Any,
    on_test: Any,
    on_enable: Any,
    on_disable: Any,
    on_select: Any,
) -> Any:
    """Create MCP servers management interface.
    
    Args:
        parent: Parent widget
        on_add: Callback for add button
        on_edit: Callback for edit button
        on_delete: Callback for delete button
        on_test: Callback for test button
        on_enable: Callback for enable button
        on_disable: Callback for disable button
        on_select: Callback for selection event
        
    Returns:
        The servers Treeview widget
    """
    try:
        from tkinter import ttk
        from ..tooltips import add_tooltip
    except Exception:
        return None
    
    # Toolbar
    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 8))
    
    add_btn = ttk.Button(toolbar, text="➕ Add Server", command=on_add)
    add_btn.pack(side="left", padx=(0, 5))
    add_tooltip(add_btn, "Add a new MCP server connection")
    
    edit_btn = ttk.Button(toolbar, text="✏️ Edit", command=on_edit)
    edit_btn.pack(side="left", padx=(0, 5))
    add_tooltip(edit_btn, "Edit selected server configuration")
    
    delete_btn = ttk.Button(toolbar, text="🗑️ Delete", command=on_delete)
    delete_btn.pack(side="left", padx=(0, 5))
    add_tooltip(delete_btn, "Remove selected server")
    
    test_btn = ttk.Button(toolbar, text="🔌 Test Connection", command=on_test)
    test_btn.pack(side="left", padx=(0, 5))
    add_tooltip(test_btn, "Test connection to selected server")
    
    ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
    
    enable_btn = ttk.Button(toolbar, text="✅ Enable", command=on_enable)
    enable_btn.pack(side="left", padx=(0, 5))
    add_tooltip(enable_btn, "Enable selected server")
    
    disable_btn = ttk.Button(toolbar, text="⏸️ Disable", command=on_disable)
    disable_btn.pack(side="left")
    add_tooltip(disable_btn, "Disable selected server")
    
    # Servers tree view
    tree_frame = ttk.Frame(parent)
    tree_frame.pack(fill="both", expand=True)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical")
    vsb.pack(side="right", fill="y")
    
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
    hsb.pack(side="bottom", fill="x")
    
    # Tree view with columns
    servers_tree = ttk.Treeview(
        tree_frame,
        columns=("name", "type", "url", "status", "tools_count", "description"),
        show="headings",
        yscrollcommand=vsb.set,
        xscrollcommand=hsb.set,
    )
    add_tooltip(servers_tree, "Configured MCP servers. Select to edit or test.")
    servers_tree.pack(fill="both", expand=True)
    
    vsb.config(command=servers_tree.yview)
    hsb.config(command=servers_tree.xview)
    
    # Column headers
    servers_tree.heading("name", text="Name")
    servers_tree.heading("type", text="Type")
    servers_tree.heading("url", text="URL/Endpoint")
    servers_tree.heading("status", text="Status")
    servers_tree.heading("tools_count", text="Tools")
    servers_tree.heading("description", text="Description")
    
    # Column widths
    servers_tree.column("name", width=150, minwidth=100)
    servers_tree.column("type", width=100, minwidth=80)
    servers_tree.column("url", width=250, minwidth=150)
    servers_tree.column("status", width=80, minwidth=60)
    servers_tree.column("tools_count", width=60, minwidth=50)
    servers_tree.column("description", width=300, minwidth=150)
    
    # Bind selection
    servers_tree.bind("<<TreeviewSelect>>", on_select)
    
    return servers_tree


def create_tools_section(
    parent: Any,
    category_var: "tk.StringVar",
    on_filter_change: Any,
    on_enable_all: Any,
    on_disable_all: Any,
    on_reset: Any,
    on_tool_toggle: Any,
) -> Any:
    """Create tool permissions interface.
    
    Args:
        parent: Parent widget
        category_var: StringVar for category filter
        on_filter_change: Callback for category filter change
        on_enable_all: Callback for enable all button
        on_disable_all: Callback for disable all button
        on_reset: Callback for reset button
        on_tool_toggle: Callback for tool toggle
        
    Returns:
        The tools Treeview widget
    """
    try:
        from tkinter import ttk
        from ..tooltips import add_tooltip
    except Exception:
        return None
    
    # Toolbar
    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 8))
    
    ttk.Label(toolbar, text="Filter by category:").pack(side="left", padx=(0, 5))
    
    category_combo = ttk.Combobox(
        toolbar,
        textvariable=category_var,
        values=("All", "File Operations", "Web & Search", "Memory & Knowledge", 
                "Code & Development", "System & Terminal", "Data Analysis"),
        state="readonly",
        width=20
    )
    category_combo.pack(side="left", padx=(0, 10))
    category_combo.bind("<<ComboboxSelected>>", on_filter_change)
    add_tooltip(category_combo, "Filter tools by functional category")
    
    ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
    
    enable_all_btn = ttk.Button(toolbar, text="✅ Enable All", command=on_enable_all)
    enable_all_btn.pack(side="left", padx=(0, 5))
    add_tooltip(enable_all_btn, "Enable all tools in current filter")
    
    disable_all_btn = ttk.Button(toolbar, text="⏸️ Disable All", command=on_disable_all)
    disable_all_btn.pack(side="left", padx=(0, 5))
    add_tooltip(disable_all_btn, "Disable all tools in current filter")
    
    reset_btn = ttk.Button(toolbar, text="🔄 Reset to Defaults", command=on_reset)
    reset_btn.pack(side="left")
    add_tooltip(reset_btn, "Reset all tool permissions to default settings")
    
    # Tools tree view with checkboxes
    tree_frame = ttk.Frame(parent)
    tree_frame.pack(fill="both", expand=True)
    
    # Scrollbars
    vsb = ttk.Scrollbar(tree_frame, orient="vertical")
    vsb.pack(side="right", fill="y")
    
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
    hsb.pack(side="bottom", fill="x")
    
    # Tree view with columns
    tools_tree = ttk.Treeview(
        tree_frame,
        columns=("enabled", "name", "category", "description", "risk", "usage_count"),
        show="tree headings",
        yscrollcommand=vsb.set,
        xscrollcommand=hsb.set,
    )
    add_tooltip(tools_tree, "Available MCP tools. Click checkbox to enable/disable individual tools.")
    tools_tree.pack(fill="both", expand=True)
    
    vsb.config(command=tools_tree.yview)
    hsb.config(command=tools_tree.xview)
    
    # Column headers
    tools_tree.heading("#0", text="")  # Checkbox column
    tools_tree.heading("enabled", text="Status")
    tools_tree.heading("name", text="Tool Name")
    tools_tree.heading("category", text="Category")
    tools_tree.heading("description", text="Description")
    tools_tree.heading("risk", text="Risk")
    tools_tree.heading("usage_count", text="Usage")
    
    # Column widths
    tools_tree.column("#0", width=30, minwidth=30, stretch=False)
    tools_tree.column("enabled", width=80, minwidth=60)
    tools_tree.column("name", width=200, minwidth=150)
    tools_tree.column("category", width=150, minwidth=100)
    tools_tree.column("description", width=350, minwidth=200)
    tools_tree.column("risk", width=80, minwidth=60)
    tools_tree.column("usage_count", width=80, minwidth=60)
    
    # Bind double-click to toggle
    tools_tree.bind("<Double-1>", on_tool_toggle)
    tools_tree.bind("<Return>", on_tool_toggle)
    
    return tools_tree
