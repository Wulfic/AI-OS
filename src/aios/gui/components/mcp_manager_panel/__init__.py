"""MCP Servers & Tools Manager Panel component.

Provides a visual interface for:
- Managing MCP (Model Context Protocol) servers
- Configuring server connections and credentials
- Enabling/disabling individual AI tools
- Categorizing tools by function
- Viewing tool usage statistics

Usage:
    from aios.gui.components.mcp_manager_panel import MCPManagerPanel
    
    panel = MCPManagerPanel(
        parent,
        append_out=my_output_callback,
        save_state_fn=my_save_callback
    )
"""

from __future__ import annotations

from .panel_main import MCPManagerPanel
from .server_dialog import MCPServerDialog

__all__ = ["MCPManagerPanel", "MCPServerDialog"]
