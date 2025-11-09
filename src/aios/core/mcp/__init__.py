"""MCP (Model Context Protocol) integration for AI-OS.

This module provides the interface between AI models and MCP servers,
enabling models to use external tools like file operations, web search,
and knowledge graph management.
"""

from __future__ import annotations

from .client import MCPClient
from .tool_executor import ToolExecutor

__all__ = ["MCPClient", "ToolExecutor"]
