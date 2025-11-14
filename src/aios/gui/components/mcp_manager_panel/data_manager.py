"""Data management for MCP servers and tool configurations."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_servers_config(config_path: str, default_callback: Any, error_callback: Any) -> List[Dict[str, Any]]:
    """Load MCP servers configuration from disk.
    
    Args:
        config_path: Path to mcp_servers.json
        default_callback: Callback to get default servers if file doesn't exist
        error_callback: Callback for logging errors
        
    Returns:
        List of server configuration dictionaries
    """
    if not os.path.exists(config_path):
        logger.info(f"MCP servers config not found at {config_path}, using defaults")
        return default_callback()
    
    try:
        logger.debug(f"Loading MCP servers config from {config_path}")
        with open(config_path, "r") as f:
            servers = json.load(f)
        logger.info(f"Loaded {len(servers)} MCP server configurations")
        return servers
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in MCP servers config at {config_path}: {e}", exc_info=True)
        error_callback(f"[MCP] Error loading servers config: {e}")
        return default_callback()
    except Exception as e:
        logger.error(f"Failed to load MCP servers config from {config_path}: {e}", exc_info=True)
        error_callback(f"[MCP] Error loading servers config: {e}")
        return default_callback()


def save_servers_config(config_path: str, servers: List[Dict[str, Any]], save_state_callback: Any, error_callback: Any) -> None:
    """Save MCP servers configuration to disk.
    
    Args:
        config_path: Path to mcp_servers.json
        servers: List of server configurations to save
        save_state_callback: Callback to save overall state
        error_callback: Callback for logging errors
    """
    try:
        logger.debug(f"Saving {len(servers)} MCP server configurations to {config_path}")
        with open(config_path, "w") as f:
            json.dump(servers, f, indent=2)
        logger.info(f"Successfully saved MCP servers config to {config_path}")
        save_state_callback()
    except PermissionError as e:
        logger.error(f"Permission denied saving MCP servers config to {config_path}: {e}", exc_info=True)
        error_callback(f"[MCP] Permission denied saving servers config: {e}")
    except Exception as e:
        logger.error(f"Failed to save MCP servers config to {config_path}: {e}", exc_info=True)
        error_callback(f"[MCP] Error saving servers config: {e}")


def get_default_servers(project_root: str) -> List[Dict[str, Any]]:
    """Get default MCP server configurations.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        List of default server configurations
    """
    return [
        {
            "name": "filesystem",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", project_root],
            "enabled": True,
            "description": "Local filesystem access",
            "tools_count": 12
        },
        {
            "name": "memory",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "enabled": True,
            "description": "Persistent knowledge graph",
            "tools_count": 8
        },
        {
            "name": "duckduckgo",
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@wulfic/server-duckduckgo"],
            "enabled": True,
            "description": "Web search via DuckDuckGo",
            "tools_count": 2
        },
        {
            "name": "pylance",
            "type": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server_pylance"],
            "enabled": True,
            "description": "Python code analysis",
            "tools_count": 7
        },
    ]


def load_tools_config(config_path: str, default_callback: Any, error_callback: Any) -> Dict[str, Dict[str, Any]]:
    """Load tool permissions configuration from disk.
    
    Args:
        config_path: Path to tool_permissions.json
        default_callback: Callback to get default tools if file doesn't exist
        error_callback: Callback for logging errors
        
    Returns:
        Dictionary mapping tool names to their configuration
    """
    if not os.path.exists(config_path):
        logger.info(f"Tool permissions config not found at {config_path}, using defaults")
        return default_callback()
    
    try:
        logger.debug(f"Loading tool permissions config from {config_path}")
        with open(config_path, "r") as f:
            tools = json.load(f)
        logger.info(f"Loaded permissions for {len(tools)} tools")
        return tools
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in tool permissions config at {config_path}: {e}", exc_info=True)
        error_callback(f"[Tools] Error loading permissions config: {e}")
        return default_callback()
    except Exception as e:
        logger.error(f"Failed to load tool permissions config from {config_path}: {e}", exc_info=True)
        error_callback(f"[Tools] Error loading permissions config: {e}")
        return default_callback()


def save_tools_config(config_path: str, tools: Dict[str, Dict[str, Any]], save_state_callback: Any, error_callback: Any) -> None:
    """Save tool permissions configuration to disk.
    
    Args:
        config_path: Path to tool_permissions.json
        tools: Dictionary of tool configurations to save
        save_state_callback: Callback to save overall state
        error_callback: Callback for logging errors
    """
    try:
        logger.debug(f"Saving permissions for {len(tools)} tools to {config_path}")
        with open(config_path, "w") as f:
            json.dump(tools, f, indent=2)
        logger.info(f"Successfully saved tool permissions config to {config_path}")
        save_state_callback()
    except PermissionError as e:
        logger.error(f"Permission denied saving tool permissions config to {config_path}: {e}", exc_info=True)
        error_callback(f"[Tools] Permission denied saving permissions config: {e}")
    except Exception as e:
        logger.error(f"Failed to save tool permissions config to {config_path}: {e}", exc_info=True)
        error_callback(f"[Tools] Error saving permissions config: {e}")


def get_default_tools() -> Dict[str, Dict[str, Any]]:
    """Get default tool permissions (all tools enabled).
    
    Returns:
        Dictionary mapping tool names to their default configuration
    """
    return {
        # File Operations
        "read_file": {"enabled": True, "category": "File Operations", "description": "Read file contents", "risk": "Low", "usage_count": 0},
        "write_file": {"enabled": True, "category": "File Operations", "description": "Write/create files", "risk": "Medium", "usage_count": 0},
        "edit_file": {"enabled": True, "category": "File Operations", "description": "Edit existing files", "risk": "Medium", "usage_count": 0},
        "list_directory": {"enabled": True, "category": "File Operations", "description": "List directory contents", "risk": "Low", "usage_count": 0},
        "create_directory": {"enabled": True, "category": "File Operations", "description": "Create directories", "risk": "Low", "usage_count": 0},
        "move_file": {"enabled": True, "category": "File Operations", "description": "Move/rename files", "risk": "Medium", "usage_count": 0},
        "search_files": {"enabled": True, "category": "File Operations", "description": "Search for files", "risk": "Low", "usage_count": 0},
        "get_file_info": {"enabled": True, "category": "File Operations", "description": "Get file metadata", "risk": "Low", "usage_count": 0},
        
        # Web & Search
        "web_search": {"enabled": True, "category": "Web & Search", "description": "Search the web (DuckDuckGo)", "risk": "Low", "usage_count": 0},
        "fetch_webpage": {"enabled": True, "category": "Web & Search", "description": "Fetch webpage content", "risk": "Low", "usage_count": 0},
        
        # Memory & Knowledge
        "create_entities": {"enabled": True, "category": "Memory & Knowledge", "description": "Create knowledge graph entities", "risk": "Low", "usage_count": 0},
        "create_relations": {"enabled": True, "category": "Memory & Knowledge", "description": "Create entity relations", "risk": "Low", "usage_count": 0},
        "add_observations": {"enabled": True, "category": "Memory & Knowledge", "description": "Add observations to entities", "risk": "Low", "usage_count": 0},
        "search_nodes": {"enabled": True, "category": "Memory & Knowledge", "description": "Search knowledge graph", "risk": "Low", "usage_count": 0},
        "read_graph": {"enabled": True, "category": "Memory & Knowledge", "description": "Read entire knowledge graph", "risk": "Low", "usage_count": 0},
        "delete_entities": {"enabled": False, "category": "Memory & Knowledge", "description": "Delete knowledge entities", "risk": "High", "usage_count": 0},
        "delete_relations": {"enabled": False, "category": "Memory & Knowledge", "description": "Delete entity relations", "risk": "Medium", "usage_count": 0},
        
        # Code & Development
        "run_code_snippet": {"enabled": True, "category": "Code & Development", "description": "Execute Python code snippets", "risk": "High", "usage_count": 0},
        "check_syntax_errors": {"enabled": True, "category": "Code & Development", "description": "Check Python syntax", "risk": "Low", "usage_count": 0},
        "get_imports": {"enabled": True, "category": "Code & Development", "description": "Analyze Python imports", "risk": "Low", "usage_count": 0},
        "invoke_refactoring": {"enabled": True, "category": "Code & Development", "description": "Apply code refactorings", "risk": "Medium", "usage_count": 0},
        
        # System & Terminal
        "run_terminal_command": {"enabled": True, "category": "System & Terminal", "description": "Execute terminal commands", "risk": "High", "usage_count": 0},
        "get_terminal_output": {"enabled": True, "category": "System & Terminal", "description": "Get command output", "risk": "Low", "usage_count": 0},
        "run_task": {"enabled": True, "category": "System & Terminal", "description": "Run VS Code tasks", "risk": "Medium", "usage_count": 0},
        
        # Data Analysis
        "semantic_search": {"enabled": True, "category": "Data Analysis", "description": "Semantic code search", "risk": "Low", "usage_count": 0},
        "grep_search": {"enabled": True, "category": "Data Analysis", "description": "Text pattern search", "risk": "Low", "usage_count": 0},
        "list_code_usages": {"enabled": True, "category": "Data Analysis", "description": "Find symbol references", "risk": "Low", "usage_count": 0},
    }
