"""MCP Client for communicating with MCP servers."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers.
    
    This client manages connections to MCP servers and handles
    tool discovery and execution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP client.
        
        Args:
            config_path: Path to mcp_servers.json config file
        """
        self.config_path = config_path or self._default_config_path()
        self.servers: List[Dict[str, Any]] = []
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    @staticmethod
    def _default_config_path() -> str:
        """Get default path to MCP servers config."""
        return str(Path(__file__).parent.parent.parent.parent.parent / "config" / "mcp_servers.json")
    
    def load_config(self) -> None:
        """Load MCP server configuration."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"MCP config not found: {self.config_path}")
            return
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self.servers = json.load(f)
            
            # Filter enabled servers
            self.servers = [s for s in self.servers if s.get("enabled", False)]
            
            logger.info(f"Loaded {len(self.servers)} enabled MCP servers")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
    
    async def discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """Discover available tools from all enabled MCP servers.
        
        Returns:
            Dict mapping tool names to their schemas
        """
        if not self._loaded:
            self.load_config()
        
        # For now, return static tool definitions based on known servers
        # TODO: Implement actual MCP protocol communication
        tools = {}
        
        for server in self.servers:
            server_name = server.get("name", "")
            
            if server_name == "filesystem":
                tools.update(self._get_filesystem_tools())
            elif server_name == "memory":
                tools.update(self._get_memory_tools())
            elif server_name == "duckduckgo":
                tools.update(self._get_duckduckgo_tools())
        
        self.available_tools = tools
        return tools
    
    def _get_filesystem_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get filesystem tool definitions."""
        return {
            "read_file": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"]
                }
            },
            "write_file": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path", "content"]
                }
            },
            "list_directory": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the directory"}
                    },
                    "required": ["path"]
                }
            },
            "search_files": {
                "name": "search_files",
                "description": "Search for files matching a pattern",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern to match"},
                        "path": {"type": "string", "description": "Starting directory path"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    
    def _get_memory_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get memory/knowledge graph tool definitions."""
        return {
            "create_entities": {
                "name": "create_entities",
                "description": "Create entities in the knowledge graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "entityType": {"type": "string"},
                                    "observations": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    },
                    "required": ["entities"]
                }
            },
            "search_nodes": {
                "name": "search_nodes",
                "description": "Search for nodes in the knowledge graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    
    def _get_duckduckgo_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get DuckDuckGo search tool definitions."""
        return {
            "web_search": {
                "name": "web_search",
                "description": "Search the web using DuckDuckGo",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum number of results", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            "fetch_webpage": {
                "name": "fetch_webpage",
                "description": "Fetch and parse content from a webpage",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        }
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call via MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
        
        # TODO: Implement actual MCP server communication
        # For now, return mock responses for testing
        logger.info(f"MCP tool call: {tool_name} with params: {parameters}")
        
        return {
            "success": True,
            "tool": tool_name,
            "result": f"[Mock] Tool {tool_name} would be executed with: {json.dumps(parameters)}"
        }
    
    def get_tools_for_prompt(self) -> List[Dict[str, Any]]:
        """Get tool definitions formatted for model prompts.
        
        Returns:
            List of tool definitions suitable for model context
        """
        if not self.available_tools:
            # Try to discover if not already done
            try:
                asyncio.run(self.discover_tools())
            except Exception as e:
                logger.warning(f"Failed to discover tools: {e}")
                return []
        
        return list(self.available_tools.values())
