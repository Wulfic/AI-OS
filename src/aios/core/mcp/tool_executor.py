"""Tool execution coordinator for MCP integration."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from .client import MCPClient

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Coordinates tool execution and result processing for chat.
    
    This class bridges between model outputs and MCP tool calls,
    handling tool call detection, execution, and result formatting.
    """
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        """Initialize tool executor.
        
        Args:
            mcp_client: MCP client instance (creates new one if None)
        """
        self.mcp_client = mcp_client or MCPClient()
        self.enabled = False
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the tool executor."""
        try:
            self.mcp_client.load_config()
            asyncio.run(self.mcp_client.discover_tools())
            self.enabled = len(self.mcp_client.available_tools) > 0
            if self.enabled:
                logger.info(f"Tool executor enabled with {len(self.mcp_client.available_tools)} tools")
        except Exception as e:
            logger.warning(f"Tool executor initialization failed: {e}")
            self.enabled = False
    
    def parse_tool_call(self, model_output: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from model output.
        
        Supports multiple formats:
        1. JSON: {"tool": "name", "parameters": {...}}
        2. XML-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        3. Function call style: tool_name(param1="value", param2="value")
        
        Args:
            model_output: Raw model output text
            
        Returns:
            Parsed tool call dict or None if no valid call found
        """
        # Try JSON format first
        try:
            # Look for JSON object in the output
            json_match = re.search(r'\{[^}]+\}', model_output)
            if json_match:
                data = json.loads(json_match.group(0))
                if "tool" in data and "parameters" in data:
                    return {
                        "tool": data["tool"],
                        "parameters": data["parameters"]
                    }
        except Exception:
            pass
        
        # Try XML-style tool_call tags
        xml_match = re.search(r'<tool_call>\s*(\{.+?\})\s*</tool_call>', model_output, re.DOTALL)
        if xml_match:
            try:
                data = json.loads(xml_match.group(1))
                return {
                    "tool": data.get("name"),
                    "parameters": data.get("arguments", {})
                }
            except Exception:
                pass
        
        # Try function call style
        func_match = re.search(r'(\w+)\(([^)]+)\)', model_output)
        if func_match:
            tool_name = func_match.group(1)
            if tool_name in self.mcp_client.available_tools:
                # Parse simple key=value parameters
                params_str = func_match.group(2)
                parameters = {}
                for param in params_str.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        parameters[key] = value
                
                return {
                    "tool": tool_name,
                    "parameters": parameters
                }
        
        return None
    
    async def execute_if_tool_call(self, model_output: str) -> Optional[Dict[str, Any]]:
        """Check if model output contains a tool call and execute it.
        
        Args:
            model_output: Model's output text
            
        Returns:
            Tool execution result if tool call detected, None otherwise
        """
        if not self.enabled:
            return None
        
        tool_call = self.parse_tool_call(model_output)
        if not tool_call:
            return None
        
        try:
            result = await self.mcp_client.execute_tool(
                tool_call["tool"],
                tool_call["parameters"]
            )
            
            return {
                "is_tool_call": True,
                "tool_name": tool_call["tool"],
                "parameters": tool_call["parameters"],
                "result": result
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "is_tool_call": True,
                "tool_name": tool_call["tool"],
                "parameters": tool_call["parameters"],
                "error": str(e)
            }
    
    def format_tool_result_for_model(self, tool_result: Dict[str, Any]) -> str:
        """Format tool execution result for model context.
        
        Args:
            tool_result: Result from execute_if_tool_call
            
        Returns:
            Formatted string to send back to model
        """
        if not tool_result.get("is_tool_call"):
            return ""
        
        tool_name = tool_result.get("tool_name", "unknown")
        
        if "error" in tool_result:
            return f"Tool {tool_name} failed: {tool_result['error']}"
        
        result = tool_result.get("result", {})
        if isinstance(result, dict):
            return f"Tool {tool_name} result: {json.dumps(result, indent=2)}"
        
        return f"Tool {tool_name} result: {result}"
    
    def get_system_prompt_addition(self) -> str:
        """Get system prompt text to add for tool support.
        
        Returns:
            System prompt addition explaining available tools
        """
        if not self.enabled:
            return ""
        
        tools = self.mcp_client.get_tools_for_prompt()
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            tool_descriptions.append(f"- {name}: {desc}")
        
        return f"""
You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with a JSON object in this format:
{{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}

Or use XML-style tags:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

Only use tools when necessary to answer the user's question.
"""
