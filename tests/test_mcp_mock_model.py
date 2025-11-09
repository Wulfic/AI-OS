"""Test MCP tool integration using a mock model.

This creates a simple mock model that can exercise MCP tools
without requiring a fully trained model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MockToolCallingModel:
    """Mock model that simulates tool calling behavior.
    
    This allows testing the MCP integration pipeline without
    needing a fully trained model.
    """
    
    def __init__(self, available_tools: List[Dict[str, Any]]):
        self.available_tools = {t["name"]: t for t in available_tools}
        self.call_history: List[Dict[str, Any]] = []
    
    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self.available_tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a specific tool."""
        return self.available_tools.get(tool_name, {})
    
    def generate_tool_call(self, user_message: str) -> Dict[str, Any]:
        """Simulate model generating a tool call based on user message.
        
        This is a simple rule-based mock - a real model would use
        its learned parameters.
        """
        message_lower = user_message.lower()
        
        # Simple keyword matching for demonstration
        if "read" in message_lower and ("file" in message_lower or "readme" in message_lower):
            return {
                "tool": "read_file",
                "parameters": {
                    "path": "README.md"
                }
            }
        
        if "search" in message_lower and "web" in message_lower:
            # Extract search query (simplified)
            query = user_message.replace("search the web for", "").strip()
            return {
                "tool": "web_search",
                "parameters": {
                    "query": query,
                    "max_results": 5
                }
            }
        
        if "list" in message_lower and ("directory" in message_lower or "files" in message_lower):
            return {
                "tool": "list_directory",
                "parameters": {
                    "path": "."
                }
            }
        
        if "create" in message_lower and "entity" in message_lower:
            return {
                "tool": "create_entities",
                "parameters": {
                    "entities": [
                        {
                            "name": "test-entity",
                            "entityType": "concept",
                            "observations": ["Created from test"]
                        }
                    ]
                }
            }
        
        # Default: return no tool call
        return {
            "tool": None,
            "text_response": f"I understand you said: '{user_message}'. I don't have a specific tool for this request yet."
        }
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Mock executing a tool call.
        
        In a real implementation, this would communicate with MCP servers.
        """
        tool_name = tool_call.get("tool")
        parameters = tool_call.get("parameters", {})
        
        # Record the call
        self.call_history.append({
            "tool": tool_name,
            "parameters": parameters
        })
        
        # Mock responses for different tools
        if tool_name == "read_file":
            return {
                "success": True,
                "content": "# Mock file content\nThis is simulated file content.",
                "tool": tool_name
            }
        
        if tool_name == "web_search":
            return {
                "success": True,
                "results": [
                    {"title": "Mock Result 1", "url": "https://example.com/1"},
                    {"title": "Mock Result 2", "url": "https://example.com/2"}
                ],
                "tool": tool_name
            }
        
        if tool_name == "list_directory":
            return {
                "success": True,
                "files": ["file1.txt", "file2.py", "subdirectory/"],
                "tool": tool_name
            }
        
        if tool_name == "create_entities":
            return {
                "success": True,
                "entities_created": len(parameters.get("entities", [])),
                "tool": tool_name
            }
        
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Full chat cycle: generate tool call, execute, return response."""
        # Step 1: Model decides if it needs a tool
        tool_call = self.generate_tool_call(user_message)
        
        # Step 2: If tool is needed, execute it
        if tool_call.get("tool"):
            tool_result = self.execute_tool_call(tool_call)
            
            # Step 3: Model uses tool result to generate final response
            return {
                "response": f"I used the {tool_call['tool']} tool and got: {json.dumps(tool_result, indent=2)}",
                "tool_call": tool_call,
                "tool_result": tool_result
            }
        else:
            return {
                "response": tool_call.get("text_response", "I'm not sure how to help with that."),
                "tool_call": None,
                "tool_result": None
            }


def load_available_tools() -> List[Dict[str, Any]]:
    """Load available tools from config."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "tool_permissions.json"
    
    if not config_path.exists():
        return []
    
    with open(config_path, "r", encoding="utf-8") as f:
        tools_config = json.load(f)
    
    # Convert to list format
    tools = []
    for name, config in tools_config.items():
        if config.get("enabled", False):
            tools.append({
                "name": name,
                "description": config.get("description", ""),
                "category": config.get("category", ""),
                "risk": config.get("risk", "")
            })
    
    return tools


def test_mock_model():
    """Test the mock model's tool calling behavior."""
    tools = load_available_tools()
    model = MockToolCallingModel(tools)
    
    # Test various user messages
    test_messages = [
        "Read the README file",
        "Search the web for model context protocol",
        "List files in the current directory",
        "Create a test entity in the knowledge graph",
        "What's the weather like?"  # Should not trigger tool
    ]
    
    print("=== Mock Model Tool Calling Test ===\n")
    
    for message in test_messages:
        print(f"User: {message}")
        result = model.chat(message)
        print(f"Model: {result['response']}")
        if result['tool_call']:
            print(f"  Tool: {result['tool_call']['tool']}")
            print(f"  Parameters: {json.dumps(result['tool_call']['parameters'], indent=4)}")
        print()
    
    # Summary
    print(f"\nTotal tool calls made: {len(model.call_history)}")
    print("Tool usage:")
    tool_counts = {}
    for call in model.call_history:
        tool = call['tool']
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count}")


if __name__ == "__main__":
    test_mock_model()
