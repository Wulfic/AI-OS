"""Test MCP server integration without a trained model.

This module provides utilities to test MCP servers directly
without requiring a fully trained model.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


class MCPServerTester:
    """Test MCP servers by launching them and checking tool availability."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
    def get_server_config(self) -> List[Dict[str, Any]]:
        """Load MCP server configuration."""
        config_path = self.project_root / "config" / "mcp_servers.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def test_server_launch(self, server: Dict[str, Any]) -> Dict[str, Any]:
        """Test if an MCP server can be launched.
        
        Args:
            server: Server configuration dict
            
        Returns:
            Dict with test results
        """
        if not server.get("enabled", False):
            return {
                "name": server.get("name"),
                "status": "skipped",
                "reason": "Server disabled in config"
            }
        
        command = server.get("command", "")
        args = server.get("args", [])
        
        if not command:
            return {
                "name": server.get("name"),
                "status": "error",
                "reason": "No command specified"
            }
        
        try:
            # Try to launch with --help to verify it exists
            full_cmd = [command] + args + ["--help"]
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "name": server.get("name"),
                "status": "ok" if result.returncode == 0 else "warning",
                "command": f"{command} {' '.join(args)}",
                "output": result.stdout[:200] if result.stdout else result.stderr[:200]
            }
            
        except FileNotFoundError:
            return {
                "name": server.get("name"),
                "status": "error",
                "reason": f"Command not found: {command}"
            }
        except subprocess.TimeoutExpired:
            return {
                "name": server.get("name"),
                "status": "warning",
                "reason": "Command timed out (may require interaction)"
            }
        except Exception as e:
            return {
                "name": server.get("name"),
                "status": "error",
                "reason": str(e)
            }
    
    def test_all_servers(self) -> Dict[str, Any]:
        """Test all configured MCP servers.
        
        Returns:
            Summary of test results
        """
        servers = self.get_server_config()
        results = []
        
        for server in servers:
            result = self.test_server_launch(server)
            results.append(result)
            
        summary = {
            "total": len(results),
            "ok": sum(1 for r in results if r["status"] == "ok"),
            "warning": sum(1 for r in results if r["status"] == "warning"),
            "error": sum(1 for r in results if r["status"] == "error"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "results": results
        }
        
        return summary


def test_tool_permissions_config():
    """Test that tool permissions config is valid."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "tool_permissions.json"
    
    assert config_path.exists(), "tool_permissions.json not found"
    
    with open(config_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    # Verify structure
    assert isinstance(tools, dict), "Tools config should be a dict"
    assert len(tools) > 0, "Should have at least one tool defined"
    
    # Verify each tool has required fields
    required_fields = ["enabled", "category", "description", "risk"]
    for tool_name, tool_config in tools.items():
        for field in required_fields:
            assert field in tool_config, f"Tool {tool_name} missing field: {field}"
        
        assert isinstance(tool_config["enabled"], bool)
        assert tool_config["category"] in [
            "File Operations",
            "Web & Search",
            "Memory & Knowledge",
            "Code & Development",
            "System & Terminal",
            "Data Analysis"
        ], f"Invalid category for {tool_name}: {tool_config['category']}"
        assert tool_config["risk"] in ["Low", "Medium", "High"]


def test_mcp_servers_config():
    """Test that MCP servers config is valid."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "mcp_servers.json"
    
    if not config_path.exists():
        pytest.skip("mcp_servers.json not found - may not be initialized yet")
    
    with open(config_path, "r", encoding="utf-8") as f:
        servers = json.load(f)
    
    # Verify structure
    assert isinstance(servers, list), "Servers config should be a list"
    
    # Verify each server has required fields
    required_fields = ["name", "type", "command", "enabled"]
    for server in servers:
        for field in required_fields:
            assert field in server, f"Server {server.get('name', 'unknown')} missing field: {field}"
        
        assert server["type"] in ["stdio", "http", "https"]


def test_mcp_server_availability():
    """Test if MCP servers can be launched."""
    project_root = Path(__file__).parent.parent
    tester = MCPServerTester(str(project_root))
    
    summary = tester.test_all_servers()
    
    print("\n=== MCP Server Test Summary ===")
    print(f"Total: {summary['total']}")
    print(f"OK: {summary['ok']}")
    print(f"Warning: {summary['warning']}")
    print(f"Error: {summary['error']}")
    print(f"Skipped: {summary['skipped']}")
    print("\nDetails:")
    for result in summary["results"]:
        print(f"\n{result['name']}: {result['status']}")
        if "command" in result:
            print(f"  Command: {result['command']}")
        if "reason" in result:
            print(f"  Reason: {result['reason']}")


if __name__ == "__main__":
    # Run tests directly
    test_tool_permissions_config()
    print("✅ Tool permissions config is valid\n")
    
    test_mcp_servers_config()
    print("✅ MCP servers config is valid\n")
    
    test_mcp_server_availability()
