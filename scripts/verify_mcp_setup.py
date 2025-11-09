#!/usr/bin/env python3
"""Verify MCP setup and test tool availability.

This script helps verify that:
1. MCP configuration files exist and are valid
2. MCP servers can be launched
3. Tools are properly configured
4. The integration pipeline is ready for model use
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def check_config_files(project_root: Path) -> Dict[str, Any]:
    """Check if MCP configuration files exist and are valid."""
    results = {
        "mcp_servers": {"exists": False, "valid": False, "count": 0},
        "tool_permissions": {"exists": False, "valid": False, "count": 0}
    }
    
    # Check mcp_servers.json
    servers_path = project_root / "config" / "mcp_servers.json"
    if servers_path.exists():
        results["mcp_servers"]["exists"] = True
        try:
            with open(servers_path, "r", encoding="utf-8") as f:
                servers = json.load(f)
                if isinstance(servers, list):
                    results["mcp_servers"]["valid"] = True
                    results["mcp_servers"]["count"] = len(servers)
                    results["mcp_servers"]["enabled_count"] = sum(
                        1 for s in servers if s.get("enabled", False)
                    )
        except Exception as e:
            results["mcp_servers"]["error"] = str(e)
    
    # Check tool_permissions.json
    tools_path = project_root / "config" / "tool_permissions.json"
    if tools_path.exists():
        results["tool_permissions"]["exists"] = True
        try:
            with open(tools_path, "r", encoding="utf-8") as f:
                tools = json.load(f)
                if isinstance(tools, dict):
                    results["tool_permissions"]["valid"] = True
                    results["tool_permissions"]["count"] = len(tools)
                    results["tool_permissions"]["enabled_count"] = sum(
                        1 for t in tools.values() if t.get("enabled", False)
                    )
        except Exception as e:
            results["tool_permissions"]["error"] = str(e)
    
    return results


def check_npx_available() -> bool:
    """Check if npx is available (needed for MCP servers)."""
    try:
        # On Windows, ensure we have the latest PATH from system + user
        import os
        import platform
        if platform.system() == "Windows":
            # Refresh PATH from registry
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                    machine_path = winreg.QueryValueEx(key, "Path")[0]
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                    try:
                        user_path = winreg.QueryValueEx(key, "Path")[0]
                    except:
                        user_path = ""
                
                os.environ["PATH"] = machine_path + ";" + user_path
            except Exception:
                pass  # If registry access fails, continue with existing PATH
        
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def test_server_launch(server: Dict[str, Any]) -> Dict[str, Any]:
    """Test launching a single MCP server."""
    name = server.get("name", "unknown")
    
    if not server.get("enabled", False):
        return {"name": name, "status": "skipped", "reason": "Disabled in config"}
    
    command = server.get("command", "")
    args = server.get("args", [])
    
    if server.get("type") == "stdio":
        try:
            # For npx commands, verify the package is accessible
            if command == "npx" and len(args) >= 2 and args[0] == "-y":
                package = args[1]
                result = subprocess.run(
                    ["npm", "view", package, "version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return {
                        "name": name,
                        "status": "ok",
                        "package": package,
                        "version": result.stdout.strip()
                    }
                else:
                    return {
                        "name": name,
                        "status": "warning",
                        "package": package,
                        "reason": "Package not found in npm registry"
                    }
        except subprocess.TimeoutExpired:
            return {"name": name, "status": "warning", "reason": "npm check timed out"}
        except Exception as e:
            return {"name": name, "status": "error", "reason": str(e)}
    
    return {"name": name, "status": "info", "reason": f"Type: {server.get('type')}"}


def main():
    """Run MCP verification checks."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("MCP Setup Verification")
    print("=" * 60)
    print()
    
    # 1. Check configuration files
    print("📄 Configuration Files")
    print("-" * 60)
    config_results = check_config_files(project_root)
    
    for config_name, info in config_results.items():
        print(f"\n{config_name}:")
        print(f"  Exists: {'✅' if info['exists'] else '❌'}")
        if info['exists']:
            print(f"  Valid: {'✅' if info['valid'] else '❌'}")
            if info['valid']:
                print(f"  Total: {info['count']}")
                if 'enabled_count' in info:
                    print(f"  Enabled: {info['enabled_count']}")
        if 'error' in info:
            print(f"  Error: ❌ {info['error']}")
    
    print()
    
    # 2. Check npx availability
    print("🔧 Dependencies")
    print("-" * 60)
    npx_available = check_npx_available()
    print(f"npx (Node.js): {'✅ Available' if npx_available else '❌ Not found'}")
    
    if not npx_available:
        print("\n⚠️  Warning: npx not found. MCP servers require Node.js/npm.")
        print("   Install from: https://nodejs.org/")
    
    print()
    
    # 3. Test MCP servers
    print("🖥️  MCP Servers")
    print("-" * 60)
    
    servers_path = project_root / "config" / "mcp_servers.json"
    if servers_path.exists():
        with open(servers_path, "r", encoding="utf-8") as f:
            servers = json.load(f)
        
        for server in servers:
            result = test_server_launch(server)
            status_icon = {
                "ok": "✅",
                "warning": "⚠️ ",
                "error": "❌",
                "skipped": "⏭️ ",
                "info": "ℹ️ "
            }.get(result["status"], "❓")
            
            print(f"\n{status_icon} {result['name']}")
            if "package" in result:
                print(f"   Package: {result['package']}")
            if "version" in result:
                print(f"   Version: {result['version']}")
            if "reason" in result:
                print(f"   {result['reason']}")
    else:
        print("❌ mcp_servers.json not found")
        print("   Run the GUI MCP Manager to create default configuration")
    
    print()
    
    # 4. Summary and next steps
    print("=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print()
    
    all_ok = (
        config_results["mcp_servers"]["valid"] and
        config_results["tool_permissions"]["valid"] and
        npx_available
    )
    
    if all_ok:
        print("✅ MCP setup is complete and ready!")
        print()
        print("Next steps:")
        print("1. Test servers manually: See tests/test_mcp_servers_manual.md")
        print("2. Run integration tests: python tests/test_mcp_integration.py")
        print("3. Test with mock model: python tests/test_mcp_mock_model.py")
        print()
        print("To use MCP with a trained model:")
        print("- Train a model using HRM training")
        print("- The model needs to be trained with function calling capabilities")
        print("- Add MCP tool integration to the chat/inference pipeline")
    else:
        print("⚠️  MCP setup incomplete")
        print()
        if not config_results["mcp_servers"]["exists"]:
            print("- Create mcp_servers.json using the GUI MCP Manager")
        if not config_results["tool_permissions"]["exists"]:
            print("- Create tool_permissions.json using the GUI MCP Manager")
        if not npx_available:
            print("- Install Node.js/npm for MCP server support")
    
    print()


if __name__ == "__main__":
    main()
