#!/usr/bin/env python3
"""Quick start guide for MCP testing.

Run this script for an interactive guide to testing your MCP setup.
"""

import sys
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(number: int, text: str):
    """Print a step."""
    print(f"\n{'━' * 70}")
    print(f"Step {number}: {text}")
    print('━' * 70)


def main():
    """Run interactive quick start."""
    print_header("MCP Testing Quick Start")
    
    print("""
This guide will help you test your MCP (Model Context Protocol) setup
WITHOUT needing a fully trained model.

The MCP tools can be tested independently, and then integrated with
your model once it's trained on function calling.
""")
    
    input("Press Enter to continue...")
    
    # Step 1: Verify setup
    print_step(1, "Verify Your Setup")
    print("""
First, let's check if everything is configured correctly:

Run this command:
    python scripts/verify_mcp_setup.py

This will check:
✓ Configuration files exist and are valid
✓ Node.js/npm is installed (required for MCP servers)
✓ MCP server packages are available
""")
    
    input("Press Enter when you've run the verification script...")
    
    # Step 2: Install Node.js if needed
    print_step(2, "Install Node.js (if needed)")
    print("""
If the verification script showed that Node.js is missing:

1. Download Node.js from: https://nodejs.org/
2. Install it (includes npm)
3. Restart your terminal
4. Verify with: npx --version

MCP servers are JavaScript/TypeScript packages that run via npx.
""")
    
    input("Press Enter to continue...")
    
    # Step 3: Initialize MCP servers config
    print_step(3, "Initialize MCP Server Configuration")
    print("""
If mcp_servers.json doesn't exist:

Option A - Use the GUI (Recommended):
    1. Run: aios gui
    2. Go to the "MCP Servers & Tools" tab
    3. The default servers will be created automatically
    4. You can enable/disable servers and configure them

Option B - Manual creation:
    1. Create: config/mcp_servers.json
    2. Copy default configuration from:
       src/aios/gui/components/mcp_manager_panel/data_manager.py
       (see get_default_servers function)
""")
    
    input("Press Enter to continue...")
    
    # Step 4: Test individual servers
    print_step(4, "Test Individual MCP Servers")
    print("""
Use the MCP Inspector to test each server:

Install Inspector (once):
    npm install -g @modelcontextprotocol/inspector

Test Filesystem Server:
    npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-filesystem .

Test Memory Server (Knowledge Graph):
    npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-memory

Test DuckDuckGo Search:
    npx @modelcontextprotocol/inspector npx -y @wulfic/server-duckduckgo

The Inspector will:
- Open a web UI in your browser
- Show all available tools
- Let you test tool calls with parameters
- Display responses
""")
    
    input("Press Enter to continue...")
    
    # Step 5: Run automated tests
    print_step(5, "Run Automated Tests")
    print("""
Test the integration with Python:

Configuration tests:
    pytest tests/test_mcp_integration.py -v

Mock model simulation:
    python tests/test_mcp_mock_model.py

This will show:
- How a model would decide to use tools
- Tool call generation examples
- Simulated tool responses
""")
    
    input("Press Enter to continue...")
    
    # Step 6: What's working
    print_step(6, "Understanding What Works NOW")
    print("""
✅ What you can test RIGHT NOW (without a trained model):

1. MCP Servers Launch
   - Servers can be started via npx
   - Tool discovery works
   - Tools execute with test parameters

2. Tool Schemas
   - Each tool's parameters are defined
   - Input validation works
   - Error handling works

3. Configuration System
   - Enable/disable servers
   - Enable/disable individual tools
   - Permission management

❌ What requires a trained model:

1. Decision Making
   - Model decides WHEN to use a tool
   - Model determines WHICH tool to use

2. Parameter Generation
   - Model generates correct arguments
   - Model handles context appropriately

3. Response Synthesis
   - Model integrates tool results
   - Model generates coherent responses
""")
    
    input("Press Enter to continue...")
    
    # Step 7: Next steps
    print_step(7, "Next Steps for Full Integration")
    print("""
To use MCP tools with your trained model:

1. Create Function Calling Training Data
   - Examples of user requests → tool calls
   - Tool results → final responses
   - Format: See docs/guide/features/MCP_TESTING_WITHOUT_MODEL.md

2. Train or Fine-tune Your Model
   - Add tool calling to training data
   - Model learns when/how to use tools
   - Test with simple examples first

3. Add Integration Code
   - Parse model output for tool calls
   - Execute tool calls via MCP servers
   - Feed results back to model

4. Test End-to-End
   - User message → model → tool → model → response
   - Verify tool selection accuracy
   - Check parameter generation quality

Sample integration in:
   src/aios/core/brains/actv1_brain.py (add run_with_tools method)
""")
    
    input("Press Enter to continue...")
    
    # Summary
    print_header("Summary")
    print("""
📝 What You've Learned:

1. MCP tools can be tested WITHOUT a trained model
2. Use MCP Inspector for manual testing
3. Use Python scripts for automated testing
4. The model is just the "decision layer" on top

📚 Documentation:
- Full guide: docs/guide/features/MCP_TESTING_WITHOUT_MODEL.md
- Manual testing: tests/test_mcp_servers_manual.md
- Tool list: config/tool_permissions.json

🚀 Ready to Test:
1. python scripts/verify_mcp_setup.py
2. npx @modelcontextprotocol/inspector [server command]
3. python tests/test_mcp_mock_model.py

💡 Key Insight:
The MCP infrastructure works independently of your model!
You're testing the "external brain" that your AI will call upon.
""")
    
    print("\nHappy testing! 🎉\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...\n")
        sys.exit(0)
