# Testing MCP Tools Without a Trained Model

This guide explains how to verify that your MCP (Model Context Protocol) tools are working correctly before you have a fully trained model capable of using them.

## Current Status

Based on your codebase analysis:

### ✅ What You Have
- **MCP Manager GUI Panel** - Configuration interface for servers and tools
- **Tool Permissions System** - 37 tools across 6 categories defined in `config/tool_permissions.json`
- **Default Server Configs** - Filesystem, Memory, and DuckDuckGo servers configured

### ⚠️ What's Missing
- **MCP Server Integration** - The actual connection between MCP servers and model inference is not yet implemented
- **Node.js/npm** - Required for running MCP servers (based on verification script)
- **Tool Calling Training Data** - Your model needs to be trained on function/tool calling examples

## Testing Approaches

### 1. Quick Setup Verification

Run the automated verification script:

```powershell
python scripts/verify_mcp_setup.py
```

This checks:
- Configuration file validity
- Node.js/npm availability
- MCP server package availability
- Overall readiness

### 2. Manual Server Testing (Most Reliable)

Use the MCP Inspector to test servers directly:

```powershell
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Test filesystem server
npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-filesystem <PROJECT_ROOT>

# Test memory (knowledge graph) server
npx @modelcontextprotocol/inspector npx -y @modelcontextprotocol/server-memory

# Test DuckDuckGo search
npx @modelcontextprotocol/inspector npx -y @wulfic/server-duckduckgo
```

The Inspector provides a web UI where you can:
- See all available tools
- Test tool calls with parameters
- View responses
- Verify error handling

### 3. Automated Integration Tests

Run the Python test suite:

```powershell
# Test configuration validity
python tests/test_mcp_integration.py

# Test with mock model
python tests/test_mcp_mock_model.py
```

The mock model test demonstrates:
- How a model would decide to use tools
- Tool call generation
- Tool execution
- Response synthesis

### 4. pytest Integration

If you have pytest installed:

```powershell
pytest tests/test_mcp_integration.py -v
```

## What Each Tool Does

### File Operations (9 tools)
- `read_file` - Read file contents
- `write_file` - Create/overwrite files
- `edit_file` - Line-based file editing
- `list_directory` - List directory contents
- `create_directory` - Create directories
- `move_file` - Move/rename files
- `search_files` - Find files by pattern
- `get_file_info` - Get metadata
- `directory_tree` - Recursive tree structure
- `read_multiple_files` - Batch file reading

### Memory & Knowledge (9 tools)
- `create_entities` - Add entities to knowledge graph
- `create_relations` - Link entities
- `add_observations` - Add entity observations
- `search_nodes` - Search knowledge graph
- `read_graph` - Read entire graph
- `open_nodes` - Get specific nodes
- `delete_entities` - Remove entities
- `delete_relations` - Remove relations
- `delete_observations` - Remove observations

### Web & Search (2 tools)
- `web_search` - DuckDuckGo search
- `fetch_webpage` - Download and parse pages

### Code & Development (6 tools)
- `run_code_snippet` - Execute Python code
- `check_syntax_errors` - Validate Python syntax
- `get_imports` - Analyze imports
- `invoke_refactoring` - Auto-refactor
- `get_python_environments` - List Python envs
- `update_python_environment` - Switch env

### System & Terminal (4 tools)
- `run_terminal_command` - Execute shell commands
- `get_terminal_output` - Read command output
- `run_task` - Execute VS Code tasks
- `get_task_output` - Read task output

### Data Analysis (7 tools)
- `semantic_search` - Semantic code search
- `grep_search` - Text pattern search
- `list_code_usages` - Find symbol references
- `file_search` - Glob pattern search
- `get_errors` - Get lint/compile errors

## Training a Model to Use Tools

For your model to actually use these tools, you'll need:

### 1. Function Calling Training Data

Create training examples in this format:

```json
{
  "messages": [
    {"role": "user", "content": "Read the README file"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "name": "read_file",
        "arguments": {"path": "README.md"}
      }]
    },
    {
      "role": "tool",
      "content": "# Project Title\n..."
    },
    {
      "role": "assistant",
      "content": "Based on the README, this project is about..."
    }
  ]
}
```

### 2. Model Architecture Considerations

Your model needs:
- **Tool detection layer** - Recognize when to use tools
- **Parameter extraction** - Generate correct tool arguments
- **Response synthesis** - Integrate tool results into responses

Options:
1. **Fine-tune existing tool-calling model** (e.g., GPT-4, Claude, Qwen2.5)
2. **Add tool-calling head to your ACTv1 architecture**
3. **Train from scratch with tool calling examples**

### 3. Integration Code Needed

You'll need to add to your inference pipeline:

```python
# In src/aios/core/brains/actv1_brain.py or similar

def run_with_tools(self, task: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference with MCP tool support."""
    # 1. Generate initial response (may include tool call)
    response = self.generate(task)
    
    # 2. Check if model wants to use a tool
    if tool_call := self._parse_tool_call(response):
        # 3. Execute the tool via MCP server
        tool_result = self._execute_mcp_tool(tool_call)
        
        # 4. Give result back to model for final response
        final_response = self.generate_with_context(task, tool_result)
        return final_response
    
    return response
```

## Verification Checklist

Before using with a trained model:

- [ ] Node.js/npm installed
- [ ] MCP servers can be launched via Inspector
- [ ] Each server's tools are discoverable
- [ ] Tool calls return expected responses
- [ ] Error handling works (invalid parameters)
- [ ] Configuration files exist and are valid
- [ ] Integration tests pass

## Next Steps

1. **Install Node.js** if you haven't already
2. **Run verification script** to check setup
3. **Test servers manually** using MCP Inspector
4. **Create function calling training data** for your domain
5. **Add tool integration** to your model's inference code
6. **Train/fine-tune** with tool calling examples

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [MCP Servers](https://github.com/modelcontextprotocol/servers)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

## Testing Strategy Summary

```
Without Trained Model:
├── Config Validation ✅
│   ├── Run verify_mcp_setup.py
│   └── Check config files exist
│
├── Server Testing ✅
│   ├── Use MCP Inspector
│   ├── Test each server individually
│   └── Verify tool discovery
│
├── Mock Testing ✅
│   ├── Use test_mcp_mock_model.py
│   └── Simulate tool calling flow
│
└── Integration Tests ✅
    ├── Run pytest suite
    └── Verify tool schemas

With Trained Model:
├── Tool Detection
│   └── Model decides when to use tools
│
├── Parameter Generation
│   └── Model creates correct arguments
│
├── Tool Execution
│   └── MCP servers process requests
│
└── Response Synthesis
    └── Model uses results in answer
```

The key insight: **You can verify the entire MCP infrastructure works without a trained model** - the model is just the decision-making layer on top!
