# Testing MCP Tools Without a Trained Model

This guide outlines what you can currently validate for MCP (Model Context Protocol) tooling inside AI-OS. The shipped code only provides a stub implementation, so verification focuses on the Python-side scaffolding rather than real server connectivity.

## Current Status

### ✅ Available Components
- GUI panel for configuring MCP servers and tool permissions.
- `MCPClient` and `ToolExecutor` classes that discover tools and parse tool calls.
- Eight mock tool definitions covering filesystem, memory, and DuckDuckGo categories.

### ⚠️ Not Implemented Yet
- Launching MCP server processes or speaking the MCP protocol.
- Any Node.js/npm integration.
- Automated verification scripts or pytest suites for MCP.
- Real tool execution; everything returns mocked responses.

## What You Can Test Today

### 1. Instantiate the Executor
```python
from aios.core.mcp.tool_executor import ToolExecutor
executor = ToolExecutor()
print(executor.enabled)           # True when mock tools load
print(executor.mcp_client.available_tools.keys())
```

### 2. Parse a Tool Call
```python
sample = '{"tool": "read_file", "parameters": {"path": "README.md"}}'
parsed = executor.parse_tool_call(sample)
print(parsed)
```

### 3. Observe Mock Execution
```python
import asyncio
result = asyncio.run(executor.execute_if_tool_call(sample))
print(result["result"])  # Contains "[Mock] Tool ... would be executed"
```

### 4. GUI Smoke Test
Launch `aios gui`, open the Chat tab, and craft a prompt that should invoke a tool. The UI will log the mock execution result.

## Limitations to Keep in Mind
- No MCP servers are contacted, so there is nothing to verify with Node.js, npm, or MCP Inspector yet.
- The repository does **not** contain `scripts/verify_mcp_setup.py`, `tests/test_mcp_integration.py`, or related utilities mentioned in earlier drafts of the docs.
- Tool outputs are placeholders; they do not read files, search the web, or modify state.

## Preparing for Real MCP Support

When you are ready to implement full MCP connectivity, plan to add:

1. Process management and RPC wiring for each configured server.
2. Discovery of tool schemas via the MCP protocol instead of hard-coded dictionaries.
3. Streaming of actual results/errors back into the chat workflow.
4. Installation instructions for any runtime dependencies (Node.js, npm packages, etc.).
5. Unit/integration tests that exercise the new codepaths.

## Model Training Considerations

Even once real tool execution exists, the model must still be trained or fine-tuned on tool-calling examples. Prepare datasets where assistant messages include tool invocations, tool responses, and follow-up reasoning so the model learns when and how to call tools.

## Summary
- ✅ You can exercise the Python scaffolding and ensure the GUI wiring behaves as expected.
- ⚠️ There is no server-side testing to perform yet; everything is mocked.
- ➡️ Future work involves implementing the MCP protocol, adding diagnostics, and training the model on function-calling data.
