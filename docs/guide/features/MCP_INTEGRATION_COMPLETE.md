# MCP Integration Status

This document reflects the current, partially implemented Model Context Protocol (MCP) support in AI-OS. The codebase ships with the beginnings of an integration, but the tooling is not yet connected to real MCP servers.

## Snapshot

- `config/mcp_servers.json` — placeholder configuration entries for filesystem, memory, and DuckDuckGo servers.
- `src/aios/core/mcp/client.py` — loads the config and exposes **mock** tool definitions.
- `src/aios/core/mcp/tool_executor.py` — parses potential tool calls in model output and routes them to the client.
- GUI wiring (`src/aios/gui/app/chat_operations.py`) — activates the tool executor so the chat UI can surface the stubbed tools.

## Current Behaviour

- Tool discovery returns eight static tool definitions (4 filesystem, 2 memory, 2 web) baked into `MCPClient`.
- `execute_tool(...)` does **not** contact an MCP server; it always returns a mocked payload of the form `"[Mock] Tool ... would be executed"`.
- No Node.js/npm pre-requisites are enforced because nothing is launched yet.
- No verification scripts or pytest suites ship in the repository for MCP. Any references to `scripts/verify_mcp_setup.py`, `tests/test_mcp_integration.py`, etc. in older docs were aspirational and have been removed.

## What’s Missing

- Real MCP protocol handshakes (JSON-RPC over stdio) with external servers.
- Process management for the configured commands in `mcp_servers.json`.
- End-to-end tool execution that streams actual results back into the chat session.
- Installation/diagnostic helpers for Node.js, npm, or server packages.
- Automated or manual tests that exercise the integration beyond the mocked responses.

## Testing the Stub

If you want to observe the current behaviour:

```python
from aios.core.mcp.tool_executor import ToolExecutor

executor = ToolExecutor()
result = executor.parse_tool_call('{"tool": "read_file", "parameters": {"path": "README.md"}}')
print(result)
# {'tool': 'read_file', 'parameters': {'path': 'README.md'}}

import asyncio
tool_run = asyncio.run(executor.execute_if_tool_call('{"tool": "read_file", "parameters": {"path": "README.md"}}'))
print(tool_run)
# {'is_tool_call': True, 'tool_name': 'read_file', 'parameters': {'path': 'README.md'}, 'result': {'success': True, 'tool': 'read_file', 'result': '[Mock] Tool read_file would be executed with: {"path": "README.md"}'}}
```

## Next Steps for a Full Integration

To complete MCP support you will need to:

1. Implement real server processes (Node or Python) and spawn them from the client.
2. Replace the mock tool registry with schema discovery via `initialize`/`tools/list` messages.
3. Stream tool execution (including stderr/stdout) back through `execute_tool`.
4. Add configuration UI/CLI that reflects actual server availability.
5. Supply diagnostics/tests for the new code paths.

## Summary

- ✅ Stubs exist so the UI can demonstrate how tool calls will be parsed.
- ⚠️ End-to-end execution is not present.
- ⚠️ No bundled scripts or tests verify MCP connectivity.

Treat the current implementation as scaffolding rather than a completed feature. Refer to `docs/guide/features/MCP_TESTING_WITHOUT_MODEL.md` for guidance on exercising the stubs and planning the remaining work.
