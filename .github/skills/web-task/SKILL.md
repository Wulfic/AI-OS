---
name: web-task
description: "Web research and browser automation. Use when: scraping a page, filling out a form, taking a screenshot, navigating a dynamic JS app, doing provenance-tracked web research, or when context7 doesn't cover a web resource. Uses playwright MCP for dynamic pages and context-mode for static provenance-tracked fetches."
argument-hint: "URL and task description (e.g. 'screenshot the login page at http://localhost:3000')"
---

# Web Tasks — Playwright and context-mode

Two tools, different use cases. Pick the right one or you're wasting resources.

## Tool Selection

| Scenario | Use |
|----------|-----|
| Dynamic JS app, login flows, forms, screenshots | `playwright` MCP |
| Static page, need citation/provenance trail | `context-mode` MCP |
| API docs / library reference | Use `research-docs` skill instead |
| Simple JSON API response | `fetch` / `curl` in terminal — don't spin up a browser |

## Playwright — Dynamic Pages

```
mcp_wulfnet-playw_playwright_get_tool_schema(tool_name)
mcp_wulfnet-playw_playwright_invoke_tool(tool_name, tool_input)
```

Always call `get_tool_schema` first.

### Common Operations

**Navigate and screenshot:**
```json
tool_name: "navigate"
tool_input: { "url": "<url>" }
// then
tool_name: "screenshot"
tool_input: { "name": "<descriptive-name>", "fullPage": true }
```

**Click an element:**
```json
tool_name: "click"
tool_input: { "selector": "<css-selector or text>" }
```

**Fill a form:**
```json
tool_name: "fill"
tool_input: { "selector": "<input selector>", "value": "<value>" }
```

**CRITICAL: Always close the page when done:**
```json
tool_name: "close"
```

### Procedure for UI Verification

1. Navigate to the target URL
2. Take a full-page screenshot to establish baseline
3. Perform the action (click, fill, submit)
4. Take another screenshot to confirm the result
5. Close the browser context
6. Report findings with screenshot references

## context-mode — Provenance-Tracked Fetch

```
mcp_wulfnet-conte_context-mode_get_tool_schema(tool_name)
mcp_wulfnet-conte_context-mode_invoke_tool(tool_name, tool_input)
```

Use for static pages where you need a verifiable citation. Slower than playwright but produces a citation block suitable for reports.

Always call `get_tool_schema` first to confirm parameter names.

## Rules

- **Never use playwright for static content** — it costs real CPU and a browser context
- **Always close browser pages** you open — leaked contexts accumulate
- Playwright screenshots go to the project's `/screenshots/` directory if available
- If a page requires authentication, note what credentials are needed before starting
- Do not store credentials in code or screenshots — redact before saving
