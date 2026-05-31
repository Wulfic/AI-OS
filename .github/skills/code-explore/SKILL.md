---
name: code-explore
description: "Explore and understand an existing codebase before implementing. Use when: starting work on an unfamiliar codebase, finding where a function is defined or called, looking for existing patterns to follow, checking if a feature already exists before building it, or doing cross-repo code intelligence. Uses gitnexus MCP and workspace search tools."
argument-hint: "What you're looking for (e.g. 'authentication logic', 'how API routes are structured', 'where UserService is defined')"
---

# Code Explore — Understand Before You Build

Never implement without first understanding what already exists. Duplicated code and ignored conventions are the two fastest ways to create a maintenance nightmare. Spend 5 minutes exploring before writing a single line.

## When to Use

- Starting work on any codebase you haven't touched in > 1 day
- Before adding a new feature — does something similar already exist?
- Before creating a new file — where do similar files live?
- Finding all callers of a function before renaming or changing its signature
- Understanding the pattern used for X so you follow it
- Cross-repo: "does this function exist in another repo we own?"

## Tool Selection

| Goal | Tool |
|------|------|
| Find callers/definitions across repos | `gitnexus` MCP |
| Find text patterns in current workspace | `grep_search` |
| Find files by name pattern | `file_search` |
| Find code by concept/meaning | `semantic_search` |
| Find all usages of a symbol | `vscode_listCodeUsages` |

## Procedure

### Step 1 — Get the lay of the land

For an unfamiliar codebase:

```
list_dir("/path/to/project")
list_dir("/path/to/project/src")
```

Then read the key entry points:
- `package.json` / `pyproject.toml` / `go.mod` — dependencies and scripts
- `README.md` — how to run it
- Main entry file (`src/index.ts`, `main.py`, `main.go`, etc.)

### Step 2 — Find relevant existing code

**In the current workspace:**

```
semantic_search("authentication middleware user login")
grep_search(query: "function handleAuth|const auth|class AuthService", isRegexp: true)
file_search(query: "**/auth/**")
```

**Across repos (gitnexus):**

```
mcp_wulfnet-gitne_gitnexus_get_tool_schema("search_symbols")  // get exact schema first
mcp_wulfnet-gitne_gitnexus_invoke_tool(tool_name, {
  "query": "AuthService",
  "type": "definition"
})
```

Always call `get_tool_schema` before invoking gitnexus — parameter names vary by version.

### Step 3 — Understand the pattern

Once you find relevant code, read it:

```
read_file(filePath, 1, 100)  // get the full picture, not just a snippet
```

Look for:
- Naming conventions (camelCase, snake_case, PascalCase, file naming)
- Import patterns (relative vs absolute, barrel files)
- Error handling pattern (throw vs return error, Result type vs try/catch)
- Logging pattern (which logger, what log levels, what gets logged)
- Test co-location (tests next to code, or in a `__tests__` dir?)

### Step 4 — Find all callers before changing anything

If you're modifying an existing function:

```
vscode_listCodeUsages(
  symbol: "functionName",
  filePath: "src/path/to/file.ts",
  lineContent: "export function functionName"
)
```

Or via gitnexus for cross-repo:

```
mcp_wulfnet-gitne_gitnexus_invoke_tool(tool_name, {
  "query": "functionName",
  "type": "references"
})
```

**Every caller must be updated or confirmed compatible before you change a signature.**

### Step 5 — Document your findings (for non-trivial exploration)

Before implementing, write a brief summary of what you found:
- Where similar code lives
- What pattern to follow
- What to avoid (and why)
- Any gotchas discovered

For significant findings, save to mem0:

```json
tool_name: "add_memory"
tool_input: {
  "content": "In [project], [pattern X] is implemented at [path]. Convention: [what to follow]. Gotcha: [what to watch for].",
  "user_id": "tyler",
  "metadata": { "project": "<project>", "type": "convention" }
}
```

## Common Exploration Patterns

### "Where does X happen?"
```
semantic_search("X behavior description")
grep_search(query: "keyword1|keyword2", isRegexp: true)
```

### "How is Y structured in this codebase?"
```
file_search(query: "**/*Y*")
// then read 2-3 examples
```

### "Does this feature already exist?"
```
semantic_search("feature description")
grep_search(query: "relatedFunctionName|RelatedClassName", isRegexp: true)
```
If you find it — don't rebuild it. Use it or extend it.

### "What calls function Z?"
```
vscode_listCodeUsages(symbol: "Z", ...)
```

## Rules

- **Never implement without exploring first** — duplication is a bug you caused
- **Follow existing patterns** — even if you think there's a better way, match conventions first, refactor in a separate PR
- **Read at least 2 examples** before concluding you understand a pattern
- **Check across repos** for shared utilities before writing new ones (gitnexus)
- If exploration takes > 15 minutes, something is wrong with the codebase structure — note it but don't fix it now
