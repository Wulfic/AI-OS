---
name: debug-errors
description: "Systematic error triage and fix loop. Use when: build fails, type errors appear, runtime exceptions occur, get_errors returns problems, tests fail with unexpected errors, logs show exceptions. Reads VS Code diagnostics, terminal output, and log files to form hypotheses and drive fixes to zero errors."
argument-hint: "Describe the error or paste the error message"
---

# Debug Errors — Triage to Zero

Errors block everything. Fix them systematically, not by guessing. Never apply a fix without understanding the root cause first.

## When to Use

- `get_errors` returns diagnostics
- Build step exits non-zero
- Test runner reports unexpected errors (not assertion failures — those go to `test-iterate`)
- Runtime exception in logs
- TypeScript/compiler errors after a change
- "It was working and now it's not"

## Procedure

### Step 1 — Collect all errors first

Run `get_errors` across the relevant files — not just the file you changed:

```
get_errors(filePaths: ["<changed file>", "<related files>"])
// or omit filePaths to scan everything
get_errors()
```

Do NOT start fixing until you have the full error list. Fixing one error in isolation often reveals it was masking others or was a symptom of a deeper problem.

### Step 2 — Read the logs

If the error is runtime (not compile-time), find the log file:

```powershell
# Common log locations — adapt per project
Get-Content .\logs\app.log -Tail 50
Get-Content .\logs\error.log -Tail 50
```

Look for the full stack trace, not just the last line. The root cause is almost always higher up in the stack.

### Step 3 — Form a hypothesis with think

Before touching code:

```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "think"
tool_input: {
  "thought": "The error is [X]. The file it occurs in is [Y]. The change that preceded it was [Z]. What is the most likely root cause? What are alternative causes? What is the minimal fix?"
}
```

Do not skip this. Blind edits without a hypothesis create new errors.

### Step 4 — Read the failing code in context

```
read_file(filePath, startLine, endLine)
```

Read at least 20 lines before and after the error location. Errors are almost never exactly where they're reported — the real problem is usually in the caller or the type definition.

For TypeScript errors, also read the type definition:
```
grep_search(query: "interface <TypeName>|type <TypeName>", isRegexp: true)
```

### Step 5 — Apply the minimal fix

- Fix the root cause, not the symptom
- Do NOT add `// @ts-ignore` or `as any` as a fix — this is a smell, not a solution
- Do NOT add empty catch blocks to hide errors
- If the fix requires understanding a library API, invoke `research-docs` first

### Step 6 — Verify to zero

After each fix:
```
get_errors()
```

Repeat until zero errors. Then run the build:
```powershell
# run whatever build command the project uses
```

Zero errors in `get_errors` + clean build = done.

### Step 7 — Post-mortem for non-trivial bugs

If the bug took more than 2 fix attempts, save it to mem0:

```json
tool_name: "add_memory"
tool_input: {
  "content": "Bug: [description]. Root cause: [cause]. Fix: [what worked]. What didn't work: [failed attempts].",
  "user_id": "tyler",
  "metadata": { "project": "<project>", "type": "bug" }
}
```

## Rules

- **Never add suppressions** (`@ts-ignore`, `eslint-disable`, `as any`) without a documented reason
- **Read before you edit** — always read the full context before applying a fix
- **Fix root cause** — if you find yourself fixing the same error in 3 places, you're fixing a symptom
- **One fix at a time** — apply one change, run `get_errors`, assess, repeat
- If you're on the 4th attempt at the same error, stop and use `think` again with the full error history

## Error Pattern Reference

| Pattern | Likely Cause |
|---------|-------------|
| `Cannot find module` | Missing import, wrong path, not installed |
| `Property X does not exist on type Y` | Wrong type, missing field in interface, outdated type def |
| `is not assignable to type` | Type mismatch — read both types fully before fixing |
| `undefined is not a function` | Missing null check, wrong variable scope, async/await missing |
| `ENOENT: no such file` | Wrong working directory, wrong path, file not created yet |
| `EADDRINUSE` | Port already in use — kill the existing process |
| `Cannot read properties of undefined` | Missing null check or incorrect async flow |
