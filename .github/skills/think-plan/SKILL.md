---
name: think-plan
description: "Structured reasoning before implementing. Use when: planning a feature, designing architecture, evaluating an approach, criticizing a draft implementation, solving a complex problem, debugging a non-obvious issue. Wraps the think MCP (think / plan / criticize)."
argument-hint: "Describe the problem or decision to reason through"
---

# Think / Plan / Criticize

Mandatory reasoning gate before non-trivial work. Do NOT skip this for anything that involves more than a single file change or a well-understood one-liner fix.

## When to Use

- Starting a new feature or project
- Evaluating two competing approaches
- Before refactoring anything that touches > 2 files
- When debugging an issue that isn't immediately obvious
- Before writing tests (what edge cases actually matter?)
- After producing a draft — stress-test it with `criticize`

## Tools

All three calls go through the `think` MCP server via the mcp-compressor pattern:

```
mcp_wulfnet-think_think_invoke_tool(tool_name, tool_input)
```

Always call `mcp_wulfnet-think_think_get_tool_schema` first if the schema is not already in context.

## Procedure

### Step 1 — Think (single-question reasoning)
Use for: "what's the right approach here?", debugging hypotheses, tradeoff analysis.

```json
tool_name: "think"
tool_input: { "thought": "<specific question or problem statement>" }
```

### Step 2 — Plan (multi-step work breakdown)
Use for: new features, project scaffolding, any task > 2 steps.

```json
tool_name: "plan"
tool_input: { "goal": "<what we're building>", "context": "<relevant constraints / stack>" }
```

The output is a numbered action list. Translate this directly into `TODO.md` entries.

### Step 3 — Criticize (stress-test a draft)
Use for: reviewing your own proposed solution before implementing, catching edge cases.

```json
tool_name: "criticize"
tool_input: { "draft": "<the plan, code, or design to stress-test>" }
```

Address every raised issue before proceeding. If you disagree, document why.

## Rules

- **Always `think` before `plan`.** Don't jump to a step list without framing the problem.
- **Always `criticize` a plan before writing code.** One extra call, saves hours.
- Output of `plan` must map to `TODO.md` — don't let plans live only in the chat.
- If `criticize` surfaces a fatal flaw, restart from `think`, not from `plan`.
