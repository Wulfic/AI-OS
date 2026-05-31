---
name: recall-session
description: "Start or resume a work session. Use when: beginning a new conversation on an existing project, resuming after a break, onboarding to a codebase, or needing to establish what was done and what's next. Searches mem0 for project history, reads TODO.md, and produces a focused next-steps plan."
argument-hint: "Project name or area of focus (e.g. 'auth service' or 'the whole project')"
---

# Session Kickoff — Recall and Plan

Never start working blind. Spend 2 minutes recalling context before touching a single file. This prevents duplicate work, contradicting past decisions, and losing track of in-progress items.

## When to Use

- First message in a new conversation about an ongoing project
- Resuming after any break longer than an hour
- Handing off work or starting a new phase
- "What was I doing?" — any time context is fuzzy

## Procedure

### Step 1 — Search mem0 for project history

```
mcp_wulfnet-mem0_mem0_get_tool_schema("search_memory")
```
Then:
```json
tool_name: "search_memory"
tool_input: {
  "query": "<project name or relevant keywords>",
  "user_id": "tyler"
}
```

Look for:
- Past architectural decisions
- Known issues or blockers
- Conventions established (naming, stack choices, patterns)
- Last known state of the work

### Step 2 — Read TODO.md

Read the workspace `TODO.md` to find:
- Unchecked items → what's still pending
- Recently checked items → what was just finished
- Blocked items → what needs attention

### Step 3 — Check recent git activity (optional but recommended)

```powershell
git log --oneline -10
git status
```

This tells you what files were last touched and if there's uncommitted work.

### Step 4 — Think about the current state

```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "think"
tool_input: {
  "thought": "Given what I know about this project from mem0 and TODO.md, what is the actual next priority? What blockers exist? What context does the user need before we start?"
}
```

### Step 5 — Produce a session brief

Report back with:
1. **What was done** (from mem0 + git log)
2. **Current blockers or open questions**
3. **Recommended next action** (single, specific, actionable)
4. **Any context the user should confirm** (decisions that may have changed)

## After the Session — Save to mem0

At the end of a productive session, persist what was learned:

```json
tool_name: "add_memory"
tool_input: {
  "content": "<what was built, decided, or discovered>",
  "user_id": "tyler",
  "metadata": { "project": "<project-name>", "date": "<YYYY-MM-DD>" }
}
```

Save:
- Architectural decisions and the reasons behind them
- Non-obvious gotchas discovered during implementation
- Which approach was tried and failed (so we don't try it again)
- Current status at end of session

## Rules

- Do NOT start writing code before completing steps 1–4
- If mem0 returns nothing, that means this is a genuinely new project — proceed to `think-plan` skill
- Session brief must be ≤ 10 bullet points — if you need more, you're not summarizing
- Always save to mem0 at session end — memory that isn't persisted is wasted work
