---
name: github-workflow
description: "GitHub operations: create/update issues, open PRs, review PR status, check workflow runs, search code across repos, manage releases, list branches. Use when working with GitHub repos, tracking work in issues, or managing the release process. Uses github MCP."
argument-hint: "What GitHub operation you need (e.g. 'create issue for login bug', 'check PR #42 status')"
---

# GitHub Workflow via github MCP

All GitHub operations go through the `github` MCP — not the `gh` CLI — because results come back as structured JSON and auth is pre-configured.

## When to Use

- Creating or updating issues
- Opening, reviewing, or merging PRs
- Checking CI/CD workflow run status
- Searching for code across public/private repos
- Managing releases and tags
- Branch management
- Linking commits to issues

## Tools

```
mcp_wulfnet-githu_github_get_tool_schema(tool_name)   # always call first
mcp_wulfnet-githu_github_invoke_tool(tool_name, tool_input)
```

Always call `get_tool_schema` for the specific operation first — do not invent parameter names.

## Common Operations

### Create an Issue

```json
tool_name: "create_issue"   // verify exact name via get_tool_schema
tool_input: {
  "owner": "<org-or-user>",
  "repo": "<repo-name>",
  "title": "<clear, specific title>",
  "body": "<description with steps to reproduce if bug, or acceptance criteria if feature>",
  "labels": ["bug" | "enhancement" | "documentation"]
}
```

### Open a Pull Request

```json
tool_name: "create_pull_request"
tool_input: {
  "owner": "<org-or-user>",
  "repo": "<repo-name>",
  "title": "<title>",
  "body": "<description, link to issue with 'Closes #N'>",
  "head": "<feature-branch>",
  "base": "main"
}
```

### Check Workflow Runs

```json
tool_name: "list_workflow_runs"
tool_input: {
  "owner": "<org-or-user>",
  "repo": "<repo-name>",
  "workflow_id": "<workflow-file.yml or ID>"
}
```

### Search Code

```json
tool_name: "search_code"
tool_input: {
  "q": "<search query> repo:<owner>/<repo>"
}
```

## Procedure for New Feature Work

1. Create an issue first — every feature/bug gets a ticket
2. Create a branch named `feature/<issue-number>-<short-slug>`
3. After implementation, open a PR that references the issue (`Closes #N`)
4. Check workflow runs pass before requesting review
5. Update `TODO.md` to mark items complete after merge

## Rules

- Always call `get_tool_schema` before invoking — parameter names change between versions
- Never push directly to `main` — always use PRs
- PR descriptions must link to the related issue
- Do not close issues manually — use `Closes #N` in the PR body so GitHub auto-closes on merge
