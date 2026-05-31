---
name: code-review
description: "Aggressive, thorough code review. Use when: reviewing a PR, auditing a file before merge, checking for security issues, OWASP Top 10 violations, performance problems, missing error handling, poor logging, or just reviewing a junior dev's work. Uses think.criticize + workspace analysis."
argument-hint: "File path(s) or feature area to review"
---

# Code Review — No Mercy

Review code like a senior dev who has seen every mistake in the book and has zero patience for slop. Every issue gets called out. No sugar-coating.

## When to Use

- Before merging any non-trivial feature
- When something "feels wrong" but you can't pinpoint it
- Security audit (OWASP Top 10 check)
- Performance review
- "Does this code actually do what it claims?"
- Post-implementation sanity check

## Procedure

### Step 1 — Read the code

Use `read_file`, `grep_search`, and `semantic_search` to fully understand the code under review. Don't start criticizing until you've read the full context.

- Check imports and dependencies
- Trace the data flow end-to-end
- Note what's missing (error handling, logging, tests, input validation)

### Step 2 — Think about it first

```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "think"
tool_input: {
  "thought": "What are the real problems in this code? Security issues, logic bugs, missing error handling, poor structure, OWASP violations, missing logging?"
}
```

### Step 3 — Criticize the implementation

```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "criticize"
tool_input: {
  "draft": "<paste the relevant code sections or description of the implementation>"
}
```

### Step 4 — Report findings

Organize issues by severity:

#### CRITICAL (must fix before merge)
- Security vulnerabilities (injection, auth bypass, secrets in code, etc.)
- Data loss / corruption risk
- Unhandled exceptions that crash the service

#### HIGH (fix in this PR)
- Missing input validation at system boundaries
- No logging on error paths
- Logic bugs that produce wrong output in edge cases
- Missing tests for critical paths

#### MEDIUM (create a ticket)
- Poor naming, confusing structure
- Unnecessary complexity / over-engineering
- Missing comments on non-obvious logic

#### LOW (nice to have)
- Style inconsistencies
- Minor refactors

### Step 5 — Check OWASP Top 10

Explicitly check:
1. Injection (SQL, command, LDAP)
2. Broken authentication / exposed secrets
3. Sensitive data exposure (PII in logs, unencrypted storage)
4. Security misconfigurations (CORS, headers, defaults)
5. Using vulnerable dependencies (check package versions)

## Rules

- Every finding must reference the exact file and line number
- "Looks fine" is never acceptable without having read the code
- If there are no tests, that is automatically a HIGH finding
- If there is no logging on error paths, that is a HIGH finding
- Don't praise mediocre work — silence means no objection, not approval
