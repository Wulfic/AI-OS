---
name: test-iterate
description: "Write tests, run them, fix failures, repeat until green. Use when: implementing new features (write tests first or after), a test suite is failing, need to add coverage for a bug fix, doing TDD, running E2E tests with playwright, or driving the red-green-refactor loop. Covers unit tests and E2E tests."
argument-hint: "Feature or file to test, and test type (unit / integration / E2E)"
---

# Test-Iterate — Red Green Refactor

Tests are not optional. No feature is done without a passing test. The loop is: write test → run → read failure → fix code (not the test) → repeat.

## When to Use

- Implementing any new feature or function
- Fixing a bug (write a failing test for it first)
- A test suite is currently failing and needs to be fixed
- Adding E2E coverage for a user flow
- Verifying a refactor didn't break anything
- CI is red and needs to be green

## Test Types

| Type | Tool | When |
|------|------|------|
| Unit | `run_in_terminal` (jest/vitest/pytest/etc.) | Pure functions, business logic, isolated components |
| Integration | `run_in_terminal` | Multiple units working together, DB queries, API handlers |
| E2E | `web-task` skill (playwright) | Full user flows through the browser |

## Procedure

### Step 1 — Understand what to test

Before writing any test:

```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "think"
tool_input: {
  "thought": "What are the critical behaviors of [feature/function]? What are the edge cases? What inputs cause incorrect output? What should I test first to get the most confidence?"
}
```

Read the code under test first — you cannot write a good test for code you haven't read:
```
read_file(filePath, startLine, endLine)
```

### Step 2 — Write the test (failing first for new features)

For new code, write the test BEFORE the implementation. For existing code, write a test that captures the expected behavior.

Test file location conventions:
- Jest/Vitest: `<file>.test.ts` or `__tests__/<file>.test.ts`
- Pytest: `test_<file>.py` or `tests/<file>_test.py`
- Go: `<file>_test.go` (same package)

Test structure (AAA pattern — Arrange, Act, Assert):
```
// Arrange — set up inputs and mocks
// Act — call the thing under test
// Assert — verify the output
```

### Step 3 — Run the tests

```powershell
# Run specific test file (faster feedback loop)
npx jest <test-file> --no-coverage
npx vitest run <test-file>
pytest tests/<test-file>.py -v
go test ./... -run TestName

# Run full suite
npm test
pytest
go test ./...
```

Capture the full output. Do not dismiss partial output — read the whole failure message.

### Step 4 — Read the failure

A test failure tells you one of three things:
1. **The code is wrong** — fix the implementation (most common)
2. **The test is wrong** — fix the test (your understanding of the spec was wrong)
3. **The setup/mock is wrong** — fix the test infrastructure

**Never change a test to make it pass** unless the test was wrong to begin with. That's not fixing — that's cheating.

Classify the failure before touching anything:
```json
mcp_wulfnet-think_think_invoke_tool
tool_name: "think"
tool_input: {
  "thought": "Test [name] failed with: [failure message]. Is the implementation wrong, the test wrong, or the mock/setup wrong? What is the minimal fix?"
}
```

### Step 5 — Fix and re-run

Apply the fix. Re-run the specific failing test (not the whole suite — faster feedback):

```powershell
npx jest <test-name-pattern> --watch=false
```

Repeat steps 3–5 until the test passes.

### Step 6 — Run the full suite

Once the target test is green, run the full suite to check for regressions:

```powershell
npm test
```

If new failures appear that weren't there before, your change broke something. Do NOT commit with a red suite.

### Step 7 — E2E tests via web-task

For E2E scenarios, invoke the `web-task` skill with playwright:
1. Start the dev server first
2. Navigate to the feature under test
3. Perform the user actions
4. Screenshot before and after
5. Assert on what you see (or use playwright assertions)

### Step 8 — Coverage check (when required)

```powershell
npm test -- --coverage
pytest --cov=src --cov-report=term-missing
```

Target: every new function has at least one happy-path test and one edge-case test. 100% coverage is a vanity metric — meaningful coverage is the goal.

## Rules

- **Write the test before you mark a feature done** — "I'll add tests later" means never
- **Never delete a failing test** — fix it or document why it's skipped
- **Don't mock what you can test directly** — over-mocking tests the mock, not the code
- **One assertion per test concept** — a test that asserts 10 things tells you nothing when it fails
- If the test suite takes > 30 seconds, check for missing `--testPathPattern` scope
- A passing test on broken code means your test is wrong
