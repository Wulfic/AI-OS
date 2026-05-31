---
name: build-run
description: "Execute build commands, dev servers, scripts, and interpret their output. Use when: running builds, starting/stopping dev servers, executing install commands, running linters or type-checkers, interpreting non-zero exit codes, or orchestrating multi-step command sequences. Manages terminal lifecycle and output parsing."
argument-hint: "What to run (e.g. 'build the project', 'start dev server', 'run lint and typecheck')"
---

# Build and Run — Command Execution

Running a command without reading its output is worthless. Every command has an exit code and output that tells you what happened. Read both.

## When to Use

- Installing dependencies
- Running the build (`tsc`, `vite build`, `go build`, `cargo build`, etc.)
- Starting the dev server for testing
- Running linters (`eslint`, `ruff`, `golangci-lint`)
- Running type-checkers (`tsc --noEmit`, `mypy`)
- Any multi-command sequence (install + build + test)
- Interpreting why a CI step failed

## Command Execution Rules

### Sync vs Async

| Scenario | Mode |
|----------|------|
| Install, build, lint, typecheck, one-off scripts | `mode=sync` with timeout |
| Dev server, watcher, long-running process | `mode=async` |

Never use `mode=async` for a build you need to wait for. Never use `mode=sync` for a server that should stay running.

### Timeouts

Set generous timeouts — don't let installs time out:
- `npm install` / `pip install`: 300000 (5 min)
- Build commands: 120000 (2 min)
- Tests: 180000 (3 min)
- Simple scripts: 30000 (30 sec)

## Standard Sequences by Project Type

### Node.js / TypeScript

```powershell
# Install
npm install

# Type-check only (no emit)
npx tsc --noEmit

# Lint
npx eslint src --ext .ts,.tsx --max-warnings 0

# Build
npm run build

# Dev server (async)
npm run dev
```

### Python

```powershell
# Install
pip install -r requirements.txt

# Type-check
mypy src/

# Lint
ruff check src/

# Build / package
python -m build
```

### Go

```powershell
# Build
go build ./...

# Vet
go vet ./...

# Lint (if golangci-lint installed)
golangci-lint run
```

## Interpreting Output

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error — read stderr |
| 2 | Misuse of command / config error |
| 127 | Command not found — tool not installed or not in PATH |
| Non-zero from `tsc` | Type errors — use `debug-errors` skill |
| Non-zero from eslint | Lint errors — read the list, fix each one |

### What to Look For in Output

1. **First error in the output** — not the last. Subsequent errors are often cascades from the first.
2. **File + line number** — always present for compiler errors
3. **"Cannot find"** / **"not found"** — missing dependency or wrong path
4. **WARN vs ERROR** — warnings can often be deferred; errors cannot

### When Build Fails

1. Read the full output — don't just look at the last line
2. Find the first error (not the last)
3. If it's a type/compile error → use `debug-errors` skill
4. If it's a missing package → `npm install <package>` or equivalent
5. If it's a config error → read the config file the error references
6. Re-run after each fix — don't stack multiple untested changes

## Dev Server Management

Starting a dev server (async mode):

```powershell
# Returns a terminal ID — save it
npm run dev
```

The terminal ID lets you:
- Check output: `get_terminal_output(id)`
- Kill it when done: `kill_terminal(id)`

**Always kill dev servers** when done with testing. Leaked servers consume ports and cause `EADDRINUSE` errors on next run.

Verify the server is actually up before testing:

```powershell
# Wait for the server to log its URL, then verify
Invoke-WebRequest http://localhost:3000 -Method Head
```

## Linting as a Gate

Lint before every commit. Zero warnings policy:

```powershell
npx eslint src --ext .ts,.tsx --max-warnings 0
```

If lint fails:
- **Don't add `eslint-disable` comments** — fix the code
- Exception: external library type issues that can't be fixed without patching the library

## Rules

- **Always read exit code AND output** — exit 0 with error messages in stdout is a thing
- **Never suppress warnings** to make a build pass — fix them
- **Kill async terminals** when done — don't litter background processes
- **Run typecheck before build** — catching type errors before bundler errors saves time
- If a command times out repeatedly, check if it's hanging on user input — some tools prompt interactively
