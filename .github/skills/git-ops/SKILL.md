---
name: git-ops
description: "Git operations with enforced conventions: branch from issues, conventional commits, clean history, tagging releases. Use when: starting work on a new feature or bug fix (create branch), committing completed work, creating a release tag, cleaning up branches, or enforcing commit message standards. Does NOT push or force-push without confirmation."
argument-hint: "What git operation (e.g. 'create branch for issue #12', 'commit current changes', 'tag release v1.2.0')"
---

# Git Ops — Branch, Commit, Tag

Git history is a permanent record. Every commit message, every branch name, every tag communicates intent to future developers (including you in 6 months). Do it right the first time.

## When to Use

- Starting work on a new feature or bug (create a branch)
- Completing work and committing
- Creating a release
- Cleaning up merged branches
- Reviewing what changed before committing

## Branch Naming Convention

```
feature/<issue-number>-<short-slug>   # new features
fix/<issue-number>-<short-slug>       # bug fixes
chore/<short-slug>                    # maintenance, deps, config
docs/<short-slug>                     # documentation only
refactor/<short-slug>                 # refactors (no behavior change)
```

Examples:
- `feature/42-user-authentication`
- `fix/17-login-redirect-loop`
- `chore/update-dependencies`

**Always branch from `main`** unless directed otherwise:

```powershell
git checkout main
git pull origin main
git checkout -b feature/<issue-number>-<slug>
```

## Conventional Commits

Every commit must follow this format:

```
<type>(<scope>): <short description>

[optional body — what and why, not how]

[optional footer: Closes #N, Breaking-Change: ...]
```

### Types

| Type | When |
|------|------|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Build, deps, config (no production code) |
| `docs` | Documentation only |
| `refactor` | Code change without behavior change |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |

### Scope

Optional but encouraged — the module or area affected:
- `feat(auth): add JWT refresh token support`
- `fix(api): handle null response from user endpoint`

### Rules for the Subject Line

- Lowercase, no period at the end
- Imperative mood: "add feature" not "added feature" or "adds feature"
- Max 72 characters
- Be specific: "fix login redirect loop on token expiry" not "fix bug"

## Commit Procedure

### Step 1 — Review what you're committing

```powershell
git status
git diff --staged
```

Never commit without reviewing the diff. `git add .` followed by a blind commit is how secrets and debug code get into repos.

### Step 2 — Stage intentionally

```powershell
# Stage specific files (preferred)
git add src/auth/handler.ts tests/auth.test.ts

# Stage all tracked changes (only after reviewing git status)
git add -u

# Never use git add . without reviewing — it picks up everything including temp files
```

### Step 3 — Commit with a conventional message

```powershell
git commit -m "feat(auth): add JWT refresh token rotation"
# or with body
git commit -m "fix(api): handle null user response

The /users endpoint can return null for deactivated accounts.
Previously this caused an unhandled exception in the serializer.

Closes #34"
```

### Step 4 — Verify the commit

```powershell
git log --oneline -3
```

Confirm the message and files look right.

## Tagging Releases

Use semantic versioning: `MAJOR.MINOR.PATCH`

- `MAJOR`: breaking change
- `MINOR`: new feature, backward compatible
- `PATCH`: bug fix, backward compatible

```powershell
# Annotated tag (required for releases — not lightweight tags)
git tag -a v1.2.0 -m "Release v1.2.0 — adds JWT refresh token rotation"

# List tags
git tag -l
```

After tagging, push to remote:
```powershell
git push origin main
git push origin v1.2.0
```

**Confirm with the user before pushing.** Pushing is not reversible without a force-push.

## Pre-Commit Checklist

Before every commit, verify:

- [ ] `get_errors()` returns zero errors
- [ ] Lint passes: `npx eslint src --max-warnings 0`
- [ ] Type-check passes: `npx tsc --noEmit`
- [ ] Tests pass: `npm test`
- [ ] `git diff --staged` reviewed — no debug code, no secrets, no `.env` files
- [ ] Commit message follows conventional commits format
- [ ] `TODO.md` updated to reflect completed items

## Branch Cleanup

After a PR is merged:

```powershell
git checkout main
git pull origin main
git branch -d feature/<slug>   # local cleanup
```

Remote branch cleanup is handled by GitHub's "auto-delete head branches" setting (enable this in repo settings).

## Rules

- **Never commit directly to `main`** — always use a branch + PR
- **Never commit secrets, API keys, or `.env` files** — check `.gitignore` before staging
- **Never use `--force` without explicit user confirmation** — force-push rewrites history for everyone
- **One logical change per commit** — don't bundle unrelated changes
- **Commit early, commit often** — small commits are easier to revert and review
- If `git status` shows files you don't recognize, investigate before staging anything
