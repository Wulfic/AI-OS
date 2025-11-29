# AI-OS Installer & Runtime TODOs

## Workstream 1 — Inno Setup experience
- [x] Embed license agreement and disk-usage estimate directly into the Inno Setup wizard using Pascal scripting (no external PowerShell window). The scripted preflight should reuse `installers/scripts/install_aios_on_windows.ps1` in quiet mode to compute space for core app + dependencies and surface the number before install starts.
- [x] Replace the separate PowerShell EULA prompt with an installer page that previews terms and includes a progress-aware status area fed by the PowerShell script’s log output (pipe to a temporary log and parse it for UI updates).
- [ ] Ensure the Inno bootstrapper shows real-time progress bars while the PowerShell helper provisions Python, dependencies, and environment assets.
- [ ] Add an explicit prerequisite check page that summarizes Python/Git/Node install decisions and admin elevation state so the user knows why additional installers might run.
- [ ] After core install, automatically launch a post-install page that will handle brain download (workstream 4) and any cache seeding actions while keeping the wizard modal.

## Workstream 2 — Privileged launch story
- [ ] Generate a lightweight EXE wrapper (e.g., C# or Win32 manifest stub) for `launcher.bat` that carries `requireAdministrator` in its application manifest.
- [ ] Update installer shortcuts (Desktop + Start Menu) to point at the elevated wrapper so AI‑OS always launches with admin rights as requested.
- [ ] Keep the PowerShell helper elevated for all follow-on actions (brain download, cache seeding) to avoid permission drift between installer steps.
- [ ] Ensure the wrapper passes through command-line args and sets the working directory to the install root so CLI flows (e.g., `aios --help`) still work from shortcuts.
- [ ] Add documentation in `README.md` explaining that AI-OS launches with admin rights by default and why (write access to ProgramData artifacts, GPU scheduling, etc.).

## Workstream 3 — Mandatory dependencies
- [ ] Update `pyproject.toml` / setup metadata so the installer always runs `pip install -e ".[ui,hf,eval]"`, guaranteeing `lm-evaluation-harness` ships with every build.
- [ ] Remove fallback code paths in `install_aios_on_windows.ps1` that skipped the HF/eval extras, and add a verification step that imports `lm_eval` before the installer exits.
- [ ] Pre-create `.lm_eval_cache` inside the user data directory after install to avoid permission errors on first evaluation run.
- [ ] Expand the verification step to check for other runtime-critical packages (tkinterweb, FlashAttention if CUDA, bitsandbytes when appropriate) so users get a single consolidated failure report.
- [ ] Update documentation/tooltips inside the evaluation panel to reflect that lm-eval is bundled and explain how to repair it via `aios doctor` if import fails later.

## Workstream 4 — Optional brain download (GitHub-driven)
- [ ] After the base install succeeds, prompt the user (default "Yes") to download the pretrained brain from `https://github.com/Wulfic/AI-OS/releases/download/Brain%2FModel/English-v1.zip`.
- [ ] Fetch the ZIP plus its metadata (size + checksum) via the GitHub release API; validate the download matches the published info before extraction.
- [ ] Extract into `%ProgramData%\AI-OS\artifacts\brains\actv1\English-v1`, update `masters.json`, and surface success/failure in the installer UI with retry instructions.
- [ ] Provide a CLI hook (e.g., `aios brains fetch --preset English-v1`) that reuses the same GitHub metadata logic so users can re-download from the UI later.
- [ ] If extraction fails (disk full, antivirus), roll back partial directories and log the reason to `logs/installer_brain_download.log` for easy troubleshooting.

## Workstream 5 — Writable storage layout
- [x] Introduce a `UserPaths` helper that maps logs/config/state/artifacts to `%LOCALAPPDATA%` and `%ProgramData%` on Windows (and the Linux/macOS equivalents).
- [x] Update modules that currently write under `C:\Program Files\AI-OS` (async logging, settings panel, config loader, state manager, training/artifact writers) to use the helper and migrate legacy files on first run.
- [x] Add sanity checks at startup to confirm the resolved directories are writable; abort with actionable guidance if not.
- [x] Modify `aios/utils/async_logging.py` session tracking so it stores `_SESSION_STATE_FILE` under the user cache dir instead of `logs/`, eliminating the WinError 5 spam from the report.
- [x] Update `aios/gui/components/resources_panel/config_persistence.py` and training workflows to default to ProgramData artifacts paths; expose per-user overrides in settings with validation messages.
- [x] When migrating config or state files, back up the old copy next to the new one (e.g., `gui_state.json.bak`) and note the action in the log.

## Workstream 6 — Runtime polish
- [x] Patch the TkinterWeb help panel wrapper so `safe_post_event` mirrors the original signature (`*args, **kwargs`) and prevents the fetch_styles thread crash.
- [x] Guard GPU signal handler registration so only the main thread attempts `signal.signal`, eliminating the "signal only works in main thread" warnings even in admin mode.
- [ ] Point training/evaluation output defaults at the new ProgramData artifact path and surface clearer GUI messages when prerequisite assets (e.g., `actv1_student.safetensors`) are missing.
- [ ] Address the application hang on shutdown by ensuring `state_management` worker threads flush/sync writes before exit (add a timeout and force-close logging handlers on `app.on_close`).
- [ ] Improve error dialogs for permission denials (logs/config/artifacts) so they direct users to rerun as admin or adjust folder permissions rather than silently logging.
- [ ] Add a diagnostics command (`aios doctor permissions`) that checks the new user-path directories for write access and warns if the launcher is not elevated.

## Verification
- [ ] Rebuild the Inno Setup installer and run it as a standard user to confirm the wizard never spawns an external console, shows disk usage up front, and optionally downloads the brain.
- [ ] Launch AI‑OS from the new admin-required shortcut; verify logs, configs, and state files land in the new user-data folders while artifacts live under ProgramData.
- [ ] Run an evaluation to confirm `lm_eval` is present and the UI no longer raises missing-dependency errors, then rerun the same flow on Ubuntu to validate cross-platform path helpers.
- [ ] Regression-test training and inference flows that touch `artifacts/brains` to ensure the ProgramData relocation didn’t break relative-path assumptions in CLI tools and background jobs.
- [ ] Validate the optional brain download logic on both online and offline installers (simulate offline error paths) to confirm retries/failures are user-friendly.
- [ ] Capture before/after install metrics (time to install, size used) to quantify improvements and feed into release notes.
