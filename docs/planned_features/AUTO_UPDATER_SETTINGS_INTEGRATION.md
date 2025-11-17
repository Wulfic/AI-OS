# PF-020: Settings Auto Updater Integration

**Status**: Planning Phase  
**Date**: November 16, 2025  
**Priority**: High  
**Complexity**: Medium-High

---

## Summary

Introduce a first-party auto updater that is discoverable and controllable from the Settings tab. The updater must run as an external helper process so it can shut down the running AI-OS instance, apply the new build, and relaunch the application without user scripting. Version detection relies on GitHub Releases: when the latest tagged semantic version exceeds the local version, the updater downloads and installs the release; otherwise it reports that the installation is current. The main application remains responsible for configuring, scheduling, and updating the updater helper through the Settings menu.

---

## Success Criteria

- Users can navigate to Settings ▸ Updates, trigger a check, and see clear status messaging (up to date, update available, update running, error).
- When an update is available and approved, the helper process gracefully shuts down the main app, replaces binaries/scripts, and restarts AI-OS without manual steps.
- Update logic works on Linux and Windows using consistently packaged artifacts and paths.
- Version comparison uses semantic version precedence and tolerates pre-release identifiers.
- Update helper is itself versioned and refreshed by the main app when a newer helper package ships.
- Update operations log to a dedicated channel consumable by diagnostics tooling.

---

## User Flow (Settings Tab)

1. Settings ▸ Updates view exposes controls for manual check, auto-check interval, update channel (stable vs pre-release), and viewing the update log.
2. Selecting "Check for updates" triggers an async call to the GitHub Releases API, comparing `latest.tag_name` to `aios.__version__`.
3. If the remote version is newer, the UI shows release notes and prompts the user to "Update and Restart"; otherwise it reports "AI-OS is up to date".
4. On acceptance, the main process persists update intent (temporary manifest) and spawns the updater helper.
5. The UI displays progress streamed from the helper; once complete the application either restarts automatically or prompts the user if relaunch fails.

---

## System Architecture

```
+---------------------+        +-------------------------+
| Main AI-OS Process  |        | Updater Helper Process  |
| (Settings Module)   |        | (Standalone Python exe) |
|                     |        |                         |
| - Render Settings UI|        | - Validate manifest     |
| - Call GitHub API   |        | - Close main process    |
| - Compare versions  |        | - Download release      |
| - Spawn helper      | -----> | - Verify checksum       |
| - Stream progress   | <----- | - Install & relaunch    |
+---------------------+        +-------------------------+
```

**Key Interfaces**
- A `VersionService` abstraction encapsulates GitHub Release discovery, environment overrides, and semantic comparison.
- An `UpdaterTransport` (likely named pipe on Windows, UNIX domain socket on Linux, TCP fallback) carries JSON-RPC style progress events between the helper and the Settings UI.
- The helper process is packaged as either a console-script entry point (`python -m aios.updater`) or self-contained executable produced via PyInstaller for Windows convenience.

---

## Functional Requirements

- **Version Source**: Query `https://api.github.com/repos/Wulfic/AI-OS/releases` (with auth token fallback) respecting rate limits and caching results for configurable intervals.
- **Artifact Selection**: Resolve the correct asset per platform (Linux tarball, Windows installer). Configurable via Settings for air-gapped environments.
- **Graceful Shutdown**: Send a structured shutdown command so the main process flushes state (`gui_state.json`, running jobs) before the helper forcibly terminates as a fallback.
- **Install Location Awareness**: Detect whether AI-OS runs from editable source, pip install, or packaged binary; block or warn if the install layout does not support automated replacement.
- **Rollback Point**: Before installing, create a restore point (copy or rename current install) to enable quick rollback if restart fails.
- **Restart Strategy**: After successful install, re-launch using the same command line the user invoked (respecting venv and CLI arguments). On failure, surface the necessary re-launch command.
- **Updater Self-Update**: During normal runs the main process checks the helper version. If mismatched, it downloads the new helper payload and updates the helper before offering core updates.

---

## Constraints & Considerations

- **Cross-Platform**: Paths, process signalling, and relaunch commands must support Linux (systemd and raw shell) and Windows (PowerShell, Start-Process). macOS support is optional but should not be broken.
- **Security**: Release downloads must be checksum-verified (SHA256 from `SHA256SUMS.txt`), optionally signed. Support GitHub API tokens stored securely in credentials manager.
- **Offline / Air-Gapped**: Provide documented override allowing manual placement of update packages and offline installation.
- **Async Integration**: All Settings-side operations must be non-blocking, using existing asyncio loop and thread executors. UI progress should update without freezing other tabs.
- **Error Handling**: Categorize errors (network, permission, disk space) with user-facing remediation tips.
- **Telemetry**: Emit structured update events to analytics with opt-out respect.

---

## Implementation Plan

### Phase 0 – Discovery & Design (1 week)
- Audit current Settings tab implementation and identify UI extension points.
- Confirm packaging formats for Linux (tar.gz or wheel) and Windows (`.zip` or `.exe`).
- Define semantic version comparison rules and finalize API contract for helper transport.
- Document shutdown sequence requirements for core subsystems.

### Phase 1 – Version & Settings Wiring (1-2 weeks)
- Implement `VersionService` with GitHub Releases polling, caching, and comparison utilities.
- Extend Settings view with "Updates" panel using existing GUI architecture.
- Provide manual check, channel selection, auto-check scheduling, and state persistence in `gui_state.json`.

### Phase 2 – Helper Process Scaffold (2 weeks)
- Create `src/aios/updater/__main__.py` containing CLI entry point for helper process.
- Implement manifest ingestion (paths, desired version, restart command).
- Add inter-process channel with heartbeat and progress events.
- Implement graceful shutdown handshake with main process (signal or RPC) with fallback kill-after-timeout logic.

### Phase 3 – Download, Verify, Install (2 weeks)
- Integrate streaming download with resume support and checksum verification.
- Extract or run platform-specific installer into target directory, backing up current installation.
- Implement rollback procedure triggered when restart health check fails.

### Phase 4 – Relaunch & UX Polish (1 week)
- Capture original invocation command (argv, environment) and reuse for restart.
- Update Settings UI to display progress, logs, and final status from helper events.
- Add notifications/log entries for success, failure, rollback actions.

### Phase 5 – Updater Maintenance Channel (1 week)
- Package helper assets alongside main releases; add manifest metadata so the main app can refresh the helper.
- Document fallback manual update process if helper fails, and provide telemetry dashboards.

---

## Testing Strategy

- **Unit Tests**: Version comparator edge cases, GitHub API response parsing, manifest serialization, restart command builder.
- **Integration Tests**: Use temporary directories to simulate install roots, running helper end-to-end against mock release server.
- **Process Tests**: Automated tests that spawn a dummy main process, verify the helper closes it, writes backups, and restarts a stub executable.
- **Windows & Linux Matrix**: CI jobs run update integration suite on both OSes; include PowerShell script validation for Windows.
- **Failure Injection**: Simulate network errors, checksum mismatch, insufficient disk, and permission errors to ensure retries and rollback behave correctly.
- **Smoke Tests**: Manual or scripted validation against staging GitHub releases prior to public release.

---

## Documentation & Support

- Update `docs/guide/` with an "Updates" section covering manual checks, auto-check scheduling, logs, and troubleshooting.
- Add FAQ entries addressing rate limits, offline installs, and helper location.
- Provide release note template entries calling out auto updater behaviour changes.
- Ensure logging.yaml includes dedicated `auto_updater` logger with rotation guidance.

---

## Open Questions & Risks

- How are release artifacts structured today, and do they contain platform installers that support unattended runs?
- What permissions does the running user require to overwrite installation directories, especially on Windows Program Files?
- Do we need signature verification beyond checksums for enterprise environments?
- How do we migrate existing users installed from source clones where automated replacement is unsafe?
- Should auto-check be opt-in or default-on (with configurable intervals)?

---

## Deliverables Checklist

- Settings ▸ Updates UI with manual and scheduled check controls.
- `VersionService` and helper manifest schema committed with unit coverage.
- Updater helper package capable of graceful shutdown, download, install, relaunch, and rollback.
- Platform-specific restart logic validated on Linux and Windows.
- CI jobs executing update integration tests across platforms.
- Documentation updates and release note templates.
- Telemetry and logging integration for update lifecycle events.

---

## Next Steps

1. Review and approve architecture and phase estimates with stakeholders.
2. Identify engineers for Settings UI, helper process, and packaging streams.
3. Schedule Phase 0 tasks and prepare GitHub Release environment (test releases, tokens).
