# PF-019: Web GUI Parity Control

**Status**: 🟡 Planning Phase  
**Date**: November 16, 2025  
**Priority**: High  
**Complexity**: Very High

---

## Summary

Deliver a browser-based control surface that mirrors the desktop GUI pixel-for-pixel in layout, flows, and behaviours. The WebGUI must expose every control, panel, status indicator, and workflow currently available through the Tk-based app, while remaining in sync with the existing state management, background workers, and analytics pipelines. Users will launch it locally via `aios web`, which should start a web server, open (or log) a URL, and run alongside the existing core services without requiring the desktop GUI to be present.

---

## Success Criteria

- **Functional parity**: All panels (`Chat`, `Brains`, `Datasets`, `Training`, `Evaluation`, `Resources`, `Debug`, `Settings`, `MCP Manager`, `Help`, `Output`) behave identically, surface the same live data, and update in real time.
- **Single source of truth**: State mutations performed through the browser immediately propagate to `gui_state.json` and the in-memory structures that the Tk GUI uses; no divergence when both UIs run simultaneously.
- **Cross-platform**: `aios web` works on Windows and Linux with Python-only dependencies plus a Node toolchain for front-end builds; optional Mac support documented.
- **Security baseline**: Local-only access by default with optional authentication and TLS knobs for remote deployment.
- **Observability**: Structured logs and metrics that distinguish WebGUI events from Tk GUI events without breaking the existing analytics pipeline.
- **Testing**: Automated parity regression test suite proving behaviour equivalence for critical workflows, and Playwright smoke tests for the WebGUI.

---

## Constraints & Considerations

- **UI fidelity**: Visual design must match colours, typography, spacing, and component behaviour of the Tk app. Introduce a design token map sourced from the existing theme configuration so that any future theme tweaks propagate to both UIs.
- **Shared runtime**: Reuse existing services in `src/aios/gui/app` and `src/aios/gui/components` rather than duplicating business logic. Where Tk widgets hold state, extract presenter/view-model layers so they can feed both Tk and web surfaces.
- **Async-first**: All new server components must use `asyncio`, integrate with the existing `setup_resources` loop, and avoid blocking the worker pools that power agents and training operations.
- **No breaking change**: Tk GUI must continue to run unchanged; users can run either UI independently or both concurrently.
- **Asset bundling**: Front-end assets must be buildable offline and distributable with the Python package; avoid heavyweight runtime dependencies.

---

## Proposed Architecture

```
┌─────────────────────────────┐        ┌────────────────────────────┐
│ aios web (CLI entrypoint)   │        │ Front-end (React/Vite)     │
│  • loads headless app core  │        │  • UI clone of Tk panels   │
│  • boots FastAPI+Socket.IO  │◀──────▶│  • State synced via WS/REST│
│  • serves static assets     │        │  • Theming via design tokens│
└──────────────┬──────────────┘        └──────────┬─────────────────┘
               │                                    │
               ▼                                    │
      ┌────────────────────┐                       │
      │ Shared GUI Core    │◀──────────────────────┘
      │  • Panel presenters│  Web view-models reuse│
      │  • State adapters  │  Tk services & storage│
      └─────────┬──────────┘
                │
                ▼
       Existing services, threads,
       async loop, analytics, etc.
```

### Backend Service (Python)

- Framework: FastAPI (for REST) + `uvicorn` runner with ASGI lifespan hooks tied into `setup_resources`.
- Reuse `setup_resources`, `setup_event_handlers`, and panel initialization to build a headless `AiosTkApp`-derived context without instantiating Tk widgets.
- Expose REST endpoints per panel (e.g., `/api/chat/send`, `/api/brains/list`) and WebSocket channels for event-driven updates (log stream, training progress, background tasks).
- Implement a state sync layer that watches `schedule_state_save` and repository-specific caches, emitting change diffs to clients.

### Front-End (TypeScript)

- Stack: React 18 + Vite + React Query/Zustand for state management; Tailwind or CSS modules constrained by design tokens extracted from Tk theme definitions.
- Component library: Build custom components that match Tk layouts; consider using Blueprint.js or Ant Design only if they can be themed precisely; otherwise, bespoke components ensure parity.
- Navigation: Mirror Tk notebook tabs as horizontal navigation; subpanels map 1:1.
- Real-time data: WebSocket subscriptions for logs, progress bars, chat streaming; REST for CRUD operations.

### CLI Entry (`aios web`)

- Add subcommand in `src/aios/cli/aios.py` wiring to `aios.cli.web_cli:run`.
- Responsibilities: ensure venv dependencies, start FastAPI service, build (or verify) front-end assets, optionally open the browser (respect `--no-open` flag), log host/port.
- Flags: `--host`, `--port`, `--theme`, `--open/--no-open`, `--auth-token`, `--certfile`, `--keyfile`, `--allow-remote` (with warnings).

---

## Implementation Phases

### Phase 0 – Discovery & Parity Mapping

- Catalogue every panel, widget, and command in the Tk UI; produce a JSON inventory describing fields, events, validation rules, and dependencies.
- Document where state lives (e.g., `state_management`, panel-specific caches) and identify Tk-only code paths that must be abstracted.
- Identify blocking dependencies (e.g., direct Tk widget calls inside services) and plan refactors to extract logic into reusable classes.

### Phase 1 – Core Refactor for Shared State

- Introduce a `src/aios/gui/core` (or expand existing `core`) module that exposes headless presenters/controllers for each panel.
- Update Tk panels to consume the new presenters via dependency injection so they remain functional.
- Ensure presenters expose async-safe methods and emit typed events (e.g., via `asyncio.Queue` or RxPy) that the Web backend can subscribe to.
- Extend state persistence to broadcast diffs whenever `schedule_state_save` runs, enabling live reload in the browser.

### Phase 2 – Web Backend Skeleton

- Scaffold FastAPI app under `src/aios/webui/server.py` integrating shared presenters.
- Implement auth middleware (token header) and optional HTTPS.
- Provide endpoints per panel in read/write parity (e.g., GET `/api/settings`, POST `/api/settings/theme`).
- Set up Socket.IO or Starlette WebSocket endpoints for:
  - Log/event stream (`/ws/logs`)
  - Status updates (resources, training progress)
  - Chat streaming (token-by-token)
- Add background tasks that bridge presenter events to WebSocket broadcasts.

### Phase 3 – Front-End Application

- Create `webui/` directory (ignored by lint when not built) containing Vite project with React + TypeScript + ESLint/Prettier.
- Define design tokens JSON generated from Tk theme definitions; load at runtime to guarantee parity.
- Build tab components mimicking Tk layout; incorporate virtualization where Tk uses scrollable lists.
- Implement data bindings with React Query for REST (CRUD) and custom hooks for WebSocket streams.
- Provide fallback skeleton states replicating Tk loading overlays.

### Phase 4 – Packaging & CLI Integration

- Integrate Vite build into Python packaging via `pyproject.toml` optional dependencies (`webui`).
- Update `setup.cfg` / entry points to include `aios web` command; ensure `aios` console script imports minimal dependencies until the command runs.
- Provide `scripts/build_webui.sh` and `.ps1` for dev ergonomics; integrate into CI using npm caches.
- Ensure built assets land under `src/aios/webui/static/` and are included with `pkgutil.get_data` for offline use.

### Phase 5 – Verification & Docs

- Implement automated parity tests:
  - Presenter unit tests covering CRUD flows.
  - Snapshot/state diff tests comparing Tk + Web responses given identical fixtures.
  - Playwright smoke suite covering critical workflows (chat generation, brain selection, dataset upload, training start/stop, settings save).
- Update `docs/guide/` with WebGUI usage, security warnings, remote deployment guide.
- Record demo walkthrough once stable.

---

## Testing Strategy

- **Unit tests**: For presenters and API contracts using pytest + httpx AsyncClient.
- **Integration tests**: Launch FastAPI app in-process, run Playwright headless tests to validate UI flows; reuse fixtures to ensure deterministic data.
- **Concurrency tests**: Simulate simultaneous Tk + Web interactions to verify lock-free state updates and absence of race conditions (use `pytest-asyncio` with `asyncio.Event` barriers).
- **Performance tests**: Benchmark WebSocket throughput and latency for chat streaming; ensure under-load behaviour matches Tk event loop responsiveness.

---

## Security & Deployment Notes

- Default bind: `127.0.0.1:8123`. Require explicit `--allow-remote` to bind to `0.0.0.0` and show warning banner in UI.
- Token-based auth stored in `~/.config/ai-os/webgui_token`; optional `AIOS_WEB_TOKEN` env override.
- Optional TLS via user-provided `--certfile/--keyfile`.
- Document reverse-proxy guidance (Caddy/Nginx) for secure remote access.

---

## Open Questions / Risks

- **Tk dependency**: Some panels might rely on direct widget references; refactor effort may be substantial.
- **Theme fidelity**: Tk theming primitives differ from CSS; will need a reliable mapping process.
- **Front-end build size**: Need to ensure packaged assets keep overall wheel size acceptable; may need optional extra.
- **Simultaneous sessions**: Determine how to handle multiple browsers; probably allow read-only by default, configurable to accept multi-writer with event conflict resolution.
- **Accessibility**: Evaluate if Tk shortcuts (e.g., keyboard accelerators) must be mapped to the Web UI for parity.

---

## Deliverables Checklist

- Shared presenter layer with parity coverage for each panel.
- FastAPI WebGUI server with REST + WebSocket endpoints.
- React/Vite front-end that mirrors the Tk UI.
- CLI command `aios web` with documented options.
- Parity test suite and Playwright automation in CI.
- Updated documentation in `docs/guide/` and release notes.
- Security review artefacts (threat model, auth configuration).

---

## Next Steps

1. Approve architecture direction (shared presenters + FastAPI + React).
2. Spin up Phase 0 discovery ticket to inventory Tk components.
3. Allocate resources for refactor + front-end build, including Node environment setup in CI.
