# Tools and Integrations
Generated: December 12, 2025
Purpose: Built-in tools and OS/HF integrations with runnable commands, inputs/outputs, and platform notes
Status: Implemented (some features are gated/experimental and noted below)

## Web tools (search, crawl, datasets)
- Source: `src/aios/tools/web.py`, `src/aios/tools/crawler.py`
- What you get:
	- DuckDuckGo HTML-only search with ad/redirect filtering (`ddg_search`)
	- Robust HTML parsing and link extraction
	- Polite crawler with robots.txt, BFS, per-origin throttling, optional Playwright rendering and Trafilatura extraction
	- Turn searches into datasets via CLI builders (images, text, videos, websites)

### Configuration knobs
- Env vars (affect search/crawl):
	- `AIOS_DDG_KL`, `AIOS_DDG_KAD` → locale/region for DDG (defaults from `config/default.yaml:web.ddg_params`)
	- `AIOS_WEB_UA` / `AIOS_WEB_UA_SUFFIX` → user agent override/suffix
- YAML: `config/default.yaml:web` mirrors the above defaults and can be edited.

### Crawl a URL → JSON summary or dataset
- Command (Windows PowerShell):
	- One page fetch/parse
		- aios crawl https://example.com --ttl-sec 3600 --progress
	- Recursive BFS within the same domain, rate-limited
		- aios crawl https://example.com --recursive --max-pages 25 --max-depth 2 --rps 2 --progress
	- Store pages as a text dataset JSONL under datasets pool
		- aios crawl https://example.com --recursive --store-dataset web_crawl/example --overwrite --progress
- Inputs/flags (subset):
	- `--ttl-sec` cache TTL seconds; `--render` Playwright render; `--trafilatura` article extraction; `--rps` or `--delay-ms` throttling
	- `--same-domain/--any-domain`, `--max-pages`, `--max-depth`, `--progress`
	- `--store-dataset NAME` outputs to datasets pool at NAME/data.jsonl; `--overwrite` to reset
- Outputs:
	- Progress: JSONL lines on stdout when `--progress` is set, e.g. {"event":"page","n":1,...}
	- Final JSON: pages summary, count, total_chars, and dataset_path/wrote_bytes when `--store-dataset` is used
- Notes:
	- Respect robots.txt by default; pass `--no-robots` for tests only
	- Playwright requires browser install; on Windows run once: playwright install

### Build datasets from the web
- Images
	- Command:
		- aios datasets-build-images "boats" --store-dataset boats_v1 --max-images 200 --per-site 40 --pages-per-site 8 --search-results 10 --rps 2 --progress
	- Inputs: `--allow-ext jpg,png,webp`, `--near-duplicate-threshold 8`, `--file-prefix boats`
	- Outputs: `artifacts path`: datasets/images/boats_v1 with image files + manifest.jsonl (path,label,source_url,page_url,title,alt)
- Text
	- Command:
		- aios datasets-build-text "boats" --store-dataset boats_text_v1 --max-docs 100 --search-results 10 --min-chars 400 --progress
	- Inputs: `--allow-ext txt,pdf,docx` (if set, fetches documents by extension/content-type); `--file-prefix`
	- Outputs: datasets/text/boats_text_v1/*.txt + manifest.jsonl (path,label,url,title,chars,excerpt)
- Videos
	- Command:
		- aios datasets-build-videos "boats" --store-dataset boats_vid_v1 --max-videos 25 --per-site 5 --min-bytes 50000 --progress
	- Inputs: `--allow-ext mp4,webm,mov,m4v`, `--file-prefix`
	- Outputs: datasets/videos/boats_vid_v1/*.mp4|*.webm|… + manifest.jsonl (path,label,source_url,page_url,bytes)
- Websites (HTML snapshots)
	- Command:
		- aios datasets-build-websites "boats" --store-dataset boats_sites_v1 --max-pages 30 --per-site 10 --search-results 10 --progress
	- Outputs: datasets/websites/boats_sites_v1/pages/*.html + manifest.jsonl (path,url,title,bytes,links)

Related docs: see `docs/guide/CORE_TRAINING.md` for using JSONL datasets; `docs/guide/DATASETS.md` for dataset pool and storage caps.

## Filesystem and OS tools (guarded)
- Source: `src/aios/tools/fs.py`, `src/aios/tools/os.py`
- What you get:
	- `write_text(path, data, cfg, conn)` writes a file with WriteGuard and SafetyBudget enforcement
	- `get_system_info()` returns basic platform info
- Guard/budget behavior:
	- Guards read allow/deny from config; budgets use DB-backed usage with tier defaults from `aios.core.budgets`
	- Domains charged: `file_writes`
- Example usage via CLI budgets helpers:
	- Show guard paths: aios guards-show
	- Simulate a service change budget decision: aios service-restart ssh --dry-run

## Root-helper and service adapters
- Source: `src/aios/tools/root_helper_client.py`, `src/aios/tools/service_adapter.py`
- What you get:
	- Optional privileged D-Bus client (Linux only). On Windows/macOS, gracefully returns via: "unavailable".
	- Read-only service diagnostics via local systemctl/journalctl fallback when root-helper is not available.
- Read status and logs for a unit
	- aios status --recent 1 --unit ssh
	- For targeted triage, prefer Agent CLI operators below
- CLI operators for triage (store artifacts):
	- aios op-run journal_summary_from_text --unit ssh --lines 200 --label ssh
	- aios op-run journal_trend_from_text --unit ssh --lines 200 --label ssh --buckets 12
	- Artifacts saved in DB (see Core CLI status for recent artifacts).
- Notes:
	- On Linux, providing a running root-helper yields via: "root-helper" in outputs; otherwise via: "local".
	- On Windows, these commands return via: "unavailable" (no systemd).

## Journal parser utilities
- Source: `src/aios/tools/journal_parser.py`
- Functions:
	- `severity_counts(text) -> Dict[str,int]` heuristic severity tallies across emerg…debug
- How it’s used:
	- The Agent CLI operators compute summaries/trends and persist to DB artifacts. See: `aios op-run ...` above.

## Package/service simulators (budgeted)
- Source: `src/aios/tools/service.py`, `src/aios/tools/pkg.py`, `src/aios/tools/privileged.py`
- What you get:
	- `restart_service(name, simulate=True)` and `pkg.install/remove(name, simulate=True)` record budget usage for service_changes/pkg_ops
	- `run_privileged(fn, ...)` wraps a function and charges privileged_calls budget
- Try it:
	- aios service-restart docker --dry-run
	- aios pkg-install git --dry-run

## MCP and external tools (GUI)
- Source: GUI MCP Manager panel under `src/aios/gui/components/mcp_manager_panel/*`
- What you get:
	- Visual editor for MCP servers and tool permissions using `config/mcp_servers.json` and `config/tool_permissions.json`
	- Enable/disable servers and toggle tool permissions; refresh state from disk
- Status:
	- GUI available. Programmatic MCP wiring is scoped to UI; CLI equivalents are not exposed yet.
	- If config files are missing, the panel initializes with defaults and saves back on change.
	- Panel screenshots in GUI doc: see `docs/guide/features/GUI_FEATURES.md` (MCP & Tools tab).

## Unlimiformer (planned)
- Source: `src/aios/integrations/unlimiformer/__init__.py`
- Status: Phase 1 scaffolding; disabled by default via config
- Config key (example):
	- config.default.yaml → brains.trainer_overrides.unlimiformer.enabled: false
- Notes:
	- When enabled in future phases, the model will be augmented for long-context eval using FAISS; Windows defaults to CPU FAISS.

## Quick reference (commands)
- Crawling
	- aios crawl <url> --recursive --max-pages 25 --max-depth 2 --rps 2 --progress
- Datasets builders
	- aios datasets-build-images "topic" --store-dataset name --max-images 200 --progress
	- aios datasets-build-text "topic" --store-dataset name --max-docs 100 --progress
	- aios datasets-build-videos "topic" --store-dataset name --max-videos 50 --progress
	- aios datasets-build-websites "topic" --store-dataset name --max-pages 30 --progress
- Budgets and guards
	- aios guards-show
	- aios service-restart ssh --dry-run
	- aios pkg-install git --dry-run

Related: Datasets, Advanced Features

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)