# GUI Features (Desktop)

Purpose: Visual management for training, brains, chat, datasets, resources, and early Subbrains (experts) administration.

Status: Implemented core panels. Subbrains Manager is WIP (view and refresh registry, actions are placeholders).

Key modules:
- App bootstrap and panel wiring: `src/aios/gui/app/app_main.py`, `src/aios/gui/app/panel_setup.py`, `src/aios/gui/app/logging_setup.py`
- Services: `src/aios/gui/services/__init__.py`, `router.py`, `log_router.py`
- Panels:
	- HRM Training: `src/aios/gui/components/hrm_training_panel/`
	- Brains: `src/aios/gui/components/brains_panel/`
	- Rich Chat: `src/aios/gui/components/rich_chat_panel/`
	- Datasets: `src/aios/gui/components/datasets_panel/`
	- Subbrains Manager (WIP): `src/aios/gui/components/subbrains_manager_panel/`
	- Resources: `src/aios/gui/components/resources_panel/`
	- Settings/Status: `src/aios/gui/components/*`

See also:
- Core training: `CORE_TRAINING.md`
- Dynamic Subbrains (MoE): `DYNAMIC_SUBBRAINS_MOE.md`
- Memory optimization: `MEMORY_OPTIMIZATION.md`
- Multi-GPU and parallel chunk system: `MULTI_GPU_DISTRIBUTED.md`, `PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md`
- GUI-specific multi-GPU behaviour: `MULTIGPU_GUI_GUIDE.md`

## Panels and flows

### HRM Training panel
- Create/train ACTv1 brains with presets, VRAM estimator, logs, and stop/resume.
- MoE awareness in estimator (shows total vs active params).
- Where: `src/aios/gui/components/hrm_training_panel/`
- Screenshot: `../../Screenshots/training_tab.png`

Tips (Windows):
- Prefer “Parallel Independent” mode for multi-GPU on Windows (DDP is CUDA-only and more finicky). See `MULTI_GPU_DISTRIBUTED.md`.
- Use Optimize to search max context/batch fitting VRAM.

### Brains panel
- List brains (bundles), load/unload, view details, export/import.
- Scans `artifacts/brains` and temporary “router-generated” brains.
- Where: `src/aios/gui/components/brains_panel/`
- Screenshot: `../../Screenshots/brains_tab.png`

### Rich Chat panel
- Chat with loaded brain, Markdown and code highlighting.
- Routing output formatter is shared service (`router.py`).
- Where: `src/aios/gui/components/rich_chat_panel/`
- Screenshot: `../../Screenshots/chat_tab.png`

### Datasets panel
- Discover/download datasets, verify metadata, manage cache.
- Where: `src/aios/gui/components/datasets_panel/`
- Screenshot: `../../Screenshots/datasets_tab.png`

### Resources panel
- CPU/GPU utilization targets, throttle modes, real-time stats for training.
- Where: `src/aios/gui/components/resources_panel/`
- Screenshot: `../../Screenshots/resources_tab.png`
- Multi-GPU usage caveats and troubleshooting: see `MULTIGPU_GUI_GUIDE.md`

### Subbrains Manager (WIP)
- View expert registry with counts (total/active/frozen), hierarchy, average routing weight, activations.
- Refresh reads `artifacts/experts/registry.json`. Actions like create/delete/freeze and goal linking currently print “CLI command needed”. Use CLI instead (see `DYNAMIC_SUBBRAINS_MOE.md`).
- Where: `src/aios/gui/components/subbrains_manager_panel/`
- Screenshots:
	- (WIP) No screenshot yet; see Brains/Resources tabs for related flows

### Debug panel
- Centralized debug output with categorized logs via LogRouter; helpful during training/inference.
- Where: `src/aios/gui/app/logging_setup.py` registers the log router; debug UI integrated in app.
- Screenshot: `../../Screenshots/debug_tab.png`

### Evaluation panel
- Run evaluation suites and view results; compare runs.
- Where: `src/aios/gui/components/evaluation_panel/` (if separated) or integrated evaluation view.
- Screenshot: `../../Screenshots/evaluation_tab.png`

### Tools & MCP panel
- Manage Tools and MCP servers; configure and observe status.
- Where: `src/aios/gui/components/tools_panel/`, `src/aios/gui/components/mcp_panel/` (module names may vary by integration status)
- Screenshots:
	- Tools: `../../Screenshots/mcp&tools_tools_tab.png`
	- MCP: `../../Screenshots/mcp&tools_mcp_tab.png`

### Settings and Themes
- Configure themes, fonts, logging level, cache, and other app settings.
- Where: `src/aios/gui/components/settings_panel/`
- Screenshots:
	- Settings: `../../Screenshots/settings_tab.png`
	- Themes: `../../Screenshots/darkmode_theme.png`, `../../Screenshots/halloween_theme.png`, `../../Screenshots/barbie_theme.png`, `../../Screenshots/matrix_theme.png`

## Running the GUI

From a configured environment, start the app:

```powershell
aios gui
```

If you need to pin a specific Python interpreter, ensure `.venv` is activated first.

## Logging and routing
- Unified log router shows categorized logs across panels (chat, training, dataset, error).
- Training logs are written to JSONL files you select in the training panel; many features (MoE, PEFT, memory) surface metrics there.

## Notes
- Subbrains actions are WIP; rely on CLI for expert training and goals linking.
- GPU stats charts may update in bursts; for stable readings, let training run for a few dozen steps.

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)