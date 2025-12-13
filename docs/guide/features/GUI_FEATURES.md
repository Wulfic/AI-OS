# Desktop GUI Features

**Last Updated**: December 12, 2025  
**Purpose**: Visual management for training, brains, chat, datasets, resources, and early Subbrains (experts) administration.

> **Note**: Screenshots are planned for a future documentation update. For now, launch `aios gui` to explore the interface.

**Status**: Implemented core panels. Subbrains Manager is WIP (view and refresh registry, actions are placeholders).

## Key Modules

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

## Related Documentation

- Core training: [CORE_TRAINING.md](CORE_TRAINING.md)
- Dynamic Subbrains (MoE): [DYNAMIC_SUBBRAINS_MOE.md](DYNAMIC_SUBBRAINS_MOE.md)
- Memory optimization: [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)
- Multi-GPU and parallel chunk system: [MULTI_GPU_DISTRIBUTED.md](MULTI_GPU_DISTRIBUTED.md), [PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md](PARALLEL_TRAINING_BLOCK_CHUNK_SYSTEM.md)
- GUI-specific multi-GPU behaviour: [MULTIGPU_GUI_GUIDE.md](MULTIGPU_GUI_GUIDE.md)

---

## Panels and Flows

### HRM Training Panel
- Create/train ACTv1 brains with presets, VRAM estimator, logs, and stop/resume.
- MoE awareness in estimator (shows total vs active params).
- **Location**: `src/aios/gui/components/hrm_training_panel/`

**Tips (Windows)**:
- Prefer "Parallel Independent" mode for multi-GPU on Windows (DDP is CUDA-only and more finicky). See [MULTI_GPU_DISTRIBUTED.md](MULTI_GPU_DISTRIBUTED.md).
- Use Optimize to search max context/batch fitting VRAM.

### Brains Panel
- List brains (bundles), load/unload, view details, export/import.
- Scans `artifacts/brains` and temporary "router-generated" brains.
- **Location**: `src/aios/gui/components/brains_panel/`

### Rich Chat Panel
- Chat with loaded brain, Markdown and code highlighting.
- Routing output formatter is shared service (`router.py`).
- **Location**: `src/aios/gui/components/rich_chat_panel/`

### Datasets Panel
- Discover/download datasets, verify metadata, manage cache.
- **Location**: `src/aios/gui/components/datasets_panel/`

### Resources Panel
- CPU/GPU utilization targets, throttle modes, real-time stats for training.
- Multi-GPU usage caveats and troubleshooting: see [MULTIGPU_GUI_GUIDE.md](MULTIGPU_GUI_GUIDE.md)
- **Location**: `src/aios/gui/components/resources_panel/`

### Subbrains Manager (WIP)
- View expert registry with counts (total/active/frozen), hierarchy, average routing weight, activations.
- Refresh reads `artifacts/experts/registry.json`. Actions like create/delete/freeze and goal linking currently print "CLI command needed". Use CLI instead (see [DYNAMIC_SUBBRAINS_MOE.md](DYNAMIC_SUBBRAINS_MOE.md)).
- **Location**: `src/aios/gui/components/subbrains_manager_panel/`

### Debug Panel
- Centralized debug output with categorized logs via LogRouter; helpful during training/inference.
- **Location**: `src/aios/gui/app/logging_setup.py` registers the log router; debug UI integrated in app.

### Evaluation Panel
- Run evaluation suites and view results; compare runs.
- **Location**: `src/aios/gui/components/evaluation_panel/` (if separated) or integrated evaluation view.

### Tools & MCP Panel
- Manage Tools and MCP servers; configure and observe status.
- **Location**: `src/aios/gui/components/tools_panel/`, `src/aios/gui/components/mcp_panel/` (module names may vary by integration status)

### Settings and Themes
- Configure themes, fonts, logging level, cache, and other app settings.
- **Location**: `src/aios/gui/components/settings_panel/`
- **Available themes**: Light Mode, Dark Mode, Matrix Mode, Halloween Mode, Barbie Mode

---

## Running the GUI

From a configured environment, start the app:

```powershell
aios gui
```

If you need to pin a specific Python interpreter, ensure `.venv` is activated first.

---

[Back to Feature Index](COMPLETE_FEATURE_INDEX.md) | [Back to Guide Index](../INDEX.MD)
