# Multi-GPU GUI Guide
**Last Updated**: October 21, 2025  
**Purpose**: Explain how the AI-OS desktop GUI exposes multi-GPU selection and fallbacks across operating systems.  
**Status**: Linux multi-GPU routing supported; Windows currently falls back to a single CUDA device.

---

## Overview
The Resources panel lets you pick one or more CUDA devices for inference. Those selections feed shared helpers in `src/aios/gui/services/resource_selection.py`, so both chat and evaluation flows inherit the same rules:
- Device selections persist between sessions via GUI state.
- Analytics events capture which GPU set the user attempted to run.
- When platform capabilities change (Linux ↔ Windows), the GUI reconciles the stored selection and surfaces warnings.

See also: [Multi-GPU & Distributed Training](MULTI_GPU_DISTRIBUTED.md) for CLI-focused orchestration details.

---

## Current Behaviour by Platform
### Windows (single-GPU fallback)
- Windows builds always run inference on the first CUDA device. Additional GPUs in the selector are kept for reference but ignored by the runtime.
- The GUI surfaces a warning banner and status message if a multi-GPU selection is detected on Windows.
- Chat and evaluation both log `windows_single_gpu_fallback` analytics events so telemetry captures the limitation.

### Linux (multi-GPU routing)
- Evaluation fan-out runs each selected device sequentially via `MultiGpuEvaluationRunner`, emitting per-device analytics entries.
- Chat currently launches tensor-parallel stubs that lock to GPU 0 while foundation work for true tensor-parallel routing continues. A warning tooltip clarifies this.
- Device lists longer than eight entries are truncated with a warning to keep tooltips concise.

---

## Verifying CUDA Visibility
Make sure the environment exposes only the GPUs you intend to target.

**PowerShell**
```powershell
Get-ChildItem Env:CUDA_VISIBLE_DEVICES
$Env:CUDA_VISIBLE_DEVICES = "0,1"
```

**Bash**
```bash
printenv CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="0,1"
```

After updating the variable, restart the GUI so the selector refreshes the list of available devices.

---

## Status Messages and Logs
- The Resources panel displays inline warnings and sets a status message any time the OS forces a fallback.
- The shared selector returns `warning_message(...)` strings that are re-used in chat, evaluation, and GUI tooltips for consistency.
- Warnings are also forwarded to the log router so they appear in the Debug panel.

---

## FAQ
**Why does chat still label GPU 0 as primary on Linux?**  
Tensor-parallel scaffolding is in place, but the router still executes on the first device until tensor slicing ships. Track progress in [planned DDP/tensor parallel work](../planned_features/DEEPSPEED_ZERO_INFINITY.md).

**Why do I see a Windows warning even after switching to Linux?**  
If you carried over a persisted state file from Windows, the GUI logs that the OS changed and re-normalises your selections. Open the Resources panel once to re-save the Linux layout.

**Where is the full roadmap for chat-time multi-GPU?**  
See [MULTIGPU_INFERENCE_TODO.md](../planned_features/MULTIGPU_INFERENCE_TODO.md) for outstanding tasks and design notes.

---

Back to Feature Index: [COMPLETE_FEATURE_INDEX.md](COMPLETE_FEATURE_INDEX.md) • Back to Guide Index: [../INDEX.MD](../INDEX.MD)
