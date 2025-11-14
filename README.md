# AI-OS v1.0.0

[![CI](https://github.com/Wulfic/AI-OS/actions/workflows/ci.yml/badge.svg)](https://github.com/Wulfic/AI-OS/actions/workflows/ci.yml)
[![Link Check](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml/badge.svg)](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Wulfic/AI-OS/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/license-ANSL--v1.0-blue.svg)](LICENSE)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S31MS4TK)

Production-ready HRM-sMoE training toolkit with a clean GUI, CLI, and installers. Future-facing architecture for OS-integrated autonomous assistance.

License: AI-OS Non‑Selling Attribution License (ANSL) v1.0. You may use, copy, modify, and redistribute this code, but you may not sell the Software or derivative works. All redistributions must retain attribution and link back to the original repository. See LICENSE and NOTICE.

## What is AI-OS?

- Today: Train Hierarchical Reasoning Models (HRM) with Sparse Mixture‑of‑Experts (MoE) on consumer GPUs. Optimizations enable long context training on almost any sized card!
- Tomorrow: Deeper OS integration and autonomous, idle‑time learning.

Key features in v1.0.0:
- Dynamic Subbrains (MoE) for specialized experts
- Long-context training strategies and memory optimizations (experimental)
- 8‑bit optimizers (bitsandbytes) to reduce memory footprint
- Multi‑GPU training (DDP)
- Modern GUI + robust CLI

## Quick start

## Minimum Recommended Specs!
- GPU: 6GB VRAM **NOTE** (Nvidia:Full Support|AMD:UNTESTED|INTEL(ARC):UNTESTED)
- System Memory: 16GB
- CPU: 4 Core

## Authors Note
While the minimum specs listed above will allow you to train and run models, please bare in mind, that training could take weeks on smaller/older cards.
The model that ships with this program was trained on 2x RTX 2080ti(11GB) cards. The model is a utilizing the qwen tokenizer, sMoE enabled 268M Params with 8 experts, context length of 1024 resulting in about 6000 characters for output. Trained 1/10th the Tiny Stories Dataset in ~8 hours.

TLDR; Have a fast ass GPU with lots of VRAM if you're impatient.

### Prerequisites

- **Python 3.10+** (included in installers)
- **Node.js 20+** (for MCP tool support - optional but recommended)
  - Windows: `winget install OpenJS.NodeJS.LTS`
  - Ubuntu: `sudo apt install nodejs npm`
  - Or download from: https://nodejs.org/
  - ~8-10GB Storage for the program, however having at a least a 1 TB SSD/NVME is ideal!

### Windows (PowerShell)
```powershell
# From repo root
./scripts/install_aios_on_windows.ps1 -Action install -Yes
aios gui
```

### Ubuntu
```bash
./scripts/install_aios_on_ubuntu.sh install --yes
aios gui
```

Manual developer setup (Windows PowerShell):
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e .
aios gui
```

## Usage

GUI (recommended):
```powershell
aios gui
```

CLI (training):
```powershell
aios hrm-hf train-actv1 `
  --model hrm-smoe `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --steps 1000
```

Minimal dry‑run tasks are available in VS Code (Terminal → Run Task):
- Run brief HRM CLI dry‑run
- Run HRM dry‑run (module)

## Documentation

- Start here: docs/INDEX.md
- Maintenance and guides: docs/README.md
- Attributions and upstream projects: REFERENCES.md
- One-page Quick Start: docs/QUICK_START.md

## Installers

See installers/README.md for Windows (.exe) and Ubuntu (.deb) build instructions and features.

## References and third‑party integrations

See REFERENCES.md for a complete list of libraries and upstream projects used (PyTorch, Transformers, DeepSpeed, bitsandbytes, FlashAttention, and more) with links and licenses.

## Acknowledgments

Thanks to the open‑source ML community and upstream projects that made this possible. See REFERENCES.md.

## License

AI‑OS Non‑Selling Attribution License (ANSL) v1.0 — see LICENSE and NOTICE.
