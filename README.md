# AI-OS v1.3.53

[![CI](https://github.com/Wulfic/AI-OS/actions/workflows/ci.yml/badge.svg)](https://github.com/Wulfic/AI-OS/actions/workflows/ci.yml)
[![Link Check](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml/badge.svg)](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml)
[![Version](https://img.shields.io/badge/version-1.3.53-blue.svg)](https://github.com/Wulfic/AI-OS/releases/tag/Official)
[![License](https://img.shields.io/badge/license-ANSL--v1.0-blue.svg)](LICENSE)
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/S6S31MS4TK)

**📚 [Fancy Documentation](https://wulfic.github.io/AI-OS/) | 💻 [GitHub Repository](https://github.com/Wulfic/AI-OS)**

HRM-sMoE LLM training toolkit with a clean GUI, CLI, and installers. Future-facing architecture for OS-integrated autonomous assistance.

License: AI-OS Non‑Selling Attribution License (ANSL) v1.0. You may use, copy, modify, and redistribute this code, but you may not sell the Software or derivative works. All redistributions must retain attribution and link back to the original repository. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## What is AI-OS?

- Today: Train Hierarchical Reasoning Models (HRM) with Sparse Mixture‑of‑Experts (MoE) on consumer GPUs. Optimizations enable long context training on almost any sized card! Designed to be a streamlined, pipeline, from data prep to model training/inference to evaluation for users of all skill levels. No need to be an ML expert to get started!
- Tomorrow: Deeper OS integration and autonomous, idle‑time learning.
- Bleeding edge technologies, pushing the boundaries of consumer hardware.
- Tons more features! Check out the Planned_Features section

Key features in v1.3.53:
- Windows and Ubuntu Support!
- Long-context training strategies and memory optimizations 
- Multi‑GPU training (DDP/Parallel/Sharding)
- GUI + CLI(TUI coming soon!)
- End to End pipeline for HRM based models
- Integration with huggingface
- (Optional) Base model with extremely basic english comprehension

## Authors Note
While the minimum specs listed below will allow you to train and run models, please bare in mind, that training could take weeks on smaller/older cards.
The Optional model that ships alongside this program was trained on 2x RTX 2080ti(11GB) cards. The model is a utilizing the qwen tokenizer, sMoE enabled 268M Params with 8 experts, context length of 1024 resulting in about ~6000 characters for output. Trained 1/10th the Tiny Stories Dataset in ~6 hours.

TLDR; Have a fast ass GPU with lots of VRAM if you're impatient.

## Minimum Recommended Specs!
- GPU: 6GB VRAM **NOTE** (Nvidia:Full Support|AMD:PARTIAL Support|INTEL:UNTESTED|CPU:Full Support)
- System Memory: 16GB
- CPU: 4 Core

### Prerequisites
- ~8-10GB Storage for the program, however having at a least a 1 TB SSD/NVME is ideal!
- Windows 11 or Ubuntu 24
- Internet connection(Faster is better)
*Note*: If you have a metered connection(limited monthly bandwidth), be mindful of the datasets you choose.

## Quick Install Info
### Installers
Find the official installers also in the [Releases section](https://github.com/Wulfic/AI-OS/releases).
See the installers README in the repository for Windows (.exe) and Ubuntu (.deb) info.
If you broke your install or are missing dependencies in terminal use: 
```bash
aios doctor --repair
```
### Ubuntu
```bash
# From repo root
sudo chmod a+x ./installers/scripts/install_aios_on_ubuntu.sh
sudo ./installers/scripts/install_aios_on_ubuntu.sh install --yes
```
### Windows (PowerShell)
```powershell
# From repo root(might need to run as admin)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./installers/scripts/install_aios_on_windows.ps1 -Action install -Yes
```

## Administrator Privileges
AI-OS requires Administrator privileges on Windows to function correctly. This is necessary for:
- Write access to `%ProgramData%` for shared artifacts and models.
- GPU scheduling and performance tuning.
- System-level integration features.

The installer creates shortcuts that automatically request elevation. If you launch AI-OS manually, please ensure you run it as Administrator.

## Usage
Use the shortcut to start the program or via terminal.

GUI (recommended):
```
aios gui
```
Interactive CLI:
```
aios
```
## Documentation

- Start here: [Guide Index](guide/INDEX.MD)
- Maintenance and guides: [README](README.md)
- Attributions and upstream projects: [REFERENCES](REFERENCES.md)
- One-page Quick Start: [Quick Start Guide](guide/QUICK_START.md)

## References and third‑party integrations

See [REFERENCES.md](REFERENCES.md) for a complete list of libraries and upstream projects used (PyTorch, Transformers, DeepSpeed, bitsandbytes, FlashAttention, and more) with links and licenses.

## Acknowledgments

Thanks to the open‑source ML community and upstream projects that made this possible. See [REFERENCES.md](REFERENCES.md).

## License

AI‑OS Non‑Selling Attribution License (ANSL) v1.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
