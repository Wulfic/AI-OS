# AI-OS Documentation

[![Link Check](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml/badge.svg)](https://github.com/Wulfic/AI-OS/actions/workflows/link-check.yml)

Welcome to the AI-OS documentation! 

AI-OS is a Hierarchical Reasoning Model (HRM) training framework designed for Windows, featuring an intuitive GUI and powerful CLI for training custom language models with extreme context lengths and advanced optimization features.

## 🎯 Quick Navigation

Start here for key references:
- ✅ **[Complete Feature Index](guide/features/COMPLETE_FEATURE_INDEX.md)** — Find every feature doc
- ✅ **[Feature Combination Matrix](guide/features/FEATURE_COMBINATION_MATRIX.md)** — Compatibility and combos

---

## Quick Links

- Start at: [Guide Index](guide/INDEX.MD)
- Training API Quick Reference: [guide/api/TRAINING_API_QUICK_REFERENCE.md](guide/api/TRAINING_API_QUICK_REFERENCE.md)

## What is AI-OS?

AI-OS implements a Hierarchical Reasoning Model (HRM) training toolkit with Mixture‑of‑Experts (MoE) support and extreme‑context optimizations, providing:

- **🧠 HRM Training** - Train hierarchical reasoning models from scratch or fine-tune existing models
- **💻 Intuitive GUI** - Full-featured training interface with real-time monitoring
- **⚡ Long Context** - Strategies for extended context lengths; see research notes
- **🎯 Dynamic Subbrains** - Goal-driven, self-organizing expert networks
- **📊 Rich Chat Interface** - Interactive chat with syntax highlighting and markdown support
- **🔧 Advanced Optimization** - 8-bit optimizers, gradient checkpointing, mixed precision training

## Installation

### Quick Start (Windows)

```powershell
# Clone the repository
git clone https://github.com/Wulfic/AI-OS.git
cd AI-OS

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -e .

# Launch GUI
aios gui
```

For installation, see scripts and installers documentation:

- Windows: `scripts/install_aios_on_windows.ps1`
- Ubuntu: `scripts/install_aios_on_ubuntu.sh`
- Windows with Ubuntu via Docker: `scripts/install_aios_ubuntu_via_docker_on_windows.ps1`

## Basic Usage

### GUI Mode
```powershell
aios gui
```

The GUI provides:
- **Training Tab** - Configure and monitor training runs
- **Brains Tab** - Manage trained models
- **Chat Tab** - Interactive chat with your models
- **Datasets Tab** - Manage and download datasets
- **Experts Tab** - Dynamic Subbrains expert management

### CLI Mode

Train a model:
```powershell
aios hrm-hf train-actv1 \
  --model "gpt2" \
  --dataset-file "path/to/dataset.txt" \
  --brain-name "MyModel" \
  --steps 1000
```

Chat with a trained model:
```powershell
aios chat --brain-path "artifacts/brains/actv1/MyModel"
```

## Key Features

### 🚀 Extreme Context Length
Train models with longer contexts using available strategies. See:
- [FLASH_ATTENTION_VS_CHUNKING.md](guide/features/FLASH_ATTENTION_VS_CHUNKING.md) (discussion)
- [EXTREME_CONTEXT_LENGTH_TRAINING.md](research/context_length/EXTREME_CONTEXT_LENGTH_TRAINING.md) (status/notes)

### 🎯 Dynamic Subbrains
Self-organizing expert networks that specialize dynamically based on goals. Subbrains manager panel is WIP; see GUI for status.

### 💾 8-Bit Optimization
Reduce memory usage with 8-bit optimizers. See: [2025-10-19_8bit_optimizer_verification.md](maintenance/2025-10-19_8bit_optimizer_verification.md)

### 📊 Rich Monitoring
Real-time training metrics, GPU monitoring, and interactive charts.

## Documentation Structure

```
docs/
├── INDEX.md             # Main documentation index
├── README.md            # Overview and quick links
├── guide/               # Guides and API quick refs (see guide/api)
│   └── api/             # CLI and tooltips quick references
├── maintenance/         # Fix logs, validation notes, production checks
├── planned_features/    # Plans and design docs for future work
└── research/            # Research notes and optimization deep dives
  ├── context_length/
  └── optimization/
```

## Common Training Recipes

### High-Performance Training
```powershell
aios hrm-hf train-actv1 \
  --model "gpt2" \
  --dataset-file "dataset.txt" \
  --max-seq-len 10000 \
  --gradient-checkpointing \
  --amp \
  --batch-size 2
```

### Memory-Efficient Training
```powershell
aios hrm-hf train-actv1 \
  --model "gpt2" \
  --dataset-file "dataset.txt" \
  --gradient-checkpointing \
  --amp \
  --batch-size 1 \
  --max-seq-len 3000
```

### Multi-GPU Training
```powershell
aios hrm-hf train-actv1 \
  --model "MyModel" \
  --dataset-file "dataset.txt" \
  --ddp \
  --cuda-ids 0,1 \
  --batch-size 1
```

## Getting Help

- Training API Reference: [guide/api/TRAINING_API_QUICK_REFERENCE.md](guide/api/TRAINING_API_QUICK_REFERENCE.md)
- GitHub Issues: https://github.com/Wulfic/AI-OS/issues

## Contributing

We welcome contributions! For development notes, see the repository README and issues:
- Code organization and architecture
- Refactoring progress and plans  
- Implementation status and roadmaps

## Link checking (docs)

We use lychee to catch broken links in documentation. From the repo root:

```powershell
# Check links in docs only
lychee --config lychee.toml docs
```

If you don’t have lychee installed, see https://github.com/lycheeverse/lychee for installation instructions.

### Local check without installing lychee
You can also run a lightweight local checker (no install required):

```powershell
.venv\Scripts\python.exe scripts\check_docs_links.py
```

This validates relative links within `docs/` and ignores code blocks to avoid false positives.

## Project Status

**Last Updated**: October 12, 2025

- ✅ Core HRM training implementation
- ✅ GUI with rich monitoring and chat
-- ⚠️ Long-context strategies: experimental; see research status
- ✅ Dynamic Subbrains Phase 1 & 2
- ✅ 8-bit optimizer integration
- ✅ Rich Chat system with markdown support
- 🔄 Ongoing: Dynamic Subbrains Phase 3

## License

See [LICENSE](../LICENSE) for details.

## References

- Project Repository: https://github.com/Wulfic/AI-OS
- Documentation Index: [Guide Index](guide/INDEX.MD)

---

Start at the documentation index: [Guide Index](guide/INDEX.MD)
