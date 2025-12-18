# AI-OS Installers

**Last Updated**: December 12, 2025  
**Purpose**: Installation scripts and pre-built packages for AI-OS

---

## Overview

This directory contains installation scripts and resources for deploying AI-OS on different platforms.

## Directory Structure

```
installers/
├── README.md              # This file
├── requirements-lock.txt  # Locked Python dependencies
├── scripts/               # Installation scripts
│   ├── install_aios_on_windows.ps1
│   └── install_aios_on_ubuntu.sh
├── releases/              # Pre-built installers
├── wrapper/               # Installer wrapper utilities
└── _builds/               # Build outputs
    └── ubuntu/            # Ubuntu-specific builds
```

## Installation Methods

### Windows Installation

**⚠️ Windows SmartScreen Warning**: Unsigned installers will trigger Windows Defender SmartScreen with "Windows protected your PC". This is expected for development builds. Click **"More info"** → **"Run anyway"** to proceed. Official signed releases coming soon.

#### PowerShell Script (Recommended)
```powershell
# From repository root - run as Administrator
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./installers/scripts/install_aios_on_windows.ps1 -Action install -Yes
```

#### Script Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `-Action` | `install`, `uninstall`, or `preflight` | `install` |
| `-Yes` | Skip confirmation prompts | `false` |
| `-Gpu` | GPU backend: `auto`, `cuda`, `dml`, `cpu` | `auto` |
| `-InstallCudaTools` | Force CUDA tools installation | `false` |
| `-SkipPythonInstall` | Skip Python installation | `false` |
| `-SkipGitInstall` | Skip Git installation | `false` |
| `-SkipNodeInstall` | Skip Node.js installation | `false` |
| `-DownloadBrain` | Download optional base model | `false` |
| `-SkipBrain` | Skip base model download | `false` |

#### Pre-built Windows Installer
Download the latest `.exe` installer from the [Releases page](https://github.com/Wulfic/AI-OS/releases).

### Ubuntu Installation

#### Shell Script (Recommended)
```bash
# From repository root
sudo chmod a+x ./installers/scripts/install_aios_on_ubuntu.sh
sudo ./installers/scripts/install_aios_on_ubuntu.sh install --yes
```

#### Pre-built .deb Package
Download the latest `.deb` installer from the [Releases page](https://github.com/Wulfic/AI-OS/releases).

## Post-Installation

After installation, you can launch AI-OS using:

```bash
# GUI Mode (recommended)
aios gui

# Interactive CLI
aios

# Verify installation
aios doctor
```

## Troubleshooting

### Common Issues

#### Dependencies Missing
If you encounter missing dependencies after installation:
```bash
aios doctor --repair
```

#### Permission Issues (Windows)
AI-OS requires Administrator privileges on Windows. Ensure you:
1. Run PowerShell as Administrator
2. Use the desktop shortcut (auto-elevates)

#### GPU Not Detected
- **NVIDIA**: Ensure CUDA toolkit and drivers are installed
- **AMD**: ROCm support is experimental
- **Intel Arc**: OneAPI support is experimental
- **CPU**: Use `-Gpu cpu` flag for CPU-only mode

### Installation Logs

Windows installer creates a log file on your Desktop:
- `AIOS_Installer.log` - Contains detailed installation progress and errors

### Getting Help

- [GitHub Issues](https://github.com/Wulfic/AI-OS/issues) - Report bugs or request features
- [Quick Start Guide](../docs/guide/QUICK_START.md) - Getting started documentation
- [Full Documentation](../docs/README.md) - Complete documentation

---

## Requirements

### Minimum System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6GB | 11GB+ |
| System RAM | 16GB | 32GB |
| CPU | 4 cores | 8+ cores |
| Storage | 10GB | 1TB SSD/NVMe |
| OS | Windows 11 / Ubuntu 24 | Same |
| Internet | Required | Fast connection |

### Software Prerequisites

The installer will attempt to install these if missing:
- Python 3.10+
- Git
- Node.js (for MCP tools)
- CUDA Toolkit (for NVIDIA GPUs)

---

## Building Installers

### Windows Installer
See `.github/workflows/build-installers.yml` for the automated build process.

### Ubuntu Installer
See `_builds/ubuntu/` for build scripts and packaging configuration.

---

[Back to Main README](../README.md) | [Documentation](../docs/README.md)
