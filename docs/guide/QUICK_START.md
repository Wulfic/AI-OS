# Quick Start

This single page gets you from zero to the GUI.

## Windows (PowerShell)

```powershell
# From repo root
./scripts/install_aios_on_windows.ps1 -Action install -Yes
aios gui
```

Developer setup:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e .
aios gui
```

## Ubuntu

```bash
./scripts/install_aios_on_ubuntu.sh install --yes
aios gui
```

## CLI training (example)

```powershell
aios hrm-hf train-actv1 `
  --model gpt2 `
  --dataset-file training_data/curated_datasets/test_sample.txt `
  --steps 1000
```

Trouble? See docs/README.md for guides and maintenance notes.
