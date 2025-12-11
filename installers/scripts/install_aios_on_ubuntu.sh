#!/usr/bin/env bash
set -euo pipefail

ACTION=""
GPU_PREF="auto"
ASSUME_YES="false"
PYTHON_BIN=""
PIP_BIN=""
PYTHON_VENV_BIN=""
PYTORCH_BUILD="cpu"
GPU_DRIVER_PENDING="false"

log_info() { printf '[i] %s\n' "$*"; }
log_warn() { printf '[!] %s\n' "$*" >&2; }
log_error() { printf '[!!!] %s\n' "$*" >&2; }
log_success() { printf '[+] %s\n' "$*"; }
die() { log_error "$*"; exit 1; }

usage() {
  cat <<'EOF'
AI-OS Ubuntu installer

Usage: ./scripts/install_aios_on_ubuntu.sh <action> [options]
  Actions:
    install              Install AI-OS into a local virtual environment
    uninstall            Remove the local AI-OS virtual environment and shims

  Options:
    --gpu <auto|cuda|rocm|cpu>   Preferred PyTorch build (default: auto)
    --yes, -y                    Assume "yes" for all prompts
    --help, -h                   Show this help message
EOF
}

parse_args() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      install|uninstall)
        if [[ -n "$ACTION" ]]; then
          die "Multiple actions specified."
        fi
        ACTION="$1"
        shift
        ;;
      --gpu)
        [[ $# -ge 2 ]] || die "--gpu requires a value."
        GPU_PREF="${2,,}"
        shift 2
        ;;
      --yes|-y)
        ASSUME_YES="true"
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  if [[ -z "$ACTION" ]]; then
    die "No action specified (install or uninstall)."
  fi

  case "$GPU_PREF" in
    auto|cuda|rocm|cpu) ;;
    *) die "Invalid --gpu option: $GPU_PREF" ;;
  esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"
VENV_PATH="$REPO_ROOT/.venv"

SUDO=""
if [[ $EUID -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    die "This script requires elevated privileges for package installation. Please run as root or install sudo."
  fi
fi

APT_UPDATED="false"
apt_install() {
  local packages=("$@")
  if [[ ${#packages[@]} -eq 0 ]]; then
    return
  fi
  if [[ "$APT_UPDATED" == "false" ]]; then
    log_info "Updating apt package index..."
    if ! $SUDO apt-get update; then
      return 1
    fi
    APT_UPDATED="true"
  fi
  log_info "Installing packages: ${packages[*]}"
  if ! $SUDO apt-get install -y "${packages[@]}"; then
    return 1
  fi
}

confirm() {
  local message="$1"
  local default="${2:-no}"
  local prompt="[y/N]"
  if [[ "$default" == "yes" ]]; then
    prompt="[Y/n]"
  fi
  if [[ "$ASSUME_YES" == "true" ]]; then
    return 0
  fi
  read -r -p "$message $prompt " reply || true
  if [[ -z "$reply" ]]; then
    [[ "$default" == "yes" ]]
  else
    [[ "$reply" =~ ^([Yy]|yes|YES)$ ]]
  fi
}

GPU_INFO_VENDOR="unknown"
GPU_INFO_MODEL=""
GPU_INFO_HAS_NVIDIA="false"
GPU_INFO_HAS_ROCM="false"

detect_gpu() {
  GPU_INFO_VENDOR="unknown"
  GPU_INFO_MODEL=""
  GPU_INFO_HAS_NVIDIA="false"
  GPU_INFO_HAS_ROCM="false"

  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO_VENDOR="nvidia"
    GPU_INFO_HAS_NVIDIA="true"
    GPU_INFO_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  fi

  if [[ "$GPU_INFO_VENDOR" == "unknown" ]] && command -v lspci >/dev/null 2>&1; then
    if lspci | grep -qi 'NVIDIA'; then
      GPU_INFO_VENDOR="nvidia"
      GPU_INFO_MODEL="$(lspci | grep -i 'NVIDIA' | head -n1 | sed 's/^.*controller: //')"
      GPU_INFO_HAS_NVIDIA="true"
    elif lspci | grep -qi 'AMD' || lspci | grep -qi 'Advanced Micro Devices'; then
      GPU_INFO_VENDOR="amd"
      GPU_INFO_MODEL="$(lspci | grep -i 'AMD' | head -n1 | sed 's/^.*controller: //')"
    elif lspci | grep -qi 'Intel'; then
      GPU_INFO_VENDOR="intel"
      GPU_INFO_MODEL="$(lspci | grep -i 'Intel' | head -n1 | sed 's/^.*controller: //')"
    fi
  fi

  if command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
    GPU_INFO_HAS_ROCM="true"
    if [[ "$GPU_INFO_VENDOR" == "unknown" ]]; then
      GPU_INFO_VENDOR="amd"
    fi
  fi

  log_info "GPU detection: Vendor=$GPU_INFO_VENDOR Model=${GPU_INFO_MODEL:-unknown} NvidiaSmi=$GPU_INFO_HAS_NVIDIA Rocm=$GPU_INFO_HAS_ROCM"
}

ensure_system_packages() {
  log_info "Ensuring base system packages are available..."
  apt_install \
    build-essential \
    python3 \
    python3-venv \
    python3-tk \
    python3-dev \
    python3-pip \
    pkg-config \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libaio-dev \
    ninja-build \
    cmake
}

ensure_cuda_toolkit() {
  if [[ "$GPU_PREF" != "cuda" && "$GPU_PREF" != "auto" ]]; then
    log_info "Skipping CUDA toolkit install for GPU preference '$GPU_PREF'."
    return
  fi

  if [[ "$GPU_PREF" == "auto" && "$GPU_INFO_HAS_NVIDIA" != "true" ]]; then
    log_info "No NVIDIA GPU detected under --gpu auto; skipping CUDA toolkit install."
    return
  fi

  if command -v nvcc >/dev/null 2>&1; then
    log_info "CUDA toolkit detected: $(nvcc --version | head -n1 || echo 'version unknown')."
    return
  fi

  log_info "Installing NVIDIA CUDA toolkit (required for DeepSpeed CUDA extensions)..."
  if ! apt_install nvidia-cuda-toolkit; then
    log_warn "Failed to install nvidia-cuda-toolkit automatically. DeepSpeed CUDA ops may require manual CUDA setup."
  fi
}

ensure_python() {
  local candidates=("python3" "python3.12" "python3.11" "python3.10")
  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; exit(0) if sys.version_info >= (3, 10) else exit(1)'; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
      fi
    fi
  done

  if [[ -z "${PYTHON_BIN:-}" ]]; then
    log_warn "Python >= 3.10 not found. Installing python3..."
    apt_install python3 python3-venv python3-dev python3-pip
    PYTHON_BIN="$(command -v python3)" || die "Failed to locate python3 after installation."
  fi

  log_info "Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN --version))"
}

ensure_git() {
  if ! command -v git >/dev/null 2>&1; then
    log_warn "Git not found. Installing..."
    apt_install git
  fi
  log_info "Git detected: $(git --version)"
}

ensure_node() {
  if command -v node >/dev/null 2>&1; then
    log_info "Node.js detected: $(node --version)"
    return
  fi
  log_warn "Node.js not found. Installing Node.js LTS..."
  apt_install nodejs npm || {
    log_warn "Failed to install Node.js via apt. Consider installing manually from NodeSource."
    return
  }
  log_info "Node.js installed: $(node --version)"
}

ensure_nvidia_driver() {
  if [[ "$GPU_PREF" == "cpu" ]]; then
    log_info "GPU preference set to CPU; skipping NVIDIA driver setup."
    return
  fi
  if [[ "$GPU_INFO_HAS_NVIDIA" != "true" ]]; then
    log_info "No NVIDIA GPU detected; skipping NVIDIA driver setup."
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local driver_version
    driver_version=$(nvidia-smi --query-driver=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
    log_info "Detected NVIDIA driver version: ${driver_version:-unknown}."
    return
  fi

  log_warn "NVIDIA GPU detected but nvidia-smi is unavailable. Installing recommended driver package."
  apt_install ubuntu-drivers-common || log_warn "Failed to install ubuntu-drivers-common; continuing without automatic driver setup."

  if ! command -v ubuntu-drivers >/dev/null 2>&1; then
    log_warn "ubuntu-drivers utility not found. Please install NVIDIA drivers manually (e.g. 'sudo ubuntu-drivers install') and rerun this installer."
    return
  fi

  local recommended_driver=""
  recommended_driver=$("${PYTHON_BIN:-python3}" - <<'PY'
import re, subprocess, sys
try:
    out = subprocess.check_output(["ubuntu-drivers", "devices"], text=True)
except Exception:
    sys.exit(0)
pattern = re.compile(r"driver\s*:\s*([^\s]+).*recommended", re.IGNORECASE)
for line in out.splitlines():
    m = pattern.search(line)
    if m:
        print(m.group(1))
        break
PY
  )

  if [[ -z "$recommended_driver" ]]; then
    log_warn "Unable to determine a recommended NVIDIA driver automatically. Run 'sudo ubuntu-drivers devices' and install a driver manually."
    return
  fi

  log_info "Installing recommended NVIDIA driver package: $recommended_driver"
  if $SUDO ubuntu-drivers install "$recommended_driver"; then
    GPU_DRIVER_PENDING="true"
    log_success "Driver package installation complete. Reboot required before CUDA will function."
    log_warn "If Secure Boot is enabled, run 'sudo mokutil --disable-validation', reboot, and complete the MOK enrollment prompts."
    log_warn "After reboot, rerun this installer with '--gpu cuda' to finish GPU configuration."
  else
    log_warn "Automatic driver installation failed. Please run 'sudo ubuntu-drivers install $recommended_driver' manually and rerun this installer."
  fi
}

create_venv() {
  if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    log_info "Creating virtual environment at $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
  else
    log_info "Reusing existing virtual environment at $VENV_PATH"
  fi
  PIP_BIN="$VENV_PATH/bin/pip"
  PYTHON_VENV_BIN="$VENV_PATH/bin/python"
}

ensure_local_bin_in_path() {
  local local_bin="$HOME/.local/bin"
  if [[ -z "$local_bin" ]]; then
    return
  fi
  if [[ ":$PATH:" == *":$local_bin:"* ]]; then
    log_info "~/.local/bin already present in PATH."
    return
  fi

  local block_start="# BEGIN AIOS PATH"
  local block_end="# END AIOS PATH"
  local snippet
  snippet=$(cat <<EOF
if [[ :"\$PATH": != *:"$local_bin":* ]]; then
  export PATH="$local_bin:\$PATH"
fi
EOF
)
  local updated="false"
  local profile
  for profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile"; do
    [[ -n "$profile" ]] || continue
    mkdir -p "$(dirname "$profile")"
    if [[ -f "$profile" ]]; then
      if grep -Fq "$block_start" "$profile"; then
        local tmp
        tmp=$(mktemp)
        sed "/$block_start/,/$block_end/d" "$profile" >"$tmp"
        mv "$tmp" "$profile"
      fi
    fi
    {
      printf '\n%s\n%s\n%s\n' "$block_start" "$snippet" "$block_end"
    } >>"$profile"
    log_info "Added PATH export to $profile"
    updated="true"
  done

  if [[ "$updated" == "true" ]]; then
    log_success "~/.local/bin will be added to PATH on new shells. Run 'source ~/.profile' to update current session."
  else
    log_info "PATH update skipped; profiles already contain AI-OS block."
  fi
}

install_pytorch() {
  local gpu_choice="$GPU_PREF"
  if [[ "$gpu_choice" == "auto" ]]; then
    if [[ "$GPU_INFO_HAS_NVIDIA" == "true" ]]; then
      gpu_choice="cuda"
    elif [[ "$GPU_INFO_HAS_ROCM" == "true" ]]; then
      gpu_choice="rocm"
    else
      gpu_choice="cpu"
    fi
  fi

  local index_url="https://download.pytorch.org/whl/cpu"
  case "$gpu_choice" in
    cuda)
      index_url="https://download.pytorch.org/whl/cu126"
      ;;
    rocm)
      index_url="https://download.pytorch.org/whl/rocm6.0"
      ;;
    cpu)
      index_url="https://download.pytorch.org/whl/cpu"
      ;;
  esac

  log_info "Installing PyTorch ($gpu_choice build)..."
  if ! "$PIP_BIN" install --upgrade pip wheel setuptools; then
    die "Failed to upgrade pip in the virtual environment."
  fi

  if [[ "$gpu_choice" == "cuda" && "$GPU_DRIVER_PENDING" == "true" ]]; then
    log_warn "NVIDIA driver installation pending reboot; installing CPU PyTorch build for now. Rerun after reboot with '--gpu cuda'."
    gpu_choice="cpu"
    index_url="https://download.pytorch.org/whl/cpu"
  fi

  if ! "$PIP_BIN" install --upgrade --no-cache-dir torch torchvision torchaudio --index-url "$index_url"; then
    log_warn "PyTorch stack installation failed for $gpu_choice build. Falling back to CPU build."
    "$PIP_BIN" install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    gpu_choice="cpu"
  fi

  local probe
  probe=$("$PYTHON_VENV_BIN" - <<'PY'
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": getattr(torch.version, "cuda", None)
}))
PY
  )
  log_info "PyTorch probe: $probe"
  if [[ "$gpu_choice" == "cuda" ]]; then
    local cuda_available
    cuda_available=$(echo "$probe" | "${PYTHON_BIN:-python3}" - <<'PY'
import json, sys
try:
    data = json.loads(sys.stdin.read())
    print("true" if data.get("cuda_available") else "false")
except Exception:
    print("false")
PY
    )
    if [[ "$cuda_available" != "true" ]]; then
      log_warn "CUDA build installed but torch.cuda.is_available() reported False. Ensure 'nvidia-smi' works after reboot, then reinstall with '--gpu cuda'."
    fi
  fi
  PYTORCH_BUILD="$gpu_choice"
  log_success "PyTorch installed ($gpu_choice)."
}

install_ui_minimal() {
  (cd "$REPO_ROOT" && "$PIP_BIN" install -e '.[ui]')
}

install_hf_stack() {
  local mode="${1:-cpu}"
  local -a base_deps=(
    "transformers>=4.41.0"
    "accelerate>=0.31.0"
    "sentencepiece>=0.1.99"
    "protobuf>=3.20.0"
    "peft>=0.11.1"
    "datasets>=2.14.0"
    "huggingface_hub>=0.19.0"
    "hf-xet>=1.0.0"
    "Pillow>=10.0.0"
    "tqdm>=4.65.0"
    "safetensors>=0.3.1"
    "lm-eval>=0.4.0"
  )
  "$PIP_BIN" install --upgrade "${base_deps[@]}"
  if [[ "$mode" == "gpu" ]]; then
    local -a gpu_deps=(
      "deepspeed>=0.14.0"
      "mpi4py>=4.0.0"
      "bitsandbytes>=0.43.0"
    )
    local pkg
    for pkg in "${gpu_deps[@]}"; do
      if ! "$PIP_BIN" install --upgrade "$pkg"; then
        log_warn "Failed to install $pkg (non-critical)."
      fi
    done
  fi
}

preinstall_flash_attn() {
  if [[ "$PYTORCH_BUILD" != "cuda" ]]; then
    return
  fi
  if "$PIP_BIN" show flash-attn >/dev/null 2>&1; then
    log_info "flash-attn already present; skipping preinstall."
    return
  fi
  log_info "Pre-installing flash-attn to avoid build-isolation failures..."
  if "$PIP_BIN" install --no-build-isolation "flash-attn>=2.3.0"; then
    log_success "flash-attn wheel installed ahead of extras."
  else
    log_warn "flash-attn pre-install failed; will retry later in best-effort mode."
  fi
}

try_install_flash_attn() {
  if [[ "$PYTORCH_BUILD" != "cuda" && "$PYTORCH_BUILD" != "rocm" ]]; then
    log_info "Skipping flash-attn (GPU build not active)."
    return
  fi
  log_info "Installing flash-attn (may take several minutes)..."
  if "$PIP_BIN" install flash-attn --no-build-isolation; then
    log_success "flash-attn installed."
  else
    log_warn "flash-attn installation failed (non-critical). You can retry manually after installing CUDA/ROCm toolchains."
  fi
}

install_aios_packages() {
  log_info "Installing AI-OS dependencies..."
  if [[ "$PYTORCH_BUILD" == "cuda" || "$PYTORCH_BUILD" == "rocm" ]]; then
    preinstall_flash_attn
    if (cd "$REPO_ROOT" && "$PIP_BIN" install -e '.[ui,hf]'); then
      log_success "Installed AI-OS with GPU extras."
    else
      log_warn "UI/HF extras installation failed. Retrying without flash-attn and rebuilding stack manually..."
      install_ui_minimal
      install_hf_stack "gpu"
      try_install_flash_attn
    fi
  else
    install_ui_minimal
    install_hf_stack "cpu"
    log_info "Skipped GPU-only packages (flash-attn, deepspeed, bitsandbytes)."
  fi
  log_info "Ensuring supporting tooling (httpx, ruff)..."
  "$PIP_BIN" install --upgrade "httpx>=0.27" "ruff>=0.4.0"
}

install_playwright() {
  local playwright_bin="$VENV_PATH/bin/playwright"
  if [[ ! -x "$playwright_bin" ]]; then
    log_warn "Playwright CLI not found in the virtual environment. Skipping browser install."
    return
  fi
  log_info "Installing Playwright dependencies (may require sudo)..."
  if [[ -n "$SUDO" ]]; then
    $SUDO env PATH="$PATH" "$playwright_bin" install-deps chromium || log_warn "Playwright install-deps failed (non-critical)."
  else
    "$playwright_bin" install-deps chromium || log_warn "Playwright install-deps failed (non-critical)."
  fi
  log_info "Installing Playwright Chromium browser..."
  "$playwright_bin" install chromium
}

install_openmpi() {
  log_info "Ensuring OpenMPI is available for DeepSpeed..."
  apt_install openmpi-bin libopenmpi-dev
}

create_cli_shim() {
  local shim="$HOME/.local/bin/aios"
  local venv_realpath="$(realpath "$VENV_PATH")"
  mkdir -p "$(dirname "$shim")"
  if [[ -f "$shim" ]] && ! confirm "Overwrite existing CLI shim at $shim?" "no"; then
    log_info "Skipping CLI shim creation."
    return
  fi
  cat >"$shim" <<EOF
#!/usr/bin/env bash
exec "$venv_realpath/bin/python" -m aios.cli.aios "\$@"
EOF
  chmod +x "$shim"
  log_success "Installed CLI shim: $shim"
}

create_desktop_entry() {
  local venv_realpath="$(realpath "$VENV_PATH")"
  local desktop_dir="$HOME/.local/share/applications"
  local shortcut="$desktop_dir/ai-os.desktop"
  mkdir -p "$desktop_dir"
  local icon_path="$REPO_ROOT/installers/AI-OS.png"
  if [[ -f "$shortcut" ]] && ! confirm "Overwrite existing desktop entry at $shortcut?" "no"; then
    log_info "Skipping desktop entry update."
    return
  fi
  cat >"$shortcut" <<EOF
[Desktop Entry]
Type=Application
Name=AI-OS
Comment=AI-OS - Artificially Intelligent Operating System
Exec=$venv_realpath/bin/python -m aios.cli.aios gui
Icon=$icon_path
Terminal=false
Categories=Development;Utility;
EOF
  chmod +x "$shortcut"
  log_success "Desktop entry created: $shortcut"

  local desktop_shortcut="$HOME/Desktop/AI-OS.desktop"
  if [[ -d "$HOME/Desktop" ]] && confirm "Create Desktop shortcut for AI-OS GUI?" "yes"; then
    cp "$shortcut" "$desktop_shortcut"
    chmod +x "$desktop_shortcut"
    log_success "Desktop shortcut created: $desktop_shortcut"
  fi
}

install_aios() {
  log_info "--- Starting AI-OS Ubuntu Installation ---"
  detect_gpu
  ensure_system_packages
  ensure_python
  ensure_git
  ensure_node
  ensure_nvidia_driver
  ensure_cuda_toolkit
  create_venv
  install_pytorch
  install_aios_packages
  install_playwright
  install_openmpi
  if confirm "Install global 'aios' CLI shim in ~/.local/bin?" "yes"; then
    create_cli_shim
    ensure_local_bin_in_path
  fi
  if confirm "Create AI-OS desktop entry for system menus?" "yes"; then
    create_desktop_entry
  fi
  log_success "AI-OS installation complete!"
  log_info "Activate the environment with: source $VENV_PATH/bin/activate"
  log_info "Then run: aios gui"
}

uninstall_aios() {
  log_info "--- Starting AI-OS Ubuntu Uninstallation ---"
  if [[ -d "$VENV_PATH" ]]; then
    if confirm "Remove virtual environment at $VENV_PATH?" "yes"; then
      rm -rf "$VENV_PATH"
      log_success "Virtual environment removed."
    else
      log_info "Virtual environment preserved."
    fi
  else
    log_info "No virtual environment found at $VENV_PATH."
  fi

  local shim="$HOME/.local/bin/aios"
  if [[ -f "$shim" ]] && confirm "Remove CLI shim at $shim?" "yes"; then
    rm -f "$shim"
    log_success "CLI shim removed."
  fi

  local desktop_entry="$HOME/.local/share/applications/ai-os.desktop"
  if [[ -f "$desktop_entry" ]] && confirm "Remove desktop entry at $desktop_entry?" "yes"; then
    rm -f "$desktop_entry"
    log_success "Desktop entry removed."
  fi

  local desktop_shortcut="$HOME/Desktop/AI-OS.desktop"
  if [[ -f "$desktop_shortcut" ]] && confirm "Remove Desktop shortcut $desktop_shortcut?" "yes"; then
    rm -f "$desktop_shortcut"
    log_success "Desktop shortcut removed."
  fi

  log_success "AI-OS uninstallation complete."
}

main() {
  parse_args "$@"
  cd "$REPO_ROOT"
  case "$ACTION" in
    install)
      install_aios
      ;;
    uninstall)
      uninstall_aios
      ;;
    *)
      die "Unknown action: $ACTION"
      ;;
  esac
}

main "$@"
