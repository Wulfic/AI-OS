#!/usr/bin/env bash
set -euo pipefail

# Debian package builder that stages AI-OS under /opt and installs dependencies
# from an offline wheelhouse during post-install.

log_info() { printf '[i] %s\n' "$*"; }
log_error() { printf '[!] %s\n' "$*" >&2; }
die() { log_error "$1"; exit 1; }

usage() {
  cat <<'EOF'
AI-OS .deb builder for Ubuntu

Usage: ./build_deb.sh [--version <version>] [--arch <arch>] [--keep-build]

Options:
  --version <version>  Override package version (defaults to pyproject version)
  --arch <arch>        Target architecture (defaults to dpkg --print-architecture)
  --keep-build         Preserve temporary build directory
  --help               Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
export REPO_ROOT
RELEASE_DIR="$REPO_ROOT/installers/releases"
BUILD_ROOT="$SCRIPT_DIR/.deb-build"
STAGING_ROOT="$BUILD_ROOT/pkg"
WHEELHOUSE="$BUILD_ROOT/wheelhouse"
TEMP_VENV="$BUILD_ROOT/.venv"
KEEP_BUILD="false"
PACKAGE_NAME="ai-os"
VERSION=""
ARCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      [[ $# -ge 2 ]] || die "--version requires a value"
      VERSION="$2"
      shift 2
      ;;
    --arch)
      [[ $# -ge 2 ]] || die "--arch requires a value"
      ARCH="$2"
      shift 2
      ;;
    --keep-build)
      KEEP_BUILD="true"
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

if [[ -z "$ARCH" ]]; then
  if command -v dpkg >/dev/null 2>&1; then
    ARCH="$(dpkg --print-architecture)"
  else
    die "dpkg not found; please install dpkg to detect architecture or pass --arch"
  fi
fi

if [[ -z "$VERSION" ]]; then
  VERSION="$(python3 - <<'PY'
import os
import pathlib

repo_root = pathlib.Path(os.environ["REPO_ROOT"])
data = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

try:
    import tomllib
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib

project = tomllib.loads(data)["project"]
print(project.get("version"))
PY
)"
fi

[[ -n "$VERSION" ]] || die "Unable to determine version"

command -v python3 >/dev/null 2>&1 || die "python3 is required to build the package"
command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required to create the .deb archive"

log_info "Building AI-OS Debian package v$VERSION ($ARCH)"

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT" "$STAGING_ROOT/DEBIAN" "$RELEASE_DIR"
mkdir -p "$WHEELHOUSE"

python3 -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"
# Build an offline wheelhouse so postinst can run without internet access.
python -m pip install --upgrade pip wheel setuptools build >/dev/null
pushd "$REPO_ROOT" >/dev/null
python -m pip wheel ".[ui]" pip setuptools wheel -w "$WHEELHOUSE"
popd >/dev/null
deactivate

mkdir -p "$STAGING_ROOT/opt/ai-os" "$STAGING_ROOT/usr/bin" "$STAGING_ROOT/usr/share/applications" "$STAGING_ROOT/usr/share/pixmaps"

copy_item() {
  local item="$1"
  local src="$REPO_ROOT/$item"
  if [[ -d "$src" ]]; then
    log_info "Copying directory $item"
    cp -a "$src" "$STAGING_ROOT/opt/ai-os/"
  elif [[ -f "$src" ]]; then
    log_info "Copying file $item"
    install -Dm644 "$src" "$STAGING_ROOT/opt/ai-os/$item"
  else
    log_error "Skipping missing item: $item"
  fi
}

PROJECT_ITEMS=(
  "artifacts"
  "config"
  "docs"
  "installers"
  "src"
  "training_data"
  "training_datasets"
  "logging.yaml"
  "pyproject.toml"
  "README.md"
  "LICENSE"
  "NOTICE"
  "ruff.toml"
  "checklist.txt"
)

for item in "${PROJECT_ITEMS[@]}"; do
  copy_item "$item"
done

rm -rf "$STAGING_ROOT/opt/ai-os/installers/_builds"
rm -rf "$STAGING_ROOT/opt/ai-os/installers/releases"
rm -rf "$STAGING_ROOT/opt/ai-os/logs"

log_info "Bundling Python wheelhouse"
mkdir -p "$STAGING_ROOT/opt/ai-os/wheels"
cp -a "$WHEELHOUSE"/* "$STAGING_ROOT/opt/ai-os/wheels/"

log_info "Creating runtime helper scripts"
cat >"$STAGING_ROOT/usr/bin/aios" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="/opt/ai-os"
VENV="$APP_ROOT/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "AI-OS virtual environment missing. Please reinstall the ai-os package." >&2
  exit 1
fi
export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$APP_ROOT"
exec "$VENV/bin/python" -m aios.cli.aios "$@"
EOF
chmod 755 "$STAGING_ROOT/usr/bin/aios"

cat >"$STAGING_ROOT/usr/bin/aios-gui" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="/opt/ai-os"
VENV="$APP_ROOT/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "AI-OS virtual environment missing. Please reinstall the ai-os package." >&2
  exit 1
fi
export PYTHONPATH="$APP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$APP_ROOT"
exec "$VENV/bin/python" -m aios.cli.aios gui "$@"
EOF
chmod 755 "$STAGING_ROOT/usr/bin/aios-gui"

ICON_SRC="$REPO_ROOT/installers/AI-OS.ico"
if [[ -f "$ICON_SRC" ]]; then
  install -Dm644 "$ICON_SRC" "$STAGING_ROOT/usr/share/pixmaps/ai-os.ico"
else
  log_error "Icon not found at $ICON_SRC"
fi

cat >"$STAGING_ROOT/usr/share/applications/ai-os.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=AI-OS GUI
Comment=Launch the AI-OS graphical interface
Exec=/usr/bin/aios-gui
Icon=/usr/share/pixmaps/ai-os.ico
Terminal=false
Categories=Utility;Development;
StartupNotify=true
EOF
chmod 644 "$STAGING_ROOT/usr/share/applications/ai-os.desktop"

log_info "Preparing maintainer scripts"
cat >"$STAGING_ROOT/DEBIAN/postinst" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="/opt/ai-os"
VENV="$APP_ROOT/venv"
WHEEL_DIR="$APP_ROOT/wheels"
PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python3 is required to finish installing ai-os." >&2
  exit 1
fi
printf '[ai-os] Preparing runtime environment...\n'
if [[ ! -d "$VENV" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
if [[ ! -x "$VENV/bin/pip" ]]; then
  "$VENV/bin/python" -m ensurepip --upgrade
fi
printf '[ai-os] Installing Python dependencies...\n'
"$VENV/bin/pip" install --no-index --find-links "$WHEEL_DIR" --upgrade pip setuptools wheel >/dev/null
"$VENV/bin/pip" install --no-index --find-links "$WHEEL_DIR" --upgrade "ai-os[ui]" >/dev/null
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q || true
fi
if command -v xdg-desktop-menu >/dev/null 2>&1; then
  xdg-desktop-menu forceupdate || true
fi
printf '[ai-os] Installation complete.\n'
EOF
chmod 755 "$STAGING_ROOT/DEBIAN/postinst"

cat >"$STAGING_ROOT/DEBIAN/prerm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="/opt/ai-os"
VENV="$APP_ROOT/venv"
if [[ "$1" == "remove" ]]; then
  rm -rf "$VENV"
fi
EOF
chmod 755 "$STAGING_ROOT/DEBIAN/prerm"

cat >"$STAGING_ROOT/DEBIAN/postrm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_ROOT="/opt/ai-os"
if [[ "$1" == "purge" ]]; then
  rm -rf "$APP_ROOT"
fi
EOF
chmod 755 "$STAGING_ROOT/DEBIAN/postrm"

# Report size for dpkg metadata while ignoring control files.
INSTALLED_SIZE="$(du -sk --exclude=DEBIAN "$STAGING_ROOT" | cut -f1)"

cat >"$STAGING_ROOT/DEBIAN/control" <<EOF
Package: $PACKAGE_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.10), python3-venv, python3-pip, python3-tk, libgdk-pixbuf2.0-0, libgl1, libsndfile1, ffmpeg
Maintainer: Wulfic <support@ai-os.invalid>
Installed-Size: $INSTALLED_SIZE
Homepage: https://github.com/Wulfic/AI-OS
Description: AI-OS human resource manager agent with GUI and CLI interfaces
 AI-OS provides CLI and GUI workflows for managing autonomous HRM agents.
EOF

log_info "Building .deb archive"
DEB_NAME="${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --build "$STAGING_ROOT" "$BUILD_ROOT/$DEB_NAME" >/dev/null

mv "$BUILD_ROOT/$DEB_NAME" "$RELEASE_DIR/$DEB_NAME"
log_info "Created $RELEASE_DIR/$DEB_NAME"

if [[ "$KEEP_BUILD" != "true" ]]; then
  rm -rf "$BUILD_ROOT"
fi

log_info "Done"
