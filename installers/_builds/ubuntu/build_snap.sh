#!/usr/bin/env bash
set -euo pipefail

# Snap package builder that stages AI-OS under /opt/ai-os inside the snap.

usage() {
  cat <<'EOF'
AI-OS snap builder

Usage: ./build_snap.sh [snapcraft arguments]

Examples:
  ./build_snap.sh            # build using default backend (LXD if available)
  ./build_snap.sh --destructive-mode  # build directly on host without LXD
  ./build_snap.sh --use-lxd           # explicit LXD backend
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done
snapcraft_args=("$@")
has_destructive=false
for arg in "${snapcraft_args[@]}"; do
  if [[ "$arg" == "--destructive-mode" ]]; then
    has_destructive=true
    break
  fi
done
if [[ "$has_destructive" == false ]]; then
  snapcraft_args+=("--destructive-mode")
fi

if ! command -v snapcraft >/dev/null 2>&1; then
  printf '[!] snapcraft is required to build the snap. Install it with: snap install snapcraft --classic\n' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
RELEASE_DIR="$REPO_ROOT/installers/releases"

mkdir -p "$RELEASE_DIR"

pushd "$SCRIPT_DIR" >/dev/null
existing_snaps=()
while IFS= read -r line; do
  existing_snaps+=("$line")
done < <(find . -maxdepth 1 -type f -name 'ai-os_*.snap' -printf '%P\n' 2>/dev/null || true)

printf '[i] Running snapcraft %s\n' "${snapcraft_args[*]}"
snapcraft "${snapcraft_args[@]}"

new_snap=""
while IFS= read -r candidate; do
  skip=false
  for old in "${existing_snaps[@]}"; do
    if [[ "$candidate" == "$old" ]]; then
      skip=true
      break
    fi
  done
  if [[ "$skip" == false ]]; then
    new_snap="$candidate"
    break
  fi
done < <(ls -t ai-os_*.snap 2>/dev/null || true)

if [[ -z "$new_snap" ]]; then
  printf '[!] No new snap artifact detected. Check snapcraft output for errors.\n' >&2
  exit 1
fi

mv "$new_snap" "$RELEASE_DIR/$new_snap"
printf '[i] Created %s/%s\n' "$RELEASE_DIR" "$new_snap"

for old in "${existing_snaps[@]}"; do
  if [[ -n "$old" && -f "$old" ]]; then
    rm -f -- "$old"
    printf '[i] Removed previous snap artifact %s\n' "$old"
  fi
done

if snapcraft clean "${snapcraft_args[@]}" >/dev/null 2>&1; then
  printf '[i] Cleaned snapcraft build artifacts.\n'
else
  printf '[w] Unable to clean snapcraft build artifacts automatically. Run "snapcraft clean" manually if needed.\n' >&2
fi

popd >/dev/null
