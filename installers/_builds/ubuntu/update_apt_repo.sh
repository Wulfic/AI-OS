#!/usr/bin/env bash
set -euo pipefail

log_info() { printf '[i] %s\n' "$*"; }
log_warn() { printf '[w] %s\n' "$*" >&2; }
log_error() { printf '[!] %s\n' "$*" >&2; }
die() { log_error "$1"; exit 1; }

usage() {
  cat <<'EOF'
Usage: update_apt_repo.sh [options]

Options:
  --deb <path>          Path to .deb to publish (defaults to newest in installers/releases)
  --repo-dir <path>     Root directory of the apt repo (default: installers/releases/apt)
  --dist <name>         Distribution codename/suite (default: stable)
  --component <name>    Repository component (default: main)
  --ppa <ppa:name>      Launchpad PPA target for dput (optional)
  --changes <path>      Source .changes file to upload with dput (requires --ppa)
  --gpg-key <key-id>    Override default GPG key (defaults to AI-OS release key)
  --no-sign             Skip signing Release files
  --force               Skip confirmation prompts
  --help                Show this help message
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
fi
DEFAULT_RELEASE_DIR="$REPO_ROOT/installers/releases"
DEFAULT_REPO_DIR="$DEFAULT_RELEASE_DIR/apt"
DEFAULT_GPG_KEY="D30BDDC813C627F5F259FA33328243A210E7BAD4"
DIST="stable"
COMPONENT="main"
DEB_PATH=""
REPO_DIR="$DEFAULT_REPO_DIR"
PPA_TARGET=""
CHANGES_FILE=""
GPG_KEY="$DEFAULT_GPG_KEY"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deb)
      [[ $# -ge 2 ]] || die "--deb requires a value"
      DEB_PATH="$2"
      shift 2
      ;;
    --repo-dir)
      [[ $# -ge 2 ]] || die "--repo-dir requires a value"
      REPO_DIR="$2"
      shift 2
      ;;
    --dist)
      [[ $# -ge 2 ]] || die "--dist requires a value"
      DIST="$2"
      shift 2
      ;;
    --component)
      [[ $# -ge 2 ]] || die "--component requires a value"
      COMPONENT="$2"
      shift 2
      ;;
    --ppa)
      [[ $# -ge 2 ]] || die "--ppa requires a value"
      PPA_TARGET="$2"
      shift 2
      ;;
    --changes)
      [[ $# -ge 2 ]] || die "--changes requires a value"
      CHANGES_FILE="$2"
      shift 2
      ;;
    --gpg-key)
      [[ $# -ge 2 ]] || die "--gpg-key requires a value"
      GPG_KEY="$2"
      shift 2
      ;;
    --no-sign)
      GPG_KEY=""
      shift
      ;;
    --force)
      FORCE="true"
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

if [[ -z "$DEB_PATH" ]]; then
  mapfile -t debs < <(ls -1t "$DEFAULT_RELEASE_DIR"/*.deb 2>/dev/null || true)
  [[ ${#debs[@]} -gt 0 ]] || die "No .deb files found in $DEFAULT_RELEASE_DIR"
  DEB_PATH="${debs[0]}"
fi

[[ -f "$DEB_PATH" ]] || die "Debian package not found: $DEB_PATH"

command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required"
command -v dpkg-scanpackages >/dev/null 2>&1 || die "dpkg-scanpackages is required (install dpkg-dev)"
command -v apt-ftparchive >/dev/null 2>&1 || die "apt-ftparchive is required (install apt-utils)"
if [[ -n "$PPA_TARGET" || -n "$CHANGES_FILE" ]]; then
  command -v dput >/dev/null 2>&1 || die "dput is required for Launchpad uploads"
fi
if [[ -n "$GPG_KEY" ]]; then
  command -v gpg >/dev/null 2>&1 || die "gpg is required for signing"
  if ! gpg --batch --list-secret-keys "$GPG_KEY" >/dev/null 2>&1; then
    die "GPG secret key $GPG_KEY not found; import it or rerun with --no-sign"
  fi
fi

PACKAGE_NAME="$(dpkg-deb -f "$DEB_PATH" Package)"
VERSION="$(dpkg-deb -f "$DEB_PATH" Version)"
ARCH="$(dpkg-deb -f "$DEB_PATH" Architecture)"
[[ -n "$PACKAGE_NAME" && -n "$VERSION" && -n "$ARCH" ]] || die "Unable to read package metadata"

log_info "Preparing apt repo in $REPO_DIR for $PACKAGE_NAME $VERSION ($ARCH)"
mkdir -p "$REPO_DIR"
POOL_DIR="$REPO_DIR/pool/$COMPONENT/${PACKAGE_NAME:0:1}/$PACKAGE_NAME"
mkdir -p "$POOL_DIR"
TARGET_DEB="$POOL_DIR/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"
cp -f "$DEB_PATH" "$TARGET_DEB"

pushd "$REPO_DIR" >/dev/null
BIN_DIR="dists/$DIST/$COMPONENT/binary-$ARCH"
mkdir -p "$BIN_DIR"
log_info "Generating Packages index"
DPKG_SCANPOUT="$BIN_DIR/Packages"
dpkg-scanpackages "pool/$COMPONENT" >"$DPKG_SCANPOUT"
gzip -kf "$DPKG_SCANPOUT"
xz -kf "$DPKG_SCANPOUT"

log_info "Creating Release metadata"
TMP_CONF="$(mktemp)"
cat >"$TMP_CONF" <<EOF
APT::FTPArchive::Release::Origin "AI-OS";
APT::FTPArchive::Release::Label "AI-OS";
APT::FTPArchive::Release::Suite "$DIST";
APT::FTPArchive::Release::Codename "$DIST";
APT::FTPArchive::Release::Architectures "$ARCH";
APT::FTPArchive::Release::Components "$COMPONENT";
APT::FTPArchive::Release::Description "AI-OS apt repository";
EOF
apt-ftparchive -c "$TMP_CONF" release "dists/$DIST" >"dists/$DIST/Release"
rm -f "$TMP_CONF"

if [[ -n "$GPG_KEY" ]]; then
  log_info "Signing Release files with key $GPG_KEY"
  gpg --batch --yes --local-user "$GPG_KEY" --armor --detach-sign \
    --output "dists/$DIST/Release.gpg" "dists/$DIST/Release"
  gpg --batch --yes --local-user "$GPG_KEY" --armor --clearsign \
    --output "dists/$DIST/InRelease" "dists/$DIST/Release"
fi
popd >/dev/null

if [[ -n "$PPA_TARGET" || -n "$CHANGES_FILE" ]]; then
  if [[ -z "$PPA_TARGET" || -z "$CHANGES_FILE" ]]; then
    log_warn "--ppa and --changes must be provided together to run dput"
  else
    [[ -f "$CHANGES_FILE" ]] || die "Changes file not found: $CHANGES_FILE"
    if [[ "$FORCE" == "false" ]]; then
      read -r -p "Upload $CHANGES_FILE to $PPA_TARGET via dput? [y/N] " reply
      case "$reply" in
        [Yy]*) ;;
        *) log_info "Skipping dput upload"; exit 0;;
      esac
    fi
    log_info "Uploading $CHANGES_FILE to $PPA_TARGET"
    dput "$PPA_TARGET" "$CHANGES_FILE"
  fi
fi

log_info "Repo update complete. Publish $REPO_DIR to your hosting target."
