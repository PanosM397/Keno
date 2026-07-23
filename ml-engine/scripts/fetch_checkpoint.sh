#!/usr/bin/env bash
# Download the paper-freeze U-Net weights and verify SHA256.
#
# Default URL points at the paper-v1 GitHub Release asset. Override with:
#   CHECKPOINT_URL=https://... ./scripts/fetch_checkpoint.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CHECKPOINT_PATH:-$ROOT/checkpoints/unet_denoiser.pt}"
# Update this URL after attaching unet_denoiser.pt to the paper-v1 Release.
DEFAULT_URL="https://github.com/PanosM397/Keno/releases/download/paper-v1/unet_denoiser.pt"
URL="${CHECKPOINT_URL:-$DEFAULT_URL}"

mkdir -p "$(dirname "$DEST")"

echo "Downloading checkpoint:"
echo "  url:  $URL"
echo "  dest: $DEST"

TMP="$(mktemp "${TMPDIR:-/tmp}/keno-unet.XXXXXX.pt")"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT

if command -v curl >/dev/null 2>&1; then
  if ! curl -fsSL --progress-bar -o "$TMP" "$URL"; then
    echo "error: download failed for $URL" >&2
    echo "hint: publish freeze weights on GitHub Release paper-v1, or set CHECKPOINT_URL" >&2
    echo "      see docs/CHECKPOINT.md" >&2
    exit 1
  fi
elif command -v wget >/dev/null 2>&1; then
  if ! wget -O "$TMP" "$URL"; then
    echo "error: download failed for $URL" >&2
    echo "see docs/CHECKPOINT.md" >&2
    exit 1
  fi
else
  echo "error: need curl or wget" >&2
  exit 1
fi

mv "$TMP" "$DEST"
trap - EXIT

"$ROOT/scripts/verify_checkpoint.sh"
echo "Installed freeze checkpoint to $DEST"
