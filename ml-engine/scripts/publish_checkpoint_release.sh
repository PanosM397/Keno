#!/usr/bin/env bash
# Verify the paper-freeze checkpoint and publish it as GitHub Release paper-v1.
#
# Prerequisites:
#   - unet_denoiser.pt with SHA256 55ce7637… (from Windows freeze machine)
#   - gh auth login  (repo write + release scope)
#
# Usage:
#   ./scripts/publish_checkpoint_release.sh
#   CHECKPOINT_PATH=/path/to/unet_denoiser.pt ./scripts/publish_checkpoint_release.sh
#   SKIP_TAG=1 ./scripts/publish_checkpoint_release.sh   # release only, tag exists
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CHECKPOINT="${CHECKPOINT_PATH:-$ROOT/checkpoints/unet_denoiser.pt}"
TAG="${RELEASE_TAG:-paper-v1}"
TITLE="${RELEASE_TITLE:-Keno paper freeze 2026-07-paper-v1}"
NOTES="${RELEASE_NOTES:-U-Net denoiser weights for reproducibility label 2026-07-paper-v1.

SHA256: 55ce7637e14dd3558d4e9ede025a5e42e1ca25048a715bd87eb4f0fd028cd49a

Download: \`cd ml-engine && ./scripts/fetch_checkpoint.sh\`
Docs: docs/CHECKPOINT.md · Zenodo: https://doi.org/10.5281/zenodo.21433068}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "error: checkpoint not found: $CHECKPOINT" >&2
  echo "Recover from Windows (see docs/CHECKPOINT.md), copy here, then re-run." >&2
  exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT" "$ROOT/scripts/verify_checkpoint.sh"

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI (gh) is required. Install: https://cli.github.com/" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "error: not logged in to GitHub. Run: gh auth login" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ "${SKIP_TAG:-0}" != "1" ]]; then
  if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "git tag $TAG already exists ($(git rev-parse --short "$TAG"))"
  else
    echo "Creating annotated tag $TAG on HEAD…"
    git tag -a "$TAG" -m "Keno scientific paper freeze 2026-07-paper-v1"
  fi
  echo "Pushing tag $TAG to origin…"
  git push origin "$TAG"
fi

ASSET_NAME="unet_denoiser.pt"
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release $TAG exists; uploading asset $ASSET_NAME…"
  gh release upload "$TAG" "$CHECKPOINT" --clobber
else
  echo "Creating release $TAG with asset $ASSET_NAME…"
  gh release create "$TAG" "$CHECKPOINT" \
    --title "$TITLE" \
    --notes "$NOTES"
fi

echo ""
echo "Published. Test download:"
echo "  rm -f ml-engine/checkpoints/unet_denoiser.pt"
echo "  cd ml-engine && ./scripts/fetch_checkpoint.sh"
