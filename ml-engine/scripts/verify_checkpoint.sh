#!/usr/bin/env bash
# Verify ml-engine/checkpoints/unet_denoiser.pt against the paper freeze hash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
CHECKPOINT="${CHECKPOINT_PATH:-$ROOT/checkpoints/unet_denoiser.pt}"
EXPECTED_FILE="${EXPECTED_SHA_FILE:-$REPO_ROOT/docs/freeze/current/checkpoint.sha256}"

if [[ ! -f "$EXPECTED_FILE" ]]; then
  echo "error: missing expected hash file: $EXPECTED_FILE" >&2
  exit 1
fi

EXPECTED="$(tr -d '[:space:]' < "$EXPECTED_FILE")"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "error: checkpoint not found: $CHECKPOINT" >&2
  echo "hint: run ./scripts/fetch_checkpoint.sh  (or train; see docs/CHECKPOINT.md)" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$CHECKPOINT" | awk '{print $1}')"
else
  echo "error: need sha256sum or shasum" >&2
  exit 1
fi

echo "path:     $CHECKPOINT"
echo "expected: $EXPECTED"
echo "actual:   $ACTUAL"

if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "FAIL: checkpoint does not match freeze 2026-07-paper-v1" >&2
  echo "See docs/CHECKPOINT.md" >&2
  exit 1
fi

echo "OK: freeze checkpoint verified"
