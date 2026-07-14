#!/usr/bin/env bash
# Run the Keno scientific proof pipeline inside the ml-engine virtualenv.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating virtualenv at ml-engine/.venv ..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/python -m app.prove "$@"
