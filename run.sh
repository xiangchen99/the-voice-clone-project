#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
./scripts/setup_environment.sh
exec .venv/bin/python run_ui.py "$@"
