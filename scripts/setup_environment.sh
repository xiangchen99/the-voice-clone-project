#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

venv_python=.venv/bin/python
if ! "$venv_python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' 2>/dev/null; then
    if [[ -e .venv || -L .venv ]]; then
        mkdir -p env
        backup_dir=$(mktemp -d env/venv-backup.XXXXXX)
        mv -- .venv "$backup_dir/.venv"
        echo "Previous environment saved to $backup_dir/.venv"
    fi
    echo "Creating the Python 3.11 environment..."
    uv venv --python 3.11 .venv
fi

if ! "$venv_python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    has_cuda_builds = all(version(package).endswith("+cu124") for package in ("torch", "torchvision", "torchaudio"))
except PackageNotFoundError:
    has_cuda_builds = False
raise SystemExit(not has_cuda_builds)
PY
then
    echo "Installing PyTorch with CUDA 12.4 support..."
    uv pip install --python "$venv_python" --index-url https://download.pytorch.org/whl/cu124 \
        --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
        torch torchvision torchaudio
fi

uv pip install --python "$venv_python" -r requirements.txt
uv pip check --python "$venv_python"

mkdir -p data/audio/raw data/audio/reference data/biography data/chat_logs data/memory_db
