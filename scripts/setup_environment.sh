#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Digital Self Preservation: Environment Setup"
echo " Target: NVIDIA RTX 4080 (16GB VRAM) / Linux WSL2"
echo "=========================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "[1/4] Installing 'uv' package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[2/4] Creating Python 3.11 isolated environment..."
uv venv --python 3.11 .venv

# Activate venv
source .venv/bin/activate

echo "[3/4] Installing PyTorch with CUDA 12.4 support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[4/4] Installing project dependencies..."
uv pip install -r requirements.txt

# Create necessary directories
mkdir -p data/audio/raw data/audio/reference data/biography data/chat_logs data/memory_db

echo "=========================================================="
echo " Setup complete! Activate the environment with:"
echo "   source .venv/bin/activate"
echo "=========================================================="

