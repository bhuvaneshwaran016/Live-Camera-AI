#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo
echo "============================================================"
echo "             LIVE VISION AI - LINUX SETUP"
echo "============================================================"
echo

echo "[1/8] Checking Python..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 was not found."
    echo "Please install Python 3.12 or newer."
    exit 1
fi

python3 --version

echo
echo "[2/8] Checking Python virtual environment support..."

if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "ERROR: Python venv support is missing."
    echo "Ubuntu/Debian:"
    echo "  sudo apt install python3-venv"
    exit 1
fi

echo
echo "[3/8] Creating virtual environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo
echo "[4/8] Upgrading pip..."

python -m pip install --upgrade pip

echo
echo "[5/8] Installing dependencies..."

python -m pip install -r requirements.txt

echo
echo "[6/8] Checking NVIDIA GPU..."

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader
else
    echo "WARNING: nvidia-smi was not found."
    echo "A supported NVIDIA GPU is recommended."
fi

echo
echo "[7/8] Downloading YOLO11m..."

python - <<'PY'
from ultralytics import YOLO

YOLO("yolo11m.pt")

print("YOLO11m ready.")
PY

echo
echo "[8/8] Downloading Qwen2.5-VL 3B..."

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path("models/Qwen2.5-VL-3B-Instruct")
target.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir=str(target),
)

print("Qwen2.5-VL 3B ready.")
PY

echo
echo "============================================================"
echo "                 SETUP COMPLETE"
echo "============================================================"
echo
echo "Run:"
echo
echo "    ./run.sh"
echo
