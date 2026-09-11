#!/usr/bin/env bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo
    echo "ERROR: Virtual environment not found."
    echo
    echo "Run:"
    echo "    ./setup.sh"
    echo
    exit 1
fi

source .venv/bin/activate

echo
echo "============================================================"
echo "                    LIVE VISION AI"
echo "============================================================"
echo

python live_vision_qa.py

EXIT_CODE=$?

echo
echo "============================================================"
echo "                 LIVE VISION AI STOPPED"
echo "============================================================"
echo

exit $EXIT_CODE
