#!/usr/bin/env bash
# Start the observe-only collector. Usage: ./run.sh [extra collect args]
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -m aviator_lab collect "$@"
