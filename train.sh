#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

echo "🚀  Start training: Mjlab-Velocity-Flat-Unitree-G1 (0.3 m/s)"

uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-Unitree-G1
