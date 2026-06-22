#!/usr/bin/env bash
set -euo pipefail

# Start TensorBoard to view training curves.
# Usage: ./tensorboard.sh [port]
#
# Default port: 6006
# Open http://localhost:6006 in your browser.

PORT="${1:-6006}"

cd "$(dirname "$0")"

echo "📊  TensorBoard: http://localhost:$PORT"
echo "   Logdir: $(pwd)/logs/rsl_rl"
echo "   Press Ctrl+C to stop."
echo ""

uv run tensorboard --logdir "$(pwd)/logs/rsl_rl" --port "$PORT" --reload_interval 5
