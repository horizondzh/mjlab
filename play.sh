#!/usr/bin/env bash
set -euo pipefail

# Play / evaluate the latest trained policy.
# Usage: ./play.sh [--num-envs N]

TASK="Mjlab-Velocity-Flat-Unitree-G1"
ARGS=()

while [[ $# -gt 0 ]]; do
    ARGS+=("$1")
    shift
done

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""

# Find latest checkpoint
LOG_DIR="logs/rsl_rl/g1_velocity_xpeng_walk"
LATEST_RUN=$(ls -dt "$LOG_DIR"/*/ 2>/dev/null | head -1)
if [[ -z "$LATEST_RUN" ]]; then
    echo "❌ No training runs found. Run ./train.sh first."
    exit 1
fi

# Find latest model file (highest number)
LATEST_MODEL=$(ls -t "$LATEST_RUN"/model_*.pt 2>/dev/null | head -1)
if [[ -z "$LATEST_MODEL" ]]; then
    echo "❌ No checkpoints found in $LATEST_RUN"
    exit 1
fi

echo "🎮  Play: $LATEST_MODEL"

uv run python src/mjlab/scripts/play.py \
    "$TASK" \
    --checkpoint-file "$LATEST_MODEL" \
    --num-envs 1 \
    --device cpu \
    "${ARGS[@]}"
