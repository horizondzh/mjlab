#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings ----

LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

# -------------------------------------------------

if [[ ! -d "${LOG_ROOT}" ]]; then
  echo "[tensorboard.sh] Log dir '${LOG_ROOT}' not found, creating..."
  mkdir -p "${LOG_ROOT}"
fi

echo "[tensorboard.sh] Log dir:  ${LOG_ROOT}"
echo "[tensorboard.sh] Listening: http://${HOST}:${PORT}"
echo ""

exec uv run tensorboard --logdir "${LOG_ROOT}" --host "${HOST}" --port "${PORT}" "$@"
