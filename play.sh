#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings ----

# Task selection.
TASK="${TASK:-Mjlab-Velocity-Flat-Unitree-G1}"

# Explicit checkpoint file — if not set, auto-detect the latest one from log dir.
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

# Log root (matches LOG_ROOT in train.sh).
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"

# Experiment name (matches EXPERIMENT_NAME in train.sh).
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_velocity_flat}"

# Auto speed (used only when no joystick is detected).
AUTO_SPEED="${AUTO_SPEED:-0.5}"

# Number of environments (default 1 for play).
NUM_ENVS="${NUM_ENVS:-1}"

# Viewer backend: "auto", "native", or "viser".
VIEWER="${VIEWER:-auto}"

# -------------------------------------------------

# Auto-detect latest checkpoint if not explicitly provided.
if [[ -z "${CHECKPOINT_FILE}" ]]; then
  EXP_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}"
  if [[ -d "${EXP_DIR}" ]]; then
    LATEST_RUN=$(ls -dt "${EXP_DIR}"/*/ 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_RUN}" ]]; then
      LATEST_CKPT=$(ls -t "${LATEST_RUN}"/*.pt 2>/dev/null | head -1 || true)
      if [[ -n "${LATEST_CKPT}" ]]; then
        CHECKPOINT_FILE="${LATEST_CKPT}"
        echo "[play.sh] Auto-detected checkpoint: ${CHECKPOINT_FILE}"
      fi
    fi
  fi
fi

# Detect joystick — use joystick_play.py if present, else auto_play.py.
HAS_JOYSTICK=0
if python3 -c "import pygame; pygame.init(); pygame.joystick.init(); print(pygame.joystick.get_count())" 2>/dev/null | grep -q '^1$'; then
  HAS_JOYSTICK=1
fi

echo "[play.sh] Task:       ${TASK}"
echo "[play.sh] Checkpoint: ${CHECKPOINT_FILE:-<auto-detect>}"

if [[ "$HAS_JOYSTICK" -eq 1 ]]; then
  echo "[play.sh] Joystick detected → joystick_play.py"
  export CHECKPOINT_FILE
  exec uv run python joystick_play.py "$@"
else
  echo "[play.sh] No joystick → auto-play at ${AUTO_SPEED} m/s"
  export CHECKPOINT_FILE
  export AUTO_SPEED
  exec uv run python auto_play.py "$@"
fi
