#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings ----

# Task selection (use 'uv run list-envs' to see all available tasks).
TASK="${TASK:-Mjlab-Velocity-Flat-Unitree-G1}"

# Logging root directory.
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl}"

# Velocity command ranges (forward-only, 0–1 m/s).
LIN_VEL_X="${LIN_VEL_X:-(0.0, 1.0)}"
LIN_VEL_Y="${LIN_VEL_Y:-(0.0, 0.0)}"
ANG_VEL_Z="${ANG_VEL_Z:-(0.0, 0.0)}"

# Number of environments (envs per GPU if multi-GPU).
NUM_ENVS="${NUM_ENVS:-4096}"

# Experiment name.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-g1_velocity_flat}"

# Max training iterations.
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"

# -------------------------------------------------

CMD=(
  uv run train "${TASK}"
  --log-root "${LOG_ROOT}"
  --agent.experiment-name "${EXPERIMENT_NAME}"
  --agent.max-iterations "${MAX_ITERATIONS}"
  --agent.logger tensorboard
  --env.scene.num-envs "${NUM_ENVS}"
  "--env.commands.twist.ranges.lin-vel-x" "${LIN_VEL_X}"
  "--env.commands.twist.ranges.lin-vel-y" "${LIN_VEL_Y}"
  "--env.commands.twist.ranges.ang-vel-z" "${ANG_VEL_Z}"
  --env.commands.twist.heading-command False
  --env.commands.twist.rel-standing-envs 0.0
  --env.commands.twist.rel-heading-envs 0.0
  --env.commands.twist.ranges.heading None
  --env.curriculum.command-vel.params.velocity-stages.0.lin-vel-x "${LIN_VEL_X}"
  --env.curriculum.command-vel.params.velocity-stages.0.ang-vel-z "${ANG_VEL_Z}"
)

# Auto-detect whether to resume from a previous run.
EXP_DIR="${LOG_ROOT}/${EXPERIMENT_NAME}"
if [[ -d "${EXP_DIR}" ]]; then
  LATEST_RUN=$(ls -dt "${EXP_DIR}"/*/ 2>/dev/null | head -1 || true)
  if [[ -n "${LATEST_RUN}" ]]; then
    LATEST_CKPT=$(ls -t "${LATEST_RUN}"/*.pt 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
      CMD+=(--agent.resume True)
      echo "[train.sh] Resuming from: ${LATEST_CKPT}"
    fi
  fi
fi

echo "[train.sh] Task:         ${TASK}"
echo "[train.sh] Log dir:      ${EXP_DIR}"
echo "[train.sh] Num envs:     ${NUM_ENVS}"
echo "[train.sh] Lin vel x:    ${LIN_VEL_X}"
echo ""

exec "${CMD[@]}" "$@"
