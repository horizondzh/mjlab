#!/usr/bin/env python
"""Play a trained G1 policy with automatic forward speed (no joystick needed)."""

import os
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def main():
  configure_torch_backends()

  task_id = "Mjlab-Velocity-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  twist_cmd = env_cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.ranges.lin_vel_x = (0.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.resampling_time_range = (1e9, 1e9)

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_root = Path(os.environ.get("LOG_ROOT", "logs/rsl_rl"))
  exp_name = os.environ.get("EXPERIMENT_NAME", "g1_velocity_flat")
  resume_path = get_checkpoint_path(log_root / exp_name, checkpoint="model_.*\\.pt")
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # Set a fixed forward speed (0.5 m/s).
  cmd_term = env.unwrapped.command_manager._terms["twist"]
  speed = float(os.environ.get("AUTO_SPEED", "0.5"))
  cmd_term.vel_command_b[0, 0] = speed  # pyright: ignore[reportAttributeAccessIssue]
  cmd_term.vel_command_b[0, 1] = 0.0  # pyright: ignore[reportAttributeAccessIssue]
  cmd_term.vel_command_b[0, 2] = 0.0  # pyright: ignore[reportAttributeAccessIssue]
  print(f"[INFO] Auto speed: {speed} m/s")

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  viewer_cls = NativeMujocoViewer if has_display else ViserPlayViewer
  viewer_cls(env, policy, frame_rate=60.0).run()
  env.close()


if __name__ == "__main__":
  main()
