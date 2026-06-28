"""Diagnostic: inspect first_contact events step-by-step in play mode.

Usage:
    uv run python scripts/diag_contact_flicker.py
"""

import torch
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def main():
  configure_torch_backends()

  task_id = "Mjlab-Velocity-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = 1
  env_cfg.scene.env_spacing = 3.0
  env_cfg.events.pop("push_robot", None)

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_root_path = Path("logs/rsl_rl")
  checkpoint_path = get_checkpoint_path(
    log_root_path / "g1_velocity_flat", checkpoint="model_.*\\.pt"
  )
  print(f"[INFO] Loading checkpoint: {checkpoint_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # Zero velocity command so robot should stand still.
  cmd_term = env.unwrapped.command_manager._terms["twist"]
  cmd_term.vel_command_b[0, 0] = 0.0
  cmd_term.vel_command_b[0, 1] = 0.0
  cmd_term.vel_command_b[0, 2] = 0.0
  cmd_term.cfg.resampling_time_range = (1e9, 1e9)

  sensor = env.unwrapped.scene["feet_ground_contact"]
  dt = env.unwrapped.step_dt
  print(f"[INFO] step_dt = {dt:.4f}")

  obs, _ = env.reset()
  print(f"[INFO] Initial obs ready, starting step-by-step inspection...")
  print()

  # Print header.
  print(
    f"{'step':>5} {'time':>8} {'L_contact':>10} {'L_ctime':>10} {'L_atime':>10} "
    f"{'R_contact':>10} {'R_ctime':>10} {'R_atime':>10} {'L_1st':>6} {'R_1st':>6} "
    f"{'freq':>8}"
  )

  for i in range(200):
    state = sensor._air_time_state

    # current_contact_time: [B, P] — how long each primary has been in contact
    ctime = state.current_contact_time[0]  # [P]
    atime = state.current_air_time[0]  # [P]
    is_contact = atime == 0.0  # in contact if air_time is 0

    first_contact = sensor.compute_first_contact(dt=dt)[0]  # [P]

    # Step freq tracking (replicate the reward logic).
    # We'll compute running freq after enough steps.

    if i % 20 == 0 or first_contact.any():
      marker = " *** FIRST CONTACT" if first_contact.any() else ""
      print(
        f"{i:5d} {i * dt:8.3f} "
        f"{is_contact[0].item():>10} {ctime[0].item():>10.5f} {atime[0].item():>10.5f} "
        f"{is_contact[1].item():>10} {ctime[1].item():>10.5f} {atime[1].item():>10.5f} "
        f"{first_contact[0].item():>6} {first_contact[1].item():>6}"
        f"{marker}"
      )

    # Step the env with a zero action (standing still).
    action = policy(obs)
    result = env.step(action)
    obs, reward, dones, extras = result

  env.close()


if __name__ == "__main__":
  main()
