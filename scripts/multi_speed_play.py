"""Multi-speed demo: 10 G1 robots walking side-by-side at 0.1–1.0 m/s."""

import os
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def main():
  configure_torch_backends()

  task_id = "Mjlab-Velocity-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # 10 environments, 10 discrete speeds.
  num_envs = 10
  speeds = [i / 10 for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0

  # Widen env spacing so robots don't overlap.
  env_cfg.scene.num_envs = num_envs
  env_cfg.scene.env_spacing = 3.0

  # Disable push and velocity randomization for a clean demo.
  env_cfg.events.pop("push_robot", None)
  env_cfg.events["reset_base"].params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  env_cfg.events["reset_base"].params["velocity_range"] = {}

  # Force CPU.
  device = "cuda:0"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Load trained policy.
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

  # Override velocity commands with fixed per-env speeds.
  cmd_term = env.unwrapped.command_manager._terms["twist"]
  for i in range(num_envs):
    cmd_term.vel_command_b[i, 0] = speeds[i]  # pyright: ignore[reportAttributeAccessIssue]
    cmd_term.vel_command_b[i, 1] = 0.0  # pyright: ignore[reportAttributeAccessIssue]
    cmd_term.vel_command_b[i, 2] = 0.0  # pyright: ignore[reportAttributeAccessIssue]
  # Disable resampling so speeds stay fixed.
  cmd_term.cfg.resampling_time_range = (1e9, 1e9)

  print(f"[INFO] Speeds: {speeds}")

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  viewer_cls = NativeMujocoViewer if has_display else ViserPlayViewer
  viewer_cls(env, policy, frame_rate=60.0).run()
  env.close()


if __name__ == "__main__":
  main()
