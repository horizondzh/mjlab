"""Velocity task metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class step_freq_from_contact:
  """Step frequency from contact sensor phase transitions over a sliding window.

  Counts how many times ``last_air_time`` was written (indicating a completed
  air→contact transition) over a *window_len* second window, averaged across
  both feet.  This mirrors the hip-pitch metric's window-based approach so the
  two are directly comparable.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self.step_dt = env.step_dt
    self.window_len: float = cfg.params.get("window_len", 2.0)
    window_steps = int(self.window_len / self.step_dt)
    self.window_steps = window_steps

    # Ring buffer tracking per-foot last_air_time: [B, window_steps, P].
    B = env.num_envs
    P = 2
    self.la_history = torch.zeros(
      (B, window_steps, P), device=env.device, dtype=torch.float32
    )
    self._ptr = torch.zeros(B, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    last_air = sensor.data.last_air_time  # [B, P]

    B = env.num_envs
    env_ids = torch.arange(B, device=env.device)
    ptr = self._ptr
    assert last_air is not None
    self.la_history[env_ids, ptr] = last_air
    self._ptr = (ptr + 1) % self.window_steps

    # Chronological order for each env: most recent at (ptr-1), oldest at ptr.
    offsets = (
      torch.arange(self.window_steps, device=env.device) - self._ptr.unsqueeze(1)
    ) % self.window_steps
    ordered = torch.gather(
      self.la_history, 1, offsets.unsqueeze(-1).expand(-1, -1, 2)
    )  # [B, T, P]

    # A "step" (completed air phase) is when last_air_time changes to a new
    # non-zero value — i.e. the foot just touched down.
    prev = ordered[:, :-1, :]  # [B, T-1, P]
    curr = ordered[:, 1:, :]  # [B, T-1, P]
    new_landing = (curr != prev) & (curr > 0)  # [B, T-1, P]
    step_count = new_landing.float().sum(dim=(1, 2))  # [B]

    freq = step_count / self.window_len  # [B]
    return freq

  def reset(self, env_ids: torch.Tensor) -> None:
    self.la_history[env_ids] = 0.0


class step_freq_from_hip_pitch:
  """Step frequency from hip pitch velocity zero-crossings.

  During walking the hip pitch joints oscillate sinusoidally.  The left-right
  difference of hip pitch velocities changes sign at each swing↔stance
  transition, giving two zero-crossings per stride cycle per leg.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self.step_dt = env.step_dt
    self.window_len: float = cfg.params.get("window_len", 2.0)
    window_steps = int(self.window_len / self.step_dt)
    self.window_steps = window_steps

    robot: Entity = env.scene["robot"]
    left_ids, _ = robot.find_joints("left_hip_pitch_joint")
    right_ids, _ = robot.find_joints("right_hip_pitch_joint")
    self.left_idx: int = left_ids[0]
    self.right_idx: int = right_ids[0]

    # Per-env ring buffer: [num_envs, window_steps] storing vel diff.
    self.vel_history = torch.zeros(
      (env.num_envs, window_steps), device=env.device, dtype=torch.float32
    )
    self._ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    left_vel = joint_vel[:, self.left_idx]
    right_vel = joint_vel[:, self.right_idx]
    diff_vel = left_vel - right_vel

    # Per-env write to ring buffer; ptr advance.
    B = env.num_envs
    env_ids = torch.arange(B, device=env.device)
    ptr = self._ptr  # [B]
    self.vel_history[env_ids, ptr] = diff_vel
    self._ptr = (ptr + 1) % self.window_steps

    # Per-env: count zero-crossings in the ring buffer.
    # Build an index tensor that aligns each env's history chronologically:
    # the most recent entry is at (ptr - 1), oldest at ptr.
    offsets = (
      torch.arange(self.window_steps, device=env.device) - self._ptr.unsqueeze(1)
    ) % self.window_steps
    ordered = torch.gather(self.vel_history, 1, offsets)  # [B, T]
    sign = ordered > 0
    crossings = (sign[:, :-1] != sign[:, 1:]).float()

    cycles = crossings.sum(dim=1) / 2.0
    return cycles / self.window_len

  def reset(self, env_ids: torch.Tensor) -> None:
    self.vel_history[env_ids] = 0.0
