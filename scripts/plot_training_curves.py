"""
训练曲线可视化工具

用法:
  uv run python scripts/plot_training_curves.py
  uv run python scripts/plot_training_curves.py --log-dir logs/rsl_rl/g1_velocity_xpeng_walk
  uv run python scripts/plot_training_curves.py --log-dir logs/rsl_rl --smooth 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )
except ImportError:
  print("需要安装 tensorboard: uv add tensorboard")
  raise

try:
  import matplotlib.pyplot as plt
  import numpy as np
except ImportError:
  print("需要安装 matplotlib: uv add matplotlib")
  raise


def _scalars(events: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
  events.Reload()
  if tag not in events.Tags().get("scalars", []):
    return np.array([]), np.array([])
  data = events.Scalars(tag)
  return np.array([d.step for d in data]), np.array([d.value for d in data])


def smooth(y: np.ndarray, weight: float = 0.85) -> np.ndarray:
  """指数移动平均平滑"""
  if len(y) < 3:
    return y
  out = np.empty_like(y)
  out[0] = y[0]
  for i in range(1, len(y)):
    out[i] = weight * out[i - 1] + (1 - weight) * y[i]
  return out


def plot_run(ax, run_dir: Path, label: str, smooth_weight: float) -> dict[str, list]:
  events = EventAccumulator(str(run_dir))
  events.Reload()

  tags = events.Tags().get("scalars", [])
  tag_groups: dict[str, list[str]] = {}
  for tag in sorted(tags):
    group = tag.split("/")[0] if "/" in tag else "Other"
    tag_groups.setdefault(group, []).append(tag)

  plotted: dict[str, list] = {}
  for group, group_tags in sorted(tag_groups.items()):
    for tag in group_tags:
      steps, values = _scalars(events, tag)
      if len(steps) == 0:
        continue

      # Use EMI loss tag name
      short_name = tag.split("/")[-1]
      if short_name in plotted:
        continue
      plotted.setdefault(short_name, []).append(run_dir)

      y_smooth = smooth(values, smooth_weight)
      (line,) = ax.plot(steps, y_smooth, label=f"{label}/{short_name}", alpha=0.8)
      ax.fill_between(
        steps,
        y_smooth - np.std(values),
        y_smooth + np.std(values),
        alpha=0.1,
        color=line.get_color(),
      )

  return plotted


def main() -> None:
  parser = argparse.ArgumentParser(description="绘制 mjlab 训练曲线")
  parser.add_argument(
    "--log-dir",
    type=str,
    default="logs/rsl_rl",
    help="TensorBoard 日志目录 (默认: logs/rsl_rl)",
  )
  parser.add_argument(
    "--smooth",
    type=float,
    default=0.85,
    help="指数移动平均平滑系数 (默认: 0.85)",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="保存图片路径 (可选，默认显示窗口)",
  )
  args = parser.parse_args()

  log_root = Path(args.log_dir)
  if not log_root.exists():
    print(f"❌ 目录不存在: {log_root}")
    print("提示: 先运行训练: uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-Unitree-G1")
    return

  # Find all run directories with tfevents.
  run_dirs = sorted(log_root.glob("*/*/")) if (log_root / "*").parent.exists() else []
  # Also search recursively
  if not run_dirs:
    run_dirs = sorted(log_root.rglob("events.out.tfevents*"))
    run_dirs = list({d.parent for d in run_dirs})
    if not run_dirs:
      run_dirs = [log_root]

  if not run_dirs or not any(
    any(f.name.startswith("events.out.tfevents") for f in d.iterdir())
    for d in run_dirs
  ):
    print(f"❌ 在 {log_root} 下未找到 TensorBoard 日志文件")
    print("请确保已运行训练并指定正确的 --log-dir")
    return

  print(f"找到 {len(run_dirs)} 个运行目录:")
  for d in run_dirs:
    print(f"  {d}")

  # Plot each run in its own figure.
  figures: list[plt.Figure] = []
  for i, run_dir in enumerate(run_dirs):
    rel = run_dir.relative_to(log_root) if run_dir != log_root else run_dir.name
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Training Curves: {rel}", fontsize=14)

    # Panel 1: Total reward
    ax = axes[0, 0]
    plot_run(ax, run_dir, "Train", args.smooth)
    ax.set_title("Episode Metrics")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Individual reward components
    ax = axes[0, 1]
    events = EventAccumulator(str(run_dir))
    events.Reload()
    reward_tags = [t for t in events.Tags().get("scalars", []) if "rewards" in t.lower()]
    for tag in reward_tags:
      steps, values = _scalars(events, tag)
      if len(steps) > 0:
        ax.plot(steps, smooth(values, args.smooth), label=tag.split("/")[-1], alpha=0.7)
    ax.set_title("Reward Components")
    ax.set_xlabel("Step")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel 3: Policy loss
    ax = axes[1, 0]
    loss_tags = [t for t in events.Tags().get("scalars", []) if "loss" in t.lower() or "surrogate" in t.lower()]
    for tag in loss_tags:
      steps, values = _scalars(events, tag)
      if len(steps) > 0:
        ax.plot(steps, smooth(values, args.smooth), label=tag.split("/")[-1], alpha=0.7)
    ax.set_title("Losses")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: KL / entropy / explained variance
    ax = axes[1, 1]
    misc_tags = ["Train/kl", "Train/explained_variance", "Train/learning_rate"]
    for tag in misc_tags:
      steps, values = _scalars(events, tag)
      if len(steps) > 0:
        ax.plot(steps, smooth(values, args.smooth), label=tag.split("/")[-1], alpha=0.7)
    ax.set_title("Training Metrics")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figures.append(fig)

  if args.output:
    for i, fig in enumerate(figures):
      path = args.output if len(figures) == 1 else f"{Path(args.output).stem}_{i}{Path(args.output).suffix}"
      fig.savefig(path, dpi=150)
      print(f"✓ 已保存: {path}")
  else:
    plt.show()


if __name__ == "__main__":
  main()
