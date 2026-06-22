# mjlab — G1 Velocity Training

Fork of `google-deepmind/mjlab` focused on Unitree G1 flat-terrain velocity training.
Based on MuJoCo + Warp + RSL-RL.

## Project

- Python 3.12+, managed with `uv`. Always use `uv run` prefix.
- Entry: `src/mjlab/scripts/train.py` (training), `src/mjlab/scripts/play.py` (inference).
- Remote: `git@github.com:horizondzh/mjlabdzh.git`
- Task ID: `Mjlab-Velocity-Flat-Unitree-G1`

## Commands

```sh
# Quick scripts
./train.sh        # Start/resume training (resume=True in config)
./play.sh         # Play latest model on CPU (1 env)
./tensorboard.sh  # View training curves (port 6006)

# Dev checks
uv run ruff format && uv run ruff check --fix  # Format & lint
make type                                       # Type-check
make check                                      # Full check (format + lint + type)

# Tests
uv run pytest tests/ -x -q   # Run all
uv run pytest tests/test_velocity_task.py  # Specific
```

## Architecture

```
src/mjlab/
├── asset_zoo/robots/unitree_g1/   # G1 XML + constants
├── envs/                           # ManagerBasedRlEnv, MDP primitives
├── tasks/velocity/
│   ├── config/g1/
│   │   ├── env_cfgs.py            # ⭐ Main config: rewards, physics, commands
│   │   └── rl_cfg.py              # PPO config, network sizes, resume
│   ├── mdp/                        # Reward/observation/termination functions
│   ├── rl/runner.py                # VelocityOnPolicyRunner
│   └── velocity_env_cfg.py         # Base velocity task (shared)
├── rl/                             # RSL-RL wrappers, vecenv, config base
└── scripts/
    ├── train.py                    # Training entry point
    └── play.py                     # Inference entry point
logs/rsl_rl/g1_velocity_xpeng_walk/ # Training runs (TensorBoard)
```

Key config locations for G1 reward tuning:
- `src/mjlab/tasks/velocity/config/g1/env_cfgs.py` — `_g1_base_env_cfg` and `unitree_g1_flat_env_cfg`
- `src/mjlab/tasks/velocity/config/g1/rl_cfg.py` — `unitree_g1_ppo_runner_cfg`

## Conventions

- Line length: 88 cols (code + comments + docstrings).
- Use functions/fixtures for tests, not test classes.
- Import order: stdlib → third-party → mjlab internal.
- Config overrides go in G1-specific files, not shared `velocity_env_cfg.py`.
- Always `uv run`, never bare `python` or `pip`.

## Notes

