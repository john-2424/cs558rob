# CS558 IRL Project

**Planner-Guided Residual Reinforcement Learning for Robust Robotic Manipulation**

## Overview

This project investigates whether integrating structured geometric planning (RRT*) with residual reinforcement learning improves robustness and sample efficiency in manipulation tasks. A Franka Emika Panda 7-DoF manipulator is simulated in PyBullet performing tabletop pick-and-place.

The hybrid control architecture consists of:

- **Planning**: Joint/configuration-space RRT* for collision-free trajectory generation
- **Classical Control**: PD joint-velocity tracking controller
- **Residual RL**: Bounded joint-position residual learned with PPO (TorchRL)

The residual is applied to the planner's IK target rather than to the velocity command:

`q*_t = q_plan_t + a_t · ρ`, with `a_t ∈ [-1, 1]^7` and cap `ρ = 0.15 rad` per joint.

The PD controller then tracks `q*_t` as its setpoint. The residual is phase-gated — active only during `PRE_GRASP` and `GRASP_DESCEND`, and exactly zero elsewhere (approach, lift, transfer, place, retreat, return-home). An earlier velocity-additive form (`qd_total = qd_PD + qd_RL`) was replaced because it compounded across steps and rewarded instantaneous distance deltas even when the motion was wrong; the position form is equivalent to a micro-waypoint shift and does not compound.

---

## Environment Setup

```bash
conda create -n cs558 python=3.10.20 -y
conda activate cs558
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Milestone 1: Classical Planning and Control Foundation

### Description

The M1 system implements the classical backbone:

- PyBullet simulation environment setup
- Panda robot loading and state extraction
- Joint-space planning with RRT* and collision checking
- Custom PD waypoint tracking controller
- Cartesian task decomposition for pick-and-place
- Grasp validation using contact and proximity checks
- Nominal pick-and-place execution with logging and plots

### How to Run

Run the nominal pick-and-place demo (launches PyBullet GUI):

```bash
python -m src.main pick-place
```

The 11-phase task sequence is:

1. Home
2. Pre-grasp
3. Grasp descend
4. Close gripper
5. Grasp validation and attachment
6. Lift
7. Transfer to place region
8. Place descend
9. Release
10. Retreat
11. Return home

Grasp validation (step 5) is a geometric check in the cube's local frame — below-top penetration plus a fingertip bracket `t ∈ [0.2, 0.8]` along a face axis with tight tolerance on the orthogonal in-plane axis. Contact-normal dot products were tried first but proved flaky across PyBullet's contact sampler at the configured step rate.

Generate M1 evaluation plots:

```bash
python -m src.evaluations.plot_pick_place_results
```

### Results

```text
results/m1/
  trajectory_log.json        -- Logged trajectory execution data
  trajectory_summary.txt     -- Summary text file
  plots/                     -- Evaluation figures
```

---

## Milestone 2: Residual Reinforcement Learning

### Description

The M2 system introduces:

- **Object pose perturbations**: Controlled randomization of cube XY (train range `‖δ_xy‖ ≤ 0.08 m`, eval extends to 0.12 m), Z (train `|δ_z| ≤ 0.01 m`), and yaw (train `|δ_yaw| ≤ 0.20 rad`). Eval scales z and yaw proportionally to the xy level (`scale = L / 0.08`), so the xy=0.12 cell sees `|δ_z| ≤ 0.015 m` and `|δ_yaw| ≤ 0.30 rad`. The 0.10 and 0.12 m eval cells are out-of-distribution on all three axes.
- **Residual PPO policy**: Bounded joint-position residual on top of the planner's IK target, trained with TorchRL PPO (`ClipPPOLoss`)
- **Evaluation framework**: Comparison across planner-only, planner+residual (hybrid), and RL-only baselines at 7 perturbation levels
- **Metrics**: Wilson-CI grasp success rate, mean reward, residual magnitude, EE-to-cube distance, episode length, phase distribution

### Architecture

The planner computes IK targets from the **nominal** (pre-perturbation) cube pose. After perturbation, the PD controller would track these now-stale targets. The RL policy observes the real-time EE-to-cube vector and outputs a bounded joint-position residual `a_t · ρ` that is added to the planner's IK target before the PD loop.

- **Observation space (41-dim, fixed-scale normalized to roughly [-1, 1])**: joint positions q (7), joint velocities q̇ (7), EE position (3) and Euler orientation (3), cube position (3) and Euler orientation (3), EE-to-cube vector (3), PD nominal velocity command (7), scalar phase indicator (1), perturbation offset δ (3), waypoint progress ω_t ∈ [0, 1] (1).
- **Action space (7-dim)**: `a_t ∈ [-1, 1]^7`, bounded joint-position residual per joint; cap `ρ = RESIDUAL_MAX_POS = 0.15 rad` (≈ 1 cm of tool-tip motion for the Panda).
- **Phase gating**: the residual is active only in `PRE_GRASP ∪ GRASP_DESCEND` (`config.RESIDUAL_ACTIVE_PHASES = {0, 1}`). It is exactly zero during approach, lift, transfer, place, retreat, and return-home.
- **Training hyperparameters** (see `src/config.py`): 1 M environment frames, 8 parallel collector workers, frames-per-batch 8192, mini-batch 256, 4 epochs per rollout, LR `5e-5` with linear decay to 0, grad-clip 0.5, critic coef 0.5, entropy coef `0.015 → 0.005` (linear anneal), target-KL early stop at 0.05 after a 100 k-frame warmup.

The classical pipeline runs the full 11-phase state machine (home → pre-grasp → grasp-descend → close → attach → lift → transfer → place-descend → release → retreat → return-home) unchanged in all three methods; only the residual injection differs.

### How to Run

**Training** (headless, no GUI):

```bash
python -m src.main train                              # hybrid (PD + residual, default)
python -m src.main train --mode rl_only               # RL-only baseline (no PD backbone)
python -m src.main train --total-timesteps 200000      # short diagnostic run
```

The `rl_only` run writes to separate paths (`results/m2/models_rl_only/`, `results/m2/tb_logs_rl_only/`) so it never overwrites the hybrid model.

Monitor progress via TensorBoard:

```bash
tensorboard --logdir results/m2/tb_logs                # hybrid
tensorboard --logdir results/m2/tb_logs_rl_only        # rl_only
```

**Evaluation**:

```bash
python -m src.main eval                                # default (per-episode diagnostics on)
python -m src.main eval --quiet                        # suppress per-episode lines
python -m src.main eval --hybrid-model PATH            # override hybrid checkpoint
python -m src.main eval --rl-only-model PATH           # override rl_only checkpoint
```

If a checkpoint exists at `results/m2/models_rl_only/final_model.pt`, the evaluator picks it up automatically; otherwise the rl-only column is skipped. Evaluates across the configured perturbation levels (see `config.PERTURB_LEVELS`) with `EVAL_EPISODES_PER_LEVEL` episodes each.

**Generate evaluation plots**:

```bash
python -m src.evaluations.plot_m2_results
```

**Full pipeline demo with residual RL** (launches PyBullet GUI):

```bash
python -m src.main residual-demo                                       # hybrid, default perturbation
python -m src.main residual-demo --mode planner_only --perturb-xy 0.04 # planner-only failure demo
python -m src.main residual-demo --mode hybrid --perturb-xy 0.02       # hybrid correction demo
python -m src.main residual-demo --no-retry                            # single-attempt grasp (matches eval)
```

### Results

```text
results/m2/
  models/                    -- Hybrid model checkpoints (final_model.pt)
  models_rl_only/            -- RL-only baseline checkpoints
  tb_logs/                   -- TensorBoard training logs (hybrid)
  tb_logs_rl_only/           -- TensorBoard training logs (rl_only)
  eval_results.json          -- Evaluation metrics across methods and perturbation levels
  plots/                     -- Comparison plots
  trajectory_log.json        -- Demo trajectory log
```

### Grasp Success Rate

Evaluation runs `EVAL_EPISODES_PER_LEVEL = 100` episodes per method per perturbation level across planner-only, hybrid (planner + position residual RL), and RL-only baselines (2100 episodes total).

Headline results from `results/m2/eval_results.json` (`‖δ_xy‖` in m; other axes at their full per-level magnitude):

| `‖δ_xy‖` | planner | hybrid | rl_only |
|---------:|--------:|-------:|--------:|
| 0.00     | 1.00    | 0.78   | 0.00    |
| 0.02     | 0.46    | 0.65   | 0.00    |
| 0.04     | 0.45    | 0.79   | 0.00    |
| 0.06     | 0.29    | 0.85   | 0.00    |
| 0.08     | 0.19    | 0.78   | 0.00    |
| 0.10     | 0.25    | 0.77   | 0.00    |
| 0.12     | 0.22    | 0.71   | 0.00    |

Hybrid stays in `[0.65, 0.85]` across every level while planner-only collapses from 1.00 to 0.22. The hybrid–planner gap exceeds 32 percentage points at every `‖δ_xy‖ ≥ 0.04 m` level and peaks at 59 points at 0.08 m. Mean residual magnitude stays between 0.0496 and 0.0549 rad — roughly one third of the 0.15 rad cap — so the policy never saturates.

---

## Project Structure

```text
project/
  src/
    config.py                    -- All configuration parameters
    main.py                      -- Entry point (subcommands)
    controller/
      pd.py                      -- Joint-space PD controller
    planner/
      rrtstar.py                 -- RRT/RRT* motion planner
    sim/
      env.py                     -- PyBullet environment
      robot.py                   -- Panda robot interface
      state.py                   -- State extraction utilities
    trajectory/
      joint_trajectory.py        -- Trajectory interpolation
    demo/
      init_dev.py                -- M1 development demo
      pick_place.py              -- M1 pick-and-place demo
      pick_place_residual.py     -- M2 residual RL demo
    rl/
      gym_env.py                 -- Gymnasium environment wrapper
      perturbation.py            -- Cube pose perturbation
      residual_policy.py         -- Residual action wrapper
      reward.py                  -- Reward function
      train.py                   -- TorchRL PPO training
      evaluate.py                -- Evaluation framework
    evaluations/
      plot_trajectory_results.py -- M1 trajectory plots
      plot_pick_place_results.py -- M1 pick-place plots
      plot_m2_results.py         -- M2 comparison plots
    utils/
      logger.py                  -- Trajectory logger
  requirements.txt
  README.md
```
