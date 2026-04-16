# CS558 IRL Project

**Planner-Guided Residual Reinforcement Learning for Robust Robotic Manipulation**

## Overview

This project investigates whether integrating structured geometric planning (RRT*) with residual reinforcement learning improves robustness and sample efficiency in manipulation tasks. A Franka Emika Panda 7-DoF manipulator is simulated in PyBullet performing tabletop pick-and-place.

The hybrid control architecture consists of:

- **Planning**: Joint/configuration-space RRT* for collision-free trajectory generation
- **Classical Control**: PD velocity tracking controller
- **Residual RL**: Bounded velocity residual learned using PPO (TorchRL)

The total velocity command applied is: `qd_total = qd_PD + qd_RL`

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

The expected task sequence is:

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

- **Object pose perturbations**: Controlled randomization of cube position (XY, Z, yaw) to simulate execution uncertainty
- **Residual PPO policy**: Bounded velocity corrections on top of the PD controller, trained with TorchRL
- **Evaluation framework**: Comparison across planner-only, planner+residual (hybrid), and RL-only baselines at multiple perturbation levels
- **Metrics**: Grasp success rate, episode reward, episode length across perturbation magnitudes

### Architecture

The planner computes IK targets from the **nominal** (pre-perturbation) cube pose. After perturbation, the classical PD controller tracks these now-stale targets. The RL policy observes the real-time EE-to-cube vector and outputs bounded velocity corrections to steer the arm toward the actual cube.

- Observation space (40-dim): joint positions, joint velocities, EE pose, cube pose, EE-to-cube vector, PD nominal velocity command, phase indicator, perturbation offset
- Action space (7-dim): bounded residual velocity correction per joint, scaled by `RESIDUAL_MAX`

The RL policy operates during the approach-and-grasp phases (pre-grasp, grasp-descend, lift). Post-grasp phases (transfer, place, retreat, return-home) use the classical pipeline unchanged.

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

Results will be populated after training completes. Evaluation runs N=50 episodes per method per perturbation level across planner-only, hybrid (PD + residual RL), and RL-only baselines.

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
