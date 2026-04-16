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

## Milestone 1: Classical Planning and Control Foundation

The M1 system implements the classical backbone:

- PyBullet simulation environment setup
- Panda robot loading and state extraction
- Joint-space planning with RRT* and collision checking
- Custom PD waypoint tracking controller
- Cartesian task decomposition for pick-and-place
- Grasp validation using contact and proximity checks
- Nominal pick-and-place execution with logging and plots

---

## Milestone 2: Residual Reinforcement Learning

The M2 system introduces:

- **Object pose perturbations**: Controlled randomization of cube position to simulate execution uncertainty
- **Residual PPO policy**: Bounded velocity corrections on top of the PD controller, trained with TorchRL
- **Evaluation framework**: Comparison across planner-only, planner+residual (hybrid), and RL-only baselines at multiple perturbation levels
- **Metrics**: Grasp success rate, episode reward, episode length across perturbation magnitudes

### Architecture

The RL policy operates during the approach-and-grasp phases (pre-grasp, grasp-descend, lift). Post-grasp phases (transfer, place, retreat, return-home) use the classical pipeline unchanged.

Observation space (37-dim): joint positions, joint velocities, EE pose, cube pose, EE-to-cube vector, PD nominal velocity command, phase indicator.

Action space (7-dim): bounded residual velocity correction per joint, scaled by `RESIDUAL_MAX`.

---

## Conda Environment Setup

### Create a new Conda environment and install from `requirements.txt`

```bash
conda create -n cs558 python=3.10.20 -y
conda activate cs558
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## How to Run

The entry point uses subcommands. Run `python -m src.main --help` to see all modes.

### M1: Nominal Pick-and-Place Demo

From the project root:

```bash
python -m src.main pick-place
```

This launches the PyBullet GUI and executes the nominal pick-and-place pipeline.

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

### M2: Training the Residual PPO Policy

Hybrid policy (PD backbone + bounded residual -- default):

```bash
python -m src.main train
python -m src.main train --mode hybrid
```

RL-only baseline (no PD backbone, full velocity from policy):

```bash
python -m src.main train --mode rl_only
```

The `rl_only` run writes to separate paths (`results/m2/models_rl_only/`,
`results/m2/tb_logs_rl_only/`, `results/m2/train_log_rl_only.txt`) so it
never overwrites the hybrid model.

Training runs headless (no GUI). Monitor progress via TensorBoard:

```bash
tensorboard --logdir results/m2/tb_logs           # hybrid
tensorboard --logdir results/m2/tb_logs_rl_only   # rl_only
```

### M2: Evaluation

```bash
python -m src.main eval                          # per-episode diagnostics on (config default)
python -m src.main eval --quiet                  # suppress per-episode lines
python -m src.main eval --hybrid-model PATH      # override hybrid checkpoint
python -m src.main eval --rl-only-model PATH     # explicitly point at rl_only checkpoint
```

If a checkpoint exists at `results/m2/models_rl_only/final_model.pt` the
evaluator picks it up automatically; otherwise the rl-only column is
skipped. `planner_only` and `hybrid` always run when their models are
available.

Or run evaluation directly:

```bash
python -m src.rl.evaluate
```

This evaluates planner-only, hybrid, and RL-only methods across the configured perturbation levels (see `config.PERTURB_LEVELS`) with `EVAL_EPISODES_PER_LEVEL` episodes each. Results are saved to `results/m2/eval_results.json`.

### M2: Generate Evaluation Plots

```bash
python -m src.evaluations.plot_m2_results
```

### M2: Full Pipeline Demo with Residual RL

```bash
python -m src.main residual-demo
```

This runs the full 12-phase pick-and-place with RL-augmented control during approach/grasp phases and classical control during transport/placement phases.

---

## Output Files and Results

### M1 Results

```text
results/m1/
  trajectory_log.json        -- Logged trajectory execution data
  trajectory_summary.txt     -- Summary text file
  plots/                     -- Evaluation figures
```

### M2 Results

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

### Grasp Success Rate (N=50 episodes per cell)

| Perturbation (m) | Planner Only | Hybrid (PD + Residual) | RL Only |
|------------------|--------------|------------------------|---------|
| 0.000            | 100%         | 100%                   | 0%      |
| 0.020            | 38%          | **78%**                | 0%      |
| 0.040            | 10%          | **20%**                | 0%      |
| 0.060            | 2%           | **10%**                | 0%      |
| 0.080            | 2%           | **6%**                 | 0%      |
| 0.100            | 0%           | 0%                     | 0%      |
| 0.120            | 4%           | 2%                     | 0%      |

**Takeaways.** At small-to-moderate perturbations (0.02--0.08 m), the hybrid policy roughly doubles the planner-only success rate while retaining perfect performance at nominal (0.000 m). The RL-only ablation converges to a do-nothing local optimum under the same 1M-frame training budget, confirming that the classical PD+planner backbone supplies the inductive bias that makes PPO trainable in this regime. At the largest perturbations (>= 0.10 m), both methods saturate near zero -- the cube leaves the original IK workspace and the waypoint trajectory becomes geometrically infeasible, a failure mode beyond what bounded residual corrections can fix.

---

## Project Structure

```text
project/
  src/
    config.py                    -- All configuration parameters
    main.py                      -- Entry point
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

---

## How to Generate M1 Evaluation Plots

```bash
python -m src.evaluations.plot_pick_place_results
```
