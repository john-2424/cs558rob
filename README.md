# cs558rob
Repository for MS Autonomy CS55800ROB Robot Learning

Project Archives: https://drive.google.com/drive/folders/1NJaikz3rToX9GwPjWIcedbMoypJbVhVO?usp=sharing

## Final Project: Planner-Guided Residual RL for Robust Pick-and-Place

The main work in this repository lives in [`project/`](project/): a full robot-learning project on **planner-guided residual reinforcement learning** for robust robotic pick-and-place under object pose perturbation.

The central question is simple: if a classical robot pipeline plans from a nominal object pose, can a small learned correction recover robustness when the object is shifted at execution time? The project answers yes for a simulated Franka Panda in PyBullet. A classical RRT* + IK + PD backbone succeeds at the nominal cube pose, but degrades sharply when the cube is perturbed. A bounded PPO residual, added on top of the planner's joint targets only during approach and grasp descent, recovers much of that lost success without throwing away the classical structure.

### Project arc

| Milestone | Focus | What is in the repo |
|---|---|---|
| M1 | Classical manipulation backbone | PyBullet scene, Franka Panda setup, RRT*/IK waypoints, custom joint-space PD tracking, 11-phase pick-place state machine, trajectory logging, and plots. |
| M2 | Residual PPO controller | A bounded joint-position residual policy trained with TorchRL PPO, evaluated against `planner_only` and `rl_only` baselines across seven perturbation levels. |
| M3 | Multi-seed ablation and extensions | Three-seed evaluation, confidence-gated residuals, a learned grasp gate, aggregate result JSONs, plots, and documented negative results for curriculum and full-orientation perturbations. |

The detailed replication guide is in [`project/README.md`](project/README.md). The final report and M2 presentation were used as references for the framing, architecture summary, evaluation protocol, and headline results summarized here.

### Core idea

The classical stack computes nominal joint targets:

```text
nominal cube pose -> RRT* plan -> IK waypoints -> PD velocity control
```

At runtime, the cube pose is perturbed after planning. The planner still tracks stale waypoints, so the gripper may close where the cube used to be. The hybrid controller keeps the planner and adds a small residual:

```text
corrected_target = planned_target + bounded_residual
```

Key design choices:

- **Position-target residual:** the policy perturbs the PD target, letting the PD loop do the actual tracking.
- **Bounded action:** residuals are capped at `0.15 rad` per joint.
- **Phase gating:** residuals are active only during `PRE_GRASP` and `GRASP_DESCEND`, then disabled for lift, transfer, place, retreat, and return-home.
- **Classical scaffold retained:** the learned policy corrects a good prior instead of relearning the whole manipulation skill from scratch.

### Headline results

From the M2 single-seed evaluation, using 100 episodes per method per perturbation level:

| Method | Nominal success | Success at 12 cm XY perturbation | Takeaway |
|---|---:|---:|---|
| `planner_only` | 1.00 | 0.21 | Strong at nominal pose, brittle under pose error. |
| `hybrid` | 0.78 | 0.67 | Holds success in the `0.57-0.83` range across all tested perturbations. |
| `rl_only` | 0.00 | 0.00 | Removing the PD/IK scaffold fails on this training budget. |

The M3 final report extends this with a 2x2 ablation across three training seeds:

- **Stage A:** baseline hybrid residual.
- **Stage B:** confidence-gated residual, where the actor learns a scalar gate for when the residual should act.
- **Stage C1:** learned grasp gate, a small MLP classifier layered on top of the geometric grasp heuristic.
- **Stage C2:** both extensions stacked.

No single M3 variant dominates everywhere. The confidence gate helps in some moderate perturbation regimes and partially addresses nominal residual noise in one seed; the learned grasp gate helps most at the hardest perturbation cell; stacking both reaches the only M3 cell that beats the M2 single-seed reference at the same perturbation level (`0.83` success at 4 cm).

### Final project structure

```text
project/
|-- README.md                         # Full replication guide and detailed results
|-- requirements.txt                   # Python dependencies
|-- src/
|   |-- main.py                        # CLI entry point
|   |-- config.py                      # Experiment constants and feature flags
|   |-- controller/pd.py               # Joint-space PD controller
|   |-- planner/rrtstar.py             # RRT/RRT* planning
|   |-- sim/                           # PyBullet world, robot, and state helpers
|   |-- trajectory/                    # Joint trajectory interpolation
|   |-- demo/                          # Classical and residual GUI demos
|   |-- rl/                            # Gym env, PPO training, eval, residual policy, grasp gate
|   |-- evaluations/                   # Plotting and aggregation scripts
|   +-- utils/                         # Logging helpers
+-- results/
    |-- m1/                            # Classical trajectory logs and plots
    |-- m2/                            # Single-seed train/eval logs, plots, checkpoints
    |-- m3/                            # Three-seed baseline aggregate
    |-- m3_gated/                      # Confidence-gated residual aggregate
    |-- m3_lgg/                        # Learned-grasp-gate aggregate
    +-- m3_full/                       # Stacked extension aggregate
```

### Running the project

From `project/`:

```bash
python -m pip install -r requirements.txt
```

Run the classical pick-and-place demo:

```bash
python -m src.main pick-place
```

Train and evaluate the residual policy:

```bash
python -m src.main train --mode hybrid
python -m src.main train --mode rl_only
python -m src.main eval --quiet
```

Run the M3 multi-seed workflow:

```bash
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3
python -m src.main aggregate-seeds --index results/m3/multi_seed_index.json --out results/m3/eval_aggregate.json
python -m src.evaluations.plot_m3_comparison
```

The full README under `project/` includes additional commands for confidence-gated residuals, learned grasp-gate training, result regeneration, and Linux file-descriptor setup for long multi-worker PPO runs.

## Other Coursework in This Repository

The repository also contains earlier CS558ROB assignments under [`assignments/`](assignments/). These are useful context for how the final project builds from planning and control into residual learning.

### Assignment 1: Sampling-Based Motion Planning

[`assignments/assignment1/`](assignments/assignment1/) contains RRT-style planning work in both 2D and 3D:

- `part1_3d/`: PyBullet UR5 planning setup, URDF/mesh assets, RRT/BiRRT-style planning code, collision utilities, and an environment image.
- `part2_2d/`: RRT* sample code adapted for a 2D planning problem with oriented-rectangle collision checks and obstacle geometry.
- `CS55800_ROB_Assignment1.pdf`: assignment handout.

### Assignment 2: Classical Control

[`assignments/assignment2/`](assignments/assignment2/) contains a notebook for classical control of a 2-DOF robot arm:

- forward kinematics and Jacobian derivation,
- task-space and joint-space PD control,
- inverse kinematics,
- trajectory tracking plots,
- a race-car control section with multiple tracks.

The folder also includes `robotArm.png` and `arm_traj.png` used by the notebook.

### Assignment 3: MPNet Neural Planning

[`assignments/assignment3/`](assignments/assignment3/) contains a PyTorch MPNet-based neural planning assignment, adapted from the MPNet homework codebase:

- MPNet model definitions and data loaders for 2D and 3D planning,
- `mpnet_train.py` and `mpnet_test.py`,
- planning/visualization helpers,
- pretrained checkpoints,
- training logs,
- generated path outputs and comparison figures.

This assignment is large because it includes model checkpoints and many generated path files.

## Repository Notes

- The repo is intentionally broad: it includes assignment code, the final project implementation, generated plots, evaluation JSONs, checkpoints, and logs.
- The final project is the most complete and polished work; start with `project/README.md` if you want to reproduce experiments.
- The top-level `LICENSE` is Apache License 2.0.
