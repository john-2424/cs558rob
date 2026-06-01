# cs558rob
Repository for MS Autonomy CS55800ROB Robot Learning

Project Archives: https://drive.google.com/drive/folders/1NJaikz3rToX9GwPjWIcedbMoypJbVhVO?usp=sharing

## Final Project: Planner-Guided Residual RL for Robust Pick-and-Place

The main work in this repository lives in [`project/`](project/): a full robot-learning project on **planner-guided residual reinforcement learning** for robust robotic pick-and-place under object pose perturbation.

The central question is simple: if a classical robot pipeline plans from a nominal object pose, can a small learned correction recover robustness when the object is shifted at execution time? The project answers yes for a simulated Franka Panda in PyBullet. A classical RRT* + IK + PD backbone succeeds at the nominal cube pose, but degrades sharply when the cube is perturbed. A bounded PPO residual, added on top of the planner's joint targets only during approach and grasp descent, recovers much of that lost success without throwing away the classical structure.

This root README is the single guide for the repository. The final project comes first because it is the most complete piece of work; the assignment folders are summarized afterward.

### Project arc

| Milestone | Focus | What is in the repo |
|---|---|---|
| M1 | Classical manipulation backbone | PyBullet scene, Franka Panda setup, RRT*/IK waypoints, custom joint-space PD tracking, 11-phase pick-place state machine, trajectory logging, and plots. |
| M2 | Residual PPO controller | A bounded joint-position residual policy trained with TorchRL PPO, evaluated against `planner_only` and `rl_only` baselines across seven perturbation levels. |
| M3 | Multi-seed ablation and extensions | Three-seed evaluation, confidence-gated residuals, a learned grasp gate, aggregate result JSONs, plots, and documented negative results for curriculum and full-orientation perturbations. |

The final report and M2 presentation were used as references for the framing, architecture summary, evaluation protocol, and headline results summarized here.

### Core idea

The classical stack computes nominal joint targets:

```text
nominal cube pose -> RRT* plan -> IK waypoints -> PD velocity control
```

At runtime, the cube pose is perturbed after planning. The planner still tracks stale waypoints, so the gripper may close where the cube used to be. The hybrid controller keeps the planner and applies a small residual correction:

```text
q_star_t = q_plan_t + rho * a_t * active_phase
```

Where:

- `q_plan_t` is the planner's joint-space target.
- `a_t` is the residual policy action, clipped to `[-1, 1]` per joint.
- `rho = 0.15 rad` is the per-joint residual cap.
- `active_phase` is 1 only during `PRE_GRASP` and `GRASP_DESCEND`, and 0 during lift, transfer, place, retreat, and return-home.

Key design choices:

- **Position-target residual:** the policy perturbs the PD target, letting the PD loop do the actual tracking.
- **Bounded action:** residuals are capped at `0.15 rad` per joint, enough to correct pose error without replacing the planner.
- **Phase gating:** the residual only acts in the phases where pose perturbation matters most.
- **Classical scaffold retained:** the learned policy corrects a strong prior instead of relearning the whole manipulation behavior from scratch.

### Architecture details

The classical pipeline plans collision-free joint-space waypoints from a nominal cube pose, maps Cartesian sub-goals to joint targets with damped least-squares IK, and tracks them with a custom PD velocity controller. The task is an 11-phase sequence:

```text
HOME -> APPROACH -> PRE_GRASP -> GRASP_DESCEND -> GRASP_CLOSE -> LIFT -> TRANSFER -> PLACE_DESCEND -> RELEASE -> RETREAT -> RETURN_HOME
```

The residual policy observes the robot state, end-effector pose, cube pose, end-effector-to-cube vector, nominal PD command, phase indicator, perturbation offset, and waypoint progress. In the M2 configuration this produces a 41-dimensional normalized observation and a 7-dimensional residual action.

Milestone 3 adds two optional extensions:

- **Confidence-gated residual:** the actor emits an extra scalar gate `g_t` in `[0, 1]`, so the residual can learn when to stay quiet.
- **Learned grasp gate:** a small MLP classifier filters grasp attach decisions on top of the geometric heuristic using logged grasp-attempt features.

Two additional ideas were tested and kept as negative results: a linear perturbation-magnitude curriculum and full-orientation perturbations with pitch/roll. Both made training worse at the same 1M-frame budget.

### Environment setup

The project trains on CPU. Multi-worker PPO with 8 PyBullet workers and 1M frames takes roughly 11 hours per seed on a modern x86 machine. No GPU is required.

The project was developed with Python 3.10. From the repository root:

```bash
cd project
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`project/requirements.txt` includes:

- `numpy==2.2.6`
- `matplotlib==3.10.8`
- `pybullet==3.2.7`
- `gymnasium>=1.0.0`
- `torch>=2.1.0`
- `torchrl>=0.6.0`
- `tensordict>=0.6.0`
- `tensorboard>=2.14.0`

Quick import check:

```bash
python -c "import torch, torchrl, tensordict, pybullet, gymnasium; print('imports ok')"
```

On Linux, long multi-worker runs can exhaust file descriptors because PyTorch opens many handles for shared tensors. Before long M3 runs, use:

```bash
ulimit -n 65535
```

### Final project structure

```text
project/
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

`project/src/main.py` is the single CLI entry point for the main workflows.

### Running the final project

From `project/`, first sanity-check the config import:

```bash
python -c "from src import config; print('config ok, perturb_xy_range =', config.PERTURB_XY_RANGE)"
```

Run the M1 classical pick-and-place GUI demo:

```bash
python -m src.main pick-place
```

Generate M1 trajectory plots:

```bash
python -m src.evaluations.plot_pick_place_results
```

Train the M2 hybrid residual policy:

```bash
python -m src.main train --mode hybrid
```

Train the M2 `rl_only` baseline:

```bash
python -m src.main train --mode rl_only
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir results/m2/tb_logs
tensorboard --logdir results/m2/tb_logs_rl_only
```

Evaluate `planner_only`, `hybrid`, and `rl_only` across all perturbation levels:

```bash
python -m src.main eval --quiet
```

Generate M2 comparison plots:

```bash
python -m src.evaluations.plot_m2_results
```

Run an optional residual GUI demo:

```bash
python -m src.main residual-demo --mode hybrid --perturb-xy 0.04
python -m src.main residual-demo --mode planner_only --perturb-xy 0.04
python -m src.main residual-demo --mode hybrid --no-retry
```

Run the M3 baseline multi-seed workflow:

```bash
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3
python -m src.main aggregate-seeds --index results/m3/multi_seed_index.json --out results/m3/eval_aggregate.json
```

The multi-seed runner is idempotent: if a per-seed checkpoint or evaluation JSON already exists, it skips that step. Use `--force-retrain` or `--force-reeval` to override.

Run only training or only evaluation:

```bash
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3 --train-only
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3 --eval-only
```

Run the confidence-gated residual variant:

```bash
# In src/config.py, set RESIDUAL_USE_GATE = True before training this variant.
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid --root results/m3_gated
python -m src.main aggregate-seeds --index results/m3_gated/multi_seed_index.json --out results/m3_gated/eval_aggregate.json
```

Train the learned grasp gate from logged grasp-attempt data:

```bash
python -m src.main train-grasp-gate --dataset results/m3/grasp_dataset.jsonl --out results/m3/grasp_gate.pt
```

Evaluate with the learned grasp gate:

```bash
# In src/config.py, set GRASP_GATE_MODE = "learned_filter" before this eval-only pass.
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3_lgg --eval-only
python -m src.main aggregate-seeds --index results/m3_lgg/multi_seed_index.json --out results/m3_lgg/eval_aggregate.json
```

Generate the M3 ablation plot:

```bash
python -m src.evaluations.plot_m3_comparison
```

### Headline results

M2 single-seed evaluation, 100 episodes per method per perturbation level:

| XY perturbation (m) | `planner_only` | `hybrid` | `rl_only` |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.78 | 0.00 |
| 0.02 | 0.53 | 0.73 | 0.00 |
| 0.04 | 0.44 | 0.78 | 0.00 |
| 0.06 | 0.40 | 0.83 | 0.00 |
| 0.08 | 0.24 | 0.72 | 0.00 |
| 0.10 | 0.20 | 0.57 | 0.00 |
| 0.12 | 0.21 | 0.67 | 0.00 |

M2 takeaways:

- `planner_only` falls from `1.00` at nominal to `0.21` at 12 cm XY perturbation.
- `hybrid` stays in the `0.57-0.83` range across all tested perturbation levels.
- `rl_only` fails at every perturbation level, showing the PD/IK scaffold is load-bearing.
- Hybrid mean residual amplitude is about `0.0496-0.0549 rad`, roughly one third of the `0.15 rad` cap, so the residual is active but not saturated.

M3 2x2 ablation, averaged across 3 seeds with 100 episodes per seed:

| XY perturbation (m) | M2 ref | A baseline | B gated | C1 learned grasp gate | C2 both |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.78 | 0.63 | 0.64 | 0.61 | 0.66 |
| 0.02 | 0.73 | 0.70 | 0.65 | 0.60 | 0.61 |
| 0.04 | 0.78 | 0.73 | 0.78 | 0.72 | 0.83 |
| 0.06 | 0.83 | 0.74 | 0.80 | 0.75 | 0.80 |
| 0.08 | 0.72 | 0.63 | 0.65 | 0.68 | 0.61 |
| 0.10 | 0.57 | 0.56 | 0.55 | 0.53 | 0.55 |
| 0.12 | 0.67 | 0.46 | 0.42 | 0.53 | 0.41 |

M3 takeaways:

- The planner cliff is reproduced across seeds.
- Every hybrid variant flattens the cliff relative to the planner at moderate and large perturbations.
- No single extension dominates everywhere.
- The learned grasp gate helps most at the hardest perturbation cell.
- The stacked variant is strongest at 4 cm and is the only M3 cell that exceeds the M2 single-seed reference at the same level.

### Important configuration flags

Most experimental switches live in [`project/src/config.py`](project/src/config.py):

| Flag | Purpose |
|---|---|
| `PERTURB_XY_RANGE` | Training-time max planar perturbation per axis. |
| `PERTURB_Z_RANGE` | Training-time max vertical perturbation. |
| `PERTURB_YAW_RANGE` | Training-time max yaw perturbation. |
| `PERTURB_PITCH_RANGE`, `PERTURB_ROLL_RANGE` | Full-orientation perturbation switches used for the M3 negative result. |
| `PERTURB_LEVELS` | Evaluation grid from 0 to 12 cm. |
| `RESIDUAL_MAX_POS` | Per-joint residual cap, `0.15 rad` in the reported experiments. |
| `RESIDUAL_USE_GATE` | Enables the M3 confidence-gated residual actor. |
| `RESIDUAL_GATE_PENALTY` | Penalizes unnecessary gate activation. |
| `GRASP_GATE_MODE` | Chooses the heuristic grasp gate, learned filter, or learned-only mode. |
| `GRASP_GATE_DATASET_PATH` | JSONL dataset of logged grasp attempts. |
| `GRASP_GATE_MODEL_PATH` | Trained learned-grasp-gate checkpoint. |
| `CURRICULUM_RAMP_EPISODES` | Linear curriculum control; the report keeps this effectively disabled after negative results. |
| `PPO_TOTAL_TIMESTEPS` | PPO training budget per seed. |
| `PPO_NUM_COLLECTOR_WORKERS` | Number of parallel PyBullet workers. |
| `EVAL_EPISODES_PER_LEVEL` | Evaluation episodes per method and perturbation level. |

For exact reproduction of a specific stage, check the relevant flags before running. Stage A uses the baseline residual setup, Stage B enables `RESIDUAL_USE_GATE`, and Stage C evaluates with `GRASP_GATE_MODE = "learned_filter"`.

### Result artifacts

| Area | Key outputs |
|---|---|
| M1 | `project/results/m1/trajectory_log.json`, `trajectory_summary.txt`, and trajectory plots. |
| M2 training | `project/results/m2/models/`, `models_rl_only/`, `tb_logs/`, and `tb_logs_rl_only/`. |
| M2 evaluation | `project/results/m2/eval_results.json` and `project/results/m2/plots/`. |
| M3 baseline | `project/results/m3/seed_NNN/`, `multi_seed_index.json`, `eval_aggregate.json`, and `grasp_dataset.jsonl`. |
| M3 confidence gate | `project/results/m3_gated/seed_NNN/` and `eval_aggregate.json`. |
| M3 learned grasp gate | `project/results/m3_lgg/seed_NNN/` and `eval_aggregate.json`. |
| M3 stacked variant | `project/results/m3_full/seed_NNN/` and `eval_aggregate.json`. |
| M3 plot | `project/results/m3/plots/m3_comparison.png`. |

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
- planning and visualization helpers,
- pretrained checkpoints,
- training logs,
- generated path outputs and comparison figures.

This assignment is large because it includes model checkpoints and many generated path files.

## Repository Notes

- The repo is intentionally broad: it includes assignment code, the final project implementation, generated plots, evaluation JSONs, checkpoints, and logs.
- Start with the final project section above if you want to understand or reproduce the main result.
- The top-level `LICENSE` is Apache License 2.0.
