# CS558ROB Project

**Planner-Guided Residual Reinforcement Learning for Robust Robotic Pick-and-Place under Object Pose Perturbation**

This repository contains the full project: classical motion planning + control (Milestone 1), a bounded residual PPO policy on top (Milestone 2), and a 2×2 architectural ablation across multiple training seeds with two architectural extensions plus two negative-result extensions (Milestone 3). The headline finding is that a small (≈ 1/3 of cap) joint-position residual on top of an RRT* + IK + PD pipeline recovers most of the planner's perturbation cliff: hybrid grasp success holds in `[0.57, 0.83]` across seven perturbation magnitudes while planner-only collapses from `1.00` to `0.21`.

This README is a self-contained replication guide — every numeric claim in the final report can be reproduced from the steps below.

---

## Architecture in one paragraph

The classical pipeline plans collision-free joint-space waypoints from a **nominal** cube pose, maps Cartesian sub-goals to joint targets via damped least-squares IK, and tracks them with a custom PD velocity controller. At runtime, the cube is perturbed by `δ = (δ_xy, δ_z, δ_yaw)` after the planner has committed; the planner is blind to `δ` and the gripper would arrive at the wrong place. The residual policy adds a bounded joint-position correction:

```
q*_t = q_plan_t + ρ · a_t · 1[φ_t ∈ Φ_r],   a_t ∈ [-1, 1]^7,   ρ = 0.15 rad
```

The PD controller then tracks `q*_t` instead of the raw planner target. The residual is **phase-gated** — active only in `PRE_GRASP` and `GRASP_DESCEND`, exactly zero during lift, transfer, place, retreat, and return-home. Milestone 3 adds two architectural extensions on top of this: a **confidence-gated residual** that learns when to act (an extra scalar gate `g_t ∈ [0,1]` multiplies the residual), and a **learned grasp gate** (a small MLP classifier that filters attach decisions on top of the geometric heuristic).

---

## Environment setup

**Hardware:** the project trains on CPU. Multi-worker PPO on 8 PyBullet workers at 1 M frames takes roughly 11 hours of wall-clock per seed on a modern x86 server. No GPU is required (or used).

**Operating system:** developed and tested on both Linux and Windows. **One Linux-specific note:** PyTorch's default cross-process tensor sharing strategy on Linux opens a file descriptor per shared tensor. With 8 workers + a long episode buffer this can exhaust the per-process FD limit. The training code calls `torch.multiprocessing.set_sharing_strategy("file_system")` early to mitigate this; before launching long multi-seed runs on Linux, also bump the FD ulimit:

```bash
ulimit -n 65535     # only needed on Linux for long multi-seed runs
```

**Python:** 3.10. Tested with conda 3.10.20.

```bash
conda create -n cs558 python=3.10.20 -y
conda activate cs558
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy 2.2.6`, `matplotlib 3.10.8`, `pybullet 3.2.7`, and floor-pins `gymnasium ≥ 1.0`, `torch ≥ 2.1`, `torchrl ≥ 0.6`, `tensordict ≥ 0.6`, `tensorboard ≥ 2.14`. These are the only dependencies — every script under `src/` imports from this set.

Quick sanity check after install:

```bash
python -c "import torch, torchrl, tensordict, pybullet, gymnasium; print('imports ok')"
```

---

## Project structure

```text
project/
├── src/
│   ├── config.py                         # All experiment constants in one place
│   ├── main.py                           # Single CLI entry point with subcommands
│   ├── controller/
│   │   └── pd.py                         # Joint-space PD controller
│   ├── planner/
│   │   └── rrtstar.py                    # RRT / RRT* sampling-based planner
│   ├── sim/
│   │   ├── env.py                        # PyBullet world setup
│   │   ├── robot.py                      # Panda interface, IK, grasp validity check
│   │   └── state.py                      # State extraction utilities
│   ├── trajectory/
│   │   └── joint_trajectory.py           # Trajectory interpolation
│   ├── demo/
│   │   ├── init_dev.py                   # M1 dev/init demo
│   │   ├── pick_place.py                 # M1 nominal pick-and-place
│   │   └── pick_place_residual.py        # M2 residual demo (GUI)
│   ├── rl/
│   │   ├── gym_env.py                    # Gymnasium env (obs, action, reward wiring)
│   │   ├── perturbation.py               # Cube pose perturbation sampler
│   │   ├── residual_policy.py            # Residual / planner_only / rl_only action wrapper
│   │   ├── reward.py                     # Reward composition
│   │   ├── train.py                      # PPO training (TorchRL) + actor-critic build
│   │   ├── evaluate.py                   # Multi-method, multi-level eval
│   │   ├── trace_rollout.py              # Per-step rollout diagnostic
│   │   ├── multi_seed.py                 # M3: orchestrator for N-seed train + eval
│   │   └── grasp_gate.py                 # M3: learned grasp-gate classifier (train + infer)
│   ├── evaluations/
│   │   ├── plot_trajectory_results.py    # M1 trajectory plots
│   │   ├── plot_pick_place_results.py    # M1 pick-place plots
│   │   ├── plot_m2_results.py            # M2 single-seed comparison plots
│   │   ├── aggregate_seeds.py            # M3: aggregate per-seed eval JSONs
│   │   └── plot_m3_comparison.py         # M3 2×2 ablation overlay plot
│   └── utils/
│       ├── logger.py                     # Trajectory logger
│       └── run_log.py                    # Stdout-tee log helper
├── results/                              # All outputs land here (generated, gitignored)
├── requirements.txt
└── README.md
```

`src/main.py` is the single entry point. All commands below are subcommands of `python -m src.main`.

---

## How to replicate the full project arc

This section is a top-to-bottom recipe. Following it exactly produces every number in the final report and every figure in `results/m3/plots/`.

Time estimates assume an 8-core x86 server with no GPU. Adjust if you have a beefier machine.

### Step 0 — Sanity check the install

```bash
python -c "from src import config; print('config ok, perturb_xy_range =', config.PERTURB_XY_RANGE)"
# Expected: perturb_xy_range = 0.08
```

### Step 1 — Milestone 1: classical pick-and-place demo

This is the visual sanity check that the classical backbone works at nominal pose. Launches the PyBullet GUI.

```bash
python -m src.main pick-place
```

The 11-phase task sequence runs once: HOME → APPROACH → PRE_GRASP → GRASP_DESCEND → GRASP_CLOSE → LIFT → TRANSFER → PLACE_DESCEND → RELEASE → RETREAT → RETURN_HOME. The grasp validity check (step 5) is a geometric test in the cube's local frame: both fingertips below the cube top, cube center inside the finger-axis bracket `t ∈ [0.2, 0.8]`, and finger axis perpendicular to the bracketed face. If the first descend fails the check, the system retries one descend deeper before attaching.

Generate the M1 trajectory plots:

```bash
python -m src.evaluations.plot_pick_place_results
```

**Outputs:**

```
results/m1/
├── trajectory_log.json
├── trajectory_summary.txt
└── plots/                             # XZ trajectory, joint-tracking error, etc.
```

### Step 2 — Milestone 2: residual PPO baseline (single seed)

This trains the M2 baseline residual on top of the classical pipeline, then evaluates planner-only / hybrid / rl-only across seven perturbation levels.

**Train hybrid (PD + residual)** — ~11 hours wall-clock, headless:

```bash
python -m src.main train                    # mode=hybrid by default
```

**Train rl_only baseline** — ~11 hours, separate output paths so it never overwrites hybrid:

```bash
python -m src.main train --mode rl_only
```

Monitor either run live via TensorBoard:

```bash
tensorboard --logdir results/m2/tb_logs                # hybrid
tensorboard --logdir results/m2/tb_logs_rl_only        # rl_only
```

**Evaluate** — ~10 minutes for the full 2,100 episodes (3 methods × 7 levels × 100 episodes):

```bash
python -m src.main eval                     # uses results/m2/models/final_model.pt and models_rl_only/
python -m src.main eval --quiet             # suppress per-episode diagnostic lines
```

**Generate M2 comparison plots:**

```bash
python -m src.evaluations.plot_m2_results
```

**M2 outputs:**

```
results/m2/
├── models/                            # hybrid checkpoints (final_model.pt, best_model.pt, …)
├── models_rl_only/                    # rl_only checkpoints
├── tb_logs/, tb_logs_rl_only/         # TensorBoard event files
├── eval_results.json                  # 7 levels × 3 methods × full diagnostics
├── eval_log.txt
├── plots/                             # success-rate, residual amplitude, phase, grasp diag
└── trajectory_log.json
```

**Optional GUI demo** — same trained model, new figure:

```bash
python -m src.main residual-demo                              # hybrid, default perturbation
python -m src.main residual-demo --mode planner_only --perturb-xy 0.04
python -m src.main residual-demo --mode hybrid --perturb-xy 0.02
python -m src.main residual-demo --no-retry                   # single-attempt grasp (matches eval)
```

### Step 3 — Milestone 3 Stage A: multi-seed M3 baseline

Stage A re-runs the M2 architecture across **three independent training seeds** with no architectural extensions, so the headline plot has across-seed percentile bands instead of just one Wilson interval. Estimated wall-clock: ~3 days for 3 seeds × 2 modes × ~11 hours per training, plus a few hours of eval.

Default config has confidence-gate OFF, learned-grasp-gate OFF, curriculum OFF, full-orientation perturbation OFF — all the M3 negative-result and not-yet-on extensions are gated off at this point.

```bash
ulimit -n 65535     # Linux only; see Environment Setup
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3
```

The orchestrator under `src/rl/multi_seed.py` is **idempotent**: if a per-seed checkpoint or eval JSON already exists, it skips that step (so you can resume after a crash without losing progress). Use `--force-retrain` or `--force-reeval` to override.

You can also split into two passes (useful if you want to verify training before evaluating):

```bash
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3 --train-only
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3 --eval-only
```

**Aggregate across seeds** (a few seconds, pure Python):

```bash
python -m src.main aggregate-seeds \
    --index results/m3/multi_seed_index.json \
    --out results/m3/eval_aggregate.json
```

**Stage A outputs:**

```
results/m3/
├── seed_000/
│   ├── models_hybrid/final_model.pt   # ~3 MB; auto-selected by composite best score
│   ├── models_rl_only/final_model.pt
│   ├── eval_results.json
│   ├── eval_log.txt, train_log_*.txt
│   └── tb_logs_*/
├── seed_001/, seed_002/               # same shape
├── multi_seed_index.json              # which seeds × modes were run
├── eval_aggregate.json                # cross-seed mean + 25/75 percentile
└── grasp_dataset.jsonl                # logged grasp-attempt rows for Stage C
```

### Step 4 — Milestone 3 Stage B: confidence-gated residual

Flip one config flag, then retrain hybrid only (rl_only does not use the gate, so its checkpoints from Stage A remain valid).

In `src/config.py`:

```python
RESIDUAL_USE_GATE = True   # was False
```

This widens the actor's action from 7 to 8 dims; the 8th channel is a scalar gate `g_t` that multiplies the residual (`q*_t = q_plan_t + ρ · g_t · a_t · 1[φ_t ∈ Φ_r]`). A per-step penalty `−λ_g · g_t` with `λ_g = 0.05` in residual-active phases pushes the gate toward zero where the planner already suffices. The gate's output-layer bias is warm-started at logit `+1.5` so `sigmoid(1.5) ≈ 0.82` at frame zero — close to M2's always-on regime.

```bash
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid --root results/m3_gated
python -m src.main aggregate-seeds \
    --index results/m3_gated/multi_seed_index.json \
    --out results/m3_gated/eval_aggregate.json
```

After Stage B, set `RESIDUAL_USE_GATE = False` again before proceeding (Stage C uses the Stage A architecture as its base).

### Step 5 — Milestone 3 Stage C: learned grasp gate

By the end of Stage A and B, every grasp attempt has been logged to `results/m3/grasp_dataset.jsonl` (≈ 18 k rows after Stages A + B). The classifier is trained offline on that snapshot.

```bash
python -m src.main train-grasp-gate \
    --dataset results/m3/grasp_dataset.jsonl \
    --out results/m3/grasp_gate.pt
```

Output prints `n_train`, `n_val`, and `best_val_acc`. Expected val_acc on the snapshot used in the report: `~0.987` on a 80/20 shuffle (15,147 train / 3,787 val). The classifier is a 10-feature MLP (10 → 64 → 64 → 1, ReLU + sigmoid, BCE-with-pos-weight to handle the natural ~30/70 class imbalance).

**Flip the inference mode** in `src/config.py`:

```python
GRASP_GATE_MODE = "learned_filter"   # was "heuristic"
```

In `learned_filter` mode the env attaches the cube only when both the geometric heuristic AND the classifier (above its 0.5 threshold) say "ready" — the classifier can only further-reject, never approve a heuristically-invalid grasp.

**Stage C is eval-only** — no retraining needed. Reuse the existing checkpoints from Stages A and B by copying their directory trees and clearing the eval JSONs:

```bash
# Stage C1: gate-OFF policy + learned grasp gate
cp -r results/m3 results/m3_lgg
rm -f results/m3_lgg/seed_*/eval_results.json
rm -f results/m3_lgg/eval_aggregate.json
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid rl_only --root results/m3_lgg --eval-only
python -m src.main aggregate-seeds \
    --index results/m3_lgg/multi_seed_index.json \
    --out results/m3_lgg/eval_aggregate.json

# Stage C2: gate-ON policy + learned grasp gate
cp -r results/m3_gated results/m3_full
rm -f results/m3_full/seed_*/eval_results.json
rm -f results/m3_full/eval_aggregate.json
python -m src.main multi-seed --seeds 0 1 2 --modes hybrid --root results/m3_full --eval-only
python -m src.main aggregate-seeds \
    --index results/m3_full/multi_seed_index.json \
    --out results/m3_full/eval_aggregate.json
```

After Stage C, set `GRASP_GATE_MODE = "heuristic"` again to leave the repo at default.

### Step 6 — Generate the headline M3 ablation plot

```bash
python -m src.evaluations.plot_m3_comparison
```

**Output:** `results/m3/plots/m3_comparison.png` — overlays planner + four hybrid variants (Stage A, B, C1, C2) with `25/75` percentile bands across the three seeds. The dashed grey reference is the M2 single-seed hybrid for visual anchoring.

The script reads the four `eval_aggregate.json` files from `results/m3/`, `results/m3_gated/`, `results/m3_lgg/`, `results/m3_full/`. If any of those paths is missing, that curve is omitted.

---

## Headline results

### M2 single-seed evaluation (Step 2)

`results/m2/eval_results.json` after a 100-episodes-per-cell eval against the trained M2 hybrid checkpoint:

| `‖δ_xy‖` (m) | planner_only | hybrid | rl_only |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.78 | 0.00 |
| 0.02 | 0.53 | 0.73 | 0.00 |
| 0.04 | 0.44 | 0.78 | 0.00 |
| 0.06 | 0.40 | 0.83 | 0.00 |
| 0.08 | 0.24 | 0.72 | 0.00 |
| 0.10 | 0.20 | 0.57 | 0.00 |
| 0.12 | 0.21 | 0.67 | 0.00 |

Hybrid stays in `[0.57, 0.83]` across every level while planner-only collapses from `1.00` to `0.21`. `rl_only` (no PD+IK backbone) fails at every level — it gets stuck in `GRASP_DESCEND` for `97%–100%` of episodes per level.

Mean residual amplitude in hybrid runs is `0.0496–0.0549` rad (≈ 33%–37% of the `0.15` rad cap), with a gentle upward slope in `δ` — the policy uses more of its budget when it has more error to cancel, but never saturates.

### M3 2×2 ablation (Steps 3–6)

Mean grasp success across 3 training seeds, 100 episodes per cell, from `results/m3*/eval_aggregate.json`:

| `‖δ_xy‖` (m) | M2 ref | A baseline | B gated | C1 lgg | C2 both |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.78 | 0.63 | 0.64 | 0.61 | **0.66** |
| 0.02 | 0.73 | **0.70** | 0.65 | 0.60 | 0.61 |
| 0.04 | 0.78 | 0.73 | 0.78 | 0.72 | **0.83** |
| 0.06 | 0.83 | 0.74 | **0.80** | 0.75 | 0.80 |
| 0.08 | 0.72 | 0.63 | 0.65 | **0.68** | 0.61 |
| 0.10 | 0.57 | **0.56** | 0.55 | 0.53 | 0.55 |
| 0.12 | 0.67 | 0.46 | 0.42 | **0.53** | 0.41 |

Four quick takeaways: (1) planner cliff is reproduced — Stage A planner mean is within 2 pp of the M2 single-seed at every cell; (2) every hybrid variant flattens the cliff; (3) no single variant dominates — each row is best in a different perturbation regime; (4) **C2 at xy = 0.04 m hits 0.83**, the only M3 cell that exceeds the M2 reference at the same level.

The full `25/75` percentile bands and the per-seed nominal cell (where Stage B at seed 2 hits **0.89**) are in the final report (`info/M3/FinalReport/FinalReport.tex`, Tables I and II).

---

## Configuration reference

Every constant the experiments touch lives in `src/config.py`. The flags below are the ones a reviewer most often wants to flip while replicating; defaults are the values used to produce the numbers in this README.

| Flag | Default | Purpose |
|---|---|---|
| `PERTURB_XY_RANGE` | 0.08 | Training-time max planar perturbation per axis (m) |
| `PERTURB_Z_RANGE` | 0.01 | Training-time max z perturbation (m) |
| `PERTURB_YAW_RANGE` | 0.20 | Training-time max yaw perturbation (rad) |
| `PERTURB_PITCH_RANGE` | 0.0 | M3 negative result; non-zero re-enables full-orientation perturbation |
| `PERTURB_ROLL_RANGE` | 0.0 | Same; gated off in the configuration that produced the report |
| `PERTURB_LEVELS` | `[0, 0.02, …, 0.12]` | Eval grid (m). Other axes scale proportionally to xy. |
| `RESIDUAL_MAX_POS` | 0.15 | `ρ` — per-joint residual cap (rad) |
| `RESIDUAL_USE_GATE` | False | Stage B flag: True enables the confidence-gated 8-dim actor |
| `RESIDUAL_GATE_PENALTY` | 0.05 | `λ_g` — per-step penalty on gate magnitude |
| `RESIDUAL_GATE_INIT_LOGIT` | 1.5 | Warm-starts gate at `sigmoid(1.5) ≈ 0.82` |
| `GRASP_GATE_MODE` | `"heuristic"` | Stage C flag: `"learned_filter"` enables the classifier on top of heuristic |
| `GRASP_GATE_THRESHOLD` | 0.5 | Classifier decision threshold |
| `GRASP_GATE_DATASET_PATH` | `results/m3/grasp_dataset.jsonl` | JSON-lines log of every grasp attempt |
| `GRASP_GATE_MODEL_PATH` | `results/m3/grasp_gate.pt` | Trained classifier checkpoint |
| `CURRICULUM_RAMP_EPISODES` | 0 | M3 negative result; non-zero re-enables linear curriculum |
| `PPO_TOTAL_TIMESTEPS` | 1,000,000 | Training budget per seed |
| `PPO_NUM_COLLECTOR_WORKERS` | 8 | PyBullet parallel env workers |
| `EVAL_EPISODES_PER_LEVEL` | 100 | Episodes per (method × level) eval cell |

---

## Where every result file lives

| Step | Output |
|---|---|
| Step 1 (M1) | `results/m1/{trajectory_log.json, plots/}` |
| Step 2 (M2 train) | `results/m2/{models/, models_rl_only/, tb_logs/, tb_logs_rl_only/}` |
| Step 2 (M2 eval) | `results/m2/eval_results.json`, `results/m2/plots/` |
| Step 3 (Stage A) | `results/m3/seed_NNN/`, `results/m3/eval_aggregate.json`, `results/m3/grasp_dataset.jsonl` |
| Step 4 (Stage B) | `results/m3_gated/seed_NNN/`, `results/m3_gated/eval_aggregate.json` |
| Step 5 (Stage C1) | `results/m3_lgg/seed_NNN/eval_results.json` (re-eval; checkpoints copied from Stage A), `results/m3_lgg/eval_aggregate.json` |
| Step 5 (Stage C2) | `results/m3_full/seed_NNN/eval_results.json` (re-eval from Stage B), `results/m3_full/eval_aggregate.json` |
| Step 5 (classifier) | `results/m3/grasp_gate.pt` |
| Step 6 (plot) | `results/m3/plots/m3_comparison.png` |

Everything under `results/` is gitignored — these directories are produced by the commands above, not committed.

---

## Acknowledgments

CS558ROB, Spring 2026 — Purdue University. Built on `pybullet`, `torchrl`, `tensordict`, `gymnasium`, and a Franka Emika Panda URDF.
