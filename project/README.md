# CS558 IRL Project Milestone 1  
**Classical Planning and Control Foundation for Planner-Guided Robotic Manipulation**

## Overview

This project implements the Milestone 1 classical component for a robotic manipulation pipeline using the Franka Emika Panda robot in PyBullet. The current system sets up a tabletop simulation environment, loads the robot and scene objects, extracts robot and environment state information, performs classical motion planning with RRT*, tracks motion using a custom joint-space PD controller, and executes a nominal pick-and-place demo.

The final Milestone 1 demo includes:

- PyBullet simulation environment setup
- Panda robot loading
- Table and cube scene initialization
- Robot and environment state extraction
- Joint-space planning with collision checking
- Custom PD waypoint tracking
- Cartesian task decomposition for pick-and-place
- Grasp validation using contact and proximity checks
- Nominal pick-and-place execution with logging and plots

---

## Conda Environment Setup

### Create a new Conda environment and install from `requirements.txt`

```bash
conda create -n cs558_m1 python=3.10.20 -y
conda activate cs558_m1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## How to Run the Nominal Pick-and-Place Demo

From the project root, run:

```bash
python -m src.main
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

---

## Output Files and Results

After a successful run, results are written to:

```text
results/m1/
```

Important outputs include:

- `results/m1/trajectory_log.json`  
  Stores the logged trajectory execution data, including phase labels, waypoint index, simulation step, target joints, actual joints, joint error, and end-effector pose.

- `results/m1/trajectory_summary.txt`  
  Summary text file for the run.

---

## How to Generate Evaluation Plots

If the trajectory log already exists, you can generate plots using the plotting script.

From the project root:

```bash
python -m src.evaluations.plot_pick_place_results
```

This generates evaluation figures in the results directory, typically under:

- `results/m1/plots/demo/pick_place/`  
  Stores generated plots from the logged data.

The main generated figures are:

- End-effector trajectory in the XZ plane
- End-effector position versus sample index
- Maximum absolute joint tracking error versus sample index
