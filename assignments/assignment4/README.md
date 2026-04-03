# CS558-ROB Assignment 4 — Policy Gradients

This repository contains the implementation for **Assignment 4: Policy Gradients**.

It includes:
- **Question 1 (Q1): CartPole-v1** with three REINFORCE variants for a **discrete** action space.
- **Question 2 (Q2): ReacherPyBulletEnv-v1** with a Gaussian policy for a **continuous** action space.

The code was developed and tested in a Conda environment and Python 3.8.20

## Environment Setup

Create and activate the Conda environment:

```bash
conda create -n cs558_ass4 python=3.8 -y
conda activate cs558_ass4
```
Please follow the instructions below to build the environment:

0. Add pybullet-gym/ and /modified-gym-env/ to the root of the codebase
1. cd pybullet-gym/
2. pip install -e .
4. Update modified-gym-env/setup.py - install_requires parameter's value list with ['gym==0.21.0', 'pybullet>=3.2.1,<3.3']
3. cd /modified-gym-env/
4. python -m pip install "pip<24.1"
5. pip install -e .
6. pip install ../requirements.txt

## 1. Question 1 — CartPole-v1

Q1 uses a **categorical policy** for the discrete CartPole action space.

Implemented variants:
- **Q1.1**: Vanilla REINFORCE with full-episode discounted return
- **Q1.2**: REINFORCE with reward-to-go
- **Q1.3**: Reward-to-go with mean subtraction / normalization
- **Q1.4**: Q1.3 with different episodes-per-iteration settings

### 1.1 Full assignment runs for Q1.1–Q1.3

These match the assignment specification of **200 iterations**, **500 episodes per iteration**, and **gamma = 0.99**.

```bash
python train_cartpole.py --mode q11 --num_iterations 200 --episodes_per_iteration 500 --gamma 0.99 --save_path results/cartpole_q11.csv
python train_cartpole.py --mode q12 --num_iterations 200 --episodes_per_iteration 500 --gamma 0.99 --save_path results/cartpole_q12.csv
python train_cartpole.py --mode q13 --num_iterations 200 --episodes_per_iteration 500 --gamma 0.99 --save_path results/cartpole_q13.csv
```

### 1.2 Q1.4 Episode-count comparison

These runs use the **Q1.3** update with different numbers of episodes per policy update.

```bash
python train_cartpole.py --mode q13 --num_iterations 200 --episodes_per_iteration 100 --gamma 0.99 --save_path results/cartpole_q14_ep100.csv
python train_cartpole.py --mode q13 --num_iterations 200 --episodes_per_iteration 300 --gamma 0.99 --save_path results/cartpole_q14_ep300.csv
python train_cartpole.py --mode q13 --num_iterations 200 --episodes_per_iteration 1000 --gamma 0.99 --save_path results/cartpole_q14_ep1000.csv
```

### 1.3 Expected outputs for Q1

After training, the `results/` folder should contain CSV files with columns:
- `iteration`
- `avg_reward`

### 1.4 Plotting commands for Q1

Run the following commands after the Q1 training CSV files have been generated:

```bash
python plots.py
```

This will generate the Q1 plots in the results/figures/ directory

---

## 2. Question 2 — 2-Link Reacher

Q2 uses a **Gaussian policy** for the continuous-action custom 2-link reacher environment.

### 2.1 Random-initialized training runs

These are the main Q2 experiments.

#### Episode-count comparison

```bash
python train_reacher.py --num_iterations 75 --episodes_per_iteration 20 --gamma 0.9 --lr 5e-4 --seed 42 --rand_init --save_path results/reacher_randinit_ep20_seed42.csv
python train_reacher.py --num_iterations 75 --episodes_per_iteration 30 --gamma 0.9 --lr 5e-4 --seed 42 --rand_init --save_path results/reacher_randinit_ep30_seed42.csv
python train_reacher.py --num_iterations 75 --episodes_per_iteration 50 --gamma 0.9 --lr 5e-4 --seed 42 --rand_init --save_path results/reacher_randinit_ep50_seed42.csv
```

#### Seed-check runs

```bash
python train_reacher.py --num_iterations 75 --episodes_per_iteration 30 --gamma 0.9 --lr 5e-4 --seed 0 --rand_init --save_path results/reacher_randinit_ep30_seed0.csv
python train_reacher.py --num_iterations 75 --episodes_per_iteration 30 --gamma 0.9 --lr 5e-4 --seed 1 --rand_init --save_path results/reacher_randinit_ep30_seed1.csv
```

#### Best final showcased model

This was the strongest final Q2 training configuration used in our experiments.

```bash
python train_reacher.py --num_iterations 150 --episodes_per_iteration 50 --gamma 0.9 --lr 5e-4 --seed 42 --rand_init --save_path results/reacher_randinit_ep50_seed42_iter150.csv
```

---

### 2.2 Evaluating the trained Q2 model

Use the evaluation script to run the trained policy with deterministic mean actions.

#### 2.2.1 Single rendered rollout

```bash
python eval_reacher.py --model_path results/reacher_randinit_ep50_seed42_iter150.pt --rand_init --num_episodes 1 --render
```

#### 2.2.2 Multi-episode evaluation

```bash
python eval_reacher.py --model_path results/reacher_randinit_ep50_seed42_iter150.pt --rand_init --num_episodes 5 --render
```

### 2.3 Plotting Results

Generate the Q2 figures with:

```bash
python plot_reacher.py
```

This will generate the Q2 plots in the results/figures/ directory