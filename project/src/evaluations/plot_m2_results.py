import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.utils.run_log import open_tee_log, close_tee_log


def load_results(path=None):
    path = path or config.M2_EVAL_RESULTS_PATH
    with open(path) as f:
        return json.load(f)


def plot_success_rate(results, save_dir):
    levels = sorted(results.keys())
    methods = set()
    for level_data in results.values():
        methods.update(level_data.keys())
    methods = sorted(methods)

    perturb_values = []
    for lk in levels:
        val = float(lk.split("_")[1])
        perturb_values.append(val)

    x = np.arange(len(perturb_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    method_labels = {
        "planner_only": "Planner Only",
        "hybrid": "Planner + Residual RL",
        "rl_only": "RL Only",
    }
    colors = {
        "planner_only": "#4878CF",
        "hybrid": "#6ACC65",
        "rl_only": "#D65F5F",
    }

    for i, method in enumerate(methods):
        success_rates = []
        for lk in levels:
            if method in results[lk]:
                success_rates.append(results[lk][method]["success_rate"] * 100)
            else:
                success_rates.append(0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, success_rates, width,
            label=method_labels.get(method, method),
            color=colors.get(method, None),
        )

    ax.set_xlabel("Perturbation Range (m)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Grasp Success Rate vs Perturbation Level")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in perturb_values])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "success_rate_vs_perturbation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_mean_reward(results, save_dir):
    levels = sorted(results.keys())
    methods = set()
    for level_data in results.values():
        methods.update(level_data.keys())
    methods = sorted(methods)

    perturb_values = [float(lk.split("_")[1]) for lk in levels]

    method_labels = {
        "planner_only": "Planner Only",
        "hybrid": "Planner + Residual RL",
        "rl_only": "RL Only",
    }
    colors = {
        "planner_only": "#4878CF",
        "hybrid": "#6ACC65",
        "rl_only": "#D65F5F",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        rewards = []
        stds = []
        for lk in levels:
            if method in results[lk]:
                rewards.append(results[lk][method]["mean_reward"])
                stds.append(results[lk][method]["std_reward"])
            else:
                rewards.append(0)
                stds.append(0)

        rewards = np.array(rewards)
        stds = np.array(stds)

        ax.plot(
            perturb_values, rewards,
            marker="o",
            label=method_labels.get(method, method),
            color=colors.get(method, None),
        )
        ax.fill_between(
            perturb_values,
            rewards - stds,
            rewards + stds,
            alpha=0.15,
            color=colors.get(method, None),
        )

    ax.set_xlabel("Perturbation Range (m)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Reward vs Perturbation Level")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "mean_reward_vs_perturbation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_episode_length(results, save_dir):
    levels = sorted(results.keys())
    methods = set()
    for level_data in results.values():
        methods.update(level_data.keys())
    methods = sorted(methods)

    perturb_values = [float(lk.split("_")[1]) for lk in levels]

    method_labels = {
        "planner_only": "Planner Only",
        "hybrid": "Planner + Residual RL",
        "rl_only": "RL Only",
    }
    colors = {
        "planner_only": "#4878CF",
        "hybrid": "#6ACC65",
        "rl_only": "#D65F5F",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        steps = []
        for lk in levels:
            if method in results[lk]:
                steps.append(results[lk][method]["mean_steps"])
            else:
                steps.append(0)

        ax.plot(
            perturb_values, steps,
            marker="s",
            label=method_labels.get(method, method),
            color=colors.get(method, None),
        )

    ax.set_xlabel("Perturbation Range (m)")
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("Episode Length vs Perturbation Level")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "episode_length_vs_perturbation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def generate_all_plots(results_path=None, save_dir=None, log_file_path=None):
    save_dir = save_dir or config.M2_PLOT_DIR
    log_file_path = log_file_path or os.path.join(config.M2_RESULTS_DIR, "plots_log.txt")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config.M2_RESULTS_DIR, exist_ok=True)

    original_stdout, log_file = open_tee_log(log_file_path, banner="Plot generation run")
    try:
        resolved_results_path = results_path or config.M2_EVAL_RESULTS_PATH
        print(f"Log file: {log_file_path}")
        print(f"Reading evaluation results from {resolved_results_path}")
        print(f"Saving plots to {save_dir}")

        results = load_results(results_path)

        plots = [
            ("success_rate", plot_success_rate),
            ("mean_reward", plot_mean_reward),
            ("episode_length", plot_episode_length),
        ]
        overall_start = time.time()
        for name, fn in plots:
            t0 = time.time()
            fn(results, save_dir)
            print(f"  [{name}] rendered in {time.time() - t0:.2f}s")

        print(f"\nAll M2 plots saved to {save_dir} (total: {time.time() - overall_start:.2f}s)")
    finally:
        close_tee_log(original_stdout, log_file)


if __name__ == "__main__":
    generate_all_plots()
