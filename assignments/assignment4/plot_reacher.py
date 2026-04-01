import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_single_curve(csv_path, title, out_path):
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["iteration"], df["avg_reward"])
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_multiple_curves(csv_paths, labels, title, out_path):
    plt.figure()

    for csv_path, label in zip(csv_paths, labels):
        df = pd.read_csv(csv_path)
        plt.plot(df["iteration"], df["avg_reward"], label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Final showcased Q2 model
    plot_single_curve(
        "results/reacher_randinit_ep50_seed42_iter150.csv",
        "Reacher Q2 Final Training Curve (50 episodes/iteration, 150 iterations)",
        "results/figures/reacher_q2_final_ep50_iter150.png",
    )

    # Episode count comparison at 75 iterations
    plot_multiple_curves(
        [
            "results/reacher_randinit_ep20_seed42.csv",
            "results/reacher_randinit_ep30_seed42.csv",
            "results/reacher_randinit_ep50_seed42.csv",
        ],
        [
            "20 episodes/iter",
            "30 episodes/iter",
            "50 episodes/iter",
        ],
        "Reacher Q2 Episode Count Comparison",
        "results/figures/reacher_q2_episode_compare.png",
    )

    # Seed comparison for the smallest successful setting
    plot_multiple_curves(
        [
            "results/reacher_randinit_ep30_seed42.csv",
            "results/reacher_randinit_ep30_seed0.csv",
            "results/reacher_randinit_ep30_seed1.csv",
        ],
        [
            "seed 42",
            "seed 0",
            "seed 1",
        ],
        "Reacher Q2 Seed Comparison (30 episodes/iteration)",
        "results/figures/reacher_q2_seed_compare_ep30.png",
    )