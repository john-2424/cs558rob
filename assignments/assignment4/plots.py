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
    plot_single_curve(
        "results/cartpole_q11.csv",
        "CartPole Q1.1: Vanilla REINFORCE",
        "results/figures/cartpole_q11.png",
    )

    plot_single_curve(
        "results/cartpole_q12.csv",
        "CartPole Q1.2: Reward-to-Go",
        "results/figures/cartpole_q12.png",
    )

    plot_single_curve(
        "results/cartpole_q13.csv",
        "CartPole Q1.3: Normalized Reward-to-Go",
        "results/figures/cartpole_q13.png",
    )

    plot_multiple_curves(
        [
            "results/cartpole_q14_ep100.csv",
            "results/cartpole_q14_ep300.csv",
            "results/cartpole_q14_ep1000.csv",
        ],
        ["100 episodes/iter", "300 episodes/iter", "1000 episodes/iter"],
        "CartPole Q1.4: Episode Count Comparison",
        "results/figures/cartpole_q14_compare.png",
    )