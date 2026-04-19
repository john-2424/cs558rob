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


def _perturb_tick_labels(perturb_values):
    """Build x-tick labels showing XY, Z, and yaw for each perturbation level."""
    max_xy = config.PERTURB_XY_RANGE or 1.0
    labels = []
    for xy in perturb_values:
        scale = xy / max_xy if max_xy > 0 else 0.0
        z = float(getattr(config, "PERTURB_Z_RANGE", 0.0)) * scale
        yaw = config.PERTURB_YAW_RANGE * scale
        labels.append(f"xy={xy:.2f}\nz={z:.3f}\nyaw={yaw:.2f}")
    return labels


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
        ci_errors_lo = []
        ci_errors_hi = []
        for lk in levels:
            if method in results[lk]:
                sr = results[lk][method]["success_rate"] * 100
                lo = results[lk][method].get("success_ci_lo", sr / 100) * 100
                hi = results[lk][method].get("success_ci_hi", sr / 100) * 100
                success_rates.append(sr)
                # Clip to non-negative: Wilson CI at p=1.0 gives upper<1.0
                # (e.g. 0.983 at n=100), so hi-sr goes slightly negative and
                # matplotlib's yerr rejects it. Same symmetry applies at p=0.
                ci_errors_lo.append(max(0.0, sr - lo))
                ci_errors_hi.append(max(0.0, hi - sr))
            else:
                success_rates.append(0)
                ci_errors_lo.append(0)
                ci_errors_hi.append(0)

        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, success_rates, width,
            yerr=[ci_errors_lo, ci_errors_hi],
            capsize=3, ecolor="gray",
            label=method_labels.get(method, method),
            color=colors.get(method, None),
        )

    ax.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Grasp Success Rate vs Perturbation Level")
    ax.set_xticks(x)
    ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
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

    ax.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Mean Reward vs Perturbation Level")
    ax.set_xticks(perturb_values)
    ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
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

    ax.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("Episode Length vs Perturbation Level")
    ax.set_xticks(perturb_values)
    ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "episode_length_vs_perturbation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_phase_breakdown(results, save_dir):
    """Stacked bar: fraction of episodes reaching each max phase, per method x level."""
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
    phase_colors = {
        "pre_grasp": "#D65F5F",
        "grasp_descend": "#F5A623",
        "lift": "#6ACC65",
    }
    phase_names = ["pre_grasp", "grasp_descend", "lift"]

    num_methods = len(methods)
    num_levels = len(perturb_values)
    x = np.arange(num_levels)
    width = 0.25

    method_short = {
        "planner_only": "Plan",
        "hybrid": "Hyb",
        "rl_only": "RL",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, method in enumerate(methods):
        bottoms = np.zeros(num_levels)
        for phase in phase_names:
            fracs = []
            for lk in levels:
                if method in results[lk] and "phase_distribution" in results[lk][method]:
                    pd = results[lk][method]["phase_distribution"]
                    total = sum(pd.values())
                    fracs.append(pd.get(phase, 0) / total * 100 if total > 0 else 0)
                else:
                    fracs.append(0)
            fracs = np.array(fracs)
            offset = (i - num_methods / 2 + 0.5) * width
            ax.bar(
                x + offset, fracs, width, bottom=bottoms,
                color=phase_colors[phase],
                label=phase.replace("_", " ").title() if i == 0 else "",
                edgecolor="white", linewidth=0.5,
            )
            bottoms += fracs

    ax.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)", labelpad=10)
    ax.set_ylabel("Episodes (%)")
    # Bar order per perturbation group
    bar_order = " | ".join(
        f"{method_short.get(m, m[:3])}={method_labels.get(m, m)}" for m in methods
    )
    ax.set_title(f"Max Phase Reached by Episode\nBar order (L\u2192R): {bar_order}", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "phase_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_residual_magnitude(results, save_dir):
    """Line plot: mean residual magnitude vs perturbation, per method."""
    levels = sorted(results.keys())
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

    for method in sorted({m for lvl in results.values() for m in lvl.keys()}):
        residuals = []
        for lk in levels:
            if method in results[lk]:
                residuals.append(results[lk][method].get("mean_residual", 0.0))
            else:
                residuals.append(0)

        ax.plot(
            perturb_values, residuals,
            marker="D", linewidth=2,
            label=method_labels.get(method, method),
            color=colors.get(method, None),
        )

    # Budget ceiling for hybrid (position-offset residual per joint)
    try:
        from src import config as _cfg
        budget = float(_cfg.RESIDUAL_MAX_POS)
        ax.axhline(
            budget, color="#6ACC65", linestyle="--", alpha=0.5,
            label=f"Hybrid budget ({budget:.2f} rad)",
        )
    except Exception:
        pass

    ax.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax.set_ylabel("Mean |residual| per joint (hybrid: rad, rl_only: rad/s)")
    ax.set_title("Residual Correction Magnitude vs Perturbation")
    ax.set_xticks(perturb_values)
    ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "residual_magnitude.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_grasp_analysis(results, save_dir):
    """Grouped bars: grasp attempted vs attached rate per method x level."""
    levels = sorted(results.keys())
    methods = sorted({m for lvl in results.values() for m in lvl.keys()})
    perturb_values = [float(lk.split("_")[1]) for lk in levels]

    method_labels = {
        "planner_only": "Planner Only",
        "hybrid": "Planner + Residual RL",
        "rl_only": "RL Only",
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        attempted = []
        attached = []
        lifted = []
        for lk in levels:
            if method in results[lk]:
                attempted.append(results[lk][method].get("grasp_attempted_rate", 0) * 100)
                attached.append(results[lk][method].get("grasp_attached_rate", 0) * 100)
                lifted.append(results[lk][method].get("cube_lifted_rate", 0) * 100)
            else:
                attempted.append(0)
                attached.append(0)
                lifted.append(0)

        x = np.arange(len(perturb_values))
        w = 0.25
        ax.bar(x - w, attempted, w, label="Attempted", color="#F5A623", alpha=0.85)
        ax.bar(x, attached, w, label="Attached", color="#6ACC65", alpha=0.85)
        ax.bar(x + w, lifted, w, label="Lifted", color="#4878CF", alpha=0.85)

        ax.set_title(method_labels.get(method, method))
        ax.set_xlabel("Perturbation Level", labelpad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=6)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Rate (%)")
    fig.suptitle("Grasp Pipeline: Attempted → Attached → Lifted", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "grasp_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_diagnostic_metrics(results, save_dir):
    """Two-panel line plot: mean forced WP advances and final EE-cube distance."""
    levels = sorted(results.keys())
    methods = sorted({m for lvl in results.values() for m in lvl.keys()})
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        forced = []
        ee_cube = []
        for lk in levels:
            if method in results[lk]:
                forced.append(results[lk][method].get("mean_forced_wp_advances", 0.0))
                ee_cube.append(results[lk][method].get("mean_ee_cube_dist", 0.0))
            else:
                forced.append(0)
                ee_cube.append(0)

        label = method_labels.get(method, method)
        c = colors.get(method, None)
        ax1.plot(perturb_values, forced, marker="o", label=label, color=c)
        ax2.plot(perturb_values, ee_cube, marker="s", label=label, color=c)

    ax1.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax1.set_ylabel("Mean Forced WP Advances / Episode")
    ax1.set_title("Forced Waypoint Advances")
    ax1.set_xticks(perturb_values)
    ax1.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Perturbation Level (XY m / Z m / Yaw rad)")
    ax2.set_ylabel("Mean Final EE-Cube Distance (m)")
    ax2.set_title("Final EE-to-Cube Distance")
    ax2.set_xticks(perturb_values)
    ax2.set_xticklabels(_perturb_tick_labels(perturb_values), fontsize=7)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "diagnostic_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_summary_table(results, save_dir):
    """Render a clean summary table as an image for slides."""
    levels = sorted(results.keys())
    methods = sorted({m for lvl in results.values() for m in lvl.keys()})
    perturb_values_raw = [float(lk.split("_")[1]) for lk in levels]
    max_xy = config.PERTURB_XY_RANGE or 1.0
    perturb_values = []
    for xy in perturb_values_raw:
        scale = xy / max_xy if max_xy > 0 else 0.0
        z = float(getattr(config, "PERTURB_Z_RANGE", 0.0)) * scale
        yaw = config.PERTURB_YAW_RANGE * scale
        perturb_values.append(f"xy={xy:.2f} z={z:.3f}\nyaw={yaw:.2f}rad")

    method_labels = {
        "planner_only": "Planner Only",
        "hybrid": "PD + Residual RL",
        "rl_only": "RL Only",
    }

    # Build table data: success rate per cell
    cell_text = []
    cell_colors = []
    for method in methods:
        row = []
        row_colors = []
        for lk in levels:
            if method in results[lk]:
                sr = results[lk][method]["success_rate"]
                mr = results[lk][method].get("mean_reward", 0)
                row.append(f"{sr:.0%}\n(r={mr:+.0f})")
                # Color gradient: green for high success, red for low
                g = sr
                row_colors.append((1 - 0.6 * g, 0.5 + 0.5 * g, 0.5 - 0.2 * g, 0.3))
            else:
                row.append("--")
                row_colors.append((0.9, 0.9, 0.9, 0.3))
        cell_text.append(row)
        cell_colors.append(row_colors)

    row_labels = [method_labels.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(max(14, len(levels) * 2.0), 1.5 + len(methods) * 1.0))
    ax.axis("off")
    ax.set_title("Evaluation Summary: Success Rate (Mean Reward)", fontsize=13, pad=20)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=perturb_values,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Apply cell colors
    for i, method in enumerate(methods):
        for j in range(len(levels)):
            table[i + 1, j].set_facecolor(cell_colors[i][j])

    plt.tight_layout()
    path = os.path.join(save_dir, "summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
            ("phase_breakdown", plot_phase_breakdown),
            ("residual_magnitude", plot_residual_magnitude),
            ("grasp_analysis", plot_grasp_analysis),
            ("diagnostic_metrics", plot_diagnostic_metrics),
            ("summary_table", plot_summary_table),
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
