from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from src import config


def load_trajectory_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Trajectory log not found: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError("Trajectory log is empty.")

    return records


def records_to_arrays(records):
    sample_idx = np.arange(len(records), dtype=int)

    sim_step = np.array([r["sim_step"] for r in records], dtype=int)
    waypoint_index = np.array([r["waypoint_index"] for r in records], dtype=int)
    phase = np.array([r["phase"] for r in records])

    q_target = np.array([r["q_target"] for r in records], dtype=float)
    q_actual = np.array([r["q_actual"] for r in records], dtype=float)
    q_error = np.array([r["q_error"] for r in records], dtype=float)

    ee_pos = np.array([r["ee_pos"] for r in records], dtype=float)
    ee_euler = np.array([r["ee_euler"] for r in records], dtype=float)

    max_abs_joint_error = np.max(np.abs(q_error), axis=1)

    return {
        "sample_idx": sample_idx,
        "sim_step": sim_step,
        "waypoint_index": waypoint_index,
        "phase": phase,
        "q_target": q_target,
        "q_actual": q_actual,
        "q_error": q_error,
        "ee_pos": ee_pos,
        "ee_euler": ee_euler,
        "max_abs_joint_error": max_abs_joint_error,
    }


def summarize(arr):
    phase = arr["phase"]
    sample_idx = arr["sample_idx"]
    waypoint_index = arr["waypoint_index"]
    max_abs_joint_error = arr["max_abs_joint_error"]
    ee_pos = arr["ee_pos"]

    phases = sorted(set(phase.tolist()))
    lines = []
    lines.append("Trajectory Execution Summary")
    lines.append("=" * 32)
    lines.append(f"Total logged samples: {len(sample_idx)}")
    lines.append(f"Global max abs joint error: {float(np.max(max_abs_joint_error)):.6f} rad")
    lines.append(f"Global mean max abs joint error: {float(np.mean(max_abs_joint_error)):.6f} rad")
    lines.append("")

    for ph in phases:
        mask = phase == ph
        phase_samples = sample_idx[mask]
        phase_waypoints = waypoint_index[mask]
        phase_err = max_abs_joint_error[mask]
        phase_ee = ee_pos[mask]

        unique_waypoints = np.unique(phase_waypoints)

        lines.append(f"Phase: {ph}")
        lines.append(f"  Logged samples: {len(phase_samples)}")
        lines.append(f"  Unique waypoints visited: {len(unique_waypoints)}")
        lines.append(f"  Max abs joint error: {float(np.max(phase_err)):.6f} rad")
        lines.append(f"  Mean max abs joint error: {float(np.mean(phase_err)):.6f} rad")
        lines.append(
            f"  EE x-range: [{float(np.min(phase_ee[:, 0])):.4f}, {float(np.max(phase_ee[:, 0])):.4f}] m"
        )
        lines.append(
            f"  EE y-range: [{float(np.min(phase_ee[:, 1])):.4f}, {float(np.max(phase_ee[:, 1])):.4f}] m"
        )
        lines.append(
            f"  EE z-range: [{float(np.min(phase_ee[:, 2])):.4f}, {float(np.max(phase_ee[:, 2])):.4f}] m"
        )
        lines.append("")

    return "\n".join(lines)


def save_summary(summary_text, summary_path):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def get_phase_boundaries(phase_array):
    boundaries = []
    for i in range(1, len(phase_array)):
        if phase_array[i] != phase_array[i - 1]:
            boundaries.append(i)
    return boundaries


def plot_joint_error_vs_sample(arr, out_dir):
    sample_idx = arr["sample_idx"]
    max_abs_joint_error = arr["max_abs_joint_error"]
    phase = arr["phase"]

    plt.figure(figsize=(9, 5))

    for ph in sorted(set(phase.tolist())):
        mask = phase == ph
        plt.plot(sample_idx[mask], max_abs_joint_error[mask], label=ph)

    for b in get_phase_boundaries(phase):
        plt.axvline(sample_idx[b], linestyle="--", alpha=0.5)

    plt.xlabel("Global sample index")
    plt.ylabel("Max abs joint error (rad)")
    plt.title("Max Joint Tracking Error vs Global Sample Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "max_joint_error_vs_sample.png", dpi=200)
    plt.close()


def plot_ee_position_vs_sample(arr, out_dir):
    sample_idx = arr["sample_idx"]
    ee_pos = arr["ee_pos"]
    phase = arr["phase"]

    plt.figure(figsize=(9, 5))
    plt.plot(sample_idx, ee_pos[:, 0], label="ee_x")
    plt.plot(sample_idx, ee_pos[:, 1], label="ee_y")
    plt.plot(sample_idx, ee_pos[:, 2], label="ee_z")

    for b in get_phase_boundaries(phase):
        plt.axvline(sample_idx[b], linestyle="--", alpha=0.5)

    plt.xlabel("Global sample index")
    plt.ylabel("Position (m)")
    plt.title("End-Effector Position vs Global Sample Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "ee_position_vs_sample.png", dpi=200)
    plt.close()


def plot_ee_xy_path(arr, out_dir):
    ee_pos = arr["ee_pos"]
    phase = arr["phase"]

    plt.figure(figsize=(6, 6))
    for ph in sorted(set(phase.tolist())):
        mask = phase == ph
        plt.plot(ee_pos[mask, 0], ee_pos[mask, 1], label=ph)

    plt.xlabel("EE x (m)")
    plt.ylabel("EE y (m)")
    plt.title("End-Effector XY Path")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "ee_xy_path.png", dpi=200)
    plt.close()


def plot_ee_xz_path(arr, out_dir):
    ee_pos = arr["ee_pos"]
    phase = arr["phase"]

    plt.figure(figsize=(6, 6))
    for ph in sorted(set(phase.tolist())):
        mask = phase == ph
        plt.plot(ee_pos[mask, 0], ee_pos[mask, 2], label=ph)

    plt.xlabel("EE x (m)")
    plt.ylabel("EE z (m)")
    plt.title("End-Effector XZ Path")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "ee_xz_path.png", dpi=200)
    plt.close()


def main():
    records = load_trajectory_log(config.TRAJECTORY_LOG_PATH)
    arr = records_to_arrays(records)

    out_dir = Path(config.PLOT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_text = summarize(arr)
    save_summary(summary_text, config.TRAJECTORY_SUMMARY_PATH)

    plot_joint_error_vs_sample(arr, out_dir)
    plot_ee_position_vs_sample(arr, out_dir)
    plot_ee_xy_path(arr, out_dir)
    plot_ee_xz_path(arr, out_dir)

    print(summary_text)
    print(f"Saved summary to: {config.TRAJECTORY_SUMMARY_PATH}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()