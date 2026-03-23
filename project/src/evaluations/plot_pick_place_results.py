import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_PATH = Path("results/m1/trajectory_log.json")
PLOT_DIR = Path("results/m1/plots/demo/pick_place")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_log(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    data = load_log(LOG_PATH)

    # Support either a plain list or dict-wrapped entries
    if isinstance(data, dict):
        entries = data.get("entries", [])
    else:
        entries = data

    if not entries:
        raise RuntimeError("No trajectory entries found in trajectory_log.json")

    ee = []
    sim_steps = []
    phases = []
    q_error_max = []

    for e in entries:
        ee_pos = e.get("ee_pos", None)
        if ee_pos is not None and len(ee_pos) == 3:
            ee.append(ee_pos)

        sim_steps.append(e.get("sim_step", len(sim_steps)))
        phases.append(e.get("phase", "unknown"))

        q_error = e.get("q_error", None)
        if q_error is not None:
            q_error_max.append(float(np.max(np.abs(q_error))))
        else:
            q_error_max.append(np.nan)

    ee = np.asarray(ee, dtype=float)
    samples = np.arange(len(ee))

    # --------------------------------------------------
    # Figure 4A: End-effector XZ path
    # --------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(ee[:, 0], ee[:, 2])
    plt.xlabel("End-effector x (m)")
    plt.ylabel("End-effector z (m)")
    plt.title("End-effector XZ Path")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ee_xz_path.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 4B: End-effector position vs sample
    # --------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(samples, ee[:, 0], label="x")
    plt.plot(samples, ee[:, 1], label="y")
    plt.plot(samples, ee[:, 2], label="z")
    plt.xlabel("Sample index")
    plt.ylabel("Position (m)")
    plt.title("End-effector Position vs Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ee_position_vs_sample.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 5: Max joint tracking error vs sample
    # --------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(q_error_max)), q_error_max)
    plt.xlabel("Sample index")
    plt.ylabel("Max abs joint error (rad)")
    plt.title("Tracking Error vs Sample")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "tracking_error_vs_sample.png", dpi=300)
    plt.close()

    print(f"Saved plots to: {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()