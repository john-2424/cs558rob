"""Headline comparison plot for the M3 2x2 ablation.

Loads four aggregate JSONs (Stage A baseline, Stage B confidence-gate,
Stage C-1 learned grasp gate, Stage C-2 both stacked) plus the planner
baseline, and overlays them as line curves on one figure with
across-seed percentile error bands. Optionally overlays the M2 single-
seed hybrid as a dashed reference.

Output: results/m3/plots/m3_comparison.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from src import config


def _load(path):
    with open(path) as f:
        return json.load(f)


def _series(agg, method, key):
    """Return (xs, mean, lo, hi) for one method across perturbation levels."""
    levels = sorted(agg.keys())
    xs = [float(lk.split("_")[1]) for lk in levels]
    mean, lo, hi = [], [], []
    for lk in levels:
        cell = agg.get(lk, {}).get(method, {})
        m = cell.get(key, 0.0) * 100.0
        mean.append(m)
        l = cell.get("success_ci_lo", cell.get(key, 0.0)) * 100.0
        h = cell.get("success_ci_hi", cell.get(key, 0.0)) * 100.0
        lo.append(max(0.0, m - l))
        hi.append(max(0.0, h - m))
    return np.asarray(xs), np.asarray(mean), np.asarray(lo), np.asarray(hi)


def main():
    p = argparse.ArgumentParser(prog="python -m src.evaluations.plot_m3_comparison",
                                description="M3 2x2 ablation comparison plot.")
    p.add_argument("--stage-a", default="results/m3/eval_aggregate.json",
                   help="Stage A (clean baseline) aggregate JSON.")
    p.add_argument("--stage-b", default="results/m3_gated/eval_aggregate.json",
                   help="Stage B (confidence gate) aggregate JSON.")
    p.add_argument("--stage-c1", default="results/m3_lgg/eval_aggregate.json",
                   help="Stage C-1 (learned grasp gate) aggregate JSON.")
    p.add_argument("--stage-c2", default="results/m3_full/eval_aggregate.json",
                   help="Stage C-2 (gate + lgg) aggregate JSON.")
    p.add_argument("--m2", default="results/m2/eval_results.json",
                   help="M2 single-seed reference JSON (optional dashed overlay).")
    p.add_argument("--out", default="results/m3/plots/m3_comparison.png",
                   help="Output PNG path.")
    p.add_argument("--no-m2", action="store_true",
                   help="Skip M2 reference overlay.")
    args = p.parse_args()

    A = _load(args.stage_a)
    B = _load(args.stage_b)
    C1 = _load(args.stage_c1)
    C2 = _load(args.stage_c2)
    M2 = None
    if not args.no_m2 and os.path.exists(args.m2):
        M2 = _load(args.m2)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Planner baseline (from Stage A — same planner across all stages)
    xs, m, lo, hi = _series(A, "planner_only", "success_rate")
    ax.errorbar(xs, m, yerr=[lo, hi], color="#4878CF", marker="o", capsize=3,
                linewidth=1.8, markersize=6, label="Planner only")

    # Four hybrid curves
    series_specs = [
        ("hybrid", A, "Stage A: hybrid (baseline)", "#6ACC65", "s"),
        ("hybrid", B, "Stage B: hybrid + confidence gate", "#956CB4", "^"),
        ("hybrid", C1, "Stage C1: hybrid + learned grasp gate", "#D5BB67", "D"),
        ("hybrid", C2, "Stage C2: hybrid + both", "#EE854A", "v"),
    ]
    for method, agg, label, color, marker in series_specs:
        xs, m, lo, hi = _series(agg, method, "success_rate")
        ax.errorbar(xs, m, yerr=[lo, hi], color=color, marker=marker,
                    capsize=3, linewidth=1.8, markersize=6, label=label)

    # M2 reference (single-seed, dashed)
    if M2 is not None:
        xs_m2 = sorted(float(lk.split("_")[1]) for lk in M2.keys())
        m2_levels = sorted(M2.keys())
        m2_h = np.asarray([M2[lk]["hybrid"]["success_rate"] * 100.0 for lk in m2_levels])
        ax.plot(xs_m2, m2_h, color="#888888", linestyle="--", linewidth=1.2,
                marker="x", markersize=5, label="M2 hybrid (single seed, ref)")

    ax.set_xlabel("Perturbation level (XY m, with Z & yaw scaled proportionally)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("M3 ablation: hybrid variants vs perturbation level\n"
                 "(3 seeds, 100 episodes/cell, error bars = 25/75 percentile across seeds)")
    ax.set_ylim(0, 105)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
    ax.grid(axis="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
