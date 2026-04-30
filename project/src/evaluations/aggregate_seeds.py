"""Aggregate per-seed eval JSONs into one summary file.

Given N per-seed eval JSONs (one per seed × one per training run), produce
a single aggregate JSON whose top-level shape matches a single-seed eval
JSON so the existing plotters (plot_m2_results.py) read it unchanged.
Each (level, method) cell carries:

  - mean of per-seed point estimates (success_rate, mean_reward, ...)
  - 25/75 percentile across seeds, exposed as success_ci_lo / _hi so
    existing plot code interprets the spread as error bars
  - per_seed: list of point estimates kept verbatim for plotting later
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def _percentile_ci(values: List[float], lo_p: float = 25.0, hi_p: float = 75.0):
    """Return (lo, hi) percentile bounds. With small N, percentiles are the
    honest summary — bootstrap on N≤5 gives a misleading sense of precision."""
    if not values:
        return 0.0, 0.0
    return float(np.percentile(values, lo_p)), float(np.percentile(values, hi_p))


def _aggregate_cell(per_seed_cells: List[dict]) -> dict:
    """Aggregate one (level, method) cell across seeds."""
    if not per_seed_cells:
        return {}

    succ_rates = [c["success_rate"] for c in per_seed_cells]
    mean_rewards = [c.get("mean_reward", 0.0) for c in per_seed_cells]
    mean_residuals = [c.get("mean_residual", 0.0) for c in per_seed_cells]
    mean_steps = [c.get("mean_steps", 0.0) for c in per_seed_cells]
    num_eps = [c.get("num_episodes", 0) for c in per_seed_cells]

    sr_mean = float(np.mean(succ_rates))
    sr_lo, sr_hi = _percentile_ci(succ_rates)
    sr_min = float(np.min(succ_rates))
    sr_max = float(np.max(succ_rates))
    sr_std = float(np.std(succ_rates, ddof=1)) if len(succ_rates) > 1 else 0.0

    out = {
        "success_rate": sr_mean,
        "success_ci_lo": sr_lo,
        "success_ci_hi": sr_hi,
        "success_min": sr_min,
        "success_max": sr_max,
        "success_std": sr_std,
        "mean_reward": float(np.mean(mean_rewards)),
        "mean_residual": float(np.mean(mean_residuals)),
        "mean_steps": float(np.mean(mean_steps)),
        "num_episodes": int(np.sum(num_eps)),
        "n_seeds": len(per_seed_cells),
        "per_seed_success_rate": succ_rates,
        "per_seed_mean_reward": mean_rewards,
    }

    # Forward any aggregate diagnostic fields (lifted_rate, attached_rate, …)
    # by averaging across seeds. Fixes nothing if a key is missing on some
    # seeds but absent on others — those keys just don't get aggregated.
    diag_keys = ("grasp_attempted_rate", "grasp_attached_rate",
                 "cube_fallen_rate", "cube_lifted_rate",
                 "mean_cube_dz", "mean_ee_cube_dist",
                 "mean_forced_wp_advances")
    for k in diag_keys:
        vals = [c[k] for c in per_seed_cells if k in c]
        if vals:
            out[k] = float(np.mean(vals))

    # Phase distribution: sum across seeds (counts, not means).
    pd_seeds = [c.get("phase_distribution") for c in per_seed_cells
                if isinstance(c.get("phase_distribution"), dict)]
    if pd_seeds:
        merged = {"pre_grasp": 0, "grasp_descend": 0, "lift": 0}
        for pd in pd_seeds:
            for ph in merged:
                merged[ph] += int(pd.get(ph, 0))
        out["phase_distribution"] = merged

    return out


def aggregate(eval_json_paths: Dict[int, str]) -> dict:
    """Load each per-seed eval JSON; return one aggregated summary."""
    per_seed = {}
    for seed, path in eval_json_paths.items():
        with open(path) as f:
            per_seed[seed] = json.load(f)

    levels = set()
    methods_per_level: Dict[str, set] = {}
    for seed_data in per_seed.values():
        for lk, methods in seed_data.items():
            levels.add(lk)
            methods_per_level.setdefault(lk, set()).update(methods.keys())

    summary = {}
    for lk in sorted(levels):
        summary[lk] = {}
        for method in sorted(methods_per_level[lk]):
            cells = []
            for seed_data in per_seed.values():
                if lk in seed_data and method in seed_data[lk]:
                    cells.append(seed_data[lk][method])
            summary[lk][method] = _aggregate_cell(cells)

    return summary


def aggregate_from_index(index_path: str, out_path: str = None) -> str:
    """Read a multi_seed_index.json, aggregate, write the summary, return path."""
    with open(index_path) as f:
        index = json.load(f)
    eval_paths = {int(k): v for k, v in index["eval_json_paths"].items()}

    summary = aggregate(eval_paths)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(index_path), "eval_aggregate.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated {len(eval_paths)} seed(s) -> {out_path}")
    return out_path


def _build_parser():
    p = argparse.ArgumentParser(
        prog="python -m src.evaluations.aggregate_seeds",
        description="Aggregate per-seed eval JSONs into one summary.",
    )
    p.add_argument("--index", required=True,
                   help="Path to multi_seed_index.json from a multi-seed run.")
    p.add_argument("--out", default=None,
                   help="Where to write the aggregate JSON "
                        "(default: <index dir>/eval_aggregate.json).")
    return p


def main():
    args = _build_parser().parse_args()
    aggregate_from_index(args.index, args.out)


if __name__ == "__main__":
    main()
