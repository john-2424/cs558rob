"""Multi-seed orchestration for hybrid + rl_only training and evaluation.

Loops over a list of seeds; for each seed, trains the requested modes,
evaluates the resulting checkpoints, and writes per-seed artefacts under
results/m3/seed_<N>/. Aggregation across seeds is the aggregate_seeds
module's job.

Designed to be re-startable: if a per-seed checkpoint already exists, the
training step is skipped (a one-line note is printed). The eval step is
likewise skipped if a per-seed eval JSON already exists. Delete the
relevant file to force a re-run.
"""

import argparse
import json
import os
import time
from typing import Iterable, Sequence

from src import config


def _seed_dir(seed: int, root: str) -> str:
    return os.path.join(root, f"seed_{seed:03d}")


def _hybrid_model_path(seed_dir: str) -> str:
    return os.path.join(seed_dir, "models_hybrid", "final_model.pt")


def _rl_only_model_path(seed_dir: str) -> str:
    return os.path.join(seed_dir, "models_rl_only", "final_model.pt")


def _eval_json_path(seed_dir: str) -> str:
    return os.path.join(seed_dir, "eval_results.json")


def train_one_seed(seed: int, modes: Sequence[str], seed_dir: str,
                   total_timesteps: int = None, num_workers: int = None,
                   force_retrain: bool = False) -> dict:
    """Train each requested mode for one seed; return paths to checkpoints."""
    from src.rl.train import train

    paths = {}
    for mode in modes:
        model_save_path = os.path.join(seed_dir, f"models_{mode}")
        tb_log_dir = os.path.join(seed_dir, f"tb_logs_{mode}")
        log_file_path = os.path.join(seed_dir, f"train_log_{mode}.txt")
        final_path = os.path.join(model_save_path, "final_model.pt")

        if os.path.exists(final_path) and not force_retrain:
            print(f"[seed {seed}] {mode}: checkpoint exists at {final_path}, skipping train")
            paths[mode] = final_path
            continue

        print(f"[seed {seed}] training {mode}: writing to {model_save_path}")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(tb_log_dir, exist_ok=True)

        final_path = train(
            mode=mode,
            total_timesteps=total_timesteps,
            model_save_path=model_save_path,
            tb_log_dir=tb_log_dir,
            log_file_path=log_file_path,
            num_workers=num_workers,
            seed=seed,
        )
        paths[mode] = final_path
    return paths


def evaluate_one_seed(seed: int, modes: Sequence[str], seed_dir: str,
                      num_workers: int = None,
                      force_reeval: bool = False) -> str:
    """Run the eval pipeline for one seed; return the eval JSON path.

    Writes config.M2_EVAL_RESULTS_PATH-style JSON into seed_dir/eval_results.json.
    """
    from src.rl.evaluate import run_evaluation

    eval_json = _eval_json_path(seed_dir)
    if os.path.exists(eval_json) and not force_reeval:
        print(f"[seed {seed}] eval JSON exists at {eval_json}, skipping eval")
        return eval_json

    hybrid_path = _hybrid_model_path(seed_dir) if "hybrid" in modes else None
    rl_only_path = _rl_only_model_path(seed_dir) if "rl_only" in modes else None

    # Patch the M2 results paths to land under seed_dir for this run, then
    # restore. run_evaluation reads config.M2_EVAL_RESULTS_PATH directly.
    original_eval_path = config.M2_EVAL_RESULTS_PATH
    original_results_dir = config.M2_RESULTS_DIR
    try:
        config.M2_EVAL_RESULTS_PATH = eval_json
        config.M2_RESULTS_DIR = seed_dir
        run_evaluation(
            model_path=hybrid_path,
            rl_only_model_path=rl_only_path,
            log_file_path=os.path.join(seed_dir, "eval_log.txt"),
            verbose_episodes=False,
            num_workers=num_workers,
        )
    finally:
        config.M2_EVAL_RESULTS_PATH = original_eval_path
        config.M2_RESULTS_DIR = original_results_dir

    return eval_json


def run_multi_seed(seeds: Iterable[int], modes: Sequence[str] = ("hybrid", "rl_only"),
                   root: str = "results/m3", total_timesteps: int = None,
                   num_workers: int = None, force_retrain: bool = False,
                   force_reeval: bool = False, train_only: bool = False,
                   eval_only: bool = False) -> dict:
    """Drive train + eval across seeds. Returns {seed: eval_json_path}."""
    os.makedirs(root, exist_ok=True)
    seed_eval_paths = {}
    t0 = time.time()
    for seed in seeds:
        seed_dir = _seed_dir(seed, root)
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n========== seed {seed} (dir={seed_dir}) ==========")

        if not eval_only:
            train_one_seed(
                seed=seed, modes=modes, seed_dir=seed_dir,
                total_timesteps=total_timesteps, num_workers=num_workers,
                force_retrain=force_retrain,
            )

        if not train_only:
            eval_json = evaluate_one_seed(
                seed=seed, modes=modes, seed_dir=seed_dir,
                num_workers=num_workers, force_reeval=force_reeval,
            )
            seed_eval_paths[seed] = eval_json

    elapsed = time.time() - t0
    print(f"\nMulti-seed run complete: {len(seed_eval_paths)} seed(s), "
          f"elapsed {elapsed/60:.1f} min")

    # Write a small index file listing seeds + their eval JSONs.
    index = {
        "seeds": list(seed_eval_paths.keys()),
        "eval_json_paths": seed_eval_paths,
        "modes": list(modes),
    }
    index_path = os.path.join(root, "multi_seed_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote index {index_path}")
    return seed_eval_paths


def _build_parser():
    p = argparse.ArgumentParser(prog="python -m src.rl.multi_seed",
                                description="Multi-seed train + eval orchestrator.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Random seeds to run (default: 0 1 2).")
    p.add_argument("--modes", nargs="+", default=["hybrid", "rl_only"],
                   choices=["hybrid", "rl_only"],
                   help="Which policy modes to train per seed (default: hybrid rl_only).")
    p.add_argument("--root", default="results/m3",
                   help="Root directory for per-seed artefacts (default: results/m3).")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Override PPO_TOTAL_TIMESTEPS (e.g. 200000 for a smoke test).")
    p.add_argument("--workers", type=int, default=None,
                   help="Collector / eval worker count override.")
    p.add_argument("--force-retrain", action="store_true",
                   help="Train every seed even if final_model.pt already exists.")
    p.add_argument("--force-reeval", action="store_true",
                   help="Re-run eval even if eval_results.json already exists.")
    p.add_argument("--train-only", action="store_true",
                   help="Train per seed, skip eval pass.")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; eval existing per-seed checkpoints only.")
    return p


def main():
    args = _build_parser().parse_args()
    if args.train_only and args.eval_only:
        raise SystemExit("--train-only and --eval-only are mutually exclusive")
    run_multi_seed(
        seeds=args.seeds, modes=tuple(args.modes), root=args.root,
        total_timesteps=args.total_timesteps, num_workers=args.workers,
        force_retrain=args.force_retrain, force_reeval=args.force_reeval,
        train_only=args.train_only, eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
