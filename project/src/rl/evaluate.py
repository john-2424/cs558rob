import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src import config
from src.rl.gym_env import PandaGraspEnv
from src.rl.train import load_trained_actor
from src.utils.run_log import open_tee_log, close_tee_log


def run_episode(env, actor=None, mode="hybrid", deterministic=True):
    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    residual_norms = []

    # In deterministic eval we want the TanhNormal's mode (mean), not a sample.
    exploration_type = ExplorationType.MODE if deterministic else ExplorationType.RANDOM

    while True:
        if actor is not None and mode != "planner_only":
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])

            with torch.no_grad(), set_exploration_type(exploration_type):
                td = actor(td)
            action = td["action"].squeeze(0).numpy()
        else:
            action = np.zeros(7, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if "r_residual" in info:
            residual_norms.append(abs(info["r_residual"]))

        if terminated or truncated:
            break

    return {
        "success": info.get("success", False),
        "total_reward": total_reward,
        "step_count": step_count,
        "mean_residual": float(np.mean(residual_norms)) if residual_norms else 0.0,
    }


def _run_episodes(mode, actor, perturb_level, num_episodes):
    """Serially run a batch of episodes in one PyBullet client."""
    env = PandaGraspEnv(
        gui=False,
        mode=mode,
        perturb_xy_range=perturb_level,
    )
    results = []
    try:
        for _ in range(num_episodes):
            results.append(
                run_episode(env, actor=actor, mode=mode, deterministic=True)
            )
    finally:
        env.close()
    return results


def _eval_worker_entry(mode, model_path, perturb_level, num_episodes):
    """Top-level worker function -- picklable under spawn.

    Loads its own actor copy from disk (avoids shipping torch modules
    through the process pool boundary) and runs its share of episodes.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    actor = None
    if model_path and mode != "planner_only":
        actor = load_trained_actor(model_path)

    return _run_episodes(mode, actor, perturb_level, num_episodes)


def evaluate_method(mode, model_path, perturb_level, num_episodes, num_workers=None):
    """Evaluate one method-x-level over num_episodes, optionally in parallel."""
    num_workers = (
        int(num_workers) if num_workers is not None else int(config.EVAL_NUM_WORKERS)
    )
    num_workers = max(1, min(num_workers, num_episodes))

    if num_workers == 1:
        actor = None
        if model_path and mode != "planner_only":
            actor = load_trained_actor(model_path)
        results = _run_episodes(mode, actor, perturb_level, num_episodes)
    else:
        # Split num_episodes as evenly as possible across workers.
        base, rem = divmod(num_episodes, num_workers)
        chunks = [base + (1 if i < rem else 0) for i in range(num_workers)]
        chunks = [c for c in chunks if c > 0]

        results = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(
                    _eval_worker_entry, mode, model_path, perturb_level, c
                )
                for c in chunks
            ]
            for f in futures:
                results.extend(f.result())

    successes = [r["success"] for r in results]
    rewards = [r["total_reward"] for r in results]
    steps = [r["step_count"] for r in results]
    residuals = [r["mean_residual"] for r in results]

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
        "num_episodes": len(results),
        "individual_results": results,
    }


def _eval_one_method(label, mode, model_path, perturb_level, num_episodes, num_workers):
    """Time a single method-x-level and print a one-line summary."""
    print(f"  Evaluating {label} ({num_episodes} episodes, {num_workers} worker(s))...")
    t0 = time.time()
    result = evaluate_method(
        mode=mode,
        model_path=model_path,
        perturb_level=perturb_level,
        num_episodes=num_episodes,
        num_workers=num_workers,
    )
    elapsed = time.time() - t0
    print(
        f"    {label:<13} | succ={result['success_rate']:6.2%} | "
        f"reward={result['mean_reward']:+7.2f}±{result['std_reward']:5.2f} | "
        f"steps={result['mean_steps']:6.1f} | "
        f"residual={result['mean_residual']:.3f} | "
        f"time={elapsed:.1f}s"
    )
    return result


def run_evaluation(model_path=None, rl_only_model_path=None, log_file_path=None):
    model_path = model_path or os.path.join(config.M2_MODEL_DIR, "final_model.pt")
    log_file_path = log_file_path or os.path.join(config.M2_RESULTS_DIR, "eval_log.txt")
    os.makedirs(config.M2_RESULTS_DIR, exist_ok=True)

    original_stdout, log_file = open_tee_log(log_file_path, banner="Evaluation run")

    try:
        print(f"Log file: {log_file_path}")
        print(f"Results path: {config.M2_EVAL_RESULTS_PATH}")

        hybrid_available = os.path.exists(model_path)
        if hybrid_available:
            print(f"Found hybrid model at {model_path}")
        else:
            print(f"No hybrid model found at {model_path}, skipping hybrid evaluation.")
            model_path = None

        rl_only_available = bool(rl_only_model_path) and os.path.exists(rl_only_model_path)
        if rl_only_available:
            print(f"Found RL-only model at {rl_only_model_path}")

        all_results = {}
        num_episodes = config.EVAL_EPISODES_PER_LEVEL
        num_workers = config.EVAL_NUM_WORKERS

        print(f"Using {num_workers} parallel worker(s) per method-level")
        run_start = time.time()

        for perturb_level in config.PERTURB_LEVELS:
            level_key = f"perturb_{perturb_level:.3f}"
            all_results[level_key] = {}

            print(f"\n=== Perturbation level: {perturb_level:.3f} m ===")

            all_results[level_key]["planner_only"] = _eval_one_method(
                label="planner_only",
                mode="planner_only",
                model_path=None,
                perturb_level=perturb_level,
                num_episodes=num_episodes,
                num_workers=num_workers,
            )

            if hybrid_available:
                all_results[level_key]["hybrid"] = _eval_one_method(
                    label="hybrid",
                    mode="hybrid",
                    model_path=model_path,
                    perturb_level=perturb_level,
                    num_episodes=num_episodes,
                    num_workers=num_workers,
                )

            if rl_only_available:
                all_results[level_key]["rl_only"] = _eval_one_method(
                    label="rl_only",
                    mode="rl_only",
                    model_path=rl_only_model_path,
                    perturb_level=perturb_level,
                    num_episodes=num_episodes,
                    num_workers=num_workers,
                )

        total_elapsed = time.time() - run_start
        print(f"\nTotal evaluation time: {total_elapsed:.1f}s")

        # Strip individual results for the summary JSON (keep it compact)
        summary = {}
        for level_key, methods in all_results.items():
            summary[level_key] = {}
            for method, data in methods.items():
                summary[level_key][method] = {
                    k: v for k, v in data.items() if k != "individual_results"
                }

        results_path = config.M2_EVAL_RESULTS_PATH
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {results_path}")

        _print_summary_table(summary)

        return summary
    finally:
        close_tee_log(original_stdout, log_file)


def _print_summary_table(summary):
    """Print a compact end-of-run summary: success rate per method per level."""
    print("\n" + "=" * 60)
    print("Summary: success rate (%) by method x perturbation level")
    print("=" * 60)
    levels = sorted(summary.keys())
    methods = sorted({m for lvl in summary.values() for m in lvl.keys()})

    header = f"{'method':<15} | " + " | ".join(f"{lk:>15}" for lk in levels)
    print(header)
    print("-" * len(header))
    for method in methods:
        row = [f"{method:<15}"]
        for lk in levels:
            if method in summary[lk]:
                row.append(f"{summary[lk][method]['success_rate']:>14.2%}")
            else:
                row.append(f"{'--':>15}")
        print(" | ".join(row))
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
