"""Load a trained checkpoint and run a few episodes with per-step tracing.

Writes one CSV (per-step state) and one JSON (summary + grasp-check debug)
per episode under results/m2/traces/<mode>/ep_NNN.{csv,json}.

Usage:
    python -m src.rl.trace_rollout --mode both --num-episodes 5
    python -m src.rl.trace_rollout --mode rl_only --num-episodes 3 --deterministic
"""
import argparse
import os

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src import config
from src.rl.gym_env import PandaGraspEnv
from src.rl.train import load_trained_actor


def _rollout(mode, model_path, num_episodes, trace_dir, deterministic, seed):
    env = PandaGraspEnv(
        gui=False,
        mode=mode,
        trace=True,
        trace_dir=trace_dir,
        verbose_episodes=True,
    )

    actor = None
    if mode != "planner_only" and model_path and os.path.exists(model_path):
        actor = load_trained_actor(model_path)
        print(f"[{mode}] loaded checkpoint: {model_path}")
    else:
        print(f"[{mode}] no checkpoint available; rolling out with zero action.")

    exploration = (
        ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    )

    summaries = []
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                if actor is not None:
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    td = TensorDict({"observation": obs_t}, batch_size=[1])
                    with torch.no_grad(), set_exploration_type(exploration):
                        td = actor(td)
                    action = td["action"].squeeze(0).numpy()
                else:
                    action = np.zeros(7, dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(action)
            summaries.append({
                "success": bool(info.get("success", False)),
                "max_phase": int(info.get("ep_max_phase", -1)),
                "grasp_attempted": bool(info.get("ep_grasp_attempted", False)),
                "grasp_attached": bool(info.get("ep_grasp_attached", False)),
                "cube_dz": float(info.get("ep_cube_dz", 0.0)),
                "ee_cube_dist": float(info.get("ep_ee_cube_dist", 0.0)),
            })
    finally:
        env.close()

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both",
                        choices=["hybrid", "rl_only", "planner_only", "both"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use tanh(loc) instead of sampling. Matches eval.py RANDOM note: "
                             "deterministic can produce constant non-zero residuals.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hybrid-model", default=None)
    parser.add_argument("--rl-only-model", default=None)
    parser.add_argument("--trace-dir", default=None)
    args = parser.parse_args()

    trace_dir = args.trace_dir or config.RL_TRACE_OUTPUT_DIR
    os.makedirs(trace_dir, exist_ok=True)

    modes = ["hybrid", "rl_only"] if args.mode == "both" else [args.mode]

    hybrid_path = args.hybrid_model or os.path.join(config.M2_MODEL_DIR, "final_model.pt")
    rl_only_path = args.rl_only_model or os.path.join(
        config.M2_RESULTS_DIR, "models_rl_only", "final_model.pt"
    )

    all_summaries = {}
    for m in modes:
        if m == "hybrid":
            model_path = hybrid_path
        elif m == "rl_only":
            model_path = rl_only_path
        else:
            model_path = None
        print(f"\n=== Tracing {m} ({args.num_episodes} episodes, "
              f"{'deterministic' if args.deterministic else 'stochastic'}) ===")
        all_summaries[m] = _rollout(
            mode=m,
            model_path=model_path,
            num_episodes=args.num_episodes,
            trace_dir=trace_dir,
            deterministic=args.deterministic,
            seed=args.seed,
        )

    print("\n=== Summary ===")
    for m, eps in all_summaries.items():
        n = len(eps)
        succ = sum(1 for e in eps if e["success"])
        tried = sum(1 for e in eps if e["grasp_attempted"])
        attached = sum(1 for e in eps if e["grasp_attached"])
        print(f"  {m:<11} | success={succ}/{n} | grasp_tried={tried}/{n} | attached={attached}/{n}")
    print(f"\nTraces written to: {trace_dir}")


if __name__ == "__main__":
    main()
