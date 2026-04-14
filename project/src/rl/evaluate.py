import json
import os

import numpy as np
import torch
from tensordict import TensorDict

from src import config
from src.rl.gym_env import PandaGraspEnv
from src.rl.train import load_trained_actor


def run_episode(env, actor=None, mode="hybrid", deterministic=True):
    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    residual_norms = []

    while True:
        if actor is not None and mode != "planner_only":
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])

            if deterministic:
                with torch.no_grad():
                    td = actor(td)
                action = td["action"].squeeze(0).numpy()
            else:
                with torch.no_grad():
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


def evaluate_method(mode, actor, perturb_level, num_episodes, seed_base=0):
    results = []

    for ep in range(num_episodes):
        env = PandaGraspEnv(
            gui=False,
            mode=mode,
            perturb_xy_range=perturb_level,
        )
        try:
            ep_result = run_episode(
                env, actor=actor, mode=mode, deterministic=True,
            )
            results.append(ep_result)
        finally:
            env.close()

    successes = [r["success"] for r in results]
    rewards = [r["total_reward"] for r in results]
    steps = [r["step_count"] for r in results]
    residuals = [r["mean_residual"] for r in results]

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "mean_residual": float(np.mean(residuals)),
        "num_episodes": num_episodes,
        "individual_results": results,
    }


def run_evaluation(model_path=None, rl_only_model_path=None):
    model_path = model_path or os.path.join(config.M2_MODEL_DIR, "final_model.pt")
    os.makedirs(config.M2_RESULTS_DIR, exist_ok=True)

    # Load trained hybrid actor
    hybrid_actor = None
    if os.path.exists(model_path):
        hybrid_actor = load_trained_actor(model_path)
        print(f"Loaded hybrid model from {model_path}")
    else:
        print(f"No hybrid model found at {model_path}, skipping hybrid evaluation.")

    # Load RL-only actor
    rl_only_actor = None
    if rl_only_model_path and os.path.exists(rl_only_model_path):
        rl_only_actor = load_trained_actor(rl_only_model_path)
        print(f"Loaded RL-only model from {rl_only_model_path}")

    all_results = {}
    num_episodes = config.EVAL_EPISODES_PER_LEVEL

    for perturb_level in config.PERTURB_LEVELS:
        level_key = f"perturb_{perturb_level:.3f}"
        all_results[level_key] = {}

        print(f"\n=== Perturbation level: {perturb_level:.3f} m ===")

        # Planner-only baseline
        print(f"  Evaluating planner_only ({num_episodes} episodes)...")
        all_results[level_key]["planner_only"] = evaluate_method(
            mode="planner_only",
            actor=None,
            perturb_level=perturb_level,
            num_episodes=num_episodes,
        )
        sr = all_results[level_key]["planner_only"]["success_rate"]
        print(f"    success_rate={sr:.2%}")

        # Hybrid (planner + residual RL)
        if hybrid_actor is not None:
            print(f"  Evaluating hybrid ({num_episodes} episodes)...")
            all_results[level_key]["hybrid"] = evaluate_method(
                mode="hybrid",
                actor=hybrid_actor,
                perturb_level=perturb_level,
                num_episodes=num_episodes,
            )
            sr = all_results[level_key]["hybrid"]["success_rate"]
            print(f"    success_rate={sr:.2%}")

        # RL-only
        if rl_only_actor is not None:
            print(f"  Evaluating rl_only ({num_episodes} episodes)...")
            all_results[level_key]["rl_only"] = evaluate_method(
                mode="rl_only",
                actor=rl_only_actor,
                perturb_level=perturb_level,
                num_episodes=num_episodes,
            )
            sr = all_results[level_key]["rl_only"]["success_rate"]
            print(f"    success_rate={sr:.2%}")

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

    return summary


if __name__ == "__main__":
    run_evaluation()
