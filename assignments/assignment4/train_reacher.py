import os
import csv
import gym
import pybullet
import modified_gym_env
import torch
import torch.optim as optim
import numpy as np
import argparse

from policies import GaussianPolicy
from utils import compute_reward_to_go, normalize_returns, set_seed


def save_rewards_to_csv(rewards, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "avg_reward"])
        for i, r in enumerate(rewards, start=1):
            writer.writerow([i, r])


def safe_reset(env):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return obs


def safe_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        next_obs, reward, done, info = out
    elif len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise ValueError(f"Unexpected env.step output length: {len(out)}")
    return next_obs, reward, done, info


def run_episode(env, policy, action_low, action_high, render=False):
    obs = safe_reset(env)
    done = False

    log_probs = []
    rewards = []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = policy(obs_t)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_np = action.detach().cpu().numpy()
        action_np = np.clip(action_np, action_low, action_high)

        next_obs, reward, done, _ = safe_step(env, action_np)

        log_probs.append(log_prob)
        rewards.append(reward)
        obs = next_obs

        if render:
            env.render()

    return log_probs, rewards


def train_reacher(
    num_iterations=50,
    episodes_per_iteration=20,
    gamma=0.9,
    lr=5e-4,
    seed=42,
    rand_init=False,
    save_path=None,
):
    set_seed(seed)

    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=rand_init)
    
    obs = safe_reset(env)
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy = GaussianPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    avg_rewards_per_iter = []

    for iteration in range(num_iterations):
        all_episode_rewards = []
        batch_log_probs = []
        batch_returns = []

        for _ in range(episodes_per_iteration):
            log_probs, rewards = run_episode(env, policy, action_low, action_high)

            returns = compute_reward_to_go(rewards, gamma)
            episode_reward = sum(rewards)
            all_episode_rewards.append(episode_reward)

            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)

        batch_returns = normalize_returns(batch_returns)

        loss_terms = []
        for log_prob, G in zip(batch_log_probs, batch_returns):
            loss_terms.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(loss_terms).mean()
        loss.backward()
        optimizer.step()

        avg_reward = np.mean(all_episode_rewards)
        avg_rewards_per_iter.append(avg_reward)

        print(
            f"[rand_init={rand_init}] Iteration {iteration+1}/{num_iterations}, "
            f"Episodes/Iter: {episodes_per_iteration}, Avg Reward: {avg_reward:.3f}"
        )

    env.close()

    if save_path is not None:
        save_rewards_to_csv(avg_rewards_per_iter, save_path)

        model_path = save_path.replace(".csv", ".pt")
        torch.save(policy.state_dict(), model_path)

    return avg_rewards_per_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--episodes_per_iteration", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand_init", action="store_true")
    parser.add_argument("--save_path", type=str, default="results/reacher_rewards.csv")

    args = parser.parse_args()

    rewards = train_reacher(
        num_iterations=args.num_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
        rand_init=args.rand_init,
        save_path=args.save_path,
    )
    print("Training finished.")