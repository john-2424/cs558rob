import gym
import torch
import torch.optim as optim
import numpy as np
import csv
import os
import argparse

from policies import DiscretePolicy
from utils import (
    compute_episode_return,
    compute_reward_to_go,
    normalize_returns,
    set_seed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def run_episode(env, policy, render=False):
    obs = env.reset()
    done = False

    log_probs = []
    rewards = []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, done, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)

        obs = next_obs

        if render:
            env.render()

    return log_probs, rewards


def get_weights_for_episode(rewards, gamma, mode):
    """
    mode:
        'q11' -> same full-episode discounted return for every timestep
        'q12' -> reward-to-go
        'q13' -> normalized reward-to-go
    """
    if mode == "q11":
        G = compute_episode_return(rewards, gamma)
        return [G] * len(rewards)

    elif mode == "q12":
        return compute_reward_to_go(rewards, gamma)

    elif mode == "q13":
        rtg = compute_reward_to_go(rewards, gamma)
        return normalize_returns(rtg)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def save_rewards_to_csv(rewards, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "avg_reward"])
        for i, r in enumerate(rewards, start=1):
            writer.writerow([i, r])


def train_cartpole(
    mode="q11",
    num_iterations=200,
    episodes_per_iteration=500,
    gamma=0.99,
    lr=1e-2,
    seed=42,
    save_path=None,
):
    set_seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = DiscretePolicy(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    avg_rewards_per_iter = []

    for iteration in range(num_iterations):
        all_episode_rewards = []
        losses = []

        for _ in range(episodes_per_iteration):
            log_probs, rewards = run_episode(env, policy)

            weights = get_weights_for_episode(rewards, gamma, mode)
            episode_reward = sum(rewards)
            all_episode_rewards.append(episode_reward)

            loss = 0.0
            for log_prob, w in zip(log_probs, weights):
                loss += -log_prob * w

            losses.append(loss)

        optimizer.zero_grad()
        loss_mean = torch.stack(losses).mean()
        loss_mean.backward()
        optimizer.step()

        avg_reward = np.mean(all_episode_rewards)
        avg_rewards_per_iter.append(avg_reward)

        print(
            f"[{mode}] Iteration {iteration+1}/{num_iterations}, "
            f"Avg Reward: {avg_reward:.2f}"
        )

    env.close()

    if save_path is not None:
        save_rewards_to_csv(avg_rewards_per_iter, save_path)

    return avg_rewards_per_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="q11", choices=["q11", "q12", "q13"])
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--episodes_per_iteration", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rewards = train_cartpole(
        mode=args.mode,
        num_iterations=args.num_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        gamma=args.gamma,
        seed=args.seed,
        save_path=args.save_path,
    )

    print("Training finished.")