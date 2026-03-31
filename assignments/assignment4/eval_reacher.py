import time
import gym
import pybullet as p
import modified_gym_env
import torch
import numpy as np
import argparse

from policies import GaussianPolicy


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


def load_policy(model_path, obs_dim, act_dim):
    policy = GaussianPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    policy.eval()
    return policy


def get_mean_action(policy, obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dist = policy(obs_t)
        action = dist.mean
    return action.cpu().numpy()


def set_camera():
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=0,
        cameraPitch=-85,
        cameraTargetPosition=[0, 0, 0]
    )


def evaluate_policy(
    model_path,
    rand_init=True,
    num_episodes=3,
    render=False,
):
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=rand_init)

    if render:
        env.render(mode="human")

    obs = safe_reset(env)

    if render:
        set_camera()

    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    policy = load_policy(model_path, obs_dim, act_dim)

    episode_rewards = []

    for ep in range(num_episodes):
        obs = safe_reset(env)

        if render:
            set_camera()

        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = get_mean_action(policy, obs)
            action = np.clip(action, action_low, action_high)

            next_obs, reward, done, _ = safe_step(env, action)
            total_reward += reward
            obs = next_obs
            step_count += 1

            if render:
                time.sleep(1.0 / 60.0)

        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}/{num_episodes} | Reward: {total_reward:.3f} | Steps: {step_count}")

    env.close()

    avg_reward = float(np.mean(episode_rewards))
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.3f}")

    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rand_init", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model_path,
        rand_init=args.rand_init,
        num_episodes=args.num_episodes,
        render=args.render,
    )