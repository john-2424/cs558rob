import os
import time
from functools import partial

import numpy as np
import torch
from torch import nn, optim

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper, TransformedEnv, StepCounter
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from torch.utils.tensorboard import SummaryWriter

from src import config
from src.rl.gym_env import PandaGraspEnv


def make_env(gui=False, mode="hybrid", perturb_xy_range=None):
    gym_env = PandaGraspEnv(
        gui=gui,
        mode=mode,
        perturb_xy_range=perturb_xy_range,
    )
    env = GymWrapper(gym_env, device="cpu")
    env = TransformedEnv(env, StepCounter(max_steps=config.RL_MAX_EPISODE_STEPS))
    return env


def _make_env_worker(mode, perturb_xy_range):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    return make_env(gui=False, mode=mode, perturb_xy_range=perturb_xy_range)


def build_actor(obs_dim, act_dim, device="cpu"):
    actor_net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 2 * act_dim),
        NormalParamExtractor(),
    )

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )

    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )
    return actor.to(device)


def build_critic(obs_dim, device="cpu"):
    critic_net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    )

    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"],
    )
    return critic.to(device)


def train(mode="hybrid", perturb_xy_range=None, total_timesteps=None,
          model_save_path=None, tb_log_dir=None):
    device = "cpu"
    total_timesteps = total_timesteps or config.PPO_TOTAL_TIMESTEPS
    model_save_path = model_save_path or config.M2_MODEL_DIR
    tb_log_dir = tb_log_dir or config.M2_TB_LOG_DIR

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_log_dir)

    obs_dim = 37
    act_dim = 7

    # Build actor-critic
    actor = build_actor(obs_dim, act_dim, device)
    critic = build_critic(obs_dim, device)

    # GAE advantage estimator
    advantage_module = GAE(
        gamma=config.PPO_GAMMA,
        lmbda=config.PPO_GAE_LAMBDA,
        value_network=critic,
    )

    # PPO loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.PPO_CLIP_EPSILON,
        entropy_coeff=config.PPO_ENT_COEFF,
        critic_coeff=0.5,
        loss_critic_type="l2",
    )

    # Optimizer
    optimizer = optim.Adam(loss_module.parameters(), lr=config.PPO_LR)

    # Collector (no main-process env -- workers each build their own PyBullet client)
    num_workers = max(1, int(config.PPO_NUM_COLLECTOR_WORKERS))
    env_fn = partial(_make_env_worker, mode, perturb_xy_range)

    if num_workers > 1:
        print(f"Using MultiSyncDataCollector with {num_workers} workers")
        collector = MultiSyncDataCollector(
            create_env_fn=[env_fn] * num_workers,
            policy=actor,
            frames_per_batch=config.PPO_FRAMES_PER_BATCH,
            total_frames=total_timesteps,
            device=device,
            storing_device=device,
        )
    else:
        collector = SyncDataCollector(
            create_env_fn=env_fn,
            policy=actor,
            frames_per_batch=config.PPO_FRAMES_PER_BATCH,
            total_frames=total_timesteps,
            device=device,
            storing_device=device,
        )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config.PPO_FRAMES_PER_BATCH),
        batch_size=config.PPO_MINI_BATCH_SIZE,
    )

    # Training loop
    total_frames = 0
    batch_idx = 0
    episode_rewards = []
    episode_successes = []
    start_time = time.time()

    print(f"Starting PPO training: mode={mode}, total_timesteps={total_timesteps}")
    print(f"Frames per batch: {config.PPO_FRAMES_PER_BATCH}")
    print(f"Mini-batch size: {config.PPO_MINI_BATCH_SIZE}")
    print(f"Epochs per batch: {config.PPO_EPOCHS}")
    print("Waiting for first batch from collector workers (this may take 30-120s on first run)...")
    iter_start = time.time()

    for batch_data in collector:
        if batch_idx == 0:
            print(f"First batch received after {time.time() - iter_start:.1f}s")
        total_frames += batch_data.numel()
        batch_idx += 1

        # Compute advantage
        with torch.no_grad():
            advantage_module(batch_data)

        # Normalize advantage
        adv = batch_data["advantage"]
        batch_data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Load batch into replay buffer
        replay_buffer.extend(batch_data.reshape(-1))

        # PPO update epochs
        num_mini_batches = max(1, config.PPO_FRAMES_PER_BATCH // config.PPO_MINI_BATCH_SIZE)
        epoch_losses = []
        for _ in range(config.PPO_EPOCHS):
            for _ in range(num_mini_batches):
                minibatch = replay_buffer.sample()
                loss_td = loss_module(minibatch)

                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()

                epoch_losses.append(loss.item())

        # Logging
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Extract episode-level info from batch
        done_mask = batch_data["done"].squeeze(-1) if batch_data["done"].dim() > 1 else batch_data["done"]
        if done_mask.any():
            done_rewards = batch_data["episode_reward"][done_mask]
            if done_rewards.numel() > 0:
                for r in done_rewards.flatten().tolist():
                    episode_rewards.append(r)

        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0

        writer.add_scalar("train/loss", mean_loss, total_frames)
        writer.add_scalar("train/fps", fps, total_frames)

        if episode_rewards:
            recent_rewards = episode_rewards[-20:]
            mean_reward = np.mean(recent_rewards)
            writer.add_scalar("train/mean_episode_reward", mean_reward, total_frames)

        print(
            f"batch={batch_idx:4d} | "
            f"frames={total_frames:8d}/{total_timesteps} | "
            f"loss={mean_loss:.4f} | "
            f"fps={fps:.0f} | "
            f"episodes={len(episode_rewards)}"
        )

        # Periodic checkpoint
        if batch_idx % 20 == 0:
            checkpoint_path = os.path.join(model_save_path, f"checkpoint_{total_frames}.pt")
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_frames": total_frames,
            }, checkpoint_path)

    # Save final model
    final_path = os.path.join(model_save_path, "final_model.pt")
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "total_frames": total_frames,
    }, final_path)
    print(f"Training complete. Model saved to {final_path}")

    collector.shutdown()
    writer.close()

    return final_path


def load_trained_actor(model_path, obs_dim=37, act_dim=7, device="cpu"):
    actor = build_actor(obs_dim, act_dim, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


if __name__ == "__main__":
    train()
