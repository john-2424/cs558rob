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
from src.utils.run_log import open_tee_log, close_tee_log


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
          model_save_path=None, tb_log_dir=None, log_file_path=None):
    device = "cpu"
    total_timesteps = total_timesteps or config.PPO_TOTAL_TIMESTEPS

    # Suffix default output paths by mode so rl_only and hybrid runs stay isolated.
    if mode == "hybrid":
        default_model_dir = config.M2_MODEL_DIR
        default_tb_dir = config.M2_TB_LOG_DIR
        default_log_name = "train_log.txt"
    else:
        default_model_dir = os.path.join(config.M2_RESULTS_DIR, f"models_{mode}")
        default_tb_dir = os.path.join(config.M2_RESULTS_DIR, f"tb_logs_{mode}")
        default_log_name = f"train_log_{mode}.txt"

    model_save_path = model_save_path or default_model_dir
    tb_log_dir = tb_log_dir or default_tb_dir
    log_file_path = log_file_path or os.path.join(config.M2_RESULTS_DIR, default_log_name)

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    # Mirror stdout (all print() output) to a persistent log file.
    original_stdout, log_file = open_tee_log(log_file_path, banner="Training run")

    writer = SummaryWriter(log_dir=tb_log_dir)

    print(f"Log file: {log_file_path}")
    print(f"TensorBoard log dir: {tb_log_dir}")
    print(f"Model checkpoint dir: {model_save_path}")

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
    episode_lengths = []
    episode_successes = []
    ep_reward_acc = None  # per-env running episode-reward accumulator (lazy init)
    ep_length_acc = None  # per-env running episode-length accumulator (lazy init)
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

        # --- Episode tracking using TorchRL's nested "next" keys ---
        # Rewards and dones live under ("next", "reward") and ("next", "done").
        step_rewards_t = batch_data.get(("next", "reward"))
        done_t = batch_data.get(("next", "done"))

        mean_step_reward = float(step_rewards_t.mean().item()) if step_rewards_t is not None else 0.0
        num_dones = int(done_t.sum().item()) if done_t is not None else 0

        # Per-env accumulation so episode returns can span batches.
        batch_new_rewards = []
        batch_new_lengths = []
        if step_rewards_t is not None and done_t is not None:
            r = step_rewards_t.squeeze(-1) if step_rewards_t.shape[-1] == 1 else step_rewards_t
            d = done_t.squeeze(-1) if done_t.shape[-1] == 1 else done_t
            if r.dim() == 1:
                r = r.unsqueeze(0)
                d = d.unsqueeze(0)
            num_envs, T = r.shape[0], r.shape[1]
            if ep_reward_acc is None or len(ep_reward_acc) != num_envs:
                ep_reward_acc = [0.0] * num_envs
                ep_length_acc = [0] * num_envs
            r_list = r.tolist()
            d_list = d.bool().tolist()
            for w in range(num_envs):
                for t in range(T):
                    ep_reward_acc[w] += r_list[w][t]
                    ep_length_acc[w] += 1
                    if d_list[w][t]:
                        ep_r_done = ep_reward_acc[w]
                        ep_len_done = ep_length_acc[w]
                        episode_rewards.append(ep_r_done)
                        episode_lengths.append(ep_len_done)
                        episode_successes.append(
                            1 if ep_r_done >= config.EP_SUCCESS_REWARD_THRESHOLD else 0
                        )
                        batch_new_rewards.append(ep_r_done)
                        batch_new_lengths.append(ep_len_done)
                        ep_reward_acc[w] = 0.0
                        ep_length_acc[w] = 0

        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0

        recent_rewards = episode_rewards[-20:]
        recent_lengths = episode_lengths[-20:]
        recent_successes = episode_successes[-20:]

        mean_ep_reward = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
        std_ep_reward = float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0
        mean_ep_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
        succ_rate = float(np.mean(recent_successes)) if recent_successes else 0.0

        # Per-batch worker-divergence signal (std of episode rewards completed this batch)
        batch_ep_std = float(np.std(batch_new_rewards)) if len(batch_new_rewards) > 1 else 0.0

        writer.add_scalar("train/loss", mean_loss, total_frames)
        writer.add_scalar("train/fps", fps, total_frames)
        writer.add_scalar("train/mean_step_reward", mean_step_reward, total_frames)
        writer.add_scalar("train/episodes_completed", len(episode_rewards), total_frames)
        if recent_rewards:
            writer.add_scalar("train/mean_episode_reward", mean_ep_reward, total_frames)
            writer.add_scalar("train/std_episode_reward", std_ep_reward, total_frames)
            writer.add_scalar("train/mean_episode_length", mean_ep_length, total_frames)
            writer.add_scalar("train/success_rate", succ_rate, total_frames)
        if len(batch_new_rewards) > 1:
            writer.add_scalar("train/batch_episode_reward_std", batch_ep_std, total_frames)

        print(
            f"batch={batch_idx:4d} | "
            f"frames={total_frames:8d}/{total_timesteps} | "
            f"loss={mean_loss:+.3f} | "
            f"step_r={mean_step_reward:+.3f} | "
            f"ep_r={mean_ep_reward:+7.2f}±{std_ep_reward:5.2f} | "
            f"len={mean_ep_length:5.0f} | "
            f"succ={succ_rate:5.0%} | "
            f"eps={len(episode_rewards):5d} (+{num_dones}) | "
            f"fps={fps:.0f}"
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

    # Restore stdout and close the log file.
    close_tee_log(original_stdout, log_file)

    return final_path


def load_trained_actor(model_path, obs_dim=37, act_dim=7, device="cpu"):
    actor = build_actor(obs_dim, act_dim, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


if __name__ == "__main__":
    train()
