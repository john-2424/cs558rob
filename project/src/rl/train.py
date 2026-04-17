import os
import random
import shutil
import time
from functools import partial

import numpy as np
import torch
from torch import nn, optim

from tensordict.nn import TensorDictModule

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


def make_env(gui=False, mode="hybrid", perturb_xy_range=None, curriculum=False):
    gym_env = PandaGraspEnv(
        gui=gui,
        mode=mode,
        perturb_xy_range=perturb_xy_range,
        curriculum=curriculum,
    )
    env = GymWrapper(gym_env, device="cpu")
    env = TransformedEnv(env, StepCounter(max_steps=config.RL_MAX_EPISODE_STEPS))
    return env


def _make_env_worker(mode, perturb_xy_range, curriculum):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    return make_env(gui=False, mode=mode, perturb_xy_range=perturb_xy_range,
                    curriculum=curriculum)




class ActorWithLearnableStd(nn.Module):
    """MLP mean network + state-independent learnable log_std.

    Standard approach in robotics PPO (Isaac Gym, Brax): the network outputs
    only the action mean; a separate nn.Parameter controls exploration noise.
    This lets the optimizer shrink std toward zero as the policy converges,
    unlike NormalParamExtractor whose scale saturates around 0.58.
    """

    def __init__(self, obs_dim, act_dim, init_log_std=-2.0, warm_start=True):
        super().__init__()
        output_layer = nn.Linear(256, act_dim)
        if warm_start:
            nn.init.uniform_(output_layer.weight, -0.01, 0.01)
            nn.init.zeros_(output_layer.bias)

        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            output_layer,
        )
        # State-independent log-std: init_log_std=-2.0 -> std=exp(-2)≈0.135
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, observation):
        loc = self.mean_net(observation)
        scale = self.log_std.exp().expand_as(loc)
        return loc, scale


def build_actor(obs_dim, act_dim, device="cpu", warm_start=True):
    actor_net = ActorWithLearnableStd(
        obs_dim, act_dim, init_log_std=-2.0, warm_start=warm_start,
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
          model_save_path=None, tb_log_dir=None, log_file_path=None,
          num_workers=None, seed=None, resume_path=None):
    device = "cpu"
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed: {seed}")
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

    obs_dim = 40
    act_dim = 7

    # Build actor-critic
    warm_start = (mode == "hybrid")
    actor = build_actor(obs_dim, act_dim, device, warm_start=warm_start)
    critic = build_critic(obs_dim, device)

    # GAE advantage estimator
    advantage_module = GAE(
        gamma=config.PPO_GAMMA,
        lmbda=config.PPO_GAE_LAMBDA,
        value_network=critic,
    )

    # PPO loss
    clip_value = bool(getattr(config, "PPO_CLIP_VALUE", False))
    ent_start = float(getattr(config, "PPO_ENT_COEFF_START", config.PPO_ENT_COEFF))
    ent_end = config.PPO_ENT_COEFF
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=config.PPO_CLIP_EPSILON,
        entropy_coeff=ent_start,
        critic_coeff=config.PPO_CRITIC_COEFF,
        loss_critic_type="l2",
        clip_value=clip_value,
    )

    # Optimizer
    optimizer = optim.Adam(loss_module.parameters(), lr=config.PPO_LR)

    # Resume from checkpoint if requested
    resumed_frames = 0
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_frames = int(ckpt.get("total_frames", 0))
        ckpt_succ = ckpt.get("success_rate", None)
        ckpt_reward = ckpt.get("mean_episode_reward", None)
        print(f"Resumed from {resume_path}")
        print(f"  checkpoint frames: {resumed_frames}"
              + (f", succ: {ckpt_succ:.0%}" if ckpt_succ is not None else "")
              + (f", ep_r: {ckpt_reward:+.1f}" if ckpt_reward is not None else ""))

    # Collector (no main-process env -- workers each build their own PyBullet client)
    num_workers = max(1, int(num_workers if num_workers is not None
                                      else config.PPO_NUM_COLLECTOR_WORKERS))
    use_curriculum = bool(getattr(config, "TRAIN_CURRICULUM", False))
    print(f"Train curriculum: {use_curriculum} "
          f"(xy_range sampled in [0, {config.PERTURB_XY_RANGE}])")
    env_fn = partial(_make_env_worker, mode, perturb_xy_range, use_curriculum)

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

    # Stability settings
    use_lr_schedule = getattr(config, "PPO_LR_SCHEDULE", "constant") == "linear"
    target_kl = float(getattr(config, "PPO_TARGET_KL", 0.0))
    kl_warmup_frames = int(getattr(config, "PPO_KL_WARMUP_FRAMES", 0))
    save_best = bool(getattr(config, "PPO_SAVE_BEST_MODEL", True))
    es_patience = int(getattr(config, "PPO_EARLY_STOP_PATIENCE", 0))
    es_min_peak = float(getattr(config, "PPO_EARLY_STOP_MIN_PEAK", 0.5))
    es_drop = float(getattr(config, "PPO_EARLY_STOP_DROP", 0.25))

    # Training loop
    total_frames = resumed_frames
    batch_idx = 0
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    ep_reward_acc = None
    ep_length_acc = None
    ep_max_step_r_acc = None
    best_succ_rate = -1.0
    best_mean_reward = -float("inf")
    best_succ_batch = 0
    best_count = 0
    below_peak_count = 0
    current_lr = config.PPO_LR
    start_time = time.time()

    print(f"Starting PPO training: mode={mode}, total_timesteps={total_timesteps}")
    print(f"Frames per batch: {config.PPO_FRAMES_PER_BATCH}")
    print(f"Mini-batch size: {config.PPO_MINI_BATCH_SIZE}")
    print(f"Epochs per batch: {config.PPO_EPOCHS}")
    print(f"LR schedule: {config.PPO_LR_SCHEDULE}, target KL: {target_kl} "
          f"(warmup: {kl_warmup_frames} frames)")
    print(f"Entropy schedule: {ent_start} -> {ent_end} (linear)")
    print(f"Early stop: patience={es_patience}, min_peak={es_min_peak}, drop={es_drop}")
    print(f"Best model tracking: {save_best}")
    print("Waiting for first batch from collector workers (this may take 30-120s on first run)...")
    iter_start = time.time()

    for batch_data in collector:
        if batch_idx == 0:
            print(f"First batch received after {time.time() - iter_start:.1f}s")
        total_frames += batch_data.numel()
        batch_idx += 1

        # Linear LR decay
        frac = max(1.0 - total_frames / total_timesteps, 0.0)
        if use_lr_schedule:
            current_lr = config.PPO_LR * frac
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        # Entropy coefficient schedule: linear decay from ent_start to ent_end
        current_ent = ent_end + (ent_start - ent_end) * frac
        loss_module.entropy_coeff.fill_(current_ent)

        # Compute advantage
        with torch.no_grad():
            advantage_module(batch_data)

        # Normalize advantage
        adv = batch_data["advantage"]
        batch_data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Load batch into replay buffer
        replay_buffer.extend(batch_data.reshape(-1))

        # PPO update epochs with KL-based early stopping
        num_mini_batches = max(1, config.PPO_FRAMES_PER_BATCH // config.PPO_MINI_BATCH_SIZE)
        epoch_losses = []
        batch_kls = []
        batch_policy_losses = []
        batch_value_losses = []
        batch_entropy_losses = []
        batch_clip_fractions = []
        batch_grad_norms = []
        batch_entropies = []
        epochs_run = 0
        kl_early_stop = False
        for epoch_i in range(config.PPO_EPOCHS):
            epoch_kls = []
            for _ in range(num_mini_batches):
                minibatch = replay_buffer.sample()
                loss_td = loss_module(minibatch)

                loss_obj = loss_td["loss_objective"]
                loss_cri = loss_td["loss_critic"]
                loss_ent = loss_td["loss_entropy"]
                loss = loss_obj + loss_cri + loss_ent

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config.PPO_MAX_GRAD_NORM)
                optimizer.step()

                epoch_losses.append(loss.item())
                batch_policy_losses.append(loss_obj.item())
                batch_value_losses.append(loss_cri.item())
                batch_entropy_losses.append(loss_ent.item())
                batch_grad_norms.append(float(grad_norm))

                kl_val = loss_td.get("kl_approx", None)
                if kl_val is not None:
                    kl_item = kl_val.mean().item() if kl_val.numel() > 1 else kl_val.item()
                    epoch_kls.append(kl_item)

                clip_frac = loss_td.get("clip_fraction", None)
                if clip_frac is not None:
                    cf = clip_frac.mean().item() if clip_frac.numel() > 1 else clip_frac.item()
                    batch_clip_fractions.append(cf)

                ent_val = loss_td.get("entropy", None)
                if ent_val is not None:
                    ev = ent_val.mean().item() if ent_val.numel() > 1 else ent_val.item()
                    batch_entropies.append(ev)

            epochs_run = epoch_i + 1
            kl_active = target_kl > 0 and total_frames >= kl_warmup_frames
            if kl_active and epoch_kls:
                mean_epoch_kl = float(np.mean(epoch_kls))
                batch_kls.append(mean_epoch_kl)
                if mean_epoch_kl > target_kl:
                    kl_early_stop = True
                    break
            elif epoch_kls:
                batch_kls.append(float(np.mean(epoch_kls)))

        mean_kl = float(np.mean(batch_kls)) if batch_kls else 0.0
        mean_grad_norm = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0
        mean_clip_frac = float(np.mean(batch_clip_fractions)) if batch_clip_fractions else 0.0
        mean_entropy = float(np.mean(batch_entropies)) if batch_entropies else 0.0
        mean_policy_loss = float(np.mean(batch_policy_losses)) if batch_policy_losses else 0.0
        mean_value_loss = float(np.mean(batch_value_losses)) if batch_value_losses else 0.0

        # Action and value statistics from the batch
        actions_t = batch_data.get("action", None)
        act_mean = float(actions_t.mean().item()) if actions_t is not None else 0.0
        act_std = float(actions_t.std().item()) if actions_t is not None else 0.0

        value_t = batch_data.get("state_value", None)
        val_mean = float(value_t.mean().item()) if value_t is not None else 0.0

        # Log learnable log_std so we can monitor exploration narrowing
        log_std_val = float("nan")
        for mod in actor.modules():
            if hasattr(mod, "log_std"):
                log_std_val = float(mod.log_std.data.mean().item())
                break

        # Logging
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # --- Episode tracking using TorchRL's nested "next" keys ---
        step_rewards_t = batch_data.get(("next", "reward"))
        done_t = batch_data.get(("next", "done"))

        mean_step_reward = float(step_rewards_t.mean().item()) if step_rewards_t is not None else 0.0
        num_dones = int(done_t.sum().item()) if done_t is not None else 0

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
                ep_max_step_r_acc = [-float("inf")] * num_envs
            r_list = r.tolist()
            d_list = d.bool().tolist()
            for w in range(num_envs):
                for t in range(T):
                    step_r = r_list[w][t]
                    ep_reward_acc[w] += step_r
                    ep_length_acc[w] += 1
                    if step_r > ep_max_step_r_acc[w]:
                        ep_max_step_r_acc[w] = step_r
                    if d_list[w][t]:
                        ep_r_done = ep_reward_acc[w]
                        ep_len_done = ep_length_acc[w]
                        episode_rewards.append(ep_r_done)
                        episode_lengths.append(ep_len_done)
                        # Detect actual cube lift: the lift bonus
                        # (REWARD_EPSILON=100) is the only single-step reward
                        # component that can exceed ~80. This is reliable
                        # regardless of cumulative proximity or approach rewards.
                        lifted = ep_max_step_r_acc[w] >= config.REWARD_EPSILON * 0.8
                        episode_successes.append(1 if lifted else 0)
                        batch_new_rewards.append(ep_r_done)
                        batch_new_lengths.append(ep_len_done)
                        ep_reward_acc[w] = 0.0
                        ep_length_acc[w] = 0
                        ep_max_step_r_acc[w] = -float("inf")

        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0

        recent_rewards = episode_rewards[-20:]
        recent_lengths = episode_lengths[-20:]
        recent_successes = episode_successes[-20:]

        mean_ep_reward = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
        std_ep_reward = float(np.std(recent_rewards)) if len(recent_rewards) > 1 else 0.0
        mean_ep_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
        succ_rate = float(np.mean(recent_successes)) if recent_successes else 0.0

        batch_ep_std = float(np.std(batch_new_rewards)) if len(batch_new_rewards) > 1 else 0.0

        # TensorBoard scalars
        writer.add_scalar("train/loss", mean_loss, total_frames)
        writer.add_scalar("train/loss_policy", mean_policy_loss, total_frames)
        writer.add_scalar("train/loss_value", mean_value_loss, total_frames)
        writer.add_scalar("train/entropy", mean_entropy, total_frames)
        writer.add_scalar("train/grad_norm", mean_grad_norm, total_frames)
        writer.add_scalar("train/clip_fraction", mean_clip_frac, total_frames)
        writer.add_scalar("train/fps", fps, total_frames)
        writer.add_scalar("train/mean_step_reward", mean_step_reward, total_frames)
        writer.add_scalar("train/episodes_completed", len(episode_rewards), total_frames)
        writer.add_scalar("train/learning_rate", current_lr, total_frames)
        writer.add_scalar("train/entropy_coeff", current_ent, total_frames)
        writer.add_scalar("train/kl_approx", mean_kl, total_frames)
        writer.add_scalar("train/epochs_run", epochs_run, total_frames)
        writer.add_scalar("train/action_mean", act_mean, total_frames)
        writer.add_scalar("train/action_std", act_std, total_frames)
        writer.add_scalar("train/value_mean", val_mean, total_frames)
        if not np.isnan(log_std_val):
            writer.add_scalar("train/log_std", log_std_val, total_frames)
        if recent_rewards:
            writer.add_scalar("train/mean_episode_reward", mean_ep_reward, total_frames)
            writer.add_scalar("train/std_episode_reward", std_ep_reward, total_frames)
            writer.add_scalar("train/mean_episode_length", mean_ep_length, total_frames)
            writer.add_scalar("train/success_rate", succ_rate, total_frames)
        if len(batch_new_rewards) > 1:
            writer.add_scalar("train/batch_episode_reward_std", batch_ep_std, total_frames)

        kl_flag = " KL!" if kl_early_stop else ""
        best_flag = ""

        # Best model tracking: higher succ_rate wins; ties broken by mean ep reward
        is_new_best = (
            save_best
            and len(recent_successes) >= 5
            and (succ_rate > best_succ_rate
                 or (succ_rate == best_succ_rate and mean_ep_reward > best_mean_reward))
        )
        if is_new_best:
            best_succ_rate = succ_rate
            best_mean_reward = mean_ep_reward
            best_succ_batch = batch_idx
            best_count += 1
            below_peak_count = 0
            checkpoint_data = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_frames": total_frames,
                "success_rate": succ_rate,
                "mean_episode_reward": mean_ep_reward,
            }
            best_path = os.path.join(model_save_path, "best_model.pt")
            torch.save(checkpoint_data, best_path)
            # Numbered copy: best_003_succ100_r152.3.pt
            succ_pct = int(round(succ_rate * 100))
            numbered_name = f"best_{run_tag}_{best_count:03d}_succ{succ_pct}_r{mean_ep_reward:.1f}.pt"
            torch.save(checkpoint_data, os.path.join(model_save_path, numbered_name))
            best_flag = f" *BEST #{best_count}*"

        print(
            f"batch={batch_idx:4d} | "
            f"frames={total_frames:8d}/{total_timesteps} | "
            f"loss={mean_loss:+.3f} (p={mean_policy_loss:+.3f} v={mean_value_loss:.3f}) | "
            f"step_r={mean_step_reward:+.4f} | "
            f"ep_r={mean_ep_reward:+7.2f}±{std_ep_reward:5.2f} | "
            f"succ={succ_rate:5.0%} | "
            f"grad={mean_grad_norm:.3f} clip={mean_clip_frac:.2f} ent={mean_entropy:.3f} | "
            f"kl={mean_kl:.4f} ep={epochs_run}/{config.PPO_EPOCHS}{kl_flag} | "
            f"lr={current_lr:.1e} ent_c={current_ent:.3f} | "
            f"act={act_mean:+.3f}±{act_std:.3f} logσ={log_std_val:+.2f} V={val_mean:+.1f} | "
            f"fps={fps:.0f}{best_flag}"
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

        # Early stopping on collapse
        if es_patience > 0 and best_succ_rate >= es_min_peak:
            if succ_rate < best_succ_rate - es_drop:
                below_peak_count += 1
                if below_peak_count >= es_patience:
                    print(
                        f"\n*** EARLY STOP: succ={succ_rate:.0%} has been "
                        f">{es_drop:.0%} below peak={best_succ_rate:.0%} "
                        f"(batch {best_succ_batch}) for {below_peak_count} "
                        f"consecutive batches ***\n"
                    )
                    break
            else:
                below_peak_count = 0

    # Save final model — use best checkpoint if available
    final_path = os.path.join(model_save_path, "final_model.pt")
    best_path = os.path.join(model_save_path, "best_model.pt")
    if save_best and os.path.exists(best_path):
        shutil.copy2(best_path, final_path)
        print(f"Training complete. Copied best model (succ={best_succ_rate:.0%}, "
              f"ep_r={best_mean_reward:+.1f}, batch={best_succ_batch}) to {final_path}")
    else:
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


def load_trained_actor(model_path, obs_dim=40, act_dim=7, device="cpu"):
    actor = build_actor(obs_dim, act_dim, device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


if __name__ == "__main__":
    train()
