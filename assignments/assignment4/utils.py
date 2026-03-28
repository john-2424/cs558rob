import torch


def compute_episode_return(rewards, gamma):
    """
    Q1.1:
    One discounted return for the entire trajectory.
    G(tau) = sum_t gamma^t r_t
    """
    G = 0.0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r
    return G


def compute_reward_to_go(rewards, gamma):
    """
    Q1.2:
    For each timestep t, compute:
    sum_{t'=t}^T gamma^(t'-t) * r(t')
    Returns a list of same length as rewards.
    """
    rtg = []
    running_return = 0.0

    for r in reversed(rewards):
        running_return = r + gamma * running_return
        rtg.insert(0, running_return)

    return rtg


def normalize_returns(returns, eps=1e-8):
    """
    Q1.3:
    Subtract mean and divide by std.
    """
    returns_t = torch.tensor(returns, dtype=torch.float32)
    mean = returns_t.mean()
    std = returns_t.std()

    if std.item() < eps:
        std = torch.tensor(1.0)

    normalized = (returns_t - mean) / (std + eps)
    return normalized.tolist()


def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)