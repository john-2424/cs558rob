import numpy as np
import torch


def normalize(x, world_size):
    if isinstance(world_size, (int, float)):
        if isinstance(x, torch.Tensor):
            return x / float(world_size)
        return x / np.float32(world_size)

    if isinstance(x, torch.Tensor):
        ws = torch.tensor(world_size, dtype=x.dtype, device=x.device)
        return x / ws

    ws = np.asarray(world_size, dtype=np.float32)
    return x / ws


def unnormalize(x, world_size):
    if isinstance(world_size, (int, float)):
        if isinstance(x, torch.Tensor):
            return x * float(world_size)
        return x * np.float32(world_size)

    if isinstance(x, torch.Tensor):
        ws = torch.tensor(world_size, dtype=x.dtype, device=x.device)
        return x * ws

    ws = np.asarray(world_size, dtype=np.float32)
    return x * ws