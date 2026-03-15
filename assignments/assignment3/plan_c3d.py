import numpy as np
import torch

OBSTACLE_SHAPES = np.array([
    [5.0, 5.0, 10.0],
    [5.0, 10.0, 5.0],
    [5.0, 10.0, 10.0],
    [10.0, 5.0, 5.0],
    [10.0, 5.0, 10.0],
    [10.0, 10.0, 5.0],
    [10.0, 10.0, 10.0],
    [5.0, 5.0, 5.0],
    [10.0, 10.0, 10.0],
    [5.0, 5.0, 5.0],
], dtype=np.float32)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def IsInCollision(x, obc, world_size=20.0):
    """
    x   : shape (3,)
    obc : shape (10, 3), obstacle centers for one environment
    """
    x = _to_numpy(x)
    obc = np.asarray(obc, dtype=np.float32).reshape(-1, 3)

    if isinstance(world_size, (int, float)):
        bounds = np.array([float(world_size)] * 3, dtype=np.float32)
    else:
        bounds = np.asarray(world_size, dtype=np.float32).reshape(3)

    # Outside workspace = collision
    if np.any(x < -bounds) or np.any(x > bounds):
        return True

    num_obs = min(len(obc), len(OBSTACLE_SHAPES))
    for i in range(num_obs):
        c = obc[i]
        sx, sy, sz = OBSTACLE_SHAPES[i]
        if (
            abs(x[0] - c[0]) <= sx / 2.0 and
            abs(x[1] - c[1]) <= sy / 2.0 and
            abs(x[2] - c[2]) <= sz / 2.0
        ):
            return True

    return False