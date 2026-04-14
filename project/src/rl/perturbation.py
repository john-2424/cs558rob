import numpy as np
import pybullet as p


def sample_perturbation(rng, xy_range, yaw_range):
    dx = rng.uniform(-xy_range, xy_range)
    dy = rng.uniform(-xy_range, xy_range)
    dyaw = rng.uniform(-yaw_range, yaw_range)
    return dx, dy, dyaw


def perturb_cube_pose(cube_id, nominal_pos, nominal_orn_euler, rng,
                      xy_range=0.04, yaw_range=0.2):
    dx, dy, dyaw = sample_perturbation(rng, xy_range, yaw_range)

    new_pos = [
        nominal_pos[0] + dx,
        nominal_pos[1] + dy,
        nominal_pos[2],
    ]

    new_euler = [
        nominal_orn_euler[0],
        nominal_orn_euler[1],
        nominal_orn_euler[2] + dyaw,
    ]
    new_orn = p.getQuaternionFromEuler(new_euler)

    p.resetBasePositionAndOrientation(cube_id, new_pos, new_orn)

    return np.array(new_pos, dtype=float), new_orn, (dx, dy, dyaw)
