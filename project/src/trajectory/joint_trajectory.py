import numpy as np


def interpolate_joint_trajectory(q_start, q_goal, num_waypoints=25):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if q_start.shape != q_goal.shape:
        raise ValueError("q_start and q_goal must have the same shape")

    if num_waypoints < 2:
        raise ValueError("num_waypoints must be at least 2")

    waypoints = []
    for alpha in np.linspace(0.0, 1.0, num_waypoints):
        q = (1.0 - alpha) * q_start + alpha * q_goal
        waypoints.append(q)

    return waypoints