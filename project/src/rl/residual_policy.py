import numpy as np

from src import config


class ResidualActionWrapper:
    def __init__(self, mode="hybrid", residual_max=None, pd_max_vel=None):
        self.mode = mode
        self.residual_max = residual_max or config.RESIDUAL_MAX
        self.pd_max_vel = np.asarray(
            pd_max_vel or config.PD_MAX_JOINT_VEL, dtype=float
        )

    def compute_action(self, qd_pd, raw_action):
        qd_pd = np.asarray(qd_pd, dtype=float)
        raw_action = np.asarray(raw_action, dtype=float)

        if self.mode == "planner_only":
            return qd_pd

        if self.mode == "rl_only":
            return np.clip(
                raw_action * self.pd_max_vel,
                -self.pd_max_vel,
                self.pd_max_vel,
            )

        # hybrid
        qd_residual = raw_action * self.residual_max
        qd_total = qd_pd + qd_residual
        return np.clip(qd_total, -self.pd_max_vel, self.pd_max_vel)

    def get_residual(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=float)
        if self.mode == "planner_only":
            return np.zeros_like(raw_action)
        if self.mode == "rl_only":
            return raw_action * self.pd_max_vel
        return raw_action * self.residual_max
