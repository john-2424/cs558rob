import numpy as np

from src import config


class ResidualActionWrapper:
    def __init__(self, mode="hybrid", residual_max_pos=None, residual_max=None,
                 pd_max_vel=None):
        self.mode = mode
        self.residual_max_pos = residual_max_pos or config.RESIDUAL_MAX_POS
        self.residual_max = residual_max or config.RESIDUAL_MAX
        self.pd_max_vel = np.asarray(
            pd_max_vel or config.PD_MAX_JOINT_VEL, dtype=float
        )

    def correct_target(self, q_target, raw_action):
        """Shift the PD position target by the RL residual (hybrid mode).

        Returns the corrected target for PD to track.  For planner_only
        the target is returned unchanged; for rl_only this method should
        not be called (use compute_action instead).
        """
        raw_action = np.asarray(raw_action, dtype=float)
        if self.mode == "planner_only":
            return np.asarray(q_target, dtype=float)
        # hybrid: shift waypoint target by learned offset
        offset = raw_action * self.residual_max_pos
        return np.asarray(q_target, dtype=float) + offset

    def compute_action(self, qd_pd, raw_action):
        """Compute velocity command (used only by rl_only mode now)."""
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

        # hybrid — should use correct_target path instead, but keep as
        # fallback: PD already computed from corrected target, just pass through.
        return qd_pd

    def get_residual(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=float)
        if self.mode == "planner_only":
            return np.zeros_like(raw_action)
        if self.mode == "rl_only":
            return raw_action * self.pd_max_vel
        return raw_action * self.residual_max_pos
