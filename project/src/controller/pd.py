from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PDControllerConfig:
    kp: np.ndarray
    kd: np.ndarray
    max_velocity: Optional[np.ndarray] = None


class JointSpacePDController:
    """
    Simple joint-space PD controller.

    For now this controller outputs a desired joint velocity command:
        qd_cmd = Kp * (q_des - q) + Kd * (qd_des - qd)

    Later, if needed, we can upgrade this same controller to output:
    - torques
    - accelerations
    - feedforward + PD
    """

    def __init__(self, config: PDControllerConfig):
        self.kp = np.asarray(config.kp, dtype=float)
        self.kd = np.asarray(config.kd, dtype=float)

        if self.kp.shape != self.kd.shape:
            raise ValueError("kp and kd must have the same shape")

        self.num_joints = self.kp.shape[0]

        if config.max_velocity is None:
            self.max_velocity = None
        else:
            self.max_velocity = np.asarray(config.max_velocity, dtype=float)
            if self.max_velocity.shape != (self.num_joints,):
                raise ValueError("max_velocity must match controller dimension")

    def compute_position_error(self, q_des, q):
        q_des = np.asarray(q_des, dtype=float)
        q = np.asarray(q, dtype=float)
        self._check_shape(q_des, "q_des")
        self._check_shape(q, "q")
        return q_des - q

    def compute_velocity_error(self, qd_des, qd):
        qd_des = np.asarray(qd_des, dtype=float)
        qd = np.asarray(qd, dtype=float)
        self._check_shape(qd_des, "qd_des")
        self._check_shape(qd, "qd")
        return qd_des - qd

    def compute_velocity_command(self, q_des, q, qd_des=None, qd=None):
        """
        Compute a custom PD velocity command.

        Args:
            q_des: desired joint positions
            q: current joint positions
            qd_des: desired joint velocities (defaults to zeros)
            qd: current joint velocities (defaults to zeros)

        Returns:
            qd_cmd: desired joint velocity command
        """
        q_des = np.asarray(q_des, dtype=float)
        q = np.asarray(q, dtype=float)

        self._check_shape(q_des, "q_des")
        self._check_shape(q, "q")

        if qd_des is None:
            qd_des = np.zeros(self.num_joints, dtype=float)
        else:
            qd_des = np.asarray(qd_des, dtype=float)

        if qd is None:
            qd = np.zeros(self.num_joints, dtype=float)
        else:
            qd = np.asarray(qd, dtype=float)

        self._check_shape(qd_des, "qd_des")
        self._check_shape(qd, "qd")

        pos_err = q_des - q
        vel_err = qd_des - qd

        qd_cmd = self.kp * pos_err + self.kd * vel_err

        if self.max_velocity is not None:
            qd_cmd = np.clip(qd_cmd, -self.max_velocity, self.max_velocity)

        return qd_cmd

    def _check_shape(self, x, name):
        if x.shape != (self.num_joints,):
            raise ValueError(
                f"{name} must have shape {(self.num_joints,)}, got {x.shape}"
            )