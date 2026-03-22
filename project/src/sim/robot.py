from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from src import config
from src.controller.pd import JointSpacePDController, PDControllerConfig


@dataclass
class JointInfo:
    joint_index: int
    joint_name: str
    joint_type: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float


class PandaRobot:
    def __init__(self):
        self.robot_id = None
        self.arm_joint_indices: List[int] = []
        self.gripper_joint_indices: List[int] = []
        self.joint_infos: List[JointInfo] = []
        self.ee_link_index = 11
        self.home_joints = np.array(config.PANDA_HOME_JOINTS, dtype=float)

        # Custom PD controller (initialized after arm joint extraction / load)
        self.pd_controller = None

        # Default arm motor force used for velocity-control mode
        self.arm_max_forces = None

    def load(self, base_position=None, base_orientation_euler=None) -> int:
        if base_position is None:
            base_position = config.PANDA_BASE_POS
        if base_orientation_euler is None:
            base_orientation_euler = config.PANDA_BASE_ORN_EULER

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        base_orientation = p.getQuaternionFromEuler(base_orientation_euler)
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
        )

        self._extract_joint_info()
        self._init_pd_controller()
        self.reset_home()
        self.hold_home_pose()
        return self.robot_id

    def _extract_joint_info(self) -> None:
        self.arm_joint_indices.clear()
        self.gripper_joint_indices.clear()
        self.joint_infos.clear()

        num_joints = p.getNumJoints(self.robot_id)

        for joint_idx in range(num_joints):
            info = p.getJointInfo(self.robot_id, joint_idx)

            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]

            self.joint_infos.append(
                JointInfo(
                    joint_index=joint_idx,
                    joint_name=joint_name,
                    joint_type=joint_type,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    max_force=max_force,
                    max_velocity=max_velocity,
                )
            )

            if joint_name.startswith("panda_joint") and joint_idx <= 6:
                self.arm_joint_indices.append(joint_idx)

            if "finger" in joint_name:
                self.gripper_joint_indices.append(joint_idx)

    def _init_pd_controller(self) -> None:
        num_arm_joints = len(self.arm_joint_indices)

        kp = np.asarray(config.PD_KP, dtype=float)
        kd = np.asarray(config.PD_KD, dtype=float)
        max_vel = np.asarray(config.PD_MAX_JOINT_VEL, dtype=float)

        if kp.shape != (num_arm_joints,):
            raise ValueError(
                f"PD_KP must have shape {(num_arm_joints,)}, got {kp.shape}"
            )
        if kd.shape != (num_arm_joints,):
            raise ValueError(
                f"PD_KD must have shape {(num_arm_joints,)}, got {kd.shape}"
            )
        if max_vel.shape != (num_arm_joints,):
            raise ValueError(
                f"PD_MAX_JOINT_VEL must have shape {(num_arm_joints,)}, got {max_vel.shape}"
            )

        self.pd_controller = JointSpacePDController(
            PDControllerConfig(
                kp=kp,
                kd=kd,
                max_velocity=max_vel,
            )
        )

        # Pull joint force limits from URDF metadata when available.
        # Fall back to a safe default if needed.
        self.arm_max_forces = []
        for joint_idx in self.arm_joint_indices:
            info = next(j for j in self.joint_infos if j.joint_index == joint_idx)
            force = float(info.max_force) if info.max_force > 0 else 200.0
            self.arm_max_forces.append(force)
    
    def reset_home(self) -> None:
        for joint_idx, joint_val in zip(self.arm_joint_indices, self.home_joints):
            p.resetJointState(self.robot_id, joint_idx, joint_val)

        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.04)

    def hold_home_pose(self) -> None:
        self.command_joint_positions_pd(self.home_joints)
        self.hold_gripper_open()

    def get_joint_positions(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        return np.array([state[0] for state in joint_states], dtype=float)

    def get_joint_velocities(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        return np.array([state[1] for state in joint_states], dtype=float)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        link_state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(link_state[4], dtype=float), np.array(link_state[5], dtype=float)

    def print_joint_summary(self) -> None:
        print("\n=== Panda Joint Summary ===")
        for info in self.joint_infos:
            print(
                f"idx={info.joint_index:2d} | "
                f"name={info.joint_name:20s} | "
                f"type={info.joint_type} | "
                f"limits=({info.lower_limit:.3f}, {info.upper_limit:.3f})"
            )
        print(f"\nArm joint indices: {self.arm_joint_indices}")
        print(f"Gripper joint indices: {self.gripper_joint_indices}")
        print(f"EE link index: {self.ee_link_index}")
    
    def command_joint_positions(
        self,
        target_joints,
        forces=None,
        position_gains=None,
        velocity_gains=None,
    ) -> None:
        target_joints = np.asarray(target_joints, dtype=float)

        if target_joints.shape[0] != len(self.arm_joint_indices):
            raise ValueError(
                f"Expected {len(self.arm_joint_indices)} arm joints, got {target_joints.shape[0]}"
            )

        if forces is None:
            forces = [200.0] * len(self.arm_joint_indices)
        if position_gains is None:
            position_gains = [0.08] * len(self.arm_joint_indices)
        if velocity_gains is None:
            velocity_gains = [1.0] * len(self.arm_joint_indices)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joints.tolist(),
            targetVelocities=[0.0] * len(self.arm_joint_indices),
            forces=forces,
            positionGains=position_gains,
            velocityGains=velocity_gains,
        )
    
    def command_joint_positions_pd(
        self,
        target_joints,
        target_joint_velocities=None,
        forces=None,
    ) -> None:
        """
        Custom PD control path.

        Computes a desired joint velocity from:
            qd_cmd = Kp (q_des - q) + Kd (qd_des - qd)

        Then sends that velocity command through PyBullet VELOCITY_CONTROL.
        """
        if self.pd_controller is None:
            raise RuntimeError("PD controller is not initialized")

        target_joints = np.asarray(target_joints, dtype=float)

        if target_joints.shape != (len(self.arm_joint_indices),):
            raise ValueError(
                f"Expected target_joints shape {(len(self.arm_joint_indices),)}, "
                f"got {target_joints.shape}"
            )

        current_q = self.get_joint_positions()
        current_qd = self.get_joint_velocities()

        if target_joint_velocities is None:
            target_joint_velocities = np.zeros_like(target_joints, dtype=float)
        else:
            target_joint_velocities = np.asarray(target_joint_velocities, dtype=float)

        qd_cmd = self.pd_controller.compute_velocity_command(
            q_des=target_joints,
            q=current_q,
            qd_des=target_joint_velocities,
            qd=current_qd,
        )

        if forces is None:
            forces = self.arm_max_forces

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.arm_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=qd_cmd.tolist(),
            forces=forces,
        )

    def hold_gripper_open(self) -> None:
        if len(self.gripper_joint_indices) == 2:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.gripper_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.04, 0.04],
                targetVelocities=[0.0, 0.0],
                forces=[50.0, 50.0],
                positionGains=[0.2, 0.2],
                velocityGains=[1.0, 1.0],
            )

    def command_arm_and_gripper(self, target_joints, target_joint_velocities=None) -> None:
        self.command_joint_positions_pd(
            target_joints=target_joints,
            target_joint_velocities=target_joint_velocities,
        )
        self.hold_gripper_open()

    def get_joint_position_error(self, target_joints) -> np.ndarray:
        target_joints = np.asarray(target_joints, dtype=float)
        return target_joints - self.get_joint_positions()

    def get_pd_debug_info(self, target_joints, target_joint_velocities=None):
        if self.pd_controller is None:
            raise RuntimeError("PD controller is not initialized")

        target_joints = np.asarray(target_joints, dtype=float)
        current_q = self.get_joint_positions()
        current_qd = self.get_joint_velocities()

        if target_joint_velocities is None:
            target_joint_velocities = np.zeros_like(target_joints, dtype=float)
        else:
            target_joint_velocities = np.asarray(target_joint_velocities, dtype=float)

        pos_err = self.pd_controller.compute_position_error(target_joints, current_q)
        vel_err = self.pd_controller.compute_velocity_error(target_joint_velocities, current_qd)
        qd_cmd = self.pd_controller.compute_velocity_command(
            q_des=target_joints,
            q=current_q,
            qd_des=target_joint_velocities,
            qd=current_qd,
        )

        return {
            "q": current_q,
            "qd": current_qd,
            "q_des": target_joints,
            "qd_des": target_joint_velocities,
            "pos_err": pos_err,
            "vel_err": vel_err,
            "qd_cmd": qd_cmd,
        }
    
    def is_at_joint_target(self, target_joints, tol: float = 0.03) -> bool:
        err = self.get_joint_position_error(target_joints)
        return bool(np.max(np.abs(err)) < tol)