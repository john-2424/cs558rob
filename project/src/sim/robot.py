from dataclasses import dataclass
from typing import List, Tuple, Optional

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


class PandaRobotDemo:
    def __init__(self):
        self.robot_id = None
        self.arm_joint_indices: List[int] = []
        self.gripper_joint_indices: List[int] = []
        self.joint_infos: List[JointInfo] = []
        self.ee_link_index = config.ATTACH_PARENT_LINK_INDEX
        self.home_joints = np.array(config.PANDA_HOME_JOINTS, dtype=float)

        # Custom PD controller (initialized after arm joint extraction / load)
        self.pd_controller = None

        # Default arm motor force used for velocity-control mode
        self.arm_max_forces = None

        # Active fixed constraint used for object attachment
        self.attached_constraint_id: Optional[int] = None
        self.attached_body_id: Optional[int] = None

        self.left_finger_link_index = config.LEFT_FINGER_LINK_INDEX
        self.right_finger_link_index = config.RIGHT_FINGER_LINK_INDEX
        self.attached_constraint_id = None
        self.attached_body_id = None

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

        self.arm_max_forces = []
        for joint_idx in self.arm_joint_indices:
            info = next(j for j in self.joint_infos if j.joint_index == joint_idx)
            force = float(info.max_force) if info.max_force > 0 else 200.0
            self.arm_max_forces.append(force)

    def reset_home(self) -> None:
        for joint_idx, joint_val in zip(self.arm_joint_indices, self.home_joints):
            p.resetJointState(self.robot_id, joint_idx, joint_val)

        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, config.GRIPPER_OPEN_WIDTH)

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
        link_state = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
        )
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

    # ---------------------------------------------------------
    # Gripper helpers
    # ---------------------------------------------------------

    def command_gripper(self, opening_width: float) -> None:
        if len(self.gripper_joint_indices) != 2:
            return

        opening_width = float(np.clip(opening_width, 0.0, config.GRIPPER_OPEN_WIDTH))
        target_positions = [opening_width, opening_width]

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.gripper_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            targetVelocities=[0.0, 0.0],
            forces=[config.GRIPPER_FORCE, config.GRIPPER_FORCE],
            positionGains=[0.4, 0.4],
            velocityGains=[1.0, 1.0],
        )

    def hold_gripper_open(self) -> None:
        self.command_gripper(config.GRIPPER_OPEN_WIDTH)

    def open_gripper(self) -> None:
        self.command_gripper(config.GRIPPER_OPEN_WIDTH)

    def close_gripper(self) -> None:
        self.command_gripper(config.GRIPPER_CLOSED_WIDTH)

    def command_arm_and_gripper(
        self,
        target_joints,
        target_joint_velocities=None,
        gripper_width: Optional[float] = None,
    ) -> None:
        self.command_joint_positions_pd(
            target_joints=target_joints,
            target_joint_velocities=target_joint_velocities,
        )
        if gripper_width is None:
            self.hold_gripper_open()
        else:
            self.command_gripper(gripper_width)

    # ---------------------------------------------------------
    # IK helpers
    # ---------------------------------------------------------

    def solve_ik(self, ee_pos, ee_orn=None) -> np.ndarray:
        ee_pos = np.asarray(ee_pos, dtype=float)

        lower_limits = []
        upper_limits = []
        joint_ranges = []
        rest_poses = self.home_joints.tolist()

        for joint_idx in self.arm_joint_indices:
            info = next(j for j in self.joint_infos if j.joint_index == joint_idx)
            ll = float(info.lower_limit)
            ul = float(info.upper_limit)
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)

        if ee_orn is None:
            ik_solution = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.ee_link_index,
                targetPosition=ee_pos.tolist(),
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=config.IK_MAX_ITERATIONS,
                residualThreshold=config.IK_RESIDUAL_THRESHOLD,
            )
        else:
            ik_solution = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.ee_link_index,
                targetPosition=ee_pos.tolist(),
                targetOrientation=ee_orn,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=config.IK_MAX_ITERATIONS,
                residualThreshold=config.IK_RESIDUAL_THRESHOLD,
            )

        return np.array(ik_solution[: len(self.arm_joint_indices)], dtype=float)

    # ---------------------------------------------------------
    # Object attach/detach helpers
    # ---------------------------------------------------------

    def attach_object(self, body_id: int) -> int:
        if self.attached_constraint_id is not None:
            self.detach_object()

        ee_pos, ee_orn = self.get_ee_pose()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(body_id)

        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos.tolist(), ee_orn.tolist())
        rel_pos, rel_orn = p.multiplyTransforms(
            inv_ee_pos,
            inv_ee_orn,
            obj_pos,
            obj_orn,
        )

        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )

        self.attached_constraint_id = constraint_id
        self.attached_body_id = body_id
        return constraint_id
    
    def detach_object(self) -> None:
        if self.attached_constraint_id is not None:
            try:
                p.removeConstraint(self.attached_constraint_id)
            except Exception:
                pass
        self.attached_constraint_id = None
        self.attached_body_id = None

    # ---------------------------------------------------------
    # Debug / error helpers
    # ---------------------------------------------------------

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

    def get_link_pose(self, link_index: int):
        link_state = p.getLinkState(
            self.robot_id,
            link_index,
            computeForwardKinematics=True,
        )
        pos = np.array(link_state[4], dtype=float)
        orn = np.array(link_state[5], dtype=float)
        return pos, orn


    def get_finger_tip_positions(self):
        left_pos, _ = self.get_link_pose(self.left_finger_link_index)
        right_pos, _ = self.get_link_pose(self.right_finger_link_index)
        return left_pos, right_pos


    def get_cube_center_distance(self, body_id: int) -> float:
        ee_pos, _ = self.get_ee_pose()
        cube_pos, _ = p.getBasePositionAndOrientation(body_id)
        return float(np.linalg.norm(np.asarray(cube_pos, dtype=float) - ee_pos))


    def get_contact_points_with_body(self, body_id: int):
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=body_id)
        return contacts


    def get_finger_contact_count_with_body(self, body_id: int) -> int:
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=body_id)

        finger_count = 0
        for c in contacts:
            link_a = c[3]  # link index on bodyA
            if link_a in (self.left_finger_link_index, self.right_finger_link_index):
                finger_count += 1

        return finger_count


    def is_grasp_ready(self, body_id: int):
        ee_to_cube_dist = self.get_cube_center_distance(body_id)
        min_finger_to_cube_dist = self.get_min_finger_to_body_distance(body_id)

        total_contacts = len(self.get_contact_points_with_body(body_id))
        finger_contacts = self.get_finger_contact_count_with_body(body_id)

        dist_ok = ee_to_cube_dist <= config.GRASP_MAX_EE_TO_CUBE_CENTER_DIST
        finger_dist_ok = min_finger_to_cube_dist <= config.GRASP_MAX_FINGER_TO_CUBE_DIST
        total_contact_ok = total_contacts >= config.GRASP_MIN_TOTAL_CONTACTS
        finger_contact_ok = finger_contacts >= config.GRASP_MIN_FINGER_CONTACTS

        if config.GRASP_REQUIRE_CONTACT:
            if config.GRASP_USE_FINGER_DISTANCE:
                ready = finger_dist_ok and total_contact_ok and finger_contact_ok
            else:
                ready = dist_ok and total_contact_ok and finger_contact_ok
        else:
            if config.GRASP_USE_FINGER_DISTANCE:
                ready = finger_dist_ok
            else:
                ready = dist_ok

        left_pos, right_pos = self.get_finger_tip_positions()

        # Fingertips must sit below the cube top (rejects top-face pinches).
        _, cube_aabb_max = p.getAABB(body_id)
        cube_top_z = cube_aabb_max[2]
        below_top = bool(
            left_pos[2] < cube_top_z - config.GRASP_FINGER_BELOW_TOP_MARGIN
            and right_pos[2] < cube_top_z - config.GRASP_FINGER_BELOW_TOP_MARGIN
        )

        # Cube center must lie BETWEEN the fingertips along the finger axis
        # (rejects same-side contacts that game the lenient contact count).
        cube_center = np.asarray(
            p.getBasePositionAndOrientation(body_id)[0], dtype=float,
        )
        finger_axis = right_pos - left_pos
        axis_sq = float(np.dot(finger_axis, finger_axis)) + 1e-9
        bracket_t = float(np.dot(cube_center - left_pos, finger_axis)) / axis_sq
        bracket_ok = bool(
            config.GRASP_BRACKET_T_MIN < bracket_t < config.GRASP_BRACKET_T_MAX
        )

        ready = ready and below_top and bracket_ok

        debug = {
            "ee_to_cube_dist": ee_to_cube_dist,
            "min_finger_to_cube_dist": min_finger_to_cube_dist,
            "total_contacts": total_contacts,
            "finger_contacts": finger_contacts,
            "dist_ok": dist_ok,
            "finger_dist_ok": finger_dist_ok,
            "total_contact_ok": total_contact_ok,
            "finger_contact_ok": finger_contact_ok,
            "below_top": below_top,
            "bracket_ok": bracket_ok,
            "bracket_t": bracket_t,
            "left_finger_pos": left_pos,
            "right_finger_pos": right_pos,
            "ready": ready,
        }
        return ready, debug

    def get_body_aabb(self, body_id: int):
        aabb_min, aabb_max = p.getAABB(body_id)
        return np.array(aabb_min, dtype=float), np.array(aabb_max, dtype=float)

    def get_cube_top_center(self, body_id: int):
        aabb_min, aabb_max = self.get_body_aabb(body_id)
        center_x = 0.5 * (aabb_min[0] + aabb_max[0])
        center_y = 0.5 * (aabb_min[1] + aabb_max[1])
        top_z = aabb_max[2]
        return np.array([center_x, center_y, top_z], dtype=float)

    def get_min_finger_to_body_distance(self, body_id: int) -> float:
        cube_top_center = self.get_cube_top_center(body_id)
        left_pos, right_pos = self.get_finger_tip_positions()

        left_dist = float(np.linalg.norm(left_pos - cube_top_center))
        right_dist = float(np.linalg.norm(right_pos - cube_top_center))
        return min(left_dist, right_dist)

    def get_finger_midpoint(self) -> np.ndarray:
        left_pos, right_pos = self.get_finger_tip_positions()
        return 0.5 * (left_pos + right_pos)

    def get_pinch_point_offset(self) -> np.ndarray:
        """
        Returns the vector from EE target frame (link 11) to the actual
        finger midpoint in world coordinates for the current wrist pose.
        """
        ee_pos, _ = self.get_ee_pose()
        finger_mid = self.get_finger_midpoint()
        return finger_mid - ee_pos

    def get_target_ee_from_desired_pinch_point(self, desired_pinch_world: np.ndarray) -> np.ndarray:
        """
        Convert a desired pinch-point world position into an EE-frame target
        by compensating for the current offset between link 11 and the
        finger midpoint.
        """
        desired_pinch_world = np.asarray(desired_pinch_world, dtype=float)

        if config.USE_MANUAL_PINCH_OFFSET:
            pinch_offset = np.array(
                [
                    config.MANUAL_PINCH_OFFSET_X,
                    config.MANUAL_PINCH_OFFSET_Y,
                    config.MANUAL_PINCH_OFFSET_Z,
                ],
                dtype=float,
            )
        else:
            pinch_offset = self.get_pinch_point_offset()

        return desired_pinch_world - pinch_offset

    def print_pinch_point_debug(self):
        ee_pos, _ = self.get_ee_pose()
        left_pos, right_pos = self.get_finger_tip_positions()
        finger_mid = self.get_finger_midpoint()
        pinch_offset = self.get_pinch_point_offset()

        print("\n=== Pinch Point Debug ===")
        print("ee_pos:", ee_pos)
        print("left_finger_pos:", left_pos)
        print("right_finger_pos:", right_pos)
        print("finger_midpoint:", finger_mid)
        print("pinch_offset (finger_mid - ee_pos):", pinch_offset)

    def has_attached_object(self) -> bool:
        return self.attached_body_id is not None