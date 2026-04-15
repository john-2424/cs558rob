import numpy as np
import pybullet as p
import gymnasium
from gymnasium import spaces

from src import config
from src.sim.env import BulletEnv
from src.sim.robot import PandaRobotDemo
from src.sim.state import (
    get_body_bottom_z,
    get_body_pose,
    get_body_top_z,
    quat_to_euler,
)
from src.rl.perturbation import perturb_cube_pose
from src.rl.reward import compute_reward
from src.rl.residual_policy import ResidualActionWrapper
from src.trajectory.joint_trajectory import interpolate_joint_trajectory

# Phase indices
PHASE_PRE_GRASP = 0
PHASE_GRASP_DESCEND = 1
PHASE_LIFT = 2
NUM_RL_PHASES = 3

OBS_DIM = 37
ACT_DIM = 7


class PandaGraspEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False, mode="hybrid", perturb_xy_range=None,
                 perturb_yaw_range=None, render_mode=None, verbose_episodes=False):
        super().__init__()

        self.gui = gui
        self.mode = mode
        self.perturb_xy_range = perturb_xy_range or config.PERTURB_XY_RANGE
        self.perturb_yaw_range = perturb_yaw_range or config.PERTURB_YAW_RANGE
        self.verbose_episodes = verbose_episodes

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32,
        )

        self.action_wrapper = ResidualActionWrapper(mode=mode)

        # PyBullet persistent state
        self._env = None
        self._robot = None
        self._objects = None
        self._table_top_z = None
        self._rng = np.random.default_rng()

        # Episode state
        self._phase = PHASE_PRE_GRASP
        self._waypoints = []
        self._waypoint_idx = 0
        self._step_count = 0
        self._prev_ee_cube_dist = None
        self._grasp_bonus_given = False
        self._cube_nominal_pos = None
        self._cube_nominal_orn_euler = None
        self._grasp_orn = None

        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        self._env = BulletEnv(gui=self.gui)
        self._env.connect()
        self._objects = self._env.load_scene()

        # Correct table to plane
        table_bottom_z = get_body_bottom_z(self._objects.table_id)
        delta_z = 0.0 - table_bottom_z + config.TABLE_PLANE_CLEARANCE
        pos, orn = p.getBasePositionAndOrientation(self._objects.table_id)
        p.resetBasePositionAndOrientation(
            self._objects.table_id,
            [pos[0], pos[1], pos[2] + delta_z],
            orn,
        )

        # Load robot
        self._robot = PandaRobotDemo()
        self._robot.load(
            base_position=config.PANDA_BASE_POS,
            base_orientation_euler=config.PANDA_BASE_ORN_EULER,
        )

        self._table_top_z = get_body_top_z(self._objects.table_id)

        # Store nominal cube pose
        self._cube_nominal_pos = list(config.CUBE_BASE_POS)
        self._cube_nominal_pos[2] = self._table_top_z + config.CUBE_TABLE_CLEARANCE
        # Correct: place cube bottom at table top
        self._place_cube_at_nominal()
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        self._cube_nominal_pos = cube_pos.tolist()
        self._cube_nominal_orn_euler = list(config.CUBE_BASE_ORN_EULER)

        self._initialized = True

    def _place_cube_at_nominal(self):
        target_xy = config.CUBE_BASE_POS[:2]
        current_pos, current_orn = p.getBasePositionAndOrientation(self._objects.cube_id)
        body_bottom_z = get_body_bottom_z(self._objects.cube_id)
        delta_z = self._table_top_z - body_bottom_z + config.CUBE_TABLE_CLEARANCE
        new_pos = [target_xy[0], target_xy[1], current_pos[2] + delta_z]
        p.resetBasePositionAndOrientation(self._objects.cube_id, new_pos, current_orn)

    def _compute_grasp_targets(self):
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        _, cube_aabb_max = p.getAABB(self._objects.cube_id)
        cube_top_z = cube_aabb_max[2]

        grasp_euler = np.array(config.GRASP_EE_EULER, dtype=float)
        self._grasp_orn = p.getQuaternionFromEuler(grasp_euler.tolist())

        grasp_bias = np.array([
            config.GRASP_TARGET_BIAS_X,
            config.GRASP_TARGET_BIAS_Y,
            config.GRASP_TARGET_BIAS_Z,
        ], dtype=float)

        pre_grasp_pos = np.array([
            cube_pos[0], cube_pos[1],
            cube_top_z + config.PICK_HOVER_Z_OFFSET,
        ], dtype=float) + grasp_bias

        grasp_pos = np.array([
            cube_pos[0], cube_pos[1],
            cube_top_z + config.PICK_DESCEND_Z_OFFSET,
        ], dtype=float) + grasp_bias

        if config.APPLY_GRASP_BIAS_TO_LIFT:
            lift_pos = np.array([
                cube_pos[0], cube_pos[1],
                cube_top_z + config.PICK_HOVER_Z_OFFSET,
            ], dtype=float) + grasp_bias
        else:
            lift_pos = np.array([
                cube_pos[0], cube_pos[1],
                cube_top_z + config.PICK_HOVER_Z_OFFSET,
            ], dtype=float)

        # Solve IK for each target
        q_pre_grasp = self._robot.solve_ik(pre_grasp_pos, self._grasp_orn)
        q_grasp = self._robot.solve_ik(grasp_pos, self._grasp_orn)
        q_lift = self._robot.solve_ik(lift_pos, self._grasp_orn)

        return q_pre_grasp, q_grasp, q_lift

    def _build_phase_waypoints(self, q_start, q_goal):
        return interpolate_joint_trajectory(
            q_start=q_start,
            q_goal=q_goal,
            num_waypoints=config.TRAJ_NUM_WAYPOINTS,
        )

    def _reset_robot(self):
        self._robot.reset_home()
        self._robot.hold_home_pose()

        # Detach any attached object
        if self._robot.has_attached_object():
            self._robot.detach_object()

        # Settle
        for _ in range(config.SETTLE_STEPS):
            self._robot.hold_home_pose()
            self._env.step(sleep=False)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._lazy_init()

        # Reset robot to home
        self._reset_robot()

        # Reset cube to nominal
        cube_orn = p.getQuaternionFromEuler(self._cube_nominal_orn_euler)
        p.resetBasePositionAndOrientation(
            self._objects.cube_id, self._cube_nominal_pos, cube_orn,
        )

        for _ in range(20):
            self._env.step(sleep=False)

        # Plan from the nominal cube pose: the classical pipeline is blind
        # to the upcoming perturbation, so its waypoints track a phantom
        # nominal target. Residual RL's job is to correct for that offset.
        q_pre_grasp, q_grasp, q_lift = self._compute_grasp_targets()

        if self.perturb_xy_range > 0:
            perturb_cube_pose(
                cube_id=self._objects.cube_id,
                nominal_pos=self._cube_nominal_pos,
                nominal_orn_euler=self._cube_nominal_orn_euler,
                rng=self._rng,
                xy_range=self.perturb_xy_range,
                yaw_range=self.perturb_yaw_range,
            )

            for _ in range(20):
                self._env.step(sleep=False)
        q_home = self._robot.home_joints.copy()

        # Build all phase waypoints
        self._phase_targets = {
            PHASE_PRE_GRASP: (q_home, q_pre_grasp),
            PHASE_GRASP_DESCEND: (q_pre_grasp, q_grasp),
            PHASE_LIFT: (q_grasp, q_lift),
        }
        self._q_grasp_target = q_grasp
        self._q_lift_target = q_lift

        # Start with pre-grasp phase
        self._phase = PHASE_PRE_GRASP
        q_start, q_goal = self._phase_targets[self._phase]
        self._waypoints = self._build_phase_waypoints(q_start, q_goal)
        self._waypoint_idx = 0
        self._waypoint_step_count = 0
        self._step_count = 0
        self._grasp_bonus_given = False

        # Episode diagnostics
        self._max_phase_reached = self._phase
        self._max_waypoint_per_phase = {PHASE_PRE_GRASP: 0, PHASE_GRASP_DESCEND: 0, PHASE_LIFT: 0}
        self._forced_waypoint_advances = 0
        self._grasp_attempted = False
        self._grasp_attached = False
        self._summary_printed = False

        # Initial EE-cube distance
        ee_pos, _ = self._robot.get_ee_pose()
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        self._prev_ee_cube_dist = float(np.linalg.norm(ee_pos - cube_pos))

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        q = self._robot.get_joint_positions()
        qd = self._robot.get_joint_velocities()
        ee_pos, ee_orn = self._robot.get_ee_pose()
        ee_euler = quat_to_euler(ee_orn)
        cube_pos, cube_orn = get_body_pose(self._objects.cube_id)
        cube_euler = quat_to_euler(cube_orn)
        ee_to_cube = cube_pos - ee_pos

        # PD nominal for current waypoint
        if self._waypoint_idx < len(self._waypoints):
            q_target = self._waypoints[self._waypoint_idx]
        else:
            q_target = q

        qd_pd = self._robot.pd_controller.compute_velocity_command(
            q_des=q_target, q=q, qd_des=np.zeros(7), qd=qd,
        )

        phase_indicator = np.array([float(self._phase)], dtype=np.float32)

        obs = np.concatenate([
            q.astype(np.float32),           # 7
            qd.astype(np.float32),          # 7
            ee_pos.astype(np.float32),      # 3
            ee_euler.astype(np.float32),    # 3
            cube_pos.astype(np.float32),    # 3
            cube_euler.astype(np.float32),  # 3
            ee_to_cube.astype(np.float32),  # 3
            qd_pd.astype(np.float32),       # 7
            phase_indicator,                # 1
        ])
        return obs

    def _advance_phase(self):
        if self._phase == PHASE_PRE_GRASP:
            # Transition to grasp descend
            self._phase = PHASE_GRASP_DESCEND
            q_start = self._robot.get_joint_positions()
            _, q_goal = self._phase_targets[PHASE_GRASP_DESCEND]
            self._waypoints = self._build_phase_waypoints(q_start, q_goal)
            self._waypoint_idx = 0
            self._waypoint_step_count = 0
            return False

        elif self._phase == PHASE_GRASP_DESCEND:
            # Auto: close gripper, validate, attach, then transition to lift
            success = self._auto_grasp_and_attach()
            if not success:
                return True  # episode over, failure

            self._phase = PHASE_LIFT
            q_start = self._robot.get_joint_positions()
            _, q_goal = self._phase_targets[PHASE_LIFT]
            self._waypoints = self._build_phase_waypoints(q_start, q_goal)
            self._waypoint_idx = 0
            self._waypoint_step_count = 0
            return False

        elif self._phase == PHASE_LIFT:
            # Lift complete -> check success
            return True

        return False

    def _auto_grasp_and_attach(self):
        self._grasp_attempted = True

        # Inline sim steps below are NOT counted toward _step_count: that
        # counter measures RL transitions for truncation budget. Counting
        # inline steps here exhausted the episode budget before LIFT could run.

        # Close gripper
        for _ in range(config.GRIPPER_SETTLE_STEPS):
            self._robot.close_gripper()
            self._env.step(sleep=False)

        # Validate grasp
        ready, _ = self._robot.is_grasp_ready(self._objects.cube_id)

        if not ready:
            # Retry deeper descend
            q_retry = self._robot.get_joint_positions()
            ee_pos, _ = self._robot.get_ee_pose()
            retry_pos = np.array(ee_pos, dtype=float)
            retry_pos[2] -= config.GRASP_DESCEND_RETRY_DELTA_Z
            q_retry_target = self._robot.solve_ik(retry_pos, self._grasp_orn)

            waypoints = self._build_phase_waypoints(q_retry, q_retry_target)
            for wp in waypoints:
                for _ in range(config.WAYPOINT_MAX_STEPS):
                    self._robot.command_arm_and_gripper(
                        wp, gripper_width=config.GRIPPER_OPEN_WIDTH,
                    )
                    self._env.step(sleep=False)
                    if self._robot.is_at_joint_target(wp, tol=config.WAYPOINT_TOL):
                        break

            for _ in range(config.GRIPPER_SETTLE_STEPS):
                self._robot.close_gripper()
                self._env.step(sleep=False)

            ready, _ = self._robot.is_grasp_ready(self._objects.cube_id)

        if ready:
            self._robot.attach_object(self._objects.cube_id)
            self._grasp_attached = True
            self._grasp_bonus_given = False  # will be given in reward
            return True

        return False

    def _check_cube_fallen(self):
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        return cube_pos[2] < self._table_top_z - 0.05

    def _check_cube_lifted(self):
        if not self._robot.has_attached_object():
            return False
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        return cube_pos[2] > self._table_top_z + 0.10

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        self._step_count += 1

        # Get current waypoint target
        if self._waypoint_idx < len(self._waypoints):
            q_target = self._waypoints[self._waypoint_idx]
        else:
            q_target = self._robot.get_joint_positions()

        # Compute PD nominal
        q = self._robot.get_joint_positions()
        qd = self._robot.get_joint_velocities()
        qd_pd = self._robot.pd_controller.compute_velocity_command(
            q_des=q_target, q=q, qd_des=np.zeros(7), qd=qd,
        )

        # Apply action through residual wrapper
        qd_total = self.action_wrapper.compute_action(qd_pd, action)
        qd_residual = self.action_wrapper.get_residual(action)

        # Apply velocity command + maintain gripper
        forces = self._robot.arm_max_forces
        gripper_width = config.GRIPPER_OPEN_WIDTH
        if self._phase == PHASE_LIFT:
            gripper_width = config.GRIPPER_CLOSED_WIDTH

        p.setJointMotorControlArray(
            bodyUniqueId=self._robot.robot_id,
            jointIndices=self._robot.arm_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=qd_total.tolist(),
            forces=forces,
        )
        self._robot.command_gripper(gripper_width)

        # Step simulation
        for _ in range(config.RL_SIM_SUBSTEPS):
            p.stepSimulation()

        # Check waypoint convergence (or force-advance on timeout)
        if self._waypoint_idx < len(self._waypoints):
            tol = config.WAYPOINT_TOL
            if self._phase == PHASE_GRASP_DESCEND:
                tol = config.GRASP_DESCEND_WAYPOINT_TOL
            self._waypoint_step_count += 1
            converged = self._robot.is_at_joint_target(q_target, tol=tol)
            timed_out = self._waypoint_step_count >= config.RL_MAX_STEPS_PER_WAYPOINT
            if converged or timed_out:
                if timed_out and not converged:
                    self._forced_waypoint_advances += 1
                self._waypoint_idx += 1
                self._waypoint_step_count = 0

        # Track per-phase max waypoint progress (diagnostic)
        self._max_waypoint_per_phase[self._phase] = max(
            self._max_waypoint_per_phase[self._phase], self._waypoint_idx,
        )

        # Check if phase trajectory is complete
        terminated = False
        episode_done = False
        if self._waypoint_idx >= len(self._waypoints):
            episode_done = self._advance_phase()

        # Track max phase reached (after advance_phase may have bumped it)
        self._max_phase_reached = max(self._max_phase_reached, self._phase)

        # Compute reward
        ee_pos, _ = self._robot.get_ee_pose()
        cube_pos, _ = get_body_pose(self._objects.cube_id)

        grasp_ready = False
        if self._phase == PHASE_LIFT and self._robot.has_attached_object():
            grasp_ready = True

        cube_lifted = self._check_cube_lifted()
        cube_fallen = self._check_cube_fallen()

        reward, self._prev_ee_cube_dist, reward_info = compute_reward(
            ee_pos=ee_pos,
            cube_pos=cube_pos,
            qd_residual=qd_residual,
            prev_ee_cube_dist=self._prev_ee_cube_dist,
            grasp_ready=grasp_ready,
            cube_lifted=cube_lifted,
            cube_fallen=cube_fallen,
            grasp_bonus_given=self._grasp_bonus_given,
        )

        if grasp_ready and not self._grasp_bonus_given:
            self._grasp_bonus_given = True

        # Termination
        truncated = self._step_count >= config.RL_MAX_EPISODE_STEPS

        if cube_lifted:
            terminated = True
        elif cube_fallen:
            terminated = True
        elif episode_done:
            # _advance_phase returns True on LIFT completion OR on failed grasp
            # in GRASP_DESCEND. Either way, the episode is over -- otherwise
            # subsequent steps would re-trigger the gripper-close/retry loop
            # every step because waypoints are exhausted.
            terminated = True

        info = {
            "success": cube_lifted,
            "phase": self._phase,
            "step_count": self._step_count,
            **reward_info,
        }

        if terminated or truncated:
            num_wp = max(1, len(self._waypoints))
            cube_dz = float(cube_pos[2] - self._table_top_z)
            ee_cube_dist = float(np.linalg.norm(ee_pos - cube_pos))
            info.update({
                "ep_max_phase": int(self._max_phase_reached),
                "ep_end_phase": int(self._phase),
                "ep_wp_pregrasp": int(self._max_waypoint_per_phase[PHASE_PRE_GRASP]),
                "ep_wp_graspdescend": int(self._max_waypoint_per_phase[PHASE_GRASP_DESCEND]),
                "ep_wp_lift": int(self._max_waypoint_per_phase[PHASE_LIFT]),
                "ep_wp_total": int(num_wp),
                "ep_forced_wp_advances": int(self._forced_waypoint_advances),
                "ep_grasp_attempted": bool(self._grasp_attempted),
                "ep_grasp_attached": bool(self._grasp_attached),
                "ep_cube_dz": cube_dz,
                "ep_ee_cube_dist": ee_cube_dist,
                "ep_cube_fallen": bool(cube_fallen),
                "ep_cube_lifted": bool(cube_lifted),
            })
            if self.verbose_episodes and not self._summary_printed:
                self._summary_printed = True
                phase_names = {0: "pre_grasp", 1: "grasp_descend", 2: "lift"}
                end_reason = (
                    "lifted" if cube_lifted else
                    "fallen" if cube_fallen else
                    "truncated" if truncated else
                    "phase_done"
                )
                print(
                    f"[ep] end={end_reason:>9s} | max_phase={phase_names[self._max_phase_reached]:<13s} "
                    f"| wp(p/g/l)={self._max_waypoint_per_phase[PHASE_PRE_GRASP]:d}/"
                    f"{self._max_waypoint_per_phase[PHASE_GRASP_DESCEND]:d}/"
                    f"{self._max_waypoint_per_phase[PHASE_LIFT]:d} of {num_wp:d} "
                    f"| forced_wp={self._forced_waypoint_advances:d} "
                    f"| grasp(try/attach)={int(self._grasp_attempted)}/{int(self._grasp_attached)} "
                    f"| cube_dz={cube_dz:+.3f} | ee_cube={ee_cube_dist:.3f} "
                    f"| steps={self._step_count:d}"
                )

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.disconnect()
            self._env = None
            self._initialized = False
