import os

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
from src.rl.grasp_gate import FEATURE_KEYS as _GRASP_GATE_FEATURE_KEYS
from src.rl.grasp_gate import LearnedGraspGate
from src.trajectory.joint_trajectory import interpolate_joint_trajectory

# Phase indices
PHASE_PRE_GRASP = 0
PHASE_GRASP_DESCEND = 1
PHASE_LIFT = 2
NUM_RL_PHASES = 3

OBS_DIM = 41  # added waypoint_progress (1)
ACT_DIM = 7
# Total action dim including the optional confidence-gate channel.
# When config.RESIDUAL_USE_GATE is True, the actor emits an 8th channel
# that gates the residual magnitude. Action space and policy net adapt.
def _act_dim_total():
    return ACT_DIM + (1 if bool(getattr(config, "RESIDUAL_USE_GATE", False)) else 0)

# Per-worker grasp-geometry dump counter. Each worker process gets its own
# copy; once this reaches GRASP_DIAG_DUMP_LIMIT, dumps stop. Dumps are
# appended to results/m2/grasp_diag.txt so they survive past terminal
# scrollback and are synced with the rest of the run artifacts.
_GRASP_DIAG_DUMPS = 0
GRASP_DIAG_DUMP_LIMIT = 200
GRASP_DIAG_PATH = os.path.join(config.M2_RESULTS_DIR, "grasp_diag.txt")

# Fixed normalization scales so all observation components are roughly [-1, 1].
_MAX_JOINT_VEL = np.asarray(config.PD_MAX_JOINT_VEL, dtype=np.float32)
_WORKSPACE_CENTER = np.array([0.5, 0.0, 0.4], dtype=np.float32)
_WORKSPACE_HALF = np.array([0.5, 0.5, 0.4], dtype=np.float32)
_PERTURB_SCALE = max(config.PERTURB_XY_RANGE, 0.01)


def _normalize_obs(q, qd, ee_pos, ee_euler, cube_pos, cube_euler,
                   ee_to_cube, qd_pd, phase_indicator, perturb_offset,
                   wp_progress):
    """Fixed-scale normalization — all components mapped to roughly [-1, 1]."""
    return np.concatenate([
        (q / np.pi).astype(np.float32),                           # 7
        (qd / _MAX_JOINT_VEL).astype(np.float32),                # 7
        ((ee_pos - _WORKSPACE_CENTER) / _WORKSPACE_HALF).astype(np.float32),   # 3
        (ee_euler / np.pi).astype(np.float32),                    # 3
        ((cube_pos - _WORKSPACE_CENTER) / _WORKSPACE_HALF).astype(np.float32), # 3
        (cube_euler / np.pi).astype(np.float32),                  # 3
        (ee_to_cube / _WORKSPACE_HALF).astype(np.float32),        # 3
        (qd_pd / _MAX_JOINT_VEL).astype(np.float32),             # 7
        (phase_indicator / NUM_RL_PHASES).astype(np.float32),     # 1
        (perturb_offset / _PERTURB_SCALE).astype(np.float32),    # 3
        wp_progress.astype(np.float32),                           # 1  (already 0-1)
    ])


class PandaGraspEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gui=False, mode="hybrid", perturb_xy_range=None,
                 perturb_yaw_range=None, perturb_z_range=None,
                 perturb_pitch_range=None, perturb_roll_range=None,
                 render_mode=None, verbose_episodes=False,
                 curriculum=False, trace=None, trace_dir=None,
                 enable_grasp_retry=False):
        super().__init__()

        self.gui = gui
        # Retry with a deeper descend when the first grasp check fails.
        # Off during training (policy must learn precise placement). On during
        # eval so planner_only and hybrid face the same physics as the M1 demo,
        # which uses retry in pick_place.py / pick_place_residual.py.
        self.enable_grasp_retry = bool(enable_grasp_retry)
        self.mode = mode
        self.perturb_xy_range = (
            config.PERTURB_XY_RANGE if perturb_xy_range is None else perturb_xy_range
        )
        self.perturb_yaw_range = (
            config.PERTURB_YAW_RANGE if perturb_yaw_range is None else perturb_yaw_range
        )
        self.perturb_z_range = (
            float(getattr(config, "PERTURB_Z_RANGE", 0.0))
            if perturb_z_range is None else perturb_z_range
        )
        self.perturb_pitch_range = (
            float(getattr(config, "PERTURB_PITCH_RANGE", 0.0))
            if perturb_pitch_range is None else perturb_pitch_range
        )
        self.perturb_roll_range = (
            float(getattr(config, "PERTURB_ROLL_RANGE", 0.0))
            if perturb_roll_range is None else perturb_roll_range
        )
        # Curriculum: when True, each env scales its effective perturbation
        # range by frac = clip((episode_count - warmup) / ramp_episodes, 0, 1)
        # so early episodes train at sub-max perturbation. After warmup +
        # ramp_episodes, frac=1 and the env runs at full PERTURB_*_RANGE.
        # ramp_episodes = 0 disables the ramp (full range from episode 1).
        self.curriculum = curriculum
        self.curriculum_warmup_episodes = int(
            getattr(config, "CURRICULUM_WARMUP_EPISODES", 0)
        )
        self.curriculum_ramp_episodes = int(
            getattr(config, "CURRICULUM_RAMP_EPISODES", 0)
        )
        self._episode_count = 0
        self.verbose_episodes = verbose_episodes

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_act_dim_total(),), dtype=np.float32,
        )
        self._use_gate = bool(getattr(config, "RESIDUAL_USE_GATE", False))

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
        self._prev_ee_target_dist = None
        self._grasp_bonus_given = False
        self._lenient_bonus_given = False
        self._attempt_fired = False
        self._attempt_lenient_ready = False
        self._attempt_bracket_score = None
        self._attempt_worst_tip_above_top = None
        self._cube_nominal_pos = None
        self._cube_nominal_orn_euler = None
        self._grasp_orn = None
        self._grasp_target_cartesian = None
        self._perturb_offset = np.zeros(3, dtype=np.float32)
        self._last_gate = 1.0
        self._last_residual_active = False

        # Learned grasp-gate setup. Mode "heuristic" means classifier is
        # never consulted. Other modes load the trained classifier; if the
        # checkpoint is missing the gate marks itself unloaded and the env
        # falls back to heuristic.
        self._grasp_gate_mode = str(getattr(config, "GRASP_GATE_MODE", "heuristic"))
        self._grasp_gate_log = bool(getattr(config, "GRASP_GATE_LOG", True))
        self._grasp_gate_dataset_path = str(
            getattr(config, "GRASP_GATE_DATASET_PATH",
                    "results/m3/grasp_dataset.jsonl")
        )
        self._learned_grasp_gate = None
        if self._grasp_gate_mode != "heuristic":
            self._learned_grasp_gate = LearnedGraspGate()
            if not self._learned_grasp_gate.loaded:
                print(f"[grasp_gate] mode={self._grasp_gate_mode} but no "
                      f"checkpoint loaded — falling back to heuristic")
        # Per-episode attempt buffer; flushed at episode end with the
        # final outcome label.
        self._grasp_attempt_log: list = []

        self._initialized = False

        # Diagnostic trace (per-step CSV + per-episode JSON summary).
        self.trace = (
            bool(getattr(config, "RL_TRACE_EPISODE", False)) if trace is None else bool(trace)
        )
        self.trace_dir = (
            trace_dir if trace_dir is not None
            else getattr(config, "RL_TRACE_OUTPUT_DIR", "results/m2/traces")
        )
        self._trace_rows = []
        self._trace_grasp_debug = None
        self._trace_episode_idx = 0

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

        # Cartesian grasp target is the reward attractor and the early-grasp
        # trigger reference. A physical, graspable pose — not cube center.
        self._grasp_target_cartesian = np.asarray(grasp_pos, dtype=float)

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

        self._episode_count += 1
        in_warmup = (self.curriculum
                     and self.curriculum_warmup_episodes > 0
                     and self._episode_count <= self.curriculum_warmup_episodes)
        if in_warmup:
            effective_xy = 0.0
            effective_yaw = 0.0
            effective_z = 0.0
            effective_pitch = 0.0
            effective_roll = 0.0
        else:
            # Linear ramp: scale all axes by frac in [0, 1] over the first
            # ramp_episodes after warmup. ramp=0 short-circuits to frac=1
            # and matches the M2 "full range from episode 1" behaviour.
            if self.curriculum and self.curriculum_ramp_episodes > 0:
                ramp_step = self._episode_count - self.curriculum_warmup_episodes
                frac = max(0.0, min(1.0, ramp_step / self.curriculum_ramp_episodes))
            else:
                frac = 1.0
            effective_xy = self.perturb_xy_range * frac
            effective_yaw = self.perturb_yaw_range * frac
            effective_z = self.perturb_z_range * frac
            effective_pitch = self.perturb_pitch_range * frac
            effective_roll = self.perturb_roll_range * frac

        if (effective_xy > 0 or effective_z > 0 or effective_yaw > 0
                or effective_pitch > 0 or effective_roll > 0):
            perturb_cube_pose(
                cube_id=self._objects.cube_id,
                nominal_pos=self._cube_nominal_pos,
                nominal_orn_euler=self._cube_nominal_orn_euler,
                rng=self._rng,
                xy_range=effective_xy,
                yaw_range=effective_yaw,
                z_range=effective_z,
                pitch_range=effective_pitch,
                roll_range=effective_roll,
            )

            for _ in range(20):
                self._env.step(sleep=False)

        # Store perturbation offset for observation (option E).
        perturbed_pos, _ = get_body_pose(self._objects.cube_id)
        self._perturb_offset = perturbed_pos - np.array(self._cube_nominal_pos)

        # Reward attractor must follow the PERTURBED cube, not the nominal one
        # the waypoints were built from. Without this, r_approach and the
        # milestone cascade reward convergence to a phantom point, and the
        # policy has no gradient to actually correct for the perturbation.
        _, perturbed_aabb_max = p.getAABB(self._objects.cube_id)
        perturbed_top_z = perturbed_aabb_max[2]
        grasp_bias = np.array([
            config.GRASP_TARGET_BIAS_X,
            config.GRASP_TARGET_BIAS_Y,
            config.GRASP_TARGET_BIAS_Z,
        ], dtype=float)
        self._grasp_target_cartesian = np.array([
            perturbed_pos[0], perturbed_pos[1],
            perturbed_top_z + config.PICK_DESCEND_Z_OFFSET,
        ], dtype=float) + grasp_bias

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
        self._lenient_bonus_given = False
        self._attempt_fired = False
        self._attempt_lenient_ready = False
        self._attempt_bracket_score = None
        self._attempt_worst_tip_above_top = None
        self._milestones_given = {"m08": False, "m05": False, "m03": False}

        # Episode diagnostics
        self._max_phase_reached = self._phase
        self._max_waypoint_per_phase = {PHASE_PRE_GRASP: 0, PHASE_GRASP_DESCEND: 0, PHASE_LIFT: 0}
        self._forced_waypoint_advances = 0
        self._grasp_attempted = False
        self._grasp_attached = False
        self._grasp_attempt_log = []
        self._summary_printed = False

        # Initial EE-to-grasp-target distance (reward attractor)
        ee_pos, _ = self._robot.get_ee_pose()
        cube_pos, _ = get_body_pose(self._objects.cube_id)
        self._prev_ee_target_dist = float(
            np.linalg.norm(ee_pos - self._grasp_target_cartesian)
        )

        if self.trace:
            self._trace_rows = []
            self._trace_grasp_debug = None
            self._trace_episode_idx += 1

        obs = self._get_obs()
        return obs, self._diag_info()

    def _diag_info(self):
        # Per-step diagnostic fields. Always emitted (not just terminal) so
        # TorchRL's info-dict reader can register them as proper spec keys
        # and train.py can aggregate attach/lift/wp-progress across the batch.
        cube_pos_now, _ = get_body_pose(self._objects.cube_id)
        cube_dz_now = float(cube_pos_now[2] - self._table_top_z)
        # TorchRL's default_info_dict_reader calls `val.dtype` on each value,
        # so every entry must be a numpy scalar/array — not a Python scalar.
        return {
            "diag_phase": np.int64(self._phase),
            "diag_max_phase": np.int64(self._max_phase_reached),
            "diag_grasp_attempted": np.bool_(self._grasp_attempted),
            "diag_grasp_attached": np.bool_(self._grasp_attached),
            "diag_cube_lifted": np.bool_(self._check_cube_lifted()),
            "diag_cube_fallen": np.bool_(self._check_cube_fallen()),
            "diag_cube_dz": np.float32(cube_dz_now),
            "diag_wp_pregrasp": np.int64(self._max_waypoint_per_phase[PHASE_PRE_GRASP]),
            "diag_wp_graspdescend": np.int64(self._max_waypoint_per_phase[PHASE_GRASP_DESCEND]),
            "diag_wp_lift": np.int64(self._max_waypoint_per_phase[PHASE_LIFT]),
            "diag_forced_wp_advances": np.int64(self._forced_waypoint_advances),
        }

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
        wp_progress = np.array([
            float(self._waypoint_idx) / max(1, len(self._waypoints))
        ], dtype=np.float32)

        obs = _normalize_obs(
            q, qd, ee_pos, ee_euler, cube_pos, cube_euler,
            ee_to_cube, qd_pd, phase_indicator, self._perturb_offset,
            wp_progress,
        )
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

    def _cartesian_settle(self, target_pos, open_gripper=True):
        """Drive PD toward IK(target_pos, grasp_orn) until EE is within
        GRASP_DESCEND_CARTESIAN_TOL of target_pos or budget expires."""
        target_pos = np.asarray(target_pos, dtype=float)
        q_settle = self._robot.solve_ik(target_pos, self._grasp_orn)
        max_vel = np.asarray(config.PD_MAX_JOINT_VEL, dtype=float)
        tol = config.GRASP_DESCEND_CARTESIAN_TOL
        budget = config.GRASP_DESCEND_CARTESIAN_SETTLE_STEPS
        for _ in range(budget):
            ee_pos, _ = self._robot.get_ee_pose()
            if float(np.linalg.norm(ee_pos - target_pos)) < tol:
                break
            q_now = self._robot.get_joint_positions()
            qd_now = self._robot.get_joint_velocities()
            qd_cmd = self._robot.pd_controller.compute_velocity_command(
                q_des=q_settle, q=q_now, qd_des=np.zeros(7), qd=qd_now,
            )
            qd_cmd = np.clip(qd_cmd, -max_vel, max_vel)
            p.setJointMotorControlArray(
                bodyUniqueId=self._robot.robot_id,
                jointIndices=self._robot.arm_joint_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=qd_cmd.tolist(),
                forces=self._robot.arm_max_forces,
            )
            if open_gripper:
                self._robot.command_gripper(config.GRIPPER_OPEN_WIDTH)
            for _ in range(config.RL_SIM_SUBSTEPS):
                p.stepSimulation()

    def _record_grasp_attempt(self, grasp_debug: dict, attempt_idx: int):
        """Append a feature snapshot of one grasp attempt to the per-episode
        buffer. Outcome label is filled in at episode end (when lifted /
        fallen are decided)."""
        if not self._grasp_gate_log:
            return
        row = {k: grasp_debug.get(k, 0.0) for k in _GRASP_GATE_FEATURE_KEYS}
        # Coerce booleans to JSON-friendly bools (numpy bool_ is not).
        for k, v in list(row.items()):
            if isinstance(v, (np.bool_,)):
                row[k] = bool(v)
        row["ready_heuristic"] = bool(grasp_debug.get("ready", False))
        row["attempt_idx"] = int(attempt_idx)
        row["perturb_xy"] = float(self.perturb_xy_range)
        row["perturb_yaw"] = float(self.perturb_yaw_range)
        self._grasp_attempt_log.append(row)

    def _flush_grasp_attempt_log(self, lifted: bool, fallen: bool):
        """Tag buffered grasp attempts with final outcome and append to JSONL."""
        import json as _json
        path = self._grasp_gate_dataset_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for row in self._grasp_attempt_log:
                    row["lifted"] = bool(lifted)
                    row["fallen"] = bool(fallen)
                    f.write(_json.dumps(row) + "\n")
        except OSError:
            pass
        self._grasp_attempt_log = []

    def _apply_learned_gate(self, heuristic_ready: bool, grasp_debug: dict) -> bool:
        """Combine heuristic decision with learned classifier (if loaded).

        learned_filter: classifier acts as additional gate on top of heuristic.
                        Returns heuristic_ready AND classifier_prob > thresh.
        learned_only:   classifier alone decides; heuristic ignored.
        Other modes (or unloaded classifier): pass heuristic through unchanged.
        """
        gate = self._learned_grasp_gate
        if gate is None or not gate.loaded:
            return heuristic_ready
        prob = gate.predict_prob(grasp_debug)
        passes_classifier = prob >= gate.threshold
        if self._grasp_gate_mode == "learned_only":
            return passes_classifier
        if self._grasp_gate_mode == "learned_filter":
            return bool(heuristic_ready) and passes_classifier
        return heuristic_ready

    def _auto_grasp_and_attach(self):
        self._grasp_attempted = True

        # Fix A: Cartesian settle before closing. Joint-space waypoint
        # tolerance can converge while EE is Cartesian-high of the grasp
        # target (~14mm in planner-nominal). Drive PD toward the target
        # until Cartesian distance is within tolerance. Skipped in rl_only.
        if (self.mode != "rl_only"
                and self._grasp_target_cartesian is not None):
            self._cartesian_settle(self._grasp_target_cartesian, open_gripper=True)

        # Close gripper (inline sim steps not counted toward _step_count)
        for _ in range(config.GRIPPER_SETTLE_STEPS):
            self._robot.close_gripper()
            self._env.step(sleep=False)

        # Single-attempt grasp validation — no retry. RL must learn to
        # position precisely enough for the physics contact check to pass.
        ready, grasp_debug = self._robot.is_grasp_ready(self._objects.cube_id)
        if self.trace:
            self._trace_grasp_debug = grasp_debug
        self._record_grasp_attempt(grasp_debug, attempt_idx=0)
        ready = self._apply_learned_gate(ready, grasp_debug)

        # Dump the first GRASP_DIAG_DUMP_LIMIT grasp geometries per worker to
        # results/m2/grasp_diag.txt. Shows local-frame fingertip coords and
        # the face-check errs so we can pick tolerances from actual data.
        # Appending from multiple workers — short writes are atomic on the
        # filesystems we care about.
        global _GRASP_DIAG_DUMPS
        if _GRASP_DIAG_DUMPS < GRASP_DIAG_DUMP_LIMIT:
            _GRASP_DIAG_DUMPS += 1
            ll = grasp_debug.get("left_local")
            rl = grasp_debug.get("right_local")
            line = (
                f"[grasp_diag pid={os.getpid()} #{_GRASP_DIAG_DUMPS}] "
                f"L_local=({ll[0]:+.4f},{ll[1]:+.4f},{ll[2]:+.4f}) "
                f"R_local=({rl[0]:+.4f},{rl[1]:+.4f},{rl[2]:+.4f}) "
                f"axis={grasp_debug.get('face_axis')} "
                f"plane_err={grasp_debug.get('face_plane_err'):.4f} "
                f"ortho_err={grasp_debug.get('face_ortho_err'):.4f} "
                f"opposite={int(grasp_debug.get('faces_opposite'))} "
                f"below_top={int(grasp_debug.get('below_top'))} "
                f"bracket_ok={int(grasp_debug.get('bracket_ok'))} "
                f"prox={int(grasp_debug.get('dist_ok'))} "
                f"contacts(t/f)={int(grasp_debug.get('total_contacts'))}/"
                f"{int(grasp_debug.get('finger_contacts'))} "
                f"ready={int(grasp_debug.get('ready'))}\n"
            )
            try:
                os.makedirs(config.M2_RESULTS_DIR, exist_ok=True)
                with open(GRASP_DIAG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass

        # Expose attempt results so step() can fire terminal geom shaping +
        # attempt bonus + lenient bonus from the geometry achieved at the
        # moment of commitment. These are consumed once in step() then reset.
        self._attempt_fired = True
        self._attempt_lenient_ready = bool(grasp_debug.get("ready_lenient", False))
        self._attempt_bracket_score = float(grasp_debug.get("bracket_score", 0.0))
        self._attempt_worst_tip_above_top = float(
            grasp_debug.get("worst_tip_above_top", 0.0)
        )

        if ready:
            self._robot.attach_object(self._objects.cube_id)
            self._grasp_attached = True
            self._grasp_bonus_given = False
            return True

        # Eval-only: one deeper-descend retry. Mirrors the M1 demo
        # (verify_and_attach_with_retry in pick_place.py). Does not affect
        # training because enable_grasp_retry defaults False.
        if self.enable_grasp_retry and self._grasp_target_cartesian is not None:
            # Fix C: re-center on the cube's CURRENT pose, not the cached
            # target. If the first attempt kicked the cube laterally, the
            # stale target aims at empty space.
            current_cube_pos, _ = get_body_pose(self._objects.cube_id)
            _, current_cube_aabb_max = p.getAABB(self._objects.cube_id)
            current_cube_top_z = current_cube_aabb_max[2]
            grasp_bias = np.array([
                config.GRASP_TARGET_BIAS_X,
                config.GRASP_TARGET_BIAS_Y,
                config.GRASP_TARGET_BIAS_Z,
            ], dtype=float)
            retry_pos = np.array([
                current_cube_pos[0], current_cube_pos[1],
                current_cube_top_z + config.PICK_DESCEND_Z_OFFSET,
            ], dtype=float) + grasp_bias
            retry_pos[2] -= config.GRASP_DESCEND_RETRY_DELTA_Z
            q_retry = self._robot.solve_ik(retry_pos, self._grasp_orn)

            # Open briefly so the retry descent can translate without the cube
            # pinching, matching the M1 demo's open-before-descend flow.
            self._robot.open_gripper()

            max_vel = np.asarray(config.PD_MAX_JOINT_VEL, dtype=float)
            retry_steps = int(getattr(config, "RL_MAX_STEPS_PER_WAYPOINT", 80))
            retry_tol = config.GRASP_DESCEND_WAYPOINT_TOL
            for _ in range(retry_steps):
                q_now = self._robot.get_joint_positions()
                qd_now = self._robot.get_joint_velocities()
                qd_cmd = self._robot.pd_controller.compute_velocity_command(
                    q_des=q_retry, q=q_now, qd_des=np.zeros(7), qd=qd_now,
                )
                qd_cmd = np.clip(qd_cmd, -max_vel, max_vel)
                p.setJointMotorControlArray(
                    bodyUniqueId=self._robot.robot_id,
                    jointIndices=self._robot.arm_joint_indices,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=qd_cmd.tolist(),
                    forces=self._robot.arm_max_forces,
                )
                self._robot.command_gripper(config.GRIPPER_OPEN_WIDTH)
                for _ in range(config.RL_SIM_SUBSTEPS):
                    p.stepSimulation()
                if self._robot.is_at_joint_target(q_retry, tol=retry_tol):
                    break

            for _ in range(config.GRIPPER_SETTLE_STEPS):
                self._robot.close_gripper()
                self._env.step(sleep=False)

            ready2, grasp_debug2 = self._robot.is_grasp_ready(self._objects.cube_id)
            if self.trace:
                self._trace_grasp_debug = grasp_debug2
            self._record_grasp_attempt(grasp_debug2, attempt_idx=1)
            ready2 = self._apply_learned_gate(ready2, grasp_debug2)
            if ready2:
                self._robot.attach_object(self._objects.cube_id)
                self._grasp_attached = True
                self._grasp_bonus_given = False
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

        # Confidence gate: split off the 8th channel (if present) and remap
        # tanh-squashed [-1, 1] -> sigmoid-style [0, 1]. The first 7 dims
        # are the joint-space residual; the 8th scales the residual's
        # effective magnitude. Falls back to gate=1.0 when the feature is off.
        if self._use_gate and action.shape[-1] >= ACT_DIM + 1:
            gate = float((action[ACT_DIM] + 1.0) * 0.5)
            action = action[:ACT_DIM]
        else:
            gate = 1.0

        # Phase-gate: zero residual in phases where PD alone suffices
        residual_active = not (getattr(config, "RESIDUAL_PHASE_GATE", False)
                               and self._phase not in getattr(config, "RESIDUAL_ACTIVE_PHASES", {1, 2}))
        if not residual_active:
            action = np.zeros_like(action)

        # Apply confidence gate to the residual magnitude.
        if self._use_gate and residual_active:
            action = (action * gate).astype(np.float32)
        self._last_gate = gate
        self._last_residual_active = residual_active

        self._step_count += 1

        # Get current waypoint target
        if self._waypoint_idx < len(self._waypoints):
            q_target = self._waypoints[self._waypoint_idx]
        else:
            q_target = self._robot.get_joint_positions()

        q = self._robot.get_joint_positions()
        qd = self._robot.get_joint_velocities()

        if self.mode == "rl_only":
            # rl_only: action directly controls velocity, no PD
            qd_total = self.action_wrapper.compute_action(np.zeros(ACT_DIM), action)
            qd_residual = np.zeros(ACT_DIM, dtype=np.float32)
        else:
            # hybrid / planner_only: RL shifts the PD position target,
            # then PD drives toward the corrected target (cooperating).
            q_target_corrected = self.action_wrapper.correct_target(q_target, action)
            qd_total = self._robot.pd_controller.compute_velocity_command(
                q_des=q_target_corrected, q=q, qd_des=np.zeros(7), qd=qd,
            )
            # Clip to max joint velocity
            max_vel = np.asarray(config.PD_MAX_JOINT_VEL, dtype=float)
            qd_total = np.clip(qd_total, -max_vel, max_vel)
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

        # Check waypoint convergence (or force-advance on timeout).
        # For hybrid mode, check against the corrected target so the robot
        # converges to where the RL wants it, not the stale nominal waypoint.
        if self._waypoint_idx < len(self._waypoints):
            tol = getattr(config, "RL_WAYPOINT_TOL", config.WAYPOINT_TOL)
            if self._phase == PHASE_GRASP_DESCEND:
                tol = config.GRASP_DESCEND_WAYPOINT_TOL
            self._waypoint_step_count += 1
            conv_target = q_target_corrected if self.mode not in ("rl_only",) else q_target
            converged = self._robot.is_at_joint_target(conv_target, tol=tol)
            timed_out = self._waypoint_step_count >= config.RL_MAX_STEPS_PER_WAYPOINT
            if converged or timed_out:
                if timed_out and not converged:
                    self._forced_waypoint_advances += 1
                self._waypoint_idx += 1
                self._waypoint_step_count = 0

        # Early grasp trigger: in GRASP_DESCEND, if EE has arrived within
        # GRASP_TRIGGER_RADIUS of the grasp target pose, force waypoint
        # exhaustion so auto_grasp_and_attach fires immediately. Without
        # this, closure is gated on wp_idx == 20, which under timeout-
        # dominated advancement exceeds the episode budget.
        if (self._phase == PHASE_GRASP_DESCEND
                and not self._grasp_attempted
                and self._grasp_target_cartesian is not None
                and self._waypoint_idx < len(self._waypoints)):
            ee_pos_now, _ = self._robot.get_ee_pose()
            dist_to_grasp = float(
                np.linalg.norm(ee_pos_now - self._grasp_target_cartesian)
            )
            if dist_to_grasp < config.GRASP_TRIGGER_RADIUS:
                self._waypoint_idx = len(self._waypoints)

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

        # Terminal geom shaping, attempt bonus, and lenient bonus all fire
        # at the moment _auto_grasp_and_attach ran. Consume flags once per
        # step; reward.compute_reward gates on attempt_fired.
        attempt_fired = bool(getattr(self, "_attempt_fired", False))
        lenient_ready = bool(getattr(self, "_attempt_lenient_ready", False))
        bracket_score = self._attempt_bracket_score if attempt_fired else None
        worst_tip_above_top = (
            self._attempt_worst_tip_above_top if attempt_fired else None
        )
        self._attempt_fired = False
        self._attempt_lenient_ready = False

        reward, self._prev_ee_target_dist, reward_info = compute_reward(
            ee_pos=ee_pos,
            grasp_target_pos=self._grasp_target_cartesian,
            qd_residual=qd_residual,
            prev_ee_target_dist=self._prev_ee_target_dist,
            grasp_ready=grasp_ready,
            cube_lifted=cube_lifted,
            cube_fallen=cube_fallen,
            grasp_bonus_given=self._grasp_bonus_given,
            phase=self._phase,
            milestones_given=self._milestones_given,
            cube_pos=cube_pos,
            bracket_score=bracket_score,
            worst_tip_above_top=worst_tip_above_top,
            lenient_ready=lenient_ready,
            lenient_bonus_given=self._lenient_bonus_given,
            attempt_fired=attempt_fired,
        )

        fired = reward_info.get("milestones_fired", {})
        for key in ("m08", "m05", "m03"):
            if fired.get(key, False):
                self._milestones_given[key] = True

        if reward_info.get("lenient_bonus_fired", False):
            self._lenient_bonus_given = True

        if grasp_ready and not self._grasp_bonus_given:
            self._grasp_bonus_given = True

        # Confidence-gate penalty: small per-step cost proportional to gate
        # value, applied only in residual-active phases. Pushes the gate
        # down when the residual isn't earning its keep at nominal poses.
        if self._use_gate and self._last_residual_active:
            gate_pen = float(getattr(config, "RESIDUAL_GATE_PENALTY", 0.0)) * self._last_gate
            reward = float(reward) - gate_pen
            reward_info["r_gate_penalty"] = -gate_pen

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

        if self.trace:
            _, cube_aabb_max = p.getAABB(self._objects.cube_id)
            cube_top_z_now = float(cube_aabb_max[2])
            ee_xy_dist = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
            ee_above = float(ee_pos[2] - cube_top_z_now)
            self._trace_rows.append({
                "step": int(self._step_count),
                "phase": int(self._phase),
                "wp_idx": int(self._waypoint_idx),
                "wp_total": int(len(self._waypoints)),
                "ee_x": float(ee_pos[0]),
                "ee_y": float(ee_pos[1]),
                "ee_z": float(ee_pos[2]),
                "cube_x": float(cube_pos[0]),
                "cube_y": float(cube_pos[1]),
                "cube_z": float(cube_pos[2]),
                "cube_top_z": cube_top_z_now,
                "ee_xy_dist": ee_xy_dist,
                "ee_above_cube_top": ee_above,
                "gripper_cmd": float(gripper_width),
                "act_norm": float(np.linalg.norm(action)),
                "grasp_ready_at_lift": bool(grasp_ready),
                "reward": float(reward),
            })

        info = {
            "success": cube_lifted,
            "phase": self._phase,
            "step_count": self._step_count,
            "residual_abs_mean": float(np.mean(np.abs(qd_residual))),
            "gate": float(self._last_gate) if self._use_gate else 1.0,
            **reward_info,
            **self._diag_info(),
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
            # Flush per-episode grasp attempt log: tag each buffered row
            # with the final outcome and append to JSONL. Append-only,
            # multi-worker-safe (short writes are atomic on the FSes we use).
            if self._grasp_gate_log and self._grasp_attempt_log:
                self._flush_grasp_attempt_log(
                    lifted=bool(cube_lifted), fallen=bool(cube_fallen),
                )
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

        if self.trace and (terminated or truncated):
            self._write_trace(
                cube_lifted=bool(cube_lifted),
                cube_fallen=bool(cube_fallen),
                truncated=bool(truncated),
            )

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _write_trace(self, cube_lifted, cube_fallen, truncated):
        import csv as _csv
        import json as _json
        import os as _os

        def _safe(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        out_dir = _os.path.join(self.trace_dir, self.mode)
        _os.makedirs(out_dir, exist_ok=True)
        name = f"ep_{self._trace_episode_idx:03d}"
        csv_path = _os.path.join(out_dir, name + ".csv")
        json_path = _os.path.join(out_dir, name + ".json")

        if self._trace_rows:
            keys = list(self._trace_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self._trace_rows)

        grasp_dbg = None
        if self._trace_grasp_debug is not None:
            grasp_dbg = {k: _safe(v) for k, v in self._trace_grasp_debug.items()}

        summary = {
            "episode": int(self._trace_episode_idx),
            "mode": self.mode,
            "success": bool(cube_lifted),
            "cube_fallen": bool(cube_fallen),
            "truncated": bool(truncated),
            "steps": int(self._step_count),
            "max_phase": int(self._max_phase_reached),
            "end_phase": int(self._phase),
            "forced_wp_advances": int(self._forced_waypoint_advances),
            "grasp_attempted": bool(self._grasp_attempted),
            "grasp_attached": bool(self._grasp_attached),
            "grasp_debug_at_close": grasp_dbg,
            "perturb_offset": [float(x) for x in self._perturb_offset],
        }
        with open(json_path, "w") as f:
            _json.dump(summary, f, indent=2, default=str)

    def close(self):
        if self._env is not None:
            self._env.disconnect()
            self._env = None
            self._initialized = False
