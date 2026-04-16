import os
import time

import numpy as np
import pybullet as p
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src import config
from src.sim.env import BulletEnv
from src.sim.robot import PandaRobotDemo
from src.sim.state import (
    get_body_aabb,
    get_body_bottom_z,
    get_body_pose,
    get_body_top_z,
    quat_to_euler,
)
from src.rl.perturbation import perturb_cube_pose
from src.rl.residual_policy import ResidualActionWrapper
from src.rl.train import load_trained_actor
from src.trajectory.joint_trajectory import interpolate_joint_trajectory
from src.utils.logger import TrajectoryLogger
from src.utils.run_log import open_tee_log, close_tee_log

# Reuse classical pipeline helpers
from src.demo.pick_place import (
    build_motion_waypoints,
    execute_waypoint_trajectory,
    hold_gripper_for_steps,
    move_body_delta_z,
    move_to_cartesian_target,
    move_to_joint_target,
    place_body_bottom_at_z,
    print_grasp_debug,
    snap_cube_to_place_pose,
    verify_and_attach_with_retry,
)


# ---------------------------------------------------------------------------
# PyBullet debug overlays for demo recording
# ---------------------------------------------------------------------------

_overlay_ids = {}  # reusable overlay item IDs for replacement


def _update_text(key, text, position, color=(1, 1, 1), size=1.5):
    """Add or replace a debug text overlay."""
    if not config.GUI:
        return
    kwargs = dict(
        text=text,
        textPosition=position,
        textColorRGB=list(color),
        textSize=size,
        lifeTime=0,
    )
    if key in _overlay_ids:
        kwargs["replaceItemUniqueId"] = _overlay_ids[key]
    _overlay_ids[key] = p.addUserDebugText(**kwargs)


def _draw_mode_label(mode):
    mode_text = {
        "hybrid": "HYBRID  (PD + Residual RL)",
        "planner_only": "PLANNER ONLY  (PD, no RL)",
        "rl_only": "RL ONLY  (no PD backbone)",
    }
    _update_text("mode", mode_text.get(mode, mode), [0.65, 0.0, 0.90],
                 color=(0.6, 0.6, 0.0), size=1.8)


def _draw_phase_label(phase_name, phase_num=None):
    prefix = f"Phase {phase_num}: " if phase_num is not None else ""
    _update_text("phase", f"{prefix}{phase_name}", [0.65, 0.0, 0.82],
                 color=(0.2, 0.3, 0.6), size=1.5)


def _draw_perturbation_info(dx, dy, dz, dyaw):
    text = (f"Perturbation: dx={dx*100:+.1f}cm dy={dy*100:+.1f}cm "
            f"dz={dz*100:+.1f}cm yaw={np.degrees(dyaw):+.1f}deg")
    _update_text("perturb", text, [0.30, 0.0, 0.75],
                 color=(0.7, 0.3, 0.0), size=1.2)


def _draw_nominal_ghost(nominal_pos, cube_id):
    """Draw a wireframe box at the nominal (planned) cube position."""
    if not config.GUI:
        return []
    # Get cube half-extents from its current AABB
    aabb_min, aabb_max = p.getAABB(cube_id)
    half = [(aabb_max[i] - aabb_min[i]) / 2.0 for i in range(3)]
    cx, cy, cz = nominal_pos

    c = {
        (sx, sy, sz): [cx + sx * half[0], cy + sy * half[1], cz + sz * half[2]]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    }
    edges = [
        # bottom face
        ((-1,-1,-1), ( 1,-1,-1)), (( 1,-1,-1), ( 1, 1,-1)),
        (( 1, 1,-1), (-1, 1,-1)), ((-1, 1,-1), (-1,-1,-1)),
        # top face
        ((-1,-1, 1), ( 1,-1, 1)), (( 1,-1, 1), ( 1, 1, 1)),
        (( 1, 1, 1), (-1, 1, 1)), ((-1, 1, 1), (-1,-1, 1)),
        # vertical edges
        ((-1,-1,-1), (-1,-1, 1)), (( 1,-1,-1), ( 1,-1, 1)),
        (( 1, 1,-1), ( 1, 1, 1)), ((-1, 1,-1), (-1, 1, 1)),
    ]
    line_ids = []
    for a, b in edges:
        line_ids.append(
            p.addUserDebugLine(c[a], c[b], [0.7, 0.1, 0.1], lineWidth=1.5, lifeTime=0)
        )
    # Label
    _update_text("ghost_label", "PLANNED", [cx, cy, cz + half[2] + 0.03],
                 color=(0.7, 0.1, 0.1), size=1.0)
    return line_ids


def _draw_residual_magnitude(residual_norm):
    bar_len = min(residual_norm / config.RESIDUAL_MAX, 1.0)
    bar_str = "|" * max(1, int(bar_len * 20))
    color = (0.0, 0.5, 0.0) if bar_len < 0.5 else (0.6, 0.5, 0.0) if bar_len < 0.8 else (0.6, 0.0, 0.0)
    _update_text("residual", f"Residual: {residual_norm:.3f} {bar_str}",
                 [0.30, 0.0, 0.70], color=color, size=1.1)


def execute_residual_trajectory(
    env, robot, waypoints, actor, action_wrapper, cube_id,
    tol=config.WAYPOINT_TOL,
    max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
    label="trajectory",
    logger=None,
    gripper_width=None,
    phase_id=0,
    perturb_offset=None,
):
    if perturb_offset is None:
        perturb_offset = np.zeros(3, dtype=np.float32)
    total_steps = 0

    for i, q_target in enumerate(waypoints):
        reached = False

        for step in range(max_steps_per_waypoint):
            q = robot.get_joint_positions()
            qd = robot.get_joint_velocities()
            ee_pos, ee_orn = robot.get_ee_pose()
            ee_euler = quat_to_euler(ee_orn)
            cube_pos, cube_orn = get_body_pose(cube_id)
            cube_euler = quat_to_euler(cube_orn)
            ee_to_cube = cube_pos - ee_pos

            # Compute PD nominal
            qd_pd = robot.pd_controller.compute_velocity_command(
                q_des=q_target, q=q, qd_des=np.zeros(7), qd=qd,
            )

            # Build observation for the policy (must match gym_env OBS_DIM=40)
            obs = np.concatenate([
                q.astype(np.float32),
                qd.astype(np.float32),
                ee_pos.astype(np.float32),
                ee_euler.astype(np.float32),
                cube_pos.astype(np.float32),
                cube_euler.astype(np.float32),
                ee_to_cube.astype(np.float32),
                qd_pd.astype(np.float32),
                np.array([float(phase_id)], dtype=np.float32),
                perturb_offset.astype(np.float32),
            ])

            # Get action from policy (deterministic: TanhNormal has no analytical mode)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = actor(td)
            raw_action = td["action"].squeeze(0).numpy()

            # Apply residual
            qd_total = action_wrapper.compute_action(qd_pd, raw_action)

            # Debug overlay: show residual magnitude
            qd_residual = qd_total - qd_pd
            _draw_residual_magnitude(float(np.linalg.norm(qd_residual)))

            # Send to robot
            forces = robot.arm_max_forces
            p.setJointMotorControlArray(
                bodyUniqueId=robot.robot_id,
                jointIndices=robot.arm_joint_indices,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=qd_total.tolist(),
                forces=forces,
            )

            if gripper_width is not None:
                robot.command_gripper(gripper_width)
            else:
                robot.hold_gripper_open()

            env.step()
            total_steps += 1

            q_error = np.asarray(q_target) - robot.get_joint_positions()

            if logger is not None:
                logger.log(
                    phase=label,
                    waypoint_index=i,
                    sim_step=total_steps,
                    q_target=q_target,
                    q_actual=robot.get_joint_positions(),
                    q_error=q_error,
                    ee_pos=ee_pos,
                    ee_euler=ee_euler,
                )

            if float(np.max(np.abs(q_error))) <= tol:
                reached = True
                break

        if not reached:
            print(f"[residual] waypoint {i} not reached in {label}")
            return False

    return True


def move_to_cartesian_target_residual(
    env, robot, target_pos, target_orn, actor, action_wrapper,
    cube_id, label, logger=None, gripper_width=None, phase_id=0,
    perturb_offset=None,
):
    q_goal = robot.solve_ik(target_pos, target_orn)
    q_start = robot.get_joint_positions()
    waypoints = interpolate_joint_trajectory(
        q_start=q_start, q_goal=q_goal,
        num_waypoints=config.TRAJ_NUM_WAYPOINTS,
    )

    tol = config.WAYPOINT_TOL
    max_steps = config.WAYPOINT_MAX_STEPS
    if label in ("grasp_descend", "grasp_descend_retry"):
        tol = config.GRASP_DESCEND_WAYPOINT_TOL
        max_steps = config.GRASP_DESCEND_WAYPOINT_MAX_STEPS

    return execute_residual_trajectory(
        env=env, robot=robot, waypoints=waypoints,
        actor=actor, action_wrapper=action_wrapper,
        cube_id=cube_id, tol=tol,
        max_steps_per_waypoint=max_steps,
        label=label, logger=logger,
        gripper_width=gripper_width,
        phase_id=phase_id,
        perturb_offset=perturb_offset,
    )


def get_pick_and_place_poses_from_cube(robot, cube_id, table_id):
    cube_pos, _ = get_body_pose(cube_id)
    _, cube_aabb_max = get_body_aabb(cube_id)
    cube_top_z = cube_aabb_max[2]
    table_top_z = get_body_top_z(table_id)

    place_xy = np.array(config.PLACE_XY, dtype=float)
    grasp_euler = np.array(config.GRASP_EE_EULER, dtype=float)
    grasp_orn = p.getQuaternionFromEuler(grasp_euler.tolist())

    grasp_bias = np.array([
        config.GRASP_TARGET_BIAS_X,
        config.GRASP_TARGET_BIAS_Y,
        config.GRASP_TARGET_BIAS_Z,
    ], dtype=float)

    pre_grasp = np.array([
        cube_pos[0], cube_pos[1],
        cube_top_z + config.PICK_HOVER_Z_OFFSET,
    ], dtype=float) + grasp_bias

    grasp = np.array([
        cube_pos[0], cube_pos[1],
        cube_top_z + config.PICK_DESCEND_Z_OFFSET,
    ], dtype=float) + grasp_bias

    if config.APPLY_GRASP_BIAS_TO_LIFT:
        lift = np.array([
            cube_pos[0], cube_pos[1],
            cube_top_z + config.PICK_HOVER_Z_OFFSET,
        ], dtype=float) + grasp_bias
    else:
        lift = np.array([
            cube_pos[0], cube_pos[1],
            cube_top_z + config.PICK_HOVER_Z_OFFSET,
        ], dtype=float)

    pre_place = np.array([
        place_xy[0], place_xy[1],
        table_top_z + config.PLACE_HOVER_Z_OFFSET,
    ], dtype=float)

    place = np.array([
        place_xy[0], place_xy[1],
        table_top_z + config.PLACE_DESCEND_Z_OFFSET,
    ], dtype=float)

    post_place = np.array([
        place_xy[0], place_xy[1],
        table_top_z + config.POST_PLACE_RETREAT_Z_OFFSET,
    ], dtype=float)

    return {
        "grasp_orn": grasp_orn,
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "lift": lift,
        "pre_place": pre_place,
        "place": place,
        "post_place": post_place,
    }


def run_pick_place_with_residual(
    model_path=None, perturb_xy_range=None, mode="hybrid", log_file_path=None,
    allow_grasp_retry=True,
):
    model_path = model_path or os.path.join(config.M2_MODEL_DIR, "final_model.pt")
    perturb_xy_range = perturb_xy_range if perturb_xy_range is not None else config.PERTURB_XY_RANGE
    log_file_path = log_file_path or os.path.join(config.M2_RESULTS_DIR, "demo_log.txt")
    os.makedirs(config.M2_RESULTS_DIR, exist_ok=True)

    original_stdout, log_file = open_tee_log(log_file_path, banner="Residual demo run")
    demo_start = time.time()

    # Load trained actor
    actor = load_trained_actor(model_path)
    action_wrapper = ResidualActionWrapper(mode=mode)
    print(f"Log file: {log_file_path}")
    print(f"Loaded model from {model_path}, mode={mode}, perturb_xy_range={perturb_xy_range}")
    _overlay_ids.clear()

    env = BulletEnv(gui=config.GUI)
    env.connect()

    try:
        objects = env.load_scene()

        # Correct table to plane
        table_bottom_z = get_body_bottom_z(objects.table_id)
        move_body_delta_z(objects.table_id, 0.0 - table_bottom_z + config.TABLE_PLANE_CLEARANCE)

        # Load robot
        robot = PandaRobotDemo()
        robot.load(
            base_position=config.PANDA_BASE_POS,
            base_orientation_euler=config.PANDA_BASE_ORN_EULER,
        )

        # Place cube on table
        table_top_z = get_body_top_z(objects.table_id)
        place_body_bottom_at_z(
            objects.cube_id, config.CUBE_BASE_POS[:2],
            table_top_z, config.CUBE_TABLE_CLEARANCE,
        )

        # Draw mode label overlay
        _draw_mode_label(mode)

        # Log nominal cube pose
        nominal_pos, nominal_orn = p.getBasePositionAndOrientation(objects.cube_id)
        nominal_euler = p.getEulerFromQuaternion(nominal_orn)
        print(f"[demo] Cube NOMINAL pose: "
              f"xy=({nominal_pos[0]:.4f}, {nominal_pos[1]:.4f}) m, "
              f"z={nominal_pos[2]:.4f} m, yaw={nominal_euler[2]:.4f} rad")

        # Plan from the NOMINAL cube pose (before perturbation), matching
        # the training env (gym_env.py). The classical planner is blind to
        # the upcoming perturbation; residual RL corrects for the offset.
        print("[demo] Planning trajectory from NOMINAL cube pose (before perturbation)")
        poses = get_pick_and_place_poses_from_cube(robot, objects.cube_id, objects.table_id)
        grasp_orn = poses["grasp_orn"]

        # In GUI mode, hold the scene at the nominal pose so the recorder can
        # frame the shot before the cube teleports to its perturbed pose.
        hold_seconds = float(getattr(config, "DEMO_PERTURB_HOLD_SECONDS", 0.0))
        if config.GUI and perturb_xy_range > 0 and hold_seconds > 0:
            print(f"[demo] Holding at nominal for {hold_seconds:.1f}s -- "
                  f"start your recorder now. Perturbation will be applied after the countdown.")
            hold_start = time.time()
            next_tick = 1.0
            while time.time() - hold_start < hold_seconds:
                robot.hold_home_pose()
                env.step()
                elapsed = time.time() - hold_start
                if elapsed >= next_tick:
                    remaining = hold_seconds - elapsed
                    print(f"[demo]   ... perturbing in {remaining:4.1f}s")
                    next_tick += 1.0
                time.sleep(1.0 / 240.0)

        # Perturb cube AFTER planning — the planner targets are now "stale"
        rng = np.random.default_rng()
        z_range = float(getattr(config, "PERTURB_Z_RANGE", 0.0))
        if perturb_xy_range > 0 or z_range > 0:
            new_pos, new_orn, (dx, dy, dz, dyaw) = perturb_cube_pose(
                cube_id=objects.cube_id,
                nominal_pos=list(nominal_pos),
                nominal_orn_euler=config.CUBE_BASE_ORN_EULER,
                rng=rng,
                xy_range=perturb_xy_range,
                yaw_range=config.PERTURB_YAW_RANGE,
                z_range=z_range,
            )
            perturbed_pos, perturbed_orn = p.getBasePositionAndOrientation(objects.cube_id)
            perturbed_euler = p.getEulerFromQuaternion(perturbed_orn)
            print(f"[demo] Cube PERTURBED pose: "
                  f"xy=({perturbed_pos[0]:.4f}, {perturbed_pos[1]:.4f}) m, "
                  f"z={perturbed_pos[2]:.4f} m, yaw={perturbed_euler[2]:.4f} rad")
            print(f"[demo] Perturbation delta: "
                  f"dx={dx:+.4f} m, dy={dy:+.4f} m, dz={dz:+.4f} m, "
                  f"dyaw={dyaw:+.4f} rad ({np.degrees(dyaw):+.2f} deg)")
            total_offset = float(np.sqrt(dx**2 + dy**2 + dz**2))
            print(f"[demo] Planner targets are from NOMINAL — "
                  f"robot will aim {total_offset*100:.1f}cm away from actual cube")
            # Debug overlays: perturbation info + ghost at nominal position
            _draw_perturbation_info(dx, dy, dz, dyaw)
            _draw_nominal_ghost(list(nominal_pos), objects.cube_id)

        # Compute perturbation offset for the policy observation (matches gym_env)
        perturbed_pos_final, _ = p.getBasePositionAndOrientation(objects.cube_id)
        perturb_offset = np.array(perturbed_pos_final, dtype=np.float32) - np.array(nominal_pos, dtype=np.float32)

        robot.reset_home()
        robot.hold_home_pose()

        for _ in range(config.SETTLE_STEPS):
            robot.hold_home_pose()
            env.step()

        logger = TrajectoryLogger() if config.ENABLE_TRAJECTORY_LOGGING else None

        # === Phases 1-6: RL-augmented control ===

        # Phase 1: Home
        print("\n=== Phase 1: home ===")
        _draw_phase_label("home", 1)
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 2: Pre-grasp (with residual)
        print("\n=== Phase 2: pre_grasp (residual) ===")
        _draw_phase_label("pre_grasp (residual)", 2)
        ok = move_to_cartesian_target_residual(
            env, robot, poses["pre_grasp"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="pre_grasp", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH, phase_id=0,
            perturb_offset=perturb_offset,
        )
        if not ok:
            print("Failed at pre_grasp with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["pre_grasp"], grasp_orn,
                                          label="pre_grasp_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_OPEN_WIDTH)

        # Phase 3: Grasp descend (with residual)
        print("\n=== Phase 3: grasp_descend (residual) ===")
        _draw_phase_label("grasp_descend (residual)", 3)
        ok = move_to_cartesian_target_residual(
            env, robot, poses["grasp"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="grasp_descend", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH, phase_id=1,
            perturb_offset=perturb_offset,
        )
        if not ok:
            print("Failed at grasp_descend with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["grasp"], grasp_orn,
                                          label="grasp_descend_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_OPEN_WIDTH)

        # Phase 4: Close gripper
        print("\n=== Phase 4: close_gripper ===")
        _draw_phase_label("close_gripper", 4)
        hold_gripper_for_steps(env, robot, config.GRIPPER_CLOSED_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 5: Validate and attach
        print("\n=== Phase 5: validate_and_attach ===")
        _draw_phase_label("validate_and_attach", 5)
        if allow_grasp_retry:
            attached = verify_and_attach_with_retry(
                env=env, robot=robot, cube_id=objects.cube_id,
                grasp_orn=grasp_orn, base_grasp_pos=poses["grasp"],
                logger=logger,
            )
        else:
            from src.demo.pick_place import print_grasp_debug
            ready, debug = robot.is_grasp_ready(objects.cube_id)
            print_grasp_debug(debug)
            if ready:
                robot.attach_object(objects.cube_id)
                print("Grasp validated. Object attached.")
                attached = True
            else:
                print("Grasp check failed (no retry -- matches eval framework).")
                attached = False
        if not attached:
            print("Grasp validation failed.")
            return

        # Phase 6: Lift (with residual)
        print("\n=== Phase 6: lift (residual) ===")
        _draw_phase_label("lift (residual)", 6)
        ok = move_to_cartesian_target_residual(
            env, robot, poses["lift"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="lift", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH, phase_id=2,
            perturb_offset=perturb_offset,
        )
        if not ok:
            print("Failed at lift with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["lift"], grasp_orn,
                                          label="lift_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_CLOSED_WIDTH)

        # === Phases 7-12: Classical pipeline (unchanged) ===

        # Phase 7: Transfer to pre-place
        print("\n=== Phase 7: transfer_pre_place ===")
        _draw_phase_label("transfer_pre_place (classical)", 7)
        ok = move_to_cartesian_target(
            env, robot, poses["pre_place"], grasp_orn,
            label="transfer_pre_place", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )

        # Phase 8: Place descend
        print("\n=== Phase 8: place_descend ===")
        _draw_phase_label("place_descend (classical)", 8)
        ok = move_to_cartesian_target(
            env, robot, poses["place"], grasp_orn,
            label="place_descend", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )

        # Phase 9: Detach
        print("\n=== Phase 9: detach_object ===")
        _draw_phase_label("detach_object", 9)
        if config.PLACE_SNAP_BEFORE_RELEASE:
            snap_cube_to_place_pose(objects)
        if robot.has_attached_object():
            robot.detach_object()

        for _ in range(config.PLACE_RELEASE_SETTLE_STEPS):
            robot.command_arm_and_gripper(
                robot.get_joint_positions(),
                gripper_width=config.GRIPPER_CLOSED_WIDTH,
            )
            env.step()

        # Phase 10: Open gripper
        print("\n=== Phase 10: open_gripper ===")
        _draw_phase_label("open_gripper", 10)
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 11: Retreat
        print("\n=== Phase 11: post_place_retreat ===")
        _draw_phase_label("post_place_retreat (classical)", 11)
        ok = move_to_cartesian_target(
            env, robot, poses["post_place"], grasp_orn,
            label="post_place_retreat", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )

        # Phase 12: Return home
        print("\n=== Phase 12: return_home ===")
        _draw_phase_label("return_home (classical)", 12)
        ok = move_to_joint_target(
            env, robot, robot.home_joints,
            label="return_home", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )

        for _ in range(config.FINAL_HOME_SETTLE_STEPS):
            robot.command_arm_and_gripper(
                robot.home_joints,
                gripper_width=config.GRIPPER_OPEN_WIDTH,
            )
            env.step()

        # Final state
        _draw_phase_label("COMPLETE", None)
        cube_pos, cube_orn = get_body_pose(objects.cube_id)
        print("\n=== Final Cube Pose ===")
        print("Cube position:", cube_pos)

        if logger is not None:
            os.makedirs(os.path.dirname(config.M2_TRAJECTORY_LOG_PATH), exist_ok=True)
            logger.save_json(config.M2_TRAJECTORY_LOG_PATH)
            print(f"Saved trajectory log to {config.M2_TRAJECTORY_LOG_PATH}")

        try:
            while True:
                robot.command_arm_and_gripper(
                    robot.home_joints,
                    gripper_width=config.GRIPPER_OPEN_WIDTH,
                )
                env.step()
        except KeyboardInterrupt:
            print("\nStopping simulation...")

    finally:
        env.disconnect()
        print(f"\nDemo total time: {time.time() - demo_start:.1f}s")
        close_tee_log(original_stdout, log_file)
