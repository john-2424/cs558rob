import numpy as np
import pybullet as p

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
from src.trajectory.joint_trajectory import interpolate_joint_trajectory
from src.utils.logger import TrajectoryLogger
from src.planner.rrtstar import JointSpaceRRTStarPlanner


def move_body_delta_z(body_id: int, delta_z: float):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    new_pos = [pos[0], pos[1], pos[2] + delta_z]
    p.resetBasePositionAndOrientation(body_id, new_pos, orn)


def place_body_bottom_at_z(body_id: int, target_xy, target_bottom_z: float, clearance: float = 0.0):
    current_pos, current_orn = p.getBasePositionAndOrientation(body_id)
    body_bottom_z = get_body_bottom_z(body_id)
    delta_z = target_bottom_z - body_bottom_z + clearance

    new_pos = [target_xy[0], target_xy[1], current_pos[2] + delta_z]
    p.resetBasePositionAndOrientation(body_id, new_pos, current_orn)


def print_scene_diagnostics(robot, objects):
    robot.print_joint_summary()

    joint_positions = robot.get_joint_positions()
    joint_velocities = robot.get_joint_velocities()
    ee_pos, ee_orn = robot.get_ee_pose()

    print("\n=== Robot State ===")
    print("Joint positions:", joint_positions)
    print("Joint velocities:", joint_velocities)
    print("End-effector position:", ee_pos)
    print("End-effector orientation (quat):", ee_orn)
    print("End-effector orientation (euler):", quat_to_euler(ee_orn))

    cube_pos, cube_orn = get_body_pose(objects.cube_id)
    print("\n=== Cube State ===")
    print("Cube position:", cube_pos)
    print("Cube orientation (quat):", cube_orn)
    print("Cube orientation (euler):", quat_to_euler(cube_orn))

    table_aabb_min, table_aabb_max = get_body_aabb(objects.table_id)
    print("\n=== Table AABB ===")
    print("Table AABB min:", table_aabb_min)
    print("Table AABB max:", table_aabb_max)
    print("Table bottom z:", table_aabb_min[2])
    print("Table top z:", table_aabb_max[2])


def execute_waypoint_trajectory(
    env,
    robot,
    waypoints,
    tol=config.WAYPOINT_TOL,
    max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
    label="trajectory",
    logger=None,
    gripper_width=None,
):
    print(f"\n=== Executing {label} ===")
    print(f"Num waypoints: {len(waypoints)}")

    total_steps = 0

    for i, q_target in enumerate(waypoints):
        reached = False
        step_max_vel_cmd = 0.0

        for step in range(max_steps_per_waypoint):
            robot.command_arm_and_gripper(q_target, gripper_width=gripper_width)
            env.step()
            total_steps += 1

            current_q = robot.get_joint_positions()
            q_error = np.asarray(q_target) - current_q
            ee_pos, ee_orn = robot.get_ee_pose()
            ee_euler = quat_to_euler(ee_orn)

            if logger is not None:
                logger.log(
                    phase=label,
                    waypoint_index=i,
                    sim_step=total_steps,
                    q_target=q_target,
                    q_actual=current_q,
                    q_error=q_error,
                    ee_pos=ee_pos,
                    ee_euler=ee_euler,
                )

            debug = robot.get_pd_debug_info(q_target)
            step_err = debug["pos_err"]
            step_max_err = float(np.max(np.abs(step_err)))
            step_max_vel_cmd = float(np.max(np.abs(debug["qd_cmd"])))

            if step_max_err <= tol:
                reached = True
                break

        final_err = robot.get_joint_position_error(q_target)
        final_max_err = float(np.max(np.abs(final_err)))

        if (i % config.PRINT_WAYPOINT_EVERY == 0) or (i == len(waypoints) - 1) or (not reached):
            print(
                f"waypoint={i:02d}/{len(waypoints)-1:02d} | "
                f"reached={reached} | "
                f"final_err={final_err} | "
                f"final_max_err={final_max_err:.6f} | "
                f"last_max_vel_cmd={step_max_vel_cmd:.5f}"
            )

        if not reached:
            print(f"Stopped early at waypoint {i} because target was not reached in time.")
            return False

    print(f"Completed {label} in {total_steps} simulation steps.")
    return True


def build_motion_waypoints(robot, q_start, q_goal, label="trajectory"):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)

    if config.USE_PLANNER:
        planner = JointSpaceRRTStarPlanner(
            robot_id=robot.robot_id,
            arm_joint_indices=robot.arm_joint_indices,
            step_size=config.PLANNER_STEP_SIZE,
            goal_threshold=config.PLANNER_GOAL_THRESHOLD,
            max_iterations=config.PLANNER_MAX_ITERATIONS,
            goal_sample_rate=config.PLANNER_GOAL_SAMPLE_RATE,
            edge_check_resolution=config.PLANNER_EDGE_CHECK_RESOLUTION,
            phase_label=label,
            attached_body_id=getattr(robot, "attached_body_id", None),
        )

        result = planner.plan(q_start, q_goal)

        print(f"\n=== Planner result: {label} ===")
        print(f"success: {result.success}")
        print(f"num_nodes: {result.num_nodes}")
        print(f"message: {result.message}")

        if result.success:
            sparse_path = result.path
            sparse_nodes_before = len(sparse_path)
            sparse_length_before = planner.path_length(sparse_path)

            if config.ENABLE_PATH_SHORTCUT_SMOOTHING:
                sparse_path = planner.shortcut_path(
                    sparse_path,
                    max_passes=config.PATH_SHORTCUT_MAX_PASSES,
                )

            sparse_nodes_after = len(sparse_path)
            sparse_length_after = planner.path_length(sparse_path)

            path = planner.interpolate_path(
                sparse_path,
                max_segment_length=config.PLANNER_INTERP_RESOLUTION,
            )

            print(f"Sparse path nodes before smoothing: {sparse_nodes_before}")
            print(f"Sparse path nodes after smoothing:  {sparse_nodes_after}")
            print(f"Sparse path length before smoothing: {sparse_length_before:.4f}")
            print(f"Sparse path length after smoothing:  {sparse_length_after:.4f}")
            print(f"Planner path waypoints: {len(path)}")

            return path

        if config.PLANNER_FALLBACK_TO_INTERPOLATION:
            print("Planner failed. Falling back to interpolated joint trajectory.")
        else:
            raise RuntimeError(f"Planner failed for {label}: {result.message}")

    fallback_path = interpolate_joint_trajectory(
        q_start=q_start,
        q_goal=q_goal,
        num_waypoints=config.TRAJ_NUM_WAYPOINTS,
    )
    print(f"Fallback interpolated waypoints: {len(fallback_path)}")
    return fallback_path


def move_to_joint_target(env, robot, q_goal, label, logger=None, gripper_width=None):
    q_start = robot.get_joint_positions().copy()

    waypoints = build_motion_waypoints(
        robot=robot,
        q_start=q_start,
        q_goal=q_goal,
        label=label,
    )

    tol = config.WAYPOINT_TOL
    max_steps = config.WAYPOINT_MAX_STEPS

    if label in ("grasp_descend", "grasp_descend_retry", "place_descend"):
        tol = getattr(config, "GRASP_DESCEND_WAYPOINT_TOL", config.WAYPOINT_TOL)
        max_steps = getattr(config, "GRASP_DESCEND_WAYPOINT_MAX_STEPS", config.WAYPOINT_MAX_STEPS)

    return execute_waypoint_trajectory(
        env=env,
        robot=robot,
        waypoints=waypoints,
        tol=tol,
        max_steps_per_waypoint=max_steps,
        label=label,
        logger=logger,
        gripper_width=gripper_width,
    )


def move_to_cartesian_target(env, robot, target_pos, target_orn, label, logger=None, gripper_width=None):
    q_goal = robot.solve_ik(target_pos, target_orn)
    print(f"{label} IK target joints: {q_goal}")
    return move_to_joint_target(
        env=env,
        robot=robot,
        q_goal=q_goal,
        label=label,
        logger=logger,
        gripper_width=gripper_width,
    )


def hold_gripper_for_steps(env, robot, open_width, steps):
    for _ in range(steps):
        robot.command_gripper(open_width)
        env.step()


def print_grasp_debug(debug_dict):
    print(
        "grasp_check | "
        f"ee_to_cube_dist={debug_dict['ee_to_cube_dist']:.5f} | "
        f"min_finger_to_cube_dist={debug_dict['min_finger_to_cube_dist']:.5f} | "
        f"total_contacts={debug_dict['total_contacts']} | "
        f"finger_contacts={debug_dict['finger_contacts']} | "
        f"dist_ok={debug_dict['dist_ok']} | "
        f"finger_dist_ok={debug_dict['finger_dist_ok']} | "
        f"total_contact_ok={debug_dict['total_contact_ok']} | "
        f"finger_contact_ok={debug_dict['finger_contact_ok']} | "
        f"ready={debug_dict['ready']}"
    )
    print(f"left_finger_pos={debug_dict['left_finger_pos']}")
    print(f"right_finger_pos={debug_dict['right_finger_pos']}")


def verify_and_attach_with_retry(env, robot, cube_id, grasp_orn, base_grasp_pos, logger=None):
    """
    1) Check current grasp readiness after closing gripper.
    2) If not ready, descend deeper and re-check.
    3) Attach only if validated.
    """
    ready, debug = robot.is_grasp_ready(cube_id)
    print_grasp_debug(debug)

    if ready:
        robot.attach_object(cube_id)
        print("Grasp validated. Object attached.")
        return True

    print("Initial grasp check failed. Trying deeper descend retry...")

    retry_grasp = np.array(base_grasp_pos, dtype=float).copy()
    retry_grasp[2] -= config.GRASP_DESCEND_RETRY_DELTA_Z

    ok = move_to_cartesian_target(
        env=env,
        robot=robot,
        target_pos=retry_grasp,
        target_orn=grasp_orn,
        label="grasp_descend_retry",
        logger=logger,
        gripper_width=config.GRIPPER_OPEN_WIDTH,
    )
    if not ok:
        print("Retry descend motion failed.")
        return False

    hold_gripper_for_steps(env, robot, config.GRIPPER_CLOSED_WIDTH, config.GRIPPER_SETTLE_STEPS)

    ready, debug = robot.is_grasp_ready(cube_id)
    print_grasp_debug(debug)

    if ready:
        robot.attach_object(cube_id)
        print("Retry grasp validated. Object attached.")
        return True

    print("Grasp validation failed even after retry. Object will NOT be attached.")
    return False


def get_pick_and_place_poses(robot, objects):
    cube_pos, _ = get_body_pose(objects.cube_id)
    _, cube_aabb_max = get_body_aabb(objects.cube_id)
    cube_top_z = cube_aabb_max[2]

    place_xy = np.array(config.PLACE_XY, dtype=float)
    table_top_z = get_body_top_z(objects.table_id)

    grasp_euler = np.array(config.GRASP_EE_EULER, dtype=float)
    grasp_orn = p.getQuaternionFromEuler(grasp_euler.tolist())

    grasp_bias = np.array(
        [
            config.GRASP_TARGET_BIAS_X,
            config.GRASP_TARGET_BIAS_Y,
            config.GRASP_TARGET_BIAS_Z,
        ],
        dtype=float,
    )

    # Desired EE targets directly, with manual XY bias applied
    pre_grasp = np.array([
        cube_pos[0],
        cube_pos[1],
        cube_top_z + config.PICK_HOVER_Z_OFFSET,
    ], dtype=float) + grasp_bias

    grasp = np.array([
        cube_pos[0],
        cube_pos[1],
        cube_top_z + config.PICK_DESCEND_Z_OFFSET,
    ], dtype=float) + grasp_bias

    if config.APPLY_GRASP_BIAS_TO_LIFT:
        lift = np.array([
            cube_pos[0],
            cube_pos[1],
            cube_top_z + config.PICK_HOVER_Z_OFFSET,
        ], dtype=float) + grasp_bias
    else:
        lift = np.array([
            cube_pos[0],
            cube_pos[1],
            cube_top_z + config.PICK_HOVER_Z_OFFSET,
        ], dtype=float)

    # Place targets can stay unchanged for now
    pre_place = np.array([
        place_xy[0],
        place_xy[1],
        table_top_z + config.PLACE_HOVER_Z_OFFSET,
    ], dtype=float)

    place = np.array([
        place_xy[0],
        place_xy[1],
        table_top_z + config.PLACE_DESCEND_Z_OFFSET,
    ], dtype=float)

    post_place = np.array([
        place_xy[0],
        place_xy[1],
        table_top_z + config.POST_PLACE_RETREAT_Z_OFFSET,
    ], dtype=float)

    print("\n=== Pick Target Debug ===")
    print("cube_pos:", cube_pos)
    print("grasp_bias:", grasp_bias)
    print("pre_grasp target:", pre_grasp)
    print("grasp target:", grasp)

    return {
        "grasp_orn": grasp_orn,
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "lift": lift,
        "pre_place": pre_place,
        "place": place,
        "post_place": post_place,
    }


def run_pick_place_demo():
    env = BulletEnv(gui=config.GUI)
    env.connect()

    try:
        objects = env.load_scene()

        # Correct table to plane
        table_bottom_z_before = get_body_bottom_z(objects.table_id)
        move_body_delta_z(
            body_id=objects.table_id,
            delta_z=(0.0 - table_bottom_z_before + config.TABLE_PLANE_CLEARANCE),
        )

        # Load robot
        robot = PandaRobotDemo()
        robot.load(
            base_position=config.PANDA_BASE_POS,
            base_orientation_euler=config.PANDA_BASE_ORN_EULER,
        )

        # Place cube on top of table
        table_top_z = get_body_top_z(objects.table_id)
        place_body_bottom_at_z(
            body_id=objects.cube_id,
            target_xy=config.CUBE_BASE_POS[:2],
            target_bottom_z=table_top_z,
            clearance=config.CUBE_TABLE_CLEARANCE,
        )

        robot.reset_home()
        robot.hold_home_pose()

        for _ in range(config.SETTLE_STEPS):
            robot.hold_home_pose()
            env.step()

        print_scene_diagnostics(robot, objects)

        logger = TrajectoryLogger() if config.ENABLE_TRAJECTORY_LOGGING else None
        robot.print_pinch_point_debug()
        poses = get_pick_and_place_poses(robot, objects)
        grasp_orn = poses["grasp_orn"]

        # 1) Home
        print("\n=== Phase 1: home ===")
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # 2) Pre-grasp
        print("\n=== Phase 2: pre_grasp ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["pre_grasp"],
            grasp_orn,
            label="pre_grasp",
            logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at pre_grasp")

        # 3) Grasp descend
        print("\n=== Phase 3: grasp_descend ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["grasp"],
            grasp_orn,
            label="grasp_descend",
            logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at grasp_descend")

        # 4) Close gripper
        print("\n=== Phase 4: close_gripper ===")
        hold_gripper_for_steps(
            env,
            robot,
            config.GRIPPER_CLOSED_WIDTH,
            config.GRIPPER_SETTLE_STEPS,
        )

        # 5) Validate grasp before attach
        print("\n=== Phase 5: validate_and_attach_object ===")
        attached = verify_and_attach_with_retry(
            env=env,
            robot=robot,
            cube_id=objects.cube_id,
            grasp_orn=grasp_orn,
            base_grasp_pos=poses["grasp"],
            logger=logger,
        )
        if not attached:
            raise RuntimeError("Grasp validation failed. Cube was not attached.")

        # 6) Lift
        print("\n=== Phase 6: lift ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["lift"],
            grasp_orn,
            label="lift",
            logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at lift")

        # 7) Transfer to pre-place
        print("\n=== Phase 7: transfer_pre_place ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["pre_place"],
            grasp_orn,
            label="transfer_pre_place",
            logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at transfer_pre_place")

        # 8) Place descend
        print("\n=== Phase 8: place_descend ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["place"],
            grasp_orn,
            label="place_descend",
            logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at place_descend")

        # 9) Snap and detach object
        print("\n=== Phase 9: detach_object ===")
        if config.PLACE_SNAP_BEFORE_RELEASE:
            snap_cube_to_place_pose(objects)

        if getattr(robot, "attached_constraint_id", None) is not None:
            robot.detach_object()
        else:
            print("No attached object found at detach phase.")

        # Let the cube settle in the snapped pose before fully retreating
        for _ in range(config.PLACE_RELEASE_SETTLE_STEPS):
            robot.command_arm_and_gripper(robot.get_joint_positions(), gripper_width=config.GRIPPER_CLOSED_WIDTH)
            env.step()

        # 10) Open gripper
        print("\n=== Phase 10: open_gripper ===")
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # 11) Retreat
        print("\n=== Phase 11: post_place_retreat ===")
        ok = move_to_cartesian_target(
            env,
            robot,
            poses["post_place"],
            grasp_orn,
            label="post_place_retreat",
            logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )
        if not ok:
            raise RuntimeError("Failed at post_place_retreat")

        # 12) Return home
        print("\n=== Phase 12: return_home ===")
        ok = move_to_joint_target(
            env,
            robot,
            robot.home_joints,
            label="return_home",
            logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )
        print(f"Return home success: {ok}")

        # Extra settle at home to reduce residual final error
        for _ in range(config.FINAL_HOME_SETTLE_STEPS):
            robot.command_arm_and_gripper(robot.home_joints, gripper_width=config.GRIPPER_OPEN_WIDTH)
            env.step()

        cube_pos, cube_orn = get_body_pose(objects.cube_id)
        print("\n=== Final Cube Pose ===")
        print("Cube position:", cube_pos)
        print("Cube orientation (quat):", cube_orn)
        print("Cube orientation (euler):", quat_to_euler(cube_orn))

        final_error = robot.get_joint_position_error(robot.home_joints)
        print("\n=== Final Home Return Error ===")
        print("Final joint error:", final_error)
        print("Max abs error:", float(abs(final_error).max()))

        if logger is not None:
            logger.save_json(config.TRAJECTORY_LOG_PATH)
            print(f"Saved trajectory log to {config.TRAJECTORY_LOG_PATH}")

        try:
            while True:
                robot.command_arm_and_gripper(robot.home_joints, gripper_width=config.GRIPPER_OPEN_WIDTH)
                env.step()
        except KeyboardInterrupt:
            print("\nStopping simulation...")

    finally:
        env.disconnect()


def snap_cube_to_place_pose(objects):
    """
    Precisely place the cube at the commanded place target before release.
    This removes the last-contact drift that causes visible placement offsets.
    """
    target_xy = np.array(config.PLACE_XY, dtype=float)
    table_top_z = get_body_top_z(objects.table_id)

    current_pos, current_orn = p.getBasePositionAndOrientation(objects.cube_id)
    body_bottom_z = get_body_bottom_z(objects.cube_id)

    delta_z = table_top_z - body_bottom_z + config.PLACE_RELEASE_CLEARANCE

    new_pos = [
        float(target_xy[0]),
        float(target_xy[1]),
        float(current_pos[2] + delta_z),
    ]

    p.resetBasePositionAndOrientation(
        objects.cube_id,
        new_pos,
        current_orn,
    )