import numpy as np
import pybullet as p

from src import config
from src.sim.env import BulletEnv
from src.sim.robot import PandaRobot
from src.sim.state import (
    get_body_aabb,
    get_body_bottom_z,
    get_body_pose,
    get_body_top_z,
    quat_to_euler,
)
from src.trajectory.joint_trajectory import interpolate_joint_trajectory
from src.utils.logger import TrajectoryLogger


def move_body_delta_z(body_id: int, delta_z: float):
    """Shift a body's base position along z by delta_z."""
    pos, orn = p.getBasePositionAndOrientation(body_id)
    new_pos = [pos[0], pos[1], pos[2] + delta_z]
    p.resetBasePositionAndOrientation(body_id, new_pos, orn)


def place_body_bottom_at_z(body_id: int, target_xy, target_bottom_z: float, clearance: float = 0.0):
    """
    Place a body's bottom surface at a desired z level.
    """
    current_pos, current_orn = p.getBasePositionAndOrientation(body_id)
    body_bottom_z = get_body_bottom_z(body_id)
    delta_z = target_bottom_z - body_bottom_z + clearance

    new_pos = [target_xy[0], target_xy[1], current_pos[2] + delta_z]
    p.resetBasePositionAndOrientation(body_id, new_pos, current_orn)


def print_body_link_aabbs(body_id: int):
    import pybullet as p

    num_joints = p.getNumJoints(body_id)

    print("\n=== Body Link AABBs ===")

    # Base link
    aabb_min, aabb_max = p.getAABB(body_id, -1)
    print(f"link_index=-1 | name=base | aabb_min={aabb_min} | aabb_max={aabb_max}")

    for link_idx in range(num_joints):
        info = p.getJointInfo(body_id, link_idx)
        link_name = info[12].decode("utf-8")
        aabb_min, aabb_max = p.getAABB(body_id, link_idx)
        print(
            f"link_index={link_idx:2d} | "
            f"name={link_name:20s} | "
            f"aabb_min={aabb_min} | "
            f"aabb_max={aabb_max}"
        )


def print_scene_diagnostics(robot, objects):
    # Print diagnostics.
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

    print("\nSimulation running. Close GUI window or Ctrl+C to stop.")


def execute_waypoint_trajectory(
    env,
    robot,
    waypoints,
    tol=config.WAYPOINT_TOL,
    max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
    label="trajectory",
    logger=None,
):
    print(f"\n=== Executing {label} ===")
    print(f"Num waypoints: {len(waypoints)}")

    total_steps = 0

    for i, q_target in enumerate(waypoints):
        reached = False

        for step in range(max_steps_per_waypoint):
            robot.command_arm_and_gripper(q_target)
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

            if step_max_err < tol:
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


def main():
    env = BulletEnv(gui=config.GUI)
    env.connect()

    objects = env.load_scene()

    print_body_link_aabbs(objects.table_id)

    # 1) Correct table so its lowest point sits exactly on the plane (z = 0).
    table_bottom_z_before = get_body_bottom_z(objects.table_id)
    move_body_delta_z(
        body_id=objects.table_id,
        delta_z=(0.0 - table_bottom_z_before + config.TABLE_PLANE_CLEARANCE),
    )

    # 2) Load robot and place it on the plane beside the table.
    robot = PandaRobot()
    robot.load(
        base_position=config.PANDA_BASE_POS,
        base_orientation_euler=config.PANDA_BASE_ORN_EULER,
    )

    # 3) Place cube on top of the table.
    table_top_z = get_body_top_z(objects.table_id)
    place_body_bottom_at_z(
        body_id=objects.cube_id,
        target_xy=config.CUBE_BASE_POS[:2],
        target_bottom_z=table_top_z,
        clearance=config.CUBE_TABLE_CLEARANCE,
    )

    # Re-apply home pose holding after repositioning.
    robot.reset_home()
    robot.hold_home_pose()

    # Let the scene settle.
    for _ in range(config.SETTLE_STEPS):
        robot.hold_home_pose()
        env.step()

    print_scene_diagnostics(robot, objects)

    home_joints = robot.home_joints
    target_joints = np.array(config.PANDA_TEST_TARGET_JOINTS, dtype=float)

    forward_waypoints = interpolate_joint_trajectory(
        q_start=home_joints,
        q_goal=target_joints,
        num_waypoints=config.TRAJ_NUM_WAYPOINTS,
    )

    return_waypoints = interpolate_joint_trajectory(
        q_start=target_joints,
        q_goal=home_joints,
        num_waypoints=config.TRAJ_NUM_WAYPOINTS,
    )

    logger = TrajectoryLogger() if config.ENABLE_TRAJECTORY_LOGGING else None

    forward_ok = execute_waypoint_trajectory(
        env=env,
        robot=robot,
        waypoints=forward_waypoints,
        tol=config.WAYPOINT_TOL,
        max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
        label="home_to_target",
        logger=logger,
    )
    print(f"Forward success: {forward_ok}")

    for _ in range(config.SETTLE_STEPS):
        robot.command_arm_and_gripper(target_joints)
        env.step()

    return_ok = execute_waypoint_trajectory(
        env=env,
        robot=robot,
        waypoints=return_waypoints,
        tol=config.WAYPOINT_TOL,
        max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
        label="target_to_home",
        logger=logger,
    )
    print(f"Return success: {return_ok}")

    # Print final error after round trip.
    final_error = robot.get_joint_position_error(robot.home_joints)
    print("\n=== Final Home Return Error ===")
    print("Final joint error:", final_error)
    print("Max abs error:", float(abs(final_error).max()))

    if logger is not None:
        logger.save_json(config.TRAJECTORY_LOG_PATH)
        print(f"Saved trajectory log to {config.TRAJECTORY_LOG_PATH}")

    try:
        while True:
            robot.hold_home_pose()
            env.step()
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()