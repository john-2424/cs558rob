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


def main():
    env = BulletEnv(gui=True)
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

    # place_body_bottom_at_z(
    #     body_id=robot.robot_id,
    #     target_xy=config.PANDA_BASE_POS[:2],
    #     target_bottom_z=0.0,
    #     clearance=config.ROBOT_PLANE_CLEARANCE,
    # )

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
    for _ in range(120):
        robot.hold_home_pose()
        env.step()

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