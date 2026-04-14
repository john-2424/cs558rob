import os

import numpy as np
import pybullet as p
import torch
from tensordict import TensorDict

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


def execute_residual_trajectory(
    env, robot, waypoints, actor, action_wrapper, cube_id,
    tol=config.WAYPOINT_TOL,
    max_steps_per_waypoint=config.WAYPOINT_MAX_STEPS,
    label="trajectory",
    logger=None,
    gripper_width=None,
    phase_id=0,
):
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

            # Build observation for the policy
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
            ])

            # Get action from policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])
            with torch.no_grad():
                td = actor(td)
            raw_action = td["action"].squeeze(0).numpy()

            # Apply residual
            qd_total = action_wrapper.compute_action(qd_pd, raw_action)

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
    model_path=None, perturb_xy_range=None, mode="hybrid",
):
    model_path = model_path or os.path.join(config.M2_MODEL_DIR, "final_model.pt")
    perturb_xy_range = perturb_xy_range if perturb_xy_range is not None else config.PERTURB_XY_RANGE

    # Load trained actor
    actor = load_trained_actor(model_path)
    action_wrapper = ResidualActionWrapper(mode=mode)
    print(f"Loaded model from {model_path}, mode={mode}")

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

        # Perturb cube
        rng = np.random.default_rng()
        if perturb_xy_range > 0:
            new_pos, _, (dx, dy, dyaw) = perturb_cube_pose(
                cube_id=objects.cube_id,
                nominal_pos=list(p.getBasePositionAndOrientation(objects.cube_id)[0]),
                nominal_orn_euler=config.CUBE_BASE_ORN_EULER,
                rng=rng,
                xy_range=perturb_xy_range,
                yaw_range=config.PERTURB_YAW_RANGE,
            )
            print(f"Cube perturbed: dx={dx:.4f}, dy={dy:.4f}, dyaw={dyaw:.4f}")

        robot.reset_home()
        robot.hold_home_pose()

        for _ in range(config.SETTLE_STEPS):
            robot.hold_home_pose()
            env.step()

        logger = TrajectoryLogger() if config.ENABLE_TRAJECTORY_LOGGING else None

        # Compute targets from perturbed cube
        poses = get_pick_and_place_poses_from_cube(robot, objects.cube_id, objects.table_id)
        grasp_orn = poses["grasp_orn"]

        # === Phases 1-6: RL-augmented control ===

        # Phase 1: Home
        print("\n=== Phase 1: home ===")
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 2: Pre-grasp (with residual)
        print("\n=== Phase 2: pre_grasp (residual) ===")
        ok = move_to_cartesian_target_residual(
            env, robot, poses["pre_grasp"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="pre_grasp", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH, phase_id=0,
        )
        if not ok:
            print("Failed at pre_grasp with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["pre_grasp"], grasp_orn,
                                          label="pre_grasp_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_OPEN_WIDTH)

        # Phase 3: Grasp descend (with residual)
        print("\n=== Phase 3: grasp_descend (residual) ===")
        ok = move_to_cartesian_target_residual(
            env, robot, poses["grasp"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="grasp_descend", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH, phase_id=1,
        )
        if not ok:
            print("Failed at grasp_descend with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["grasp"], grasp_orn,
                                          label="grasp_descend_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_OPEN_WIDTH)

        # Phase 4: Close gripper
        print("\n=== Phase 4: close_gripper ===")
        hold_gripper_for_steps(env, robot, config.GRIPPER_CLOSED_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 5: Validate and attach
        print("\n=== Phase 5: validate_and_attach ===")
        attached = verify_and_attach_with_retry(
            env=env, robot=robot, cube_id=objects.cube_id,
            grasp_orn=grasp_orn, base_grasp_pos=poses["grasp"],
            logger=logger,
        )
        if not attached:
            print("Grasp validation failed.")
            return

        # Phase 6: Lift (with residual)
        print("\n=== Phase 6: lift (residual) ===")
        ok = move_to_cartesian_target_residual(
            env, robot, poses["lift"], grasp_orn,
            actor, action_wrapper, objects.cube_id,
            label="lift", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH, phase_id=2,
        )
        if not ok:
            print("Failed at lift with residual, falling back to classical")
            ok = move_to_cartesian_target(env, robot, poses["lift"], grasp_orn,
                                          label="lift_fallback", logger=logger,
                                          gripper_width=config.GRIPPER_CLOSED_WIDTH)

        # === Phases 7-12: Classical pipeline (unchanged) ===

        # Phase 7: Transfer to pre-place
        print("\n=== Phase 7: transfer_pre_place ===")
        ok = move_to_cartesian_target(
            env, robot, poses["pre_place"], grasp_orn,
            label="transfer_pre_place", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )

        # Phase 8: Place descend
        print("\n=== Phase 8: place_descend ===")
        ok = move_to_cartesian_target(
            env, robot, poses["place"], grasp_orn,
            label="place_descend", logger=logger,
            gripper_width=config.GRIPPER_CLOSED_WIDTH,
        )

        # Phase 9: Detach
        print("\n=== Phase 9: detach_object ===")
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
        hold_gripper_for_steps(env, robot, config.GRIPPER_OPEN_WIDTH, config.GRIPPER_SETTLE_STEPS)

        # Phase 11: Retreat
        print("\n=== Phase 11: post_place_retreat ===")
        ok = move_to_cartesian_target(
            env, robot, poses["post_place"], grasp_orn,
            label="post_place_retreat", logger=logger,
            gripper_width=config.GRIPPER_OPEN_WIDTH,
        )

        # Phase 12: Return home
        print("\n=== Phase 12: return_home ===")
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
