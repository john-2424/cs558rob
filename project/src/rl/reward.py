import numpy as np

from src import config


def compute_reward(ee_pos, grasp_target_pos, qd_residual,
                   prev_ee_target_dist, grasp_ready, cube_lifted,
                   cube_fallen, grasp_bonus_given, phase=None,
                   milestones_given=None, cube_pos=None,
                   bracket_score=None, worst_tip_above_top=None,
                   lenient_ready=False, lenient_bonus_given=False,
                   attempt_fired=False):
    # Distance to the physical grasp target pose (cube XY + cube_top_z +
    # PICK_DESCEND_Z_OFFSET). Using this instead of cube-center distance
    # prevents the "plant EE inside the cube" local optimum: the reward
    # attractor is now a physically graspable pose, not a point inside the
    # cube's body.
    ee_target_dist = float(
        np.linalg.norm(np.asarray(ee_pos) - np.asarray(grasp_target_pos))
    )

    # Dense approach reward: positive when EE moves closer to grasp target
    r_approach = config.REWARD_ALPHA * (prev_ee_target_dist - ee_target_dist)

    # Penalize large residual to keep corrections small
    residual = np.asarray(qd_residual, dtype=float)
    r_residual = -config.REWARD_BETA * float(np.mean(np.abs(residual)))

    # Smoothness penalty
    r_smooth = -config.REWARD_GAMMA * float(np.mean(residual ** 2))

    # Milestone cascade on distance to grasp pose. One-time bonuses only.
    r_milestone = 0.0
    milestones_fired = {"m08": False, "m05": False, "m03": False}
    if phase is not None and phase in (0, 1) and milestones_given is not None:
        if not milestones_given.get("m08", False) and ee_target_dist < config.MILESTONE_DIST_08:
            r_milestone += config.REWARD_MILESTONE_08
            milestones_fired["m08"] = True
        if not milestones_given.get("m05", False) and ee_target_dist < config.MILESTONE_DIST_05:
            r_milestone += config.REWARD_MILESTONE_05
            milestones_fired["m05"] = True
        if not milestones_given.get("m03", False) and ee_target_dist < config.MILESTONE_DIST_03:
            r_milestone += config.REWARD_MILESTONE_03
            milestones_fired["m03"] = True

    # Terminal geometric shaping: paid ONCE at the grasp attempt moment,
    # scaled by geometry achieved at that instant. Per-step version created a
    # hover-and-farm local optimum (zero grasps across 8000 episodes). Now the
    # reward only pays when the policy commits to an attempt.
    r_geom = 0.0
    if attempt_fired and bracket_score is not None and worst_tip_above_top is not None:
        r_geom += config.REWARD_W_BRACKET * float(bracket_score)
        below_top_depth = max(0.0, -float(worst_tip_above_top))
        r_geom += config.REWARD_W_BELOW_TOP * min(1.0, below_top_depth / 0.01)

    # One-shot attempt bonus: rewards the policy for committing to a grasp,
    # regardless of outcome. Dominates any park-forever strategy.
    r_attempt = config.REWARD_ATTEMPT_BONUS if attempt_fired else 0.0

    # Per-step time cost: applied only pre-attach (phases 0=PRE_GRASP,
    # 1=GRASP_DESCEND). Once the grasp is attached and phase==LIFT, time
    # pressure would cause the policy to rush the lift and drop the cube;
    # the classical PD+IK lift trajectory is already reliable when
    # undisturbed, so we stop the clock after attach.
    r_time = -config.REWARD_TIME_COST if phase in (0, 1) else 0.0

    # Lenient grasp bonus: fires on proximity+contact readiness without the
    # strict bracket/below-top gate. Decouples learning signal from attach
    # physics so the policy gets reward for near-grasps and can iterate
    # toward the strict geometry.
    r_grasp_lenient = 0.0
    if lenient_ready and not lenient_bonus_given:
        r_grasp_lenient = config.REWARD_DELTA_LENIENT

    # One-time strict grasp readiness bonus (attach succeeded)
    r_grasp = 0.0
    if grasp_ready and not grasp_bonus_given:
        r_grasp = config.REWARD_DELTA

    # Terminal lift success
    r_lift = config.REWARD_EPSILON if cube_lifted else 0.0

    # Terminal failure
    r_fail = -config.REWARD_ZETA if cube_fallen else 0.0

    total = (r_approach + r_residual + r_smooth + r_milestone + r_geom
             + r_attempt + r_time + r_grasp_lenient + r_grasp + r_lift + r_fail)

    # Diagnostic: distance to cube center (kept for eval parity with older runs).
    ee_cube_dist_info = ee_target_dist
    if cube_pos is not None:
        ee_cube_dist_info = float(
            np.linalg.norm(np.asarray(ee_pos) - np.asarray(cube_pos))
        )

    info = {
        "r_approach": r_approach,
        "r_residual": r_residual,
        "r_smooth": r_smooth,
        "r_milestone": r_milestone,
        "r_geom": r_geom,
        "r_attempt": r_attempt,
        "r_time": r_time,
        "r_grasp_lenient": r_grasp_lenient,
        "r_grasp": r_grasp,
        "r_lift": r_lift,
        "r_fail": r_fail,
        "ee_target_dist": ee_target_dist,
        "ee_cube_dist": ee_cube_dist_info,
        "milestones_fired": milestones_fired,
        "lenient_bonus_fired": bool(r_grasp_lenient > 0),
    }

    return total, ee_target_dist, info
