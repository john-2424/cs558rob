import numpy as np

from src import config


def compute_reward(ee_pos, cube_pos, qd_residual,
                   prev_ee_cube_dist, grasp_ready, cube_lifted,
                   cube_fallen, grasp_bonus_given, phase=None,
                   milestones_given=None):
    ee_cube_dist = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(cube_pos)))

    # Dense approach reward: positive when EE moves closer to cube
    r_approach = config.REWARD_ALPHA * (prev_ee_cube_dist - ee_cube_dist)

    # Penalize large residual to keep corrections small
    residual = np.asarray(qd_residual, dtype=float)
    r_residual = -config.REWARD_BETA * float(np.mean(np.abs(residual)))

    # Smoothness penalty
    r_smooth = -config.REWARD_GAMMA * float(np.mean(residual ** 2))

    # Milestone cascade: one-time bonuses on first crossing of each distance
    # threshold during approach phases. Each flag in milestones_given ("m08",
    # "m05", "m03") is set by the caller once the bonus fires; reward is zero
    # on subsequent steps so hovering pays nothing.
    r_milestone = 0.0
    milestones_fired = {"m08": False, "m05": False, "m03": False}
    if phase is not None and phase in (0, 1) and milestones_given is not None:
        if not milestones_given.get("m08", False) and ee_cube_dist < config.MILESTONE_DIST_08:
            r_milestone += config.REWARD_MILESTONE_08
            milestones_fired["m08"] = True
        if not milestones_given.get("m05", False) and ee_cube_dist < config.MILESTONE_DIST_05:
            r_milestone += config.REWARD_MILESTONE_05
            milestones_fired["m05"] = True
        if not milestones_given.get("m03", False) and ee_cube_dist < config.MILESTONE_DIST_03:
            r_milestone += config.REWARD_MILESTONE_03
            milestones_fired["m03"] = True

    # One-time grasp readiness bonus
    r_grasp = 0.0
    if grasp_ready and not grasp_bonus_given:
        r_grasp = config.REWARD_DELTA

    # Terminal lift success
    r_lift = config.REWARD_EPSILON if cube_lifted else 0.0

    # Terminal failure
    r_fail = -config.REWARD_ZETA if cube_fallen else 0.0

    total = r_approach + r_residual + r_smooth + r_milestone + r_grasp + r_lift + r_fail

    info = {
        "r_approach": r_approach,
        "r_residual": r_residual,
        "r_smooth": r_smooth,
        "r_milestone": r_milestone,
        "r_grasp": r_grasp,
        "r_lift": r_lift,
        "r_fail": r_fail,
        "ee_cube_dist": ee_cube_dist,
        "milestones_fired": milestones_fired,
    }

    return total, ee_cube_dist, info
