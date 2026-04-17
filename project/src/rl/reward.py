import numpy as np

from src import config


def compute_reward(ee_pos, cube_pos, qd_residual,
                   prev_ee_cube_dist, grasp_ready, cube_lifted,
                   cube_fallen, grasp_bonus_given, phase=None):
    ee_cube_dist = float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(cube_pos)))

    # Dense approach reward: positive when EE moves closer to cube
    r_approach = config.REWARD_ALPHA * (prev_ee_cube_dist - ee_cube_dist)

    # Penalize large residual to keep corrections small
    residual = np.asarray(qd_residual, dtype=float)
    r_residual = -config.REWARD_BETA * float(np.mean(np.abs(residual)))

    # Smoothness penalty
    r_smooth = -config.REWARD_GAMMA * float(np.mean(residual ** 2))

    # Dense proximity bonus during pre_grasp and grasp_descend phases.
    # Linear ramp in [0, REWARD_ETA] for ee_cube_dist in [PROXIMITY_RADIUS, 0].
    # Active in both phases so the policy gets a guiding signal during approach,
    # not only during the final descent.
    # NOTE: gym_env caps cumulative proximity per episode (PROXIMITY_REWARD_CAP)
    # to prevent reward farming (hovering near cube indefinitely).
    r_proximity = 0.0
    if phase is not None and phase in (0, 1):  # PRE_GRASP or GRASP_DESCEND
        proximity = max(0.0, config.PROXIMITY_RADIUS - ee_cube_dist) / config.PROXIMITY_RADIUS
        r_proximity = config.REWARD_ETA * proximity

    # One-time grasp readiness bonus
    r_grasp = 0.0
    if grasp_ready and not grasp_bonus_given:
        r_grasp = config.REWARD_DELTA

    # Terminal lift success
    r_lift = config.REWARD_EPSILON if cube_lifted else 0.0

    # Terminal failure
    r_fail = -config.REWARD_ZETA if cube_fallen else 0.0

    total = r_approach + r_residual + r_smooth + r_proximity + r_grasp + r_lift + r_fail

    info = {
        "r_approach": r_approach,
        "r_residual": r_residual,
        "r_smooth": r_smooth,
        "r_proximity": r_proximity,
        "r_grasp": r_grasp,
        "r_lift": r_lift,
        "r_fail": r_fail,
        "ee_cube_dist": ee_cube_dist,
    }

    return total, ee_cube_dist, info
