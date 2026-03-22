import numpy as np
import pybullet as p


def get_body_pose(body_id: int):
    """Return body world position and orientation quaternion."""
    pos, orn = p.getBasePositionAndOrientation(body_id)
    return np.array(pos, dtype=float), np.array(orn, dtype=float)


def quat_to_euler(quat):
    """Convert quaternion to Euler angles."""
    return np.array(p.getEulerFromQuaternion(quat), dtype=float)


def get_body_aabb(body_id: int, link_index: int = -1):
    """Return AABB min/max corners for a body or a specific link."""
    aabb_min, aabb_max = p.getAABB(body_id, link_index)
    return np.array(aabb_min, dtype=float), np.array(aabb_max, dtype=float)


def get_body_bottom_z(body_id: int, link_index: int = -1) -> float:
    """Return lowest z point of a body."""
    aabb_min, _ = get_body_aabb(body_id, link_index)
    return float(aabb_min[2])


def get_body_top_z(body_id: int, link_index: int = -1) -> float:
    """Return highest z point of a body."""
    _, aabb_max = get_body_aabb(body_id, link_index)
    return float(aabb_max[2])


def get_body_height(body_id: int, link_index: int = -1) -> float:
    """Return height of a body from AABB."""
    aabb_min, aabb_max = get_body_aabb(body_id, link_index)
    return float(aabb_max[2] - aabb_min[2])