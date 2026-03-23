from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

from src import config


@dataclass
class Node:
    q: np.ndarray
    parent: Optional[int]
    cost: float = 0.0


@dataclass
class PlannerResult:
    success: bool
    path: List[np.ndarray]
    num_nodes: int
    message: str


class JointSpaceRRTStarPlanner:
    """
    Joint-space planner scaffold for a 7-DoF Panda arm.

    This is intentionally a scaffold:
    - supports direct path validation
    - defines all core utilities needed for later RRT*
    - returns a path as a list of joint vectors

    Later we will add:
    - tree expansion
    - rewiring
    - neighbor search
    - goal biasing
    - collision-aware edge validation
    """

    def __init__(
        self,
        robot_id: int,
        arm_joint_indices: List[int],
        step_size: float = 0.08,
        goal_threshold: float = 0.10,
        max_iterations: int = 1000,
        goal_sample_rate: float = 0.10,
        edge_check_resolution: float = 0.03,
        phase_label: str = "",
        attached_body_id: Optional[int] = None,
    ):
        self.robot_id = robot_id
        self.arm_joint_indices = arm_joint_indices
        self.num_joints = len(arm_joint_indices)

        self.step_size = float(step_size)
        self.goal_threshold = float(goal_threshold)
        self.max_iterations = int(max_iterations)
        self.goal_sample_rate = float(goal_sample_rate)
        self.edge_check_resolution = float(edge_check_resolution)

        self.lower_limits, self.upper_limits = self._get_joint_limits()

        self.rng = np.random.default_rng(config.PLANNER_RANDOM_SEED)

        self.phase_label = phase_label
        self.attached_body_id = attached_body_id

    # ------------------------------------------------------------------
    # Joint-space utilities
    # ------------------------------------------------------------------

    def _get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []

        for joint_idx in self.arm_joint_indices:
            info = p.getJointInfo(self.robot_id, joint_idx)
            lower.append(info[8])
            upper.append(info[9])

        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.clip(q, self.lower_limits, self.upper_limits)

    def in_joint_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(
            np.all(q >= self.lower_limits) and np.all(q <= self.upper_limits)
        )

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        return float(np.linalg.norm(q2 - q1))

    def sample_random_configuration(self) -> np.ndarray:
        return self.rng.uniform(self.lower_limits, self.upper_limits)

    def sample_configuration(self, q_goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_sample_rate:
            return np.asarray(q_goal, dtype=float).copy()
        return self.sample_random_configuration()

    def nearest_node_index(self, nodes: List[Node], q_query: np.ndarray) -> int:
        dists = [self.distance(node.q, q_query) for node in nodes]
        return int(np.argmin(dists))

    def near_node_indices(self, nodes: List[Node], q_query: np.ndarray) -> List[int]:
        """
        Return nearby node indices within a fixed radius, capped for efficiency.
        """
        pairs = []
        for idx, node in enumerate(nodes):
            d = self.distance(node.q, q_query)
            if d <= config.RRT_STAR_NEIGHBOR_RADIUS:
                pairs.append((idx, d))

        pairs.sort(key=lambda x: x[1])

        if config.RRT_STAR_MAX_NEIGHBORS is not None:
            pairs = pairs[: config.RRT_STAR_MAX_NEIGHBORS]

        return [idx for idx, _ in pairs]

    def edge_cost(self, q_from: np.ndarray, q_to: np.ndarray) -> float:
        return self.distance(q_from, q_to)
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)

        delta = q_to - q_from
        dist = np.linalg.norm(delta)

        if dist < 1e-9:
            return q_from.copy()

        if dist <= self.step_size:
            q_new = q_to.copy()
        else:
            q_new = q_from + (self.step_size / dist) * delta

        return self.clip_to_limits(q_new)

    def is_goal_reached(self, q: np.ndarray, q_goal: np.ndarray) -> bool:
        return self.distance(q, q_goal) <= self.goal_threshold
    
    # ------------------------------------------------------------------
    # Simulator / collision helpers
    # ------------------------------------------------------------------

    def get_current_joint_state(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        return np.array([state[0] for state in joint_states], dtype=float)

    def set_joint_state(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.num_joints,):
            raise ValueError(f"Expected shape {(self.num_joints,)}, got {q.shape}")

        for joint_idx, joint_val in zip(self.arm_joint_indices, q):
            p.resetJointState(self.robot_id, joint_idx, float(joint_val))

    def restore_joint_state(self, q_saved: np.ndarray) -> None:
        q_saved = np.asarray(q_saved, dtype=float)
        for joint_idx, joint_val in zip(self.arm_joint_indices, q_saved):
            p.resetJointState(self.robot_id, joint_idx, float(joint_val))
        
    def in_collision(self, q: np.ndarray) -> bool:
        """
        State-safe collision check for a single configuration.

        Temporarily moves the robot to q, performs collision detection,
        then restores the original joint state before returning.
        """
        q = self.clip_to_limits(q)
        q_saved = self.get_current_joint_state()

        try:
            self.set_joint_state(q)
            p.performCollisionDetection()
            contacts = p.getContactPoints(bodyA=self.robot_id)
            return len(contacts) > 0
        finally:
            self.restore_joint_state(q_saved)
            p.performCollisionDetection()
    
    def collision_free_configuration(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)

        if not self.in_joint_limits(q):
            return False

        current_q = self.get_current_joint_state()
        self.set_joint_state(q)

        try:
            p.performCollisionDetection()

            contacts = p.getContactPoints(bodyA=self.robot_id)
            for c in contacts:
                if self._is_allowed_contact(c):
                    continue

                contact_distance = c[8] if len(c) > 8 else -1.0
                if contact_distance < 0.0:
                    return False

            # Self collision
            self_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
            for c in self_contacts:
                if self._is_allowed_contact(c):
                    continue

                contact_distance = c[8] if len(c) > 8 else -1.0
                if contact_distance < 0.0:
                    return False

            return True

        finally:
            self.set_joint_state(current_q)

    def edge_in_collision(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        """
        State-safe edge collision check.

        Interpolates from q_from to q_to and checks whether any sampled
        intermediate configuration is in collision, while respecting
        allowed contacts for attached-motion phases.
        """
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)

        dist = self.distance(q_from, q_to)
        num_checks = max(2, int(np.ceil(dist / self.edge_check_resolution)) + 1)

        q_saved = self.get_current_joint_state()

        try:
            for alpha in np.linspace(0.0, 1.0, num_checks):
                q_interp = (1.0 - alpha) * q_from + alpha * q_to
                q_interp = self.clip_to_limits(q_interp)

                self.set_joint_state(q_interp)
                p.performCollisionDetection()

                contacts = p.getContactPoints(bodyA=self.robot_id)
                for c in contacts:
                    if self._is_allowed_contact(c):
                        continue

                    contact_distance = c[8] if len(c) > 8 else -1.0
                    if contact_distance < 0.0:
                        return True

                self_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
                for c in self_contacts:
                    if self._is_allowed_contact(c):
                        continue

                    contact_distance = c[8] if len(c) > 8 else -1.0
                    if contact_distance < 0.0:
                        return True

            return False
        finally:
            self.restore_joint_state(q_saved)
            p.performCollisionDetection()
    
    def collision_free_edge(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        return not self.edge_in_collision(q_from, q_to)

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def reconstruct_path(self, nodes: List[Node], goal_idx: int) -> List[np.ndarray]:
        path = []
        idx = goal_idx

        while idx is not None:
            path.append(nodes[idx].q.copy())
            idx = nodes[idx].parent

        path.reverse()
        return path

    def try_connect_to_goal(
        self,
        nodes: List[Node],
        parent_idx: int,
        q_goal: np.ndarray,
    ) -> Optional[int]:
        """
        Try to connect a node directly to the goal.
        """
        q_goal = self.clip_to_limits(q_goal)
        q_parent = nodes[parent_idx].q

        if not self.collision_free_edge(q_parent, q_goal):
            return None

        goal_cost = nodes[parent_idx].cost + self.edge_cost(q_parent, q_goal)
        goal_node = Node(q=q_goal.copy(), parent=parent_idx, cost=goal_cost)
        nodes.append(goal_node)
        return len(nodes) - 1
    
    def choose_best_parent(
        self,
        nodes: List[Node],
        near_indices: List[int],
        nearest_idx: int,
        q_new: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Choose the lowest-cost valid parent for q_new.

        Falls back to nearest_idx if no better valid parent exists.
        """
        best_parent_idx = nearest_idx
        best_cost = nodes[nearest_idx].cost + self.edge_cost(nodes[nearest_idx].q, q_new)

        for idx in near_indices:
            q_near = nodes[idx].q

            if not self.collision_free_edge(q_near, q_new):
                continue

            candidate_cost = nodes[idx].cost + self.edge_cost(q_near, q_new)

            if candidate_cost < best_cost:
                best_parent_idx = idx
                best_cost = candidate_cost

        return best_parent_idx, best_cost
    
    def propagate_descendant_costs(self, nodes: List[Node], parent_idx: int) -> None:
        """
        After rewiring a node, recursively update descendant costs.
        """
        for idx, node in enumerate(nodes):
            if node.parent == parent_idx:
                nodes[idx].cost = nodes[parent_idx].cost + self.edge_cost(nodes[parent_idx].q, nodes[idx].q)
                self.propagate_descendant_costs(nodes, idx)
    
    def rewire(
        self,
        nodes: List[Node],
        new_idx: int,
        near_indices: List[int],
    ) -> int:
        """
        Rewire nearby nodes through the newly added node if that lowers cost.

        Returns the number of rewired nodes.
        """
        rewired_count = 0
        q_new = nodes[new_idx].q
        new_cost = nodes[new_idx].cost

        for idx in near_indices:
            if idx == new_idx:
                continue
            if idx == nodes[new_idx].parent:
                continue

            q_near = nodes[idx].q
            edge_cost = self.edge_cost(q_new, q_near)
            candidate_cost = new_cost + edge_cost

            if candidate_cost >= nodes[idx].cost:
                continue

            if not self.collision_free_edge(q_new, q_near):
                continue

            nodes[idx].parent = new_idx
            nodes[idx].cost = candidate_cost
            self.propagate_descendant_costs(nodes, idx)
            rewired_count += 1

        return rewired_count
    
    def interpolate_path(self, path: List[np.ndarray], max_segment_length: float = 0.05) -> List[np.ndarray]:
        if len(path) == 0:
            return []

        dense_path = [np.asarray(path[0], dtype=float)]

        for i in range(1, len(path)):
            q_prev = np.asarray(path[i - 1], dtype=float)
            q_curr = np.asarray(path[i], dtype=float)

            dist = self.distance(q_prev, q_curr)
            num_steps = max(2, int(np.ceil(dist / max_segment_length)) + 1)

            for alpha in np.linspace(0.0, 1.0, num_steps)[1:]:
                q_interp = (1.0 - alpha) * q_prev + alpha * q_curr
                dense_path.append(np.asarray(q_interp, dtype=float))

        return dense_path

    def shortcut_path(
        self,
        path: List[np.ndarray],
        max_passes: int = 3,
    ) -> List[np.ndarray]:
        """
        Shortcut-smooth a sparse path by removing unnecessary intermediate nodes.

        If path[i] can connect directly to path[j] collision-free, then all
        nodes between i and j are removed.
        """
        if len(path) <= 2:
            return [np.asarray(q, dtype=float).copy() for q in path]

        smoothed = [np.asarray(q, dtype=float).copy() for q in path]

        for _ in range(max_passes):
            changed = False
            new_path = [smoothed[0]]
            i = 0

            while i < len(smoothed) - 1:
                best_j = i + 1

                # Try to jump as far forward as possible
                for j in range(len(smoothed) - 1, i + 1, -1):
                    if self.collision_free_edge(smoothed[i], smoothed[j]):
                        best_j = j
                        break

                new_path.append(smoothed[best_j])

                if best_j > i + 1:
                    changed = True

                i = best_j

            smoothed = new_path

            if not changed:
                break

        return smoothed
    
    def path_length(self, path: List[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(path)):
            total += self.distance(path[i - 1], path[i])
        return total

    def node_path_cost(self, nodes: List[Node], idx: int) -> float:
        return nodes[idx].cost
    
    # ------------------------------------------------------------------
    # Initial scaffold planning API
    # ------------------------------------------------------------------

    def direct_path(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        q_start = self.clip_to_limits(q_start)
        q_goal = self.clip_to_limits(q_goal)

        if not self.in_joint_limits(q_start):
            return PlannerResult(False, [], 0, "Start is outside joint limits.")
        if not self.in_joint_limits(q_goal):
            return PlannerResult(False, [], 0, "Goal is outside joint limits.")
        if not self.collision_free_configuration(q_start):
            return PlannerResult(False, [], 0, "Start is in collision.")
        if not self.collision_free_configuration(q_goal):
            return PlannerResult(False, [], 0, "Goal is in collision.")
        if not self.collision_free_edge(q_start, q_goal):
            return PlannerResult(False, [], 0, "Direct edge is in collision.")

        return PlannerResult(True, [q_start.copy(), q_goal.copy()], 2, "Direct path found.")

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> PlannerResult:
        """
        RRT / RRT* planner:
        1. Try direct path first
        2. If blocked, grow a tree in joint space
        3. For RRT*: choose best parent from nearby nodes and rewire
        4. Try to connect toward the goal
        5. Return a sparse path if successful
        """
        q_start = self.clip_to_limits(q_start)
        q_goal = self.clip_to_limits(q_goal)

        if not self.in_joint_limits(q_start):
            return PlannerResult(False, [], 0, "Start is outside joint limits.")
        if not self.in_joint_limits(q_goal):
            return PlannerResult(False, [], 0, "Goal is outside joint limits.")
        if not self.collision_free_configuration(q_start):
            return PlannerResult(False, [], 0, "Start is in collision.")
        if not self.collision_free_configuration(q_goal):
            return PlannerResult(False, [], 0, "Goal is in collision.")

        # Fast shortcut
        direct_result = self.direct_path(q_start, q_goal)
        if direct_result.success:
            return direct_result

        nodes: List[Node] = [Node(q=q_start.copy(), parent=None, cost=0.0)]
        best_goal_idx = None
        total_rewires = 0

        for iteration in range(self.max_iterations):
            q_rand = self.sample_configuration(q_goal)
            nearest_idx = self.nearest_node_index(nodes, q_rand)
            q_near = nodes[nearest_idx].q

            q_new = self.steer(q_near, q_rand)

            if self.distance(q_near, q_new) < 1e-9:
                continue

            if not self.collision_free_configuration(q_new):
                continue

            if not self.collision_free_edge(q_near, q_new):
                continue

            if config.USE_RRT_STAR:
                near_indices = self.near_node_indices(nodes, q_new)
                parent_idx, new_cost = self.choose_best_parent(
                    nodes=nodes,
                    near_indices=near_indices,
                    nearest_idx=nearest_idx,
                    q_new=q_new,
                )
            else:
                near_indices = []
                parent_idx = nearest_idx
                new_cost = nodes[parent_idx].cost + self.edge_cost(nodes[parent_idx].q, q_new)

            new_node = Node(q=q_new.copy(), parent=parent_idx, cost=new_cost)
            nodes.append(new_node)
            new_idx = len(nodes) - 1

            if config.USE_RRT_STAR:
                total_rewires += self.rewire(
                    nodes=nodes,
                    new_idx=new_idx,
                    near_indices=near_indices,
                )

            goal_idx = None

            if self.is_goal_reached(q_new, q_goal):
                goal_idx = self.try_connect_to_goal(nodes, new_idx, q_goal)
            elif self.collision_free_edge(q_new, q_goal):
                goal_idx = self.try_connect_to_goal(nodes, new_idx, q_goal)

            if goal_idx is not None:
                if best_goal_idx is None:
                    best_goal_idx = goal_idx
                else:
                    if nodes[goal_idx].cost < nodes[best_goal_idx].cost:
                        best_goal_idx = goal_idx

                # For plain RRT, return immediately on first solution.
                if not config.USE_RRT_STAR:
                    path = self.reconstruct_path(nodes, best_goal_idx)
                    return PlannerResult(
                        True,
                        path,
                        len(nodes),
                        f"RRT path found in {iteration + 1} iterations.",
                    )

            if (iteration + 1) % config.PLANNER_DEBUG_PRINT_EVERY == 0:
                if best_goal_idx is None:
                    best_goal_cost_str = "None"
                else:
                    best_goal_cost_str = f"{nodes[best_goal_idx].cost:.4f}"

                print(
                    f"[planner] iter={iteration+1} | "
                    f"nodes={len(nodes)} | "
                    f"last_dist_to_goal={self.distance(q_new, q_goal):.4f} | "
                    f"best_goal_cost={best_goal_cost_str} | "
                    f"rewires={total_rewires}"
                )

        # RRT* returns the best solution found across all iterations.
        if best_goal_idx is not None:
            path = self.reconstruct_path(nodes, best_goal_idx)
            planner_name = "RRT*" if config.USE_RRT_STAR else "RRT"
            return PlannerResult(
                True,
                path,
                len(nodes),
                f"{planner_name} path found with best cost {nodes[best_goal_idx].cost:.4f} after {self.max_iterations} iterations.",
            )

        planner_name = "RRT*" if config.USE_RRT_STAR else "RRT"
        return PlannerResult(
            False,
            [],
            len(nodes),
            f"{planner_name} failed after {self.max_iterations} iterations.",
        )
    
    def get_contact_count(self, q: np.ndarray) -> int:
        q = self.clip_to_limits(q)
        q_saved = self.get_current_joint_state()

        try:
            self.set_joint_state(q)
            p.performCollisionDetection()
            contacts = p.getContactPoints(bodyA=self.robot_id)
            return len(contacts)
        finally:
            self.restore_joint_state(q_saved)
            p.performCollisionDetection()

    def _is_allowed_contact(self, contact) -> bool:
        body_a = contact[1]
        body_b = contact[2]
        link_a = contact[3]
        link_b = contact[4]

        # Ignore contacts involving the attached object during attached-motion phases
        if (
            config.PLANNER_IGNORE_ATTACHED_OBJECT_COLLISIONS
            and self.attached_body_id is not None
            and self.phase_label in config.PLANNER_ATTACHED_MOTION_LABELS
        ):
            if body_a == self.attached_body_id or body_b == self.attached_body_id:
                return True

        # During attached motion, allow gentle robot/table contact checks to be skipped.
        # This prevents near-table placement starts from being labeled invalid too aggressively.
        if (
            config.PLANNER_ALLOW_CONTACT_DURING_ATTACHED_MOTION
            and self.phase_label in config.PLANNER_ATTACHED_MOTION_LABELS
        ):
            # body 0 is often the plane/table/cube depending on scene ordering,
            # so we only skip if one body is the robot and the other is NOT another robot link.
            if body_a == self.robot_id or body_b == self.robot_id:
                other_body = body_b if body_a == self.robot_id else body_a
                if other_body != self.robot_id:
                    return True

        return False