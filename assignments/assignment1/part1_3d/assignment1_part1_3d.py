from __future__ import division
from logging import config
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import sys


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.parent = None
        self.children = []

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

def sample_conf():
    is_goal_conf = False
    low = np.array([-2*np.pi, -2*np.pi, -np.pi])
    high = np.array([2*np.pi, 2*np.pi, np.pi])

    p_goal = 0.05
    if np.random.rand() < p_goal:
        sample_q = goal_conf
        is_goal_conf = True
    else:
        sample_q = tuple(np.random.uniform(low, high))

    return sample_q, is_goal_conf

def find_nearest(rand_node, node_list):
    q_rand = np.array(rand_node.conf)

    best = None
    best_dist = float("inf")

    for node in node_list:
        q = np.array(node.conf)
        d = np.linalg.norm(q_rand - q)
        if d < best_dist:
            best_dist = d
            best = node

    return best
        
def steer_to(rand_node, nearest_node):
    q_near = np.array(nearest_node.conf, dtype=float)
    q_rand = np.array(rand_node.conf, dtype=float)

    direction = q_rand - q_near
    dist = np.linalg.norm(direction)

    if dist == 0:
        return True

    step = 0.05
    num = int(np.ceil(dist / step))

    for i in range(1, num + 1):
        t = i / num
        q = q_near + t * direction
        q_tuple = tuple(q.tolist())

        if collision_fn(q_tuple):
            return False

    return True

def steer_to_until(rand_node, nearest_node):
    q_near = np.array(nearest_node.conf, dtype=float)
    q_rand = np.array(rand_node.conf, dtype=float)

    direction = q_rand - q_near
    dist = np.linalg.norm(direction)

    if dist == 0:
        return None

    step = 0.05
    num = int(np.ceil(dist / step))

    last_valid = None

    for i in range(1, num + 1):
        t = i / num
        q = q_near + t * direction
        q_tuple = tuple(q.tolist())

        if collision_fn(q_tuple):
            break
        last_valid = q_tuple

    if last_valid is None:
        return None

    new_node = RRT_Node(last_valid)
    return new_node

def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################
    # Root node
    root = RRT_Node(start_conf)
    node_list = [root]

    while True:
        # 1) sample
        q_rand, is_goal = sample_conf()
        rand_node = RRT_Node(q_rand)

        # 2) nearest
        near_node = find_nearest(rand_node, node_list)

        # 3) attempt connection
        if not steer_to(rand_node, near_node):
            continue

        # 4) link into tree
        rand_node.set_parent(near_node)
        near_node.add_child(rand_node)
        node_list.append(rand_node)

        # 5) if this was the goal sample, then done
        if is_goal:
            path = []
            cur = rand_node
            while cur is not None:
                path.append(cur.conf)
                cur = cur.parent
            path.reverse()
            return path

def BiRRT():
    #################################################
    # TODO your code to implement the birrt algorithm
    #################################################
    # Initialize two trees
    start_root = RRT_Node(start_conf)
    goal_root  = RRT_Node(goal_conf)

    Ta = [start_root]
    Tb = [goal_root]

    while True:
        # 1) sample random configuration
        q_rand, _ = sample_conf()
        rand_node = RRT_Node(q_rand)

        # 2) extend Ta toward q_rand
        near_a = find_nearest(rand_node, Ta)
        new_a = steer_to_until(rand_node, near_a)

        if new_a is not None:
            new_a.set_parent(near_a)
            near_a.add_child(new_a)
            Ta.append(new_a)

            # 3) try to connect Tb to new_a
            near_b = find_nearest(new_a, Tb)
            new_b = steer_to_until(new_a, near_b)

            if new_b is not None:
                new_b.set_parent(near_b)
                near_b.add_child(new_b)
                Tb.append(new_b)

                # 4) did we connect the trees?
                if np.allclose(new_b.conf, new_a.conf):
                    # reconstruct path
                    path_a = []
                    cur = new_a
                    while cur is not None:
                        path_a.append(cur.conf)
                        cur = cur.parent

                    path_b = []
                    cur = new_b.parent
                    while cur is not None:
                        path_b.append(cur.conf)
                        cur = cur.parent

                    path_a.reverse()
                    return path_a + path_b

        # 5) swap roles
        Ta, Tb = Tb, Ta

def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    # Get a path from BiRRT
    path = BiRRT()

    if path is None or len(path) < 3:
        return path

    N = 100
    step_size = 0.05

    for _ in range(N):
        L = len(path)
        if L < 3:
            break

        # Pick two non-adjacent indices
        i = np.random.randint(0, L - 2)
        j = np.random.randint(i + 2, L)

        q_start = np.array(path[i], dtype=float)
        q_end   = np.array(path[j], dtype=float)

        direction = q_end - q_start
        dist = np.linalg.norm(direction)

        if dist == 0:
            continue

        num_steps = int(np.ceil(dist / step_size))
        collision = False

        # Check straight-line interpolation
        for k in range(1, num_steps + 1):
            t = k / num_steps
            q = q_start + t * direction
            q_tuple = tuple(q.tolist())

            if collision_fn(q_tuple):
                collision = True
                break

        # If collision-free, remove intermediate points
        if not collision:
            path = path[:i+1] + path[j:]

    return path    

###############################################################################
#your implementation ends here

if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    global start_conf, goal_conf
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    
	# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    global collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    while True:
        if args.birrt:
            if args.smoothing:
                # using birrt with smoothing
                path_conf = BiRRT_smoothing()
            else:
                # using birrt without smoothing
                path_conf = BiRRT()
        else:
            # using rrt
            path_conf = RRT()

        if path_conf is None:
            # pause here
            input("no collision-free path is found within the time budget, finish?")
        else:
            # execute the path
            while True:
                for q in path_conf:
                    set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                    time.sleep(0.5)
            break
