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
        if child.parent == self.parent:
            self.children.append(child)
        else:
            print(f'{child} This child does not belong to parent {self.parent}')

def sample_conf():
    is_goal_conf = False
    low = np.array([-2*np.pi, -2*np.pi, -np.pi])
    high = np.array([2*np.pi, 2*np.pi, np.pi])

    # Based on probability we choose to either select goal or rand conf
    p_goal = 0.05
    if np.random.rand() < p_goal:
        # selects goal conf point
        sample_q = goal_conf
        is_goal_conf = True
    else:
        # selects random conf point
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
    pass

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
    pass

def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    pass     

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
