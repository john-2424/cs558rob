# =========================
# Simulation
# =========================
GUI = True
TIME_STEP = 1.0 / 240.0
GRAVITY = -9.81
USE_REALTIME = False

# =========================
# Scene geometry / placement
# =========================

# Initial placeholder positions; final z values will be corrected from geometry.
TABLE_BASE_POS = [0.65, 0.0, 0.0]
TABLE_BASE_ORN_EULER = [0.0, 0.0, 0.0]

# Robot is beside the table on the plane.
PANDA_BASE_POS = [0.65, -0.75, 0.0]
PANDA_BASE_ORN_EULER = [0.0, 0.0, 1.5708]

# Cube should sit on the table.
CUBE_BASE_POS = [0.70, 0.0, 0.0]
CUBE_BASE_ORN_EULER = [0.0, 0.0, 0.0]

TABLE_PLANE_CLEARANCE = 0.0
ROBOT_PLANE_CLEARANCE = 0.002
CUBE_TABLE_CLEARANCE = 0.002

# =========================
# Robot nominal joint states
# =========================
PANDA_HOME_JOINTS = [
    0.0,
    -0.785,
    0.0,
    -2.356,
    0.0,
    1.571,
    0.785,
]

PANDA_TEST_TARGET_JOINTS = [
    0.30,
    -0.55,
    0.00,
    -1.95,
    0.00,
    1.85,
    1.00,
]

# =========================
# Motion execution
# =========================
MAX_MOTION_STEPS = 1200
SETTLE_STEPS = 60
POST_MOTION_SETTLE_STEPS = 40

TRAJ_NUM_WAYPOINTS = 20
WAYPOINT_TOL = 0.010
WAYPOINT_MAX_STEPS = 220

# =========================
# Logging / results
# =========================
RESULTS_DIR = "results"
M1_RESULTS_DIR = "results/m1"
TRAJECTORY_LOG_PATH = "results/m1/trajectory_log.json"

ENABLE_TRAJECTORY_LOGGING = True
PRINT_WAYPOINT_EVERY = 1

PLOT_DIR = "results/m1/plots"
TRAJECTORY_SUMMARY_PATH = "results/m1/trajectory_summary.txt"

# =========================
# Custom PD controller
# =========================
PD_KP = [5.0, 5.0, 5.0, 5.5, 4.0, 4.0, 3.5]
PD_KD = [0.50, 0.50, 0.50, 0.55, 0.40, 0.40, 0.35]
PD_MAX_JOINT_VEL = [2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0]
# PD_KP = [5.0, 5.0, 5.0, 5.5, 4.0, 4.0, 3.5]
# PD_KD = [0.50, 0.50, 0.50, 0.55, 0.40, 0.40, 0.35]
# PD_MAX_JOINT_VEL = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]

# =========================
# Planner scaffold / RRT*
# =========================
PLANNER_RANDOM_SEED = 7
PLANNER_STEP_SIZE = 0.06
PLANNER_GOAL_THRESHOLD = 0.10
PLANNER_MAX_ITERATIONS = 3000
PLANNER_GOAL_SAMPLE_RATE = 0.20
PLANNER_EDGE_CHECK_RESOLUTION = 0.03
PLANNER_INTERP_RESOLUTION = 0.080
USE_PLANNER = True
PLANNER_FALLBACK_TO_INTERPOLATION = True
PLANNER_DEBUG_PRINT_EVERY = 100
ENABLE_PATH_SHORTCUT_SMOOTHING = True
PATH_SHORTCUT_MAX_PASSES = 1
PATH_SHORTCUT_MAX_SEGMENT_CHECKS = 200
USE_RRT_STAR = True
RRT_STAR_NEIGHBOR_RADIUS = 0.50
RRT_STAR_MAX_NEIGHBORS = 80

# =========================
# Planner validation targets
# =========================
USE_BLOCKED_PATH_TEST = False
PLANNER_BLOCKED_SEARCH_MAX_SAMPLES = 300
PLANNER_BLOCKED_SEARCH_SEED = 7
PLANNER_BLOCKED_TEST_TARGETS = [
    [0.35, -0.25, 0.20, -2.10, 0.15, 1.95, 0.60],
    [0.55, -0.10, -0.35, -2.00, 0.35, 2.10, 1.10],
    [-0.25, -0.55, 0.40, -1.95, -0.30, 1.85, 0.25],
    [0.70, -0.60, -0.20, -1.80, 0.45, 2.00, 1.40],
    [-0.45, -0.20, 0.55, -2.25, 0.20, 1.75, -0.10],
]

# =========================
# Pick-and-place demo
# =========================
RUN_PICK_PLACE_DEMO = True

# Gripper
GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_FORCE = 80.0
GRIPPER_SETTLE_STEPS = 40

# Cartesian task geometry
PICK_HOVER_Z_OFFSET = 0.18

# Lower than before so the hand comes closer to the cube before grasp.
PICK_DESCEND_Z_OFFSET = -0.010

PLACE_HOVER_Z_OFFSET = 0.18
PLACE_DESCEND_Z_OFFSET = 0.030
PLACE_RELEASE_CLEARANCE = 0.001
PLACE_SNAP_BEFORE_RELEASE = False
PLACE_RELEASE_SETTLE_STEPS = 80
POST_PLACE_RETREAT_Z_OFFSET = 0.15

# Optional retry if the first descend is still too high
GRASP_DESCEND_RETRY_DELTA_Z = 0.030

# Place target on the same table
PLACE_XY = [0.58, -0.12]

# End-effector top-down grasp orientation
GRASP_EE_EULER = [3.14159, 0.0, 1.5708]

# IK / grasp helpers
IK_MAX_ITERATIONS = 200
IK_RESIDUAL_THRESHOLD = 1e-4
ATTACH_PARENT_LINK_INDEX = 11

# Grasp validation
ENABLE_GRASP_VALIDATION = True
GRASP_MAX_EE_TO_CUBE_CENTER_DIST = 0.110
GRASP_REQUIRE_CONTACT = True
GRASP_MIN_TOTAL_CONTACTS = 1
GRASP_MIN_FINGER_CONTACTS = 1
GRASP_MAX_FINGER_TO_CUBE_DIST = 0.030
GRASP_USE_FINGER_DISTANCE = True

GRASP_DESCEND_WAYPOINT_TOL = 0.012
GRASP_DESCEND_WAYPOINT_MAX_STEPS = 260

# Finger links on Panda
LEFT_FINGER_LINK_INDEX = 9
RIGHT_FINGER_LINK_INDEX = 10

# Results / phase naming
PICK_PLACE_PHASE_PREFIX = "pick_place"

# Pinch-point targeting
USE_PINCH_POINT_TARGETING = False

# Optional manual fallback if you ever want to override automatically computed offset.
MANUAL_PINCH_OFFSET_X = 0.0
MANUAL_PINCH_OFFSET_Y = 0.0
MANUAL_PINCH_OFFSET_Z = 0.0
USE_MANUAL_PINCH_OFFSET = False

# Manual Cartesian grasp bias to compensate for wrist-frame / visual contact offset
GRASP_TARGET_BIAS_X = 0.0
GRASP_TARGET_BIAS_Y = 0.0
GRASP_TARGET_BIAS_Z = 0.0

# Apply the same XY bias to lift as well
APPLY_GRASP_BIAS_TO_LIFT = True

# Final home refinement
FINAL_HOME_SETTLE_STEPS = 180

# Planner behavior in contact-rich phases
PLANNER_IGNORE_ATTACHED_OBJECT_COLLISIONS = True
PLANNER_ALLOW_CONTACT_DURING_ATTACHED_MOTION = True
PLANNER_ATTACHED_MOTION_LABELS = {
    "lift",
    "transfer_pre_place",
    "place_descend",
}

# =========================
# M2: Residual RL
# =========================
RUN_M2_TRAINING = True
RUN_M2_EVALUATION = False
RUN_M2_RESIDUAL_DEMO = False

# Residual bounds
RESIDUAL_MAX = 0.5

# RL simulation
RL_SIM_SUBSTEPS = 4
RL_MAX_EPISODE_STEPS = 2000

# Perturbation
PERTURB_XY_RANGE = 0.10
PERTURB_YAW_RANGE = 0.4
PERTURB_LEVELS = [0.06, 0.08, 0.10, 0.12]

# Reward shaping
REWARD_ALPHA = 10.0
REWARD_BETA = 0.1
REWARD_GAMMA = 0.05
REWARD_DELTA = 50.0
REWARD_EPSILON = 100.0
REWARD_ZETA = 50.0

# PPO hyperparameters
PPO_TOTAL_TIMESTEPS = 1_000_000
PPO_LR = 3e-4
PPO_FRAMES_PER_BATCH = 2048
PPO_MINI_BATCH_SIZE = 64
PPO_EPOCHS = 10
PPO_CLIP_EPSILON = 0.2
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEFF = 0.01

# Parallel env workers for data collection (set to 1 to disable multiprocessing)
PPO_NUM_COLLECTOR_WORKERS = 8

# Episode-reward threshold for counting an episode as a "success" in training logs.
# Lift bonus is REWARD_EPSILON (100.0); grasp bonus is REWARD_DELTA (50.0).
EP_SUCCESS_REWARD_THRESHOLD = 50.0

# Evaluation
EVAL_EPISODES_PER_LEVEL = 50
EVAL_NUM_WORKERS = 8  # parallel workers for episode rollout; set to 1 for serial

# M2 paths
M2_RESULTS_DIR = "results/m2"
M2_MODEL_DIR = "results/m2/models"
M2_TB_LOG_DIR = "results/m2/tb_logs"
M2_EVAL_RESULTS_PATH = "results/m2/eval_results.json"
M2_PLOT_DIR = "results/m2/plots"
M2_TRAJECTORY_LOG_PATH = "results/m2/trajectory_log.json"