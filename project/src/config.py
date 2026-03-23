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
PANDA_BASE_POS = [0.65, -0.90, 0.0]
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
SETTLE_STEPS = 120
POST_MOTION_SETTLE_STEPS = 120

TRAJ_NUM_WAYPOINTS = 25
WAYPOINT_TOL = 0.03
WAYPOINT_MAX_STEPS = 240

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
PD_KP = [3.2, 3.2, 3.2, 3.5, 2.6, 2.6, 2.3]
PD_KD = [0.35, 0.35, 0.35, 0.40, 0.28, 0.28, 0.24]
PD_MAX_JOINT_VEL = [1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2]

# =========================
# Planner scaffold / RRT*
# =========================
PLANNER_RANDOM_SEED = 7
PLANNER_STEP_SIZE = 0.06
PLANNER_GOAL_THRESHOLD = 0.10
PLANNER_MAX_ITERATIONS = 3000
PLANNER_GOAL_SAMPLE_RATE = 0.20
PLANNER_EDGE_CHECK_RESOLUTION = 0.03
PLANNER_INTERP_RESOLUTION = 0.05
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
USE_BLOCKED_PATH_TEST = True
PLANNER_BLOCKED_SEARCH_MAX_SAMPLES = 300
PLANNER_BLOCKED_SEARCH_SEED = 7
PLANNER_BLOCKED_TEST_TARGETS = [
    [0.35, -0.25, 0.20, -2.10, 0.15, 1.95, 0.60],
    [0.55, -0.10, -0.35, -2.00, 0.35, 2.10, 1.10],
    [-0.25, -0.55, 0.40, -1.95, -0.30, 1.85, 0.25],
    [0.70, -0.60, -0.20, -1.80, 0.45, 2.00, 1.40],
    [-0.45, -0.20, 0.55, -2.25, 0.20, 1.75, -0.10],
]