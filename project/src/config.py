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