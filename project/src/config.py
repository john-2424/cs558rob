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

# Residual bounds
# 0.3 rad/s (~12% of PD max). Planner targets are computed from the nominal
# (pre-perturbation) cube pose, so the PD drives toward the wrong spot.
# At 0.04m perturbation the PD pulls ~0.2 rad/s toward nominal; the residual
# needs >= 0.2 to counteract. 0.3 gives enough headroom for RL to override.
RESIDUAL_MAX = 0.3

# RL simulation
RL_SIM_SUBSTEPS = 4
RL_MAX_EPISODE_STEPS = 2000
# Per-waypoint timeout (sim steps). Forces _waypoint_idx to advance even if
# WAYPOINT_TOL never converges, mirroring classical demo's WAYPOINT_MAX_STEPS.
# Without this a small constant residual can stall the policy on a single
# waypoint until RL_MAX_EPISODE_STEPS truncation.
RL_MAX_STEPS_PER_WAYPOINT = 150

# Perturbation
# Training range: XY ±4cm, Z ±1cm, yaw ±0.2rad. Planner sees nominal pose;
# RL must correct for the offset. Curriculum samples per-episode ranges.
PERTURB_XY_RANGE = 0.04
PERTURB_Z_RANGE = 0.01
PERTURB_YAW_RANGE = 0.2
PERTURB_LEVELS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
# During training, sample per-episode xy_range uniformly in
# [0, PERTURB_XY_RANGE]. Gives agent easy episodes early so it can
# discover the grasp cliff; still exposed to full range at train cap.
TRAIN_CURRICULUM = True

# Reward shaping
REWARD_ALPHA = 10.0
# With RESIDUAL_MAX=0.3, BETA=0.15 gives max per-step residual penalty
# ~= -0.045, comparable in magnitude to per-step approach reward.
# (Scaled down from 0.5 to compensate for 3x larger RESIDUAL_MAX.)
# Note: residual penalty is disabled for rl_only mode since there is no
# PD backbone — penalizing the only control signal trains it to do nothing.
REWARD_BETA = 0.15
REWARD_GAMMA = 0.02
REWARD_DELTA = 50.0
REWARD_EPSILON = 100.0
REWARD_ZETA = 50.0
# Dense proximity bonus during grasp_descend phase: linear ramp in
# [0, REWARD_ETA] for ee_cube_dist in [0.10, 0.0]. Smooths the cliff
# between "approached but didn't grasp" and the one-shot REWARD_DELTA.
REWARD_ETA = 2.0
PROXIMITY_RADIUS = 0.10

# PPO hyperparameters
# Tuned after run #3 oscillated 60-95% on bimodal rewards. Lower LR + lower
# entropy reduce per-batch swing; larger batch + fewer epochs cut the
# update/sample ratio from 320 to 64 (4x less over-fitting per batch).
PPO_TOTAL_TIMESTEPS = 1_000_000
PPO_LR = 5e-5
PPO_FRAMES_PER_BATCH = 4096
PPO_MINI_BATCH_SIZE = 64
PPO_EPOCHS = 4
PPO_CLIP_EPSILON = 0.2
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEFF = 0.01

# Training stability
PPO_LR_SCHEDULE = "linear"  # "linear" (decay to 0) or "constant"
PPO_TARGET_KL = 0.02  # stop PPO epochs early when mean approx KL exceeds this
PPO_CLIP_VALUE = True  # clip value function updates (same epsilon as policy)
PPO_SAVE_BEST_MODEL = True  # track and save the peak-performance checkpoint
PPO_EARLY_STOP_PATIENCE = 15  # consecutive batches below peak before stopping
PPO_EARLY_STOP_MIN_PEAK = 0.50  # early stop only activates after peak >= this
PPO_EARLY_STOP_DROP = 0.25  # success must drop this much below peak to count

# Parallel env workers for data collection (set to 1 to disable multiprocessing)
PPO_NUM_COLLECTOR_WORKERS = 8

# Episode-reward threshold for counting an episode as a "success" in training logs.
# Lift bonus is REWARD_EPSILON (100.0); grasp bonus is REWARD_DELTA (50.0). Set
# at 90 so only episodes that actually received the lift terminal bonus count --
# previous value 50 also counted grasp-only-no-lift episodes.
EP_SUCCESS_REWARD_THRESHOLD = 90.0

# Evaluation
EVAL_EPISODES_PER_LEVEL = 50
EVAL_NUM_WORKERS = 8  # parallel workers for episode rollout; set to 1 for serial
EVAL_VERBOSE_EPISODES = True  # print one diagnostic line per episode (phase/waypoint/grasp)

# Demo recording: seconds to hold at the nominal cube pose before applying
# the perturbation in residual-demo runs. Lets you start your screen recorder
# and see the nominal frame on camera before the cube teleports. GUI only;
# has no effect on training or evaluation (headless).
DEMO_PERTURB_HOLD_SECONDS = 10.0

# M2 paths
M2_RESULTS_DIR = "results/m2"
M2_MODEL_DIR = "results/m2/models"
M2_TB_LOG_DIR = "results/m2/tb_logs"
M2_EVAL_RESULTS_PATH = "results/m2/eval_results.json"
M2_PLOT_DIR = "results/m2/plots"
M2_TRAJECTORY_LOG_PATH = "results/m2/trajectory_log.json"