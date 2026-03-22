GUI = True
TIME_STEP = 1.0 / 240.0
GRAVITY = -9.81
USE_REALTIME = False

# Initial placeholder positions; final z values will be corrected from geometry.
TABLE_BASE_POS = [0.65, 0.0, 0.0]
TABLE_BASE_ORN_EULER = [0.0, 0.0, 0.0]

# Robot is beside the table on the plane.
PANDA_BASE_POS = [0.65, -0.90, 0.0]
PANDA_BASE_ORN_EULER = [0.0, 0.0, 1.5708]

# Cube should sit on the table.
CUBE_BASE_POS = [0.70, 0.0, 0.0]
CUBE_BASE_ORN_EULER = [0.0, 0.0, 0.0]

PANDA_HOME_JOINTS = [
    0.0,
    -0.785,
    0.0,
    -2.356,
    0.0,
    1.571,
    0.785,
]

SIM_STEPS_PER_SEC = 240

TABLE_PLANE_CLEARANCE = 0.0
ROBOT_PLANE_CLEARANCE = 0.002
CUBE_TABLE_CLEARANCE = 0.002