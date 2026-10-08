MANISKILL_CAMERA_WIDTH = 320
MANISKILL_CAMERA_HEIGHT = 240
POINTCLOUD_CAM_WIDTH = 75
POINTCLOUD_CAM_HEIGHT = 75

PICK_CUBE_MAX_N_STEPS = 100
PULL_CUBE_TOOL_MAX_N_STEPS = 350
LIFT_PEG_UPRIGHT_MAX_N_STEPS = 225
PUSH_CUBE_MAX_N_STEPS = 80

# LiftPegUpright / PushCube were built for a panda base at this x. The base is at the origin,
# so those objects and the table are shifted by the negation.
FORMER_TABLE_BASE_X = -0.615

ALL_CAMERA_NAMES = [
    "camera_center",
    "camera_left",
    "camera_right",
    "camera_base",
    "camera_low_pos_y",
    "camera_low_neg_y",
    "camera_wrist",
    "base_camera",
]
