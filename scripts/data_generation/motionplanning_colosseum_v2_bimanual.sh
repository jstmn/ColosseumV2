# !/bin/bash

# bash scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh
# bash scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh --included-cameras "external1_camera"
# bash scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh --included-cameras "hand_camera external1_camera"

PERTURBATION_SET=none
# ^ Must be one of: none, all, distractor_object_cfg, MO_color_cfg, MO_texture_cfg, RO_color_cfg, RO_texture_cfg, table_color_cfg, table_texture_cfg, camera_pose_cfg

INCLUDED_CAMERAS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --included-cameras)
            INCLUDED_CAMERAS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


ENVS=(
    "DualArmPickCube-v1"
    "DualArmPickBottle-v1"
    "DualArmLiftPot-v1"
    "DualArmLiftTray-v1"
    "DualArmPushBox-v1"
    "DualArmPourPot-v1"
    "DualArmThreading-v1"
    "DualArmPenCap-v1"
    "DualArmDrawerPlace-v1"
    "DualArmDrawerOpen-v1"
    "DualArmStackCube-v1"
    "DualArmStack3Cube-v1"
)
NUM_PROCS=5
N_TRAJ=100
TARGET_CONTROL_MODE=pd_joint_pos
OBS_MODE=rgb
REWARD_MODE=none

INCLUDED_CAMERAS_ARG=""
if [ -n "$INCLUDED_CAMERAS" ]; then
    INCLUDED_CAMERAS_ARG="--included-cameras ${INCLUDED_CAMERAS}"
fi

for ENV_ID in "${ENVS[@]}"; do

    TRAJ_PATH=demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.h5
    if [ -f "$TRAJ_PATH" ]; then
        echo -e "\033[1;33mTrajectory file $TRAJ_PATH already exists\033[0m"
        continue
    fi



    echo ""
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "            ---  ENV_ID: $ENV_ID ---"
    echo ""

    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id ${ENV_ID} \
        --num-traj ${N_TRAJ} \
        --perturbation-set ${PERTURBATION_SET} \
        ${INCLUDED_CAMERAS_ARG} \
        --num-procs ${NUM_PROCS} \
        --obs-mode "rgb" \
        --reward-mode ${REWARD_MODE} \
        --random-seed \
        --only-count-success \
        --traj-name "trajectory__pd_joint_pos__${N_TRAJ}"
done

# Merge trajectories
INPUT_DIRS=""
for ENV in "${ENVS[@]}"; do
    INPUT_DIRS="${INPUT_DIRS}demos/${ENV}/motionplanning "
done
INPUT_DIRS=$(echo "${INPUT_DIRS}" | xargs)
OUTPUT_PATH=demos/trajectory__cv2-full__${TARGET_CONTROL_MODE}__${N_TRAJ}.h5

echo "Input directories: ${INPUT_DIRS}"
echo ""
python mani_skill/trajectory/merge_multitask_trajectories.py \
    --pattern "trajectory__pd_joint_pos__${N_TRAJ}.h5" \
    --input-dirs ${INPUT_DIRS} \
    --output-path ${OUTPUT_PATH}

# Done
echo "----------------------------------------------------------------"
echo "----------------------------------------------------------------"
echo "  ---  MERGED TRAJECTORY: ${OUTPUT_PATH} ---  "
echo "Saved to: ${OUTPUT_PATH}"

N_FINAL_DEMOS=$(h5ls -r ${OUTPUT_PATH} | grep -c "/actions")
echo -e "Num. demonstrations final: ${N_FINAL_DEMOS}"