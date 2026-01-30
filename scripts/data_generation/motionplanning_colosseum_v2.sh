# !/bin/bash

DISTRACTION_SET=none
# ^ Must be one of: none, all, distractor_object_cfg, MO_color_cfg, MO_texture_cfg, RO_color_cfg, RO_texture_cfg, table_color_cfg, table_texture_cfg, camera_pose_cfg

ENVS=(
    "RaiseCube-v1"
    "PickSodaFromCabinet-v1"
    "PickDishFromRack-v1"
    "StackCube-v1"
    "PlaceBookInShelf-v1"
    "PlaceDishInRack-v1"
    "LiftPegUpright-v1"
    "RotateArrow-v1"
    "PegInsertionSide-v2"
    "PlugCharger-v1"
    "HammerNail-v1"
    "ScoopBanana-v1"
    "OpenDrawer-v1"
    "OpenCabinet-v1"
    "PlaceCubeInDrawer-v1"
    "CookItemInPan-v1"
)
NUM_PROCS=15
N_TRAJ=100
TARGET_CONTROL_MODE=pd_ee_delta_pose
OBS_MODE=rgb
REWARD_MODE=none

for ENV_ID in "${ENVS[@]}"; do

    echo ""
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "            ---  ENV_ID: $ENV_ID ---"
    echo ""

    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id ${ENV_ID} \
        --num-traj ${N_TRAJ} \
        --distraction-set ${DISTRACTION_SET} \
        --num-procs ${NUM_PROCS} \
        --obs-mode "rgb" \
        --reward-mode ${REWARD_MODE} \
        --random-seed \
        --only-count-success \
        --traj-name "trajectory__pd_joint_pos__${N_TRAJ}"

    TRAJ_PATH=demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.h5
    if [ ! -f "$TRAJ_PATH" ]; then
        TRAJ_PATH_ALT=demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.0.h5
        if [ -f "$TRAJ_PATH_ALT" ]; then
            echo -e "\033[1;33mUsing alternate trajectory file $TRAJ_PATH_ALT\033[0m"
            TRAJ_PATH=$TRAJ_PATH_ALT
        else
            echo -e "\033[0;31mTrajectory file $TRAJ_PATH does not exist.\033[0m"
            echo -e "\033[0;31m(alternate) Trajectory file $TRAJ_PATH_ALT does not exist.\033[0m"
            exit 1
        fi
    fi
    python scripts/extract_h5_images.py --h5-file ${TRAJ_PATH}

    echo ""
    echo "----------------------------------------------------------------"
    echo "            ---  REPLAYING TRAJECTORY ---"
    echo "----------------------------------------------------------------"
    echo ""

    python mani_skill/trajectory/replay_trajectory.py \
        --traj-path ${TRAJ_PATH} \
        --sim-backend physx_cpu \
        --obs-mode ${OBS_MODE} \
        --target-control-mode ${TARGET_CONTROL_MODE} \
        --no-verbose \
        --reward-mode ${REWARD_MODE} \
        --save-traj \
        --no-save-video \
        --no-discard-timeout \
        --no-allow-failure \
        --no-vis \
        --no-use-env-states \
        --use-first-env-state \
        --no-record-rewards \
        --num-envs ${NUM_PROCS}


    N_DEMOS_0=$(h5ls -r ${TRAJ_PATH} | grep -c "/actions")
    N_DEMOS_TRANSLATED=$(h5ls -r demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}*.h5 | grep -c "/actions")
    echo -e "${ENV_ID}:\tNumber of demonstrations original, translated: ${N_DEMOS_0}, ${N_DEMOS_TRANSLATED}"
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
python mani_skill/trajectory/merge_trajectory.py \
    --pattern "trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}*.h5" \
    --input-dirs ${INPUT_DIRS} \
    --output-path ${OUTPUT_PATH}


# if [ $? -ne 0 ]; then
#     echo -e "\033[0;31mError: Trajectory merging failed\033[0m"
#     exit 1
# fi


# Done
echo "----------------------------------------------------------------"
echo "----------------------------------------------------------------"
echo "  ---  MERGED TRAJECTORY: ${OUTPUT_PATH} ---  "
echo "Saved to: ${OUTPUT_PATH}"

N_FINAL_DEMOS=$(h5ls -r ${OUTPUT_PATH} | grep -c "/actions")
echo -e "Number of demonstrations in final h5 file: ${N_FINAL_DEMOS}"