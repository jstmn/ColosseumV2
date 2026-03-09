# !/bin/bash

DISTRACTION_SET=none
# ^ Must be one of: none, all, distractor_object_cfg, MO_color_cfg, MO_texture_cfg, RO_color_cfg, RO_texture_cfg, table_color_cfg, table_texture_cfg, camera_pose_cfg

ENVS=(
    "PickSodaFromCabinet-v1"
    "PickDishFromRack-v1"
    "StackCubeColosseumV2-v1"
    "PlaceDishInRack-v1"
    "LiftPegUprightColosseumV2-v1"
    "RotateArrow-v1"
    "PegInsertionSideColosseumV2-v1"
    "PlugChargerColosseumV2-v1"
    "HammerNail-v1"
    "ScoopBanana-v1"
    "OpenDrawer-v1"
    "OpenCabinet-v1"
    "PlaceCubeInDrawer-v1"
    "PlaceBookInShelf-v1"
    "CookItemInPan-v1"
    "RaiseCube-v1"
)
NUM_PROCS=7
N_TRAJ=100
TARGET_CONTROL_MODE=pd_ee_delta_pose
OBS_MODE=rgb
REWARD_MODE=none

for ENV_ID in "${ENVS[@]}"; do

    TRAJ_PATH=demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.h5
    TRANSLATED_TRAJ_PATH=demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}.physx_cpu.h5

    if [ -f "$TRANSLATED_TRAJ_PATH" ]; then
        echo -e "\033[1;32m Converted trajectory file $TRANSLATED_TRAJ_PATH already exists\033[0m"
        continue
    else
        echo -e "\033[1;33m Converted trajectory file $TRANSLATED_TRAJ_PATH does not exist\033[0m"
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "            ---  ENV_ID: $ENV_ID ---"
    echo ""

    if [ ! -f "$TRAJ_PATH" ]; then

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
    else
        echo -e "\033[1;32mTrajectory file $TRAJ_PATH already exists\033[0m"
    fi

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
    N_DEMOS_TRANSLATED=$(h5ls -r ${TRANSLATED_TRAJ_PATH} | grep -c "/actions")
    echo -e "${ENV_ID}:\tNumber of demonstrations (original, translated): (${N_DEMOS_0}, ${N_DEMOS_TRANSLATED})"
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