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
NUM_PROCS=1
N_TRAJ=1
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
        --shader "rt" \
        --only-count-success \
        --sim-backend "physx_cpu" \
        --traj-name "trajectory__pd_joint_pos__${N_TRAJ}"
        # --shader "rt-fast" \
        # --sim-backend "gpu" \

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

    python scripts/extract_h5_images.py --h5-file ${TRAJ_PATH} --first-k 10

    # echo ""
    # echo "----------------------------------------------------------------"
    # echo "            ---  REPLAYING TRAJECTORY ---"
    # echo "----------------------------------------------------------------"
    # echo ""

    # python mani_skill/trajectory/replay_trajectory.py \
    #     --traj-path ${TRAJ_PATH} \
    #     --sim-backend physx_cpu \
    #     --obs-mode ${OBS_MODE} \
    #     --target-control-mode ${TARGET_CONTROL_MODE} \
    #     --no-verbose \
    #     --reward-mode ${REWARD_MODE} \
    #     --save-traj \
    #     --no-save-video \
    #     --no-discard-timeout \
    #     --no-allow-failure \
    #     --no-vis \
    #     --no-use-env-states \
    #     --use-first-env-state \
    #     --no-record-rewards \
    #     --num-envs ${NUM_PROCS}


    # N_DEMOS_0=$(h5ls -r ${TRAJ_PATH} | grep -c "/actions")
    # N_DEMOS_TRANSLATED=$(h5ls -r demos/${ENV_ID}/motionplanning/trajectory__pd_joint_pos__${N_TRAJ}.${OBS_MODE}.${TARGET_CONTROL_MODE}*.h5 | grep -c "/actions")
    # echo -e "${ENV_ID}:\tNumber of demonstrations original, translated: ${N_DEMOS_0}, ${N_DEMOS_TRANSLATED}"
done
