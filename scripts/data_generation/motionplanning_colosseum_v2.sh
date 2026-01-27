# !/bin/bash

DISTRACTION_SET=none
# ^ Must be one of: none, all, distractor_object_cfg, MO_color_cfg, MO_texture_cfg, RO_color_cfg, RO_texture_cfg, table_color_cfg, table_texture_cfg, camera_pose_cfg

ENVS=(
    "RaiseCube-v1"
    "PickSodaFromCabinet-v1"
    # "PickDishFromRack-v1"
    # "StackCube-v1"
    # "PlaceBookInShelf-v1"
    # "PlaceDishInRack-v1"
    # "LiftPegUpright-v1"
    # "RotateArrow-v1"
    # "PegInsertionSide-v2"
    # "PlugCharger-v1"
    # "HammerNail-v1"
    # "ScoopBanana-v1"
    # "OpenDrawer-v1"
    # "OpenCabinet-v1"
    # "PlaceCubeInDrawer-v1"
    # "CookItemInPan-v1"
)
N_TRAJ=2
CONTROL_MODE=pd_joint_pos

for ENV_ID in "${ENVS[@]}"; do

    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "----------------------------------------------------------------"
    echo "            ---  ENV_ID: $ENV_ID ---"
    echo ""

    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id ${ENV_ID} \
        --num-traj ${N_TRAJ} \
        --distraction-set ${DISTRACTION_SET} \
        --num-procs 2 \
        --obs-mode "rgb" \
        --reward-mode "sparse" \
        --random-seed \
        --only-count-success \
        --traj-name "trajectory__${CONTROL_MODE}" \
        --ignore-keys "foo/bar"

    # todo: add optional control_mode conversion here
done

# Merge trajectories
INPUT_DIRS=""
for ENV in "${ENVS[@]}"; do
    INPUT_DIRS="${INPUT_DIRS}demos/${ENV}/motionplanning "
done
INPUT_DIRS=$(echo "${INPUT_DIRS}" | xargs)
echo "Input directories: ${INPUT_DIRS}"
echo ""
python mani_skill/trajectory/merge_trajectory.py \
    --pattern "trajectory__${CONTROL_MODE}.h5" \
    --input-dirs ${INPUT_DIRS} \
    --output-path demos/trajectory__cv2-full__${CONTROL_MODE}.h5


if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError: Trajectory merging failed\033[0m"
    exit 1
fi


# Done
echo "----------------------------------------------------------------"
echo "----------------------------------------------------------------"
echo "  ---  MERGED TRAJECTORY: trajectory__cv2-full__${CONTROL_MODE}.h5 ---  "
echo "Saved to: demos/trajectory__cv2-full__${CONTROL_MODE}.h5"