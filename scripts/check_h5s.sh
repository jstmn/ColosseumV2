#! /bin/bash

TASKS=(
    # "OpenDrawer-v1"
    "RaiseCube-v1"
    # "PickSodaFromCabinet-v1"
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
    # "OpenCabinet-v1"
    # "ObjectInCabinet-v1"
    # "CookItemInPan-v1"
)
PERTURBATION_SET=none

for task in "${TASKS[@]}"; do
    echo "Checking $task"

    # 
    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id ${task} \
        --record-dir /tmp \
        --obs-mode "rgb" \
        --num-traj 1 \
        --perturbation-set ${PERTURBATION_SET} \
        --num-procs 1 \
        --reward-mode "sparse" \
        --random-seed \
        --only-count-success \
        --traj-name "trajectory_${task}" \
        --ignore-keys "obs/extra"

    echo "Done checking $task"
    path=/tmp/${task}/motionplanning/trajectory_${task}.h5
    if [ -f "$path" ]; then
        echo "✓ File exists: $path"
    else
        echo "✗ File missing: $path"
        exit 1
    fi
done


# ================================
echo "H5LS:"
for task in "${TASKS[@]}"; do
    echo "--------------------------------"
    echo "--------------------------------"
    echo "Checking $task"
    path=/tmp/${task}/motionplanning/trajectory_${task}.h5
    h5ls -r $path
    echo ""
done