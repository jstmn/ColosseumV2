#!/bin/bash

# 
# ENV_ID="ScoopBanana-v1"

ENVS=(
    "RaiseCube-v1"
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
)

PERTURBATION_SETS=(
    "none"
    "all"
    "distractor_object"
    "MO_color"
    "MO_texture"
    "MO_size"
    "RO_color"
    "RO_texture"
    "RO_size"
    "table_color"
    "table_texture"
    "camera_pose"
    "light_color"
    "background_texture"
    "background_color"
    "language"
    "pose_randomization"
)


now=$(date +%H:%M:%S)



for ENV_ID in "${ENVS[@]}"; do


    echo " | --------------- ENV_ID: ${ENV_ID} --------------- |"
    echo ""


    for DS in "${PERTURBATION_SETS[@]}"; do

        echo ""
        echo "------------------------------------------------------------------------------------------------"
        echo " --- Perturbation: ${DS} ---"
        echo ""

        # Check if the movie file already exists. Skip generation if it does.
        MOVIE_FILE="demos/${ENV_ID}/motionplanning/trajectory_movie__${DS}___n:1.mp4"
        if [ -f "${MOVIE_FILE}" ]; then
            # Print in green if the file exists
            echo -e "\033[0;32mFile ${MOVIE_FILE} already exists. Skipping.\033[0m"
            continue
        fi

        python mani_skill/examples/motionplanning/panda/run.py \
            --env-id ${ENV_ID} \
            --num-traj 1 \
            --perturbation-set ${DS} \
            --num-procs 1 \
            --obs-mode "state" \
            --reward-mode "none" \
            --random-seed \
            --save-video \
            --shader "rt" \
            --only-count-success \
            --traj-name "trajectory_movie__${DS}"
    done
done