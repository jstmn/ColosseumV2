#!/bin/bash

# 
# ENV_ID="ScoopBanana-v1"

# Colosseum v2 bimanual tasks
# ENV_ID="DualArmPickCube-v1"
# ENV_ID="DualArmPickBottle-v1"
# ENV_ID="DualArmLiftPot-v1"
# ENV_ID="DualArmLiftTray-v1"
# ENV_ID="DualArmPushBox-v1"
# ENV_ID="DualArmPourPot-v1"
# ENV_ID="DualArmThreading-v1"
# ENV_ID="DualArmPenCap-v1"
# ENV_ID="DualArmDrawerPlace-v1"
# ENV_ID="DualArmDrawerOpen-v1"
# ENV_ID="DualArmStackCube-v1"
ENV_ID="DualArmStack3Cube-v1"
# ENV_ID="RaiseCube-v1"


PERTURBATION_SETS=(
    # "none"
    # "all"
    # "MO_color"
    # "RO_color"
    # "MO_texture"
    # "RO_texture"
    "MO_size"
    "RO_size"
    # "table_color"
    # "light_color"
    # "table_texture"
    # "distractor_object"
    # "background_texture"
    # "background_color"
    "pose_randomization"
    # "camera_pose"
    # "MO_mass"
    # "language"
)


now=$(date +%H:%M:%S)


for DS in "${PERTURBATION_SETS[@]}"; do

    echo ""
    echo ""
    echo "------------------------------------------------------------------------------------------------"
    echo "Generating perturbation images for ${DS}"
    echo "--------------------------------"
    echo "--------------------------------"
    echo "--------------------------------"
    echo ""
    echo ""

    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id ${ENV_ID} \
        --num-traj 3 \
        --perturbation-set ${DS} \
        --num-procs 1 \
        --obs-mode "rgb" \
        --reward-mode "none" \
        --random-seed \
        --traj-name "trajectory__${DS}"

    python scripts/extract_h5_images.py \
        --h5-file demos/${ENV_ID}/motionplanning/trajectory__${DS}.h5 \
        --first-n-trajectories 3 \
        --first-n-timesteps 65 \
        --prefix "${DS}__${now}__"
done