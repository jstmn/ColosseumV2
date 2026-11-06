#!/bin/bash
# Generate human-render-camera PNG frames under RT for each perturbation.
# Unlike generate_variation_images.sh (sensor RGB from H5), this saves
# render_mode=rgb_array videos with --shader rt, then dumps the first N frames.

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
# ENV_ID="DualArmStack3Cube-v1"
ENV_ID="RaiseCube-v1"

NUM_TRAJ=3
FIRST_N_FRAMES=20

PERTURBATION_SETS=(
    "none"
    # "all"
    # "MO_color"
    # "RO_color"
    # "MO_texture"
    # "RO_texture"
    # "MO_size"
    # "RO_size"
    # "table_color"
    # "light_color"
    # "table_texture"
    # "distractor_object"
    # "background_texture"
    # "background_color"
    # "pose_randomization"
    # "camera_pose"
    # "MO_mass"
    # "language"
)

for DS in "${PERTURBATION_SETS[@]}"; do
    OUT_DIR="demos/${ENV_ID}/motionplanning/render_camera_rt/${DS}"
    EXPECTED_PNGS=$((NUM_TRAJ * FIRST_N_FRAMES))
    EXISTING_PNGS=$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)

    if [ "${EXISTING_PNGS}" -ge "${EXPECTED_PNGS}" ]; then
        echo -e "\033[0;32mSkipping ${ENV_ID} / ${DS}: ${EXISTING_PNGS} PNGs already in ${OUT_DIR}\033[0m"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------------------------------------"
    echo "Generating human-camera RT images for ${ENV_ID} / ${DS}"
    echo "------------------------------------------------------------------------------------------------"

    python mani_skill/examples/motionplanning/panda/run.py \
        --env-id "${ENV_ID}" \
        --num-traj "${NUM_TRAJ}" \
        --perturbation-set "${DS}" \
        --num-procs 1 \
        --obs-mode "none" \
        --reward-mode "none" \
        --render-mode "rgb_array" \
        --shader "rt" \
        --save-video \
        --only-count-success \
        --random-seed \
        --traj-name "trajectory_human_rt__${DS}"

    mkdir -p "${OUT_DIR}"
    shopt -s nullglob
    videos=(demos/"${ENV_ID}"/motionplanning/trajectory_human_rt__"${DS}"___n:*.mp4)
    shopt -u nullglob

    if [ "${#videos[@]}" -eq 0 ]; then
        echo -e "\033[0;31mNo videos found for ${ENV_ID} / ${DS}; skipping frame extract.\033[0m"
        continue
    fi

    # Sort by the ___n:k index so traj prefixes are stable.
    IFS=$'\n' videos_sorted=($(printf '%s\n' "${videos[@]}" | sort -t: -k2 -n))
    unset IFS

    traj_i=0
    for video in "${videos_sorted[@]}"; do
        traj_i=$((traj_i + 1))
        python scripts/extract_video_frames.py \
            --video "${video}" \
            --output-dir "${OUT_DIR}" \
            --prefix "traj_${traj_i}___" \
            --first-n-frames "${FIRST_N_FRAMES}"
    done

    echo "Wrote frames under: ${OUT_DIR}"
done

echo "Done."
