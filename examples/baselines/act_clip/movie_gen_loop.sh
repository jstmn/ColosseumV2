#!/bin/bash
# Generate K movies per (task, perturbation) for all Colosseum-V2 tasks.
# One env at a time, rolled out K times. Uses the matching single-arm /
# bimanual checkpoints from eval_rgbd_loop.sh.

K=10

PERTURBATIONS=(
    none
    all
    MO_color
    RO_color
    MO_texture
    RO_texture
    MO_size
    RO_size
    table_color
    light_color
    table_texture
    distractor_object
    background_texture
    background_color
    camera_pose
    pose_randomization
    language_paraphrase
    language_other_task
    language_random
    language_none
)

SINGLE_ARM_TASKS=(
    RaiseCube-v1
    PickSodaFromCabinet-v1
    PickDishFromRack-v1
    StackCubeColosseumV2-v1
    PlaceBookInShelf-v1
    PlaceDishInRack-v1
    LiftPegUprightColosseumV2-v1
    RotateArrow-v1
    PegInsertionSideColosseumV2-v1
    PlugChargerColosseumV2-v1
    HammerNail-v1
    ScoopBanana-v1
    OpenDrawer-v1
    OpenCabinet-v1
    PlaceCubeInDrawer-v1
    CookItemInPan-v1
)

BIMANUAL_TASKS=(
    DualArmPickCube-v1
    DualArmPickBottle-v1
    DualArmLiftPot-v1
    DualArmLiftTray-v1
    DualArmPushBox-v1
    DualArmPourPot-v1
    DualArmThreading-v1
    DualArmPenCap-v1
    DualArmDrawerPlace-v1
    DualArmDrawerOpen-v1
    DualArmStackCube-v1
    DualArmStack3Cube-v1
)

SINGLE_ARM_CKPT="checkpoints/hyeonho_mar17/hyeonho_mar17_act_clip_single_arm_3cameras_15687623_checkpoints_best_eval_success_once.pt"
BIMANUAL_CKPT="checkpoints/hyeonho_mar17/hyeonho_mar17_act_clip_bimanual_4cameras_15689642_checkpoints_best_eval_success_once.pt"

run_eval() {
    local ckpt="$1"
    local env_id="$2"
    local pert="$3"
    local control_mode="$4"
    local n_cams="$5"

    echo "=== ${env_id} / ${pert} (1 env x ${K} episodes) ==="
    python examples/baselines/act_clip/eval_rgbd.py \
        --checkpoint-path "$ckpt" \
        --control-mode "$control_mode" \
        --no-include-depth \
        --sim-backend "physx_cpu" \
        --is-multi-task True \
        --target-num-cams "$n_cams" \
        --num-eval-episodes "$K" \
        --num-eval-envs 1 \
        --max-episode-steps-from-lookup \
        --internal-instruction \
        --capture-video \
        --no-metrics-on-video \
        --env-id "$env_id" \
        --human-render-shader "rt" \
        --perturbation-set "$pert"
}

for env_id in "${SINGLE_ARM_TASKS[@]}"; do
    for pert in "${PERTURBATIONS[@]}"; do
        run_eval "$SINGLE_ARM_CKPT" "$env_id" "$pert" "pd_ee_delta_pose" 3
    done
done

for env_id in "${BIMANUAL_TASKS[@]}"; do
    for pert in "${PERTURBATIONS[@]}"; do
        run_eval "$BIMANUAL_CKPT" "$env_id" "$pert" "pd_joint_pos" 4
    done
done

echo "Done. Videos under:"
echo "  ${SINGLE_ARM_CKPT%.pt}__videos/"
echo "  ${BIMANUAL_CKPT%.pt}__videos/"
echo "Outcomes CSV (filepath ↔ success/fail):"
echo "  ${SINGLE_ARM_CKPT%.pt}__videos/video_outcomes.csv"
echo "  ${BIMANUAL_CKPT%.pt}__videos/video_outcomes.csv"
