# !/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniskill_jm


# ============================================================
# Env ID                              | Count     
# ------------------------------------------------------------
# DualArmPickCube-v1                  | 98
# DualArmPickBottle-v1                | 98
# DualArmLiftPot-v1                   | 98
# DualArmLiftTray-v1                  | 98
# DualArmPushBox-v1                   | 96
# DualArmPourPot-v1                   | 98
# DualArmThreading-v1                 | 98
# DualArmPenCap-v1                    | 98
# DualArmDrawerPlace-v1               | 100       
# DualArmDrawerOpen-v1                | 100       
# DualArmStackCube-v1                 | 96        
# DualArmStack3Cube-v1                | 96        
# ------------------------------------------------------------
# Total Merged Episodes               | 1174      
# Max Episode Length (Steps)          | 265       
# ============================================================

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Env ID                              | Count     
# ------------------------------------------------------------
# PickSodaFromCabinet-v1              | 90
# PickDishFromRack-v1                 | 90
# StackCubeColosseumV2-v1             | 90
# PlaceDishInRack-v1                  | 72
# LiftPegUprightColosseumV2-v1        | 90
# RotateArrow-v1                      | 90
# PegInsertionSideColosseumV2-v1      | 86
# PlugChargerColosseumV2-v1           | 74
# HammerNail-v1                       | 82
# ScoopBanana-v1                      | 88
# OpenDrawer-v1                       | 90
# OpenCabinet-v1                      | 73
# PlaceCubeInDrawer-v1                | 99
# PlaceBookInShelf-v1                 | 88
# CookItemInPan-v1                    | 100       
# RaiseCube-v1                        | 100       
# ------------------------------------------------------------
# Total Merged Episodes               | 1402      
# Max Episode Length (Steps)          | 702       
# ============================================================


NOW=$(date +%H:%M:%S)

while true; do
    python examples/baselines/act_clip/eval_rgbd.py \
        --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
        --control-mode "pd_ee_delta_pose" \
        --no-include-depth \
        --sim-backend "physx_cuda" \
        --is-multi-task True \
        --target-num-cams 1 \
        --num-eval-episodes 200 \
        --num-eval-envs 20 \
        --max-episode-steps 350 \
        --internal-instruction \
        --distraction-set "BLANK" \
        --results-path logs/results_single_arm.csv

    python examples/baselines/act_clip/eval_rgbd.py \
        --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_bimanual_lang/best_eval_success_once.pt \
        --control-mode "pd_joint_pos" \
        --no-include-depth \
        --sim-backend "physx_cuda" \
        --is-multi-task True \
        --target-num-cams 1 \
        --num-eval-episodes 200 \
        --num-eval-envs 14 \
        --max-episode-steps 850 \
        --internal-instruction \
        --distraction-set "BLANK" \
        --results-path logs/results_bimanual.csv

    # =============================== VIDEO MODE ===============================
    # python examples/baselines/act_clip/eval_rgbd.py \
    #     --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_single_lang/best_eval_success_once.pt \
    #     --control-mode "pd_ee_delta_pose" \
    #     --no-include-depth \
    #     --sim-backend "physx_cuda" \
    #     --is-multi-task True \
    #     --target-num-cams 1 \
    #     --num-eval-episodes 6 \
    #     --num-eval-envs 6 \
    #     --max-episode-steps 5 \
    #     --capture-video \
    #     --hidden-dim 512 --dim-feedforward 1600 --enc-layers 4 --dec-layers 7 \
    #     --internal-instruction \
    #     --distraction-set "BLANK" \
    #     --results-path logs/results_single_arm_VIDEOS_${NOW}.csv

    # python examples/baselines/act_clip/eval_rgbd.py \
    #     --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_bimanual_lang/best_eval_success_once.pt \
    #     --control-mode "pd_joint_pos" \
    #     --no-include-depth \
    #     --sim-backend "physx_cuda" \
    #     --is-multi-task True \
    #     --target-num-cams 1 \
    #     --num-eval-episodes 5 \
    #     --num-eval-envs 5 \
    #     --max-episode-steps 5 \
    #     --internal-instruction \
    #     --distraction-set "BLANK" \
    #     --capture-video \
    #     --results-path logs/results_bimanual.csv
        # --results-path logs/results_bimanual_VIDEOS_${NOW}.csv
done