# !/bin/bash

# bash scripts/colosseum_v2_paper/runtime_act.sh

# Run the rlbench timing script for different batch sizes
TIMESTEPS_PER_TASK=200
# TIMESTEPS_PER_TASK=10
# NUM_EVAL_EPISODES=100

# "PickSodaFromCabinet-v1": int(193 + 4*2.5),
# "PickDishFromRack-v1": int(119 + 4*9.26),
# "StackCubeColosseumV2-v1": int(107 + 4*8.97),
# "PlaceDishInRack-v1": int(251 + 4*19.07),
# "LiftPegUprightColosseumV2-v1": int(198 + 4*6.51),
# "RotateArrow-v1": int(328 + 4*6.94),
# "PegInsertionSideColosseumV2-v1": int(151 + 4*28.91),
# "PlugChargerColosseumV2-v1": int(179 + 4*13.83),
# "HammerNail-v1": int(225 + 4*5.6),
# "ScoopBanana-v1": int(242 + 4*20.05),
# "OpenDrawer-v1": int(118 + 4*3.72),
# "OpenCabinet-v1": int(475 + 4*4.15),
# "PlaceCubeInDrawer-v1": int(333 + 4*13.62),
# "PlaceBookInShelf-v1": int(182 + 4*8.54),
# "CookItemInPan-v1": int(473 + 4*15.12),
# "RaiseCube-v1": int(78 + 4*3.55),
# "DualArmPickCube-v1": int(201.1 + 4*3.2),
# "DualArmPickBottle-v1": int(130.72 + 4*6.23),
# "DualArmLiftPot-v1": int(98.06 + 4*6.94),
# "DualArmLiftTray-v1": int(104.72 + 4*4.43),
# "DualArmPushBox-v1": int(93.04 + 4*9.43),
# "DualArmPourPot-v1": int(200.72 + 4*3.5),
# "DualArmThreading-v1": int(164.97 + 4*6.92),
# "DualArmPenCap-v1": int(186.1 + 4*11.54),
# "DualArmDrawerPlace-v1": int(186.35 + 4*4.0),
# "DualArmDrawerOpen-v1": int(81.0 + 4*9.4),
# "DualArmStackCube-v1": int(137.03 + 4*7.27),
# "DualArmStack3Cube-v1": int(242.08 + 4*10.5),


# for N_PROCESSES in 5 10 25 34 50 67 100; do
# for N_PROCESSES in 1 2 3 4 5 6 7 8 9 10 20 30 40 50; do
# for N_PROCESSES in 67 100 150 200; do
for N_PROCESSES in 250 300 350 400 450 500; do

    # NUM_EVAL_EPISODES=$((N_PROCESSES * 5))
    # NUM_EVAL_EPISODES=200
    NUM_EVAL_EPISODES=$N_PROCESSES

    T_START=$(date +%s%3N)
    # 
    python examples/baselines/act_clip/eval_rgbd.py \
        --checkpoint-path checkpoints/hyeonho_mar17/hyeonho_mar17_act_clip_single_arm_3cameras_15687623_checkpoints_best_eval_success_once.pt \
        --env-id "RaiseCube-v1" \
        --control-mode "pd_ee_delta_pose" \
        --no-include-depth \
        --sim-backend "physx_cuda" \
        --is-multi-task True \
        --target-num-cams 3 \
        --num-eval-episodes $NUM_EVAL_EPISODES \
        --num-eval-envs $N_PROCESSES \
        --max-episode-steps $TIMESTEPS_PER_TASK \
        --internal-instruction \
        --perturbation-set "none"

    #
    T_END=$(date +%s%3N)
    ELAPSED_SEC=$(echo "scale=4; ($T_END - $T_START) / 1000" | bc)

    python scripts/colosseum_v2_paper/runtime_rlbench_logging.py \
        --t_elapsed_sec $ELAPSED_SEC \
        --timesteps_per_task $TIMESTEPS_PER_TASK \
        --n_processes $N_PROCESSES \
        --num_eval_episodes $NUM_EVAL_EPISODES \
        --results_filepath logs/fps/act_runtime.csv
done