# !/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate maniskill_jm

while true; do
    python examples/baselines/act/eval_rgbd.py \
        --checkpoint-path checkpoints/hyeonho_simul_results/Multi-task_bimanual_lang/best_eval_success_once.pt \
        --control-mode "pd_joint_pos" \
        --no-include-depth \
        --sim-backend "physx_cuda" \
        --capture-video \
        --num-eval-episodes 100 \
        --num-eval-envs 50 \
        --max-episode-steps 500 \
        --distraction-set "BLANK" \
        --results-path logs/results_bimanual.csv

    # python examples/baselines/act/eval_rgbd.py \
    #     --checkpoint-path checkpoints/best_eval_success_once__BIMANUAL_JAN30.pt \
    #     --control-mode "pd_joint_pos" \
    #     --no-include-depth \
    #     --sim-backend "physx_cuda" \
    #     --capture-video \
    #     --num-eval-episodes 100 \
    #     --num-eval-envs 50 \
    #     --max-episode-steps 500 \
    #     --distraction-set "BLANK" \
    #     --results-path logs/results_bimanual.csv
done