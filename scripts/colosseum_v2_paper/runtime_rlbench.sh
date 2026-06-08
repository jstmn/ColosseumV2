# !/bin/bash

# bash scripts/colosseum_v2_paper/runtime_rlbench.sh

# Run the rlbench timing script for different batch sizes
TIMESTEPS_PER_TASK=200

# for N_PROCESSES in 1 2 3; do
for N_PROCESSES in 1 2 3 4 5 6 7 8 9 10 20 25 30 40 50; do

    T_START=$(date +%s%3N)
    for i in $(seq 1 $N_PROCESSES); do
        python scripts/colosseum_v2_paper/runtime_rlbench.py --n_steps $TIMESTEPS_PER_TASK &
    done
    wait
    T_END=$(date +%s%3N)
    ELAPSED_SEC=$(echo "scale=4; ($T_END - $T_START) / 1000" | bc)

    python scripts/colosseum_v2_paper/runtime_rlbench_logging.py \
        --t_elapsed_sec $ELAPSED_SEC \
        --timesteps_per_task $TIMESTEPS_PER_TASK \
        --n_processes $N_PROCESSES \
        --results_filepath logs/fps/rlbench_runtime.csv
done