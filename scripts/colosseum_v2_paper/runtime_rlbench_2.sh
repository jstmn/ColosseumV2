# !/bin/bash

# bash scripts/colosseum_v2_paper/runtime_rlbench_2.sh

conda activate maniskill_jm

# Run the rlbench timing script for different batch sizes
TIMESTEPS_PER_TASK=200
N_PROCESSES=5

T_START=$(date +%s%3N)
for i in $(seq 1 $N_PROCESSES); do
    python scripts/colosseum_v2_paper/runtime_rlbench_2.py --n_steps $TIMESTEPS_PER_TASK &
done
wait
T_END=$(date +%s%3N)
ELAPSED_SEC=$(echo "scale=4; ($T_END - $T_START) / 1000" | bc)


python3 -c "
t_elapsed_sec = $ELAPSED_SEC
n_processes = $N_PROCESSES
time_per_task_sec = t_elapsed_sec / n_processes  # amortized: N_PROCESSES tasks completed in t_elapsed
NUM_VARIATIONS=17  # includes none, all
N_TASKS=20
total_sec = NUM_VARIATIONS * N_TASKS * time_per_task_sec
print(f'Estimated total sim time: {total_sec:.0f}s  |  {total_sec/60:.1f} min  |  {total_sec/3600:.2f} hours')
"