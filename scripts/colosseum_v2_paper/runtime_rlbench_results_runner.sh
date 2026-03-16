# !/bin/bash

# ./scripts/colosseum_v2_paper/runtime_rlbench_results_runner.sh

conda activate maniskill_jm

# Run the rlbench timing script for different batch sizes
for batch_size in 2 3 4 5 6 7 8 9 10 20 30 40 50; do
    python scripts/colosseum_v2_paper/runtime_rlbench.py \
        --results_filepath "logs/fps/rlbench_fps.csv" --batch_size $batch_size --n_steps 100
done

