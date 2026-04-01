import argparse
import psutil
from time import time, sleep
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import rlbench
import gymnasium as gym
import multiprocessing as mp
from typing import Any
from rlbench.action_modes.action_mode import JointPositionActionMode
import numpy as np

import sys
sys.path.append("examples/baselines/act_clip")
from eval_rgbd import MAX_EPISODE_STEPS_BY_TASK


"""
This script measures maniskill's FPS for difference batch sizes. Note that you need to run each environment in a 
separate process.


###### How to install rlbench:
mkdir thirdparty/
# Download, install Coppelia Sim. These instructions assume you are using Ubuntu 20. There are __no__ instructions for Ubuntu 22 or 24 on the PyRep github (https://github.com/stepjam/PyRep). I recommend opening an issue there if you need to run this but are on 22/24.
curl -L -o thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -C thirdparty/ -xf thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
echo "" >> ~/.bashrc
echo "export COPPELIASIM_ROOT=$(realpath thirdparty/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/)" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
source ~/.bashrc
pip install git+https://github.com/stepjam/RLBench.git

# Example usage:
python scripts/colosseum_v2_paper/runtime_rlbench_2.py \
    --n_steps 200

"""


def measure_runtime(
    n_steps: int,
    env_id: str = "rlbench/slide_block_to_target-vision-v0",
):
    t0 = time()
    env = gym.make(
        env_id,
        action_mode=JointPositionActionMode(),
    )
    env.reset()

    for i in range(n_steps):
        env.step(env.action_space.sample())
        # This dict setting is counted as part of the computation time which is slightly unfair, but in practice it's
        # runs in microseconds so it's negligible compared to the 10+ seconds from RLBench.
    env.close()
    tf = time()
    print(f"Time taken:")
    print(f"  - seconds: {tf - t0}")
    print(f"  - minutes: {round((tf - t0) / 60, 5)}")
    print(f"  - hours:   {round((tf - t0) / 3600, 5)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="rlbench/slide_block_to_target-vision-v0")
    parser.add_argument("--n_steps", type=int, required=True)
    args = parser.parse_args()
    measure_runtime(n_steps=args.n_steps, env_id=args.env_id)
