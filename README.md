# ColosseumV2

This repository contains the code for the ColosseumV2 project.

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2605.27759)

## Installation

```bash
conda create -n colosseum_v2 python=3.10
conda activate colosseum_v2
pip install torch

pip install -e .

# download the assets
python -m mani_skill.utils.download_asset all

# install the dependencies for the ACT model
pip install -e examples/baselines/act_clip

# (optional) install torch for cuda 12
# python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```


## Train a multi task language conditioned ACT model

https://storage.googleapis.com/bucket-colosseum-v2/trajectory__cv2-full__pd_ee_delta_pose__100.h5

``` bash

# Download the multitask datasets.
# - trajectory__cv2-full__pd_ee_delta_pose__100: Single-Arm dataset
# - trajectory__cv2-full__pd_joint_pos__100: Bimanual dataset
curl --create-dirs -o demos/trajectory__cv2-full__pd_ee_delta_pose__100.h5 https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_ee_delta_pose__100.h5
curl --create-dirs -o demos/trajectory__cv2-full__pd_ee_delta_pose__100.json https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_ee_delta_pose__100.json
curl --create-dirs -o demos/trajectory__cv2-full__pd_joint_pos__100.h5 https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_joint_pos__100.h5
curl --create-dirs -o demos/trajectory__cv2-full__pd_joint_pos__100.json https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_joint_pos__100.json

# (OPTIONAL) Alternatively, create the datasets from scratch. Note: if you see 'Directory not found: demos/TASK-ID/demos__cv2-full_pd_joint_pos__100.h5', simply rerun the script.
./scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh
./scripts/data_generation/motionplanning_colosseum_v2_single_arm.sh

# Single-Arm
python examples/baselines/act_clip/train_rgbd.py \
    --seed 1 --perturbation-set none --demo-path demos/trajectory__cv2-full__pd_ee_delta_pose__100.h5 --sim-backend physx_cuda --num_eval_envs 1 --exp-name=SingleArm_ACT_Clip --control_mode pd_ee_delta_pose --track --batch_size 256 --eval-freq 10000 --max-episode-steps 300 --log-freq 1000 --total-iters 300000 --lr 0.0001 --kl_weight 10.0 --num_queries 30 --hidden_dim 512 --dim_feedforward 1600 --enc_layers 4 --dec_layers 7 --save_freq 10000 --num_eval_episodes 5 --is_multi_task True --target_num_cams 3 --internal_instruction

# Bimanual
python examples/baselines/act_clip/train_rgbd.py \
    --seed 1 --perturbation-set none --demo-path demos/trajectory__cv2-full__pd_joint_pos__100.h5 --sim-backend physx_cuda --num_eval_envs 1 --exp-name=Bimanual_ACT_Clip --control_mode pd_joint_pos --track --batch_size 2 --eval-freq 10000 --max-episode-steps 300 --log-freq 1000 --total-iters 300000 --lr 0.0001 --kl_weight 10.0 --num_queries 30 --hidden_dim 512 --dim_feedforward 1600 --enc_layers 4 --dec_layers 7 --save_freq 10000 --num_eval_episodes 5 --is_multi_task True --target_num_cams 3 --internal_instruction

# Evaluate the models
# Run the entire evaluation loop:
bash examples/baselines/act_clip/eval_rgbd_loop.sh

# Alternatively, run a single evaluation:
# Single-Arm
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path /PATH/TO/CHECKPOINT/best_eval_success_once.pt \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 3 \
    --num-eval-episodes 200 \
    --num-eval-envs 34 \
    --max-episode-steps-from-lookup \
    --internal-instruction \
    --perturbation-set "BLANK" \
    --results-path $LOGS_DIR/results_single_arm__table.csv

# Bimanual
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path /PATH/TO/CHECKPOINT/best_eval_success_once.pt \
    --control-mode "pd_joint_pos" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 4 \
    --num-eval-episodes 200 \
    --num-eval-envs 34 \
    --max-episode-steps-from-lookup \
    --internal-instruction \
    --perturbation-set "BLANK" \
    --results-path $LOGS_DIR/results_bimanual_act.csv
```



## Finetune Pi0.5

This is not done in this repository. Instead, we use the [lerobot](https://github.com/huggingface/lerobot) repository to finetune Pi0.5.

``` bash
cd <SOMEWHERE_TO_STORE_LEROBOT>
git clone https://github.com/huggingface/lerobot.git
cd lerobot
# Follow the installation guide at https://huggingface.co/docs/lerobot/installation, I recommend using uv.
uv pip install '.[dataset]'
uv pip install '.[training]'

# Single-Arm
uv run lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-single-arm-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/single_arm \
  --job_name=pi05_training_single_arm \
  --policy.repo_id=pythonsong/pi05_single_arm \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --steps=30000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda \
  --batch_size=8 \
  --save_freq=1000000000

# Bimanual
uv run lerobot-train \
  --dataset.repo_id=pythonsong/colosseum-bimanual-jan27 \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/bimanual \
  --job_name=pi05_training_bimanual \
  --policy.repo_id=pythonsong/pi05_bimanual \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --steps=30000 \
  --policy.scheduler_decay_steps=30000 \
  --policy.device=cuda \
  --batch_size=8 \
  --save_freq=1000000000
```
