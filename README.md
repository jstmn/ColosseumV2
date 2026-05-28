# ManiSkill 3, ColosseumV2 fork

This fork of ManiSkill3 contains the changes that support the ColosseumV2 project. The changes include:
1. The `DistractionSet` class which enables variations in lighting, texture, object pose / shape, etc.
2. Additional tasks, such as `DoubleStackCubes`

**Installation**
```bash
conda create --name=poseInv python=3.10
conda activate poseInv
pip install -e .
pip install -e examples/baselines/diffusion_policy_poseinv
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu124 torch # for cuda 12.4
```



```bash
ENV_ID="RaiseCube-v1"
NUM_TRAJ=2
INCLUDED_CAMERAS="external1_camera"  # comma separated list of cameras
OBS_MODE="rgb"
TRAJ_NAME="trajectory__${INCLUDED_CAMERAS}__${OBS_MODE}__${NUM_TRAJ}"

python mani_skill/examples/motionplanning/panda/run.py \
    --env-id ${ENV_ID} \
    --num-traj ${NUM_TRAJ} \
    --distraction-set none \
    --num-procs 1 \
    --obs-mode ${OBS_MODE} \
    --reward-mode "none" \
    --random-seed \
    --only-count-success \
    --included-cameras ${INCLUDED_CAMERAS} \
    --traj-name ${TRAJ_NAME}

h5ls -r demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.h5

# obs_mode="rgb"
# /traj_8                  Group
# /traj_8/actions          Dataset {79, 8}
# /traj_8/env_states       Group
# /traj_8/env_states/actors Group
# /traj_8/env_states/actors/cube Dataset {80, 13}
# /traj_8/env_states/actors/table Dataset {80, 13}
# /traj_8/env_states/articulations Group
# /traj_8/env_states/articulations/panda_wristcam Dataset {80, 31}
# /traj_8/obs              Group
# /traj_8/obs/agent        Group
# /traj_8/obs/agent/qpos   Dataset {80, 9}
# /traj_8/obs/agent/qvel   Dataset {80, 9}
# /traj_8/obs/agent/world__T__ee Dataset {80, 4, 4}
# /traj_8/obs/agent/world__T__root Dataset {80, 4, 4}
# /traj_8/obs/extra        Group
# /traj_8/obs/extra/tcp_pose Dataset {80, 7}
# /traj_8/obs/sensor_data  Group
# /traj_8/obs/sensor_data/external1_camera Group
# /traj_8/obs/sensor_data/external1_camera/rgb Dataset {80, 224, 224, 3}
# /traj_8/obs/sensor_data/external2_camera Group
# /traj_8/obs/sensor_data/external2_camera/rgb Dataset {80, 224, 224, 3}
# /traj_8/obs/sensor_data/hand_camera Group
# /traj_8/obs/sensor_data/hand_camera/rgb Dataset {80, 128, 128, 3}
# /traj_8/obs/sensor_param Group
# /traj_8/obs/sensor_param/external1_camera Group
# /traj_8/obs/sensor_param/external1_camera/cam2world_gl Dataset {80, 4, 4}
# /traj_8/obs/sensor_param/external1_camera/extrinsic_cv Dataset {80, 3, 4}
# /traj_8/obs/sensor_param/external1_camera/intrinsic_cv Dataset {80, 3, 3}
# /traj_8/obs/sensor_param/external2_camera Group
# /traj_8/obs/sensor_param/external2_camera/cam2world_gl Dataset {80, 4, 4}
# /traj_8/obs/sensor_param/external2_camera/extrinsic_cv Dataset {80, 3, 4}
# /traj_8/obs/sensor_param/external2_camera/intrinsic_cv Dataset {80, 3, 3}
# /traj_8/obs/sensor_param/hand_camera Group
# /traj_8/obs/sensor_param/hand_camera/cam2world_gl Dataset {80, 4, 4}
# /traj_8/obs/sensor_param/hand_camera/extrinsic_cv Dataset {80, 3, 4}
# /traj_8/obs/sensor_param/hand_camera/intrinsic_cv Dataset {80, 3, 3}
# /traj_8/success          Dataset {79}
# /traj_8/terminated       Dataset {79}
# /traj_8/truncated        Dataset {79}


# Convert to ee_delta_pose with:
python mani_skill/trajectory/replay_trajectory.py \
    --traj-path demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.h5 \
    --obs-mode ${OBS_MODE} \
    --reward_mode "none" \
    --target_control_mode "pd_ee_delta_pose" \
    --save-traj

cp demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.${OBS_MODE}.pd_ee_delta_pose.physx_cpu.h5 examples/baselines/diffusion_policy_poseinv/tests

h5ls -r demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.${OBS_MODE}.pd_ee_delta_pose.physx_cpu.h5


# Train
python examples/baselines/diffusion_policy_poseinv/train_rgbd.py \
    --env-id ${ENV_ID} \
    --demo-path demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.${OBS_MODE}.pd_ee_delta_pose.physx_cpu.h5 \
    --seed 1 \
    --batch-size 128 \
    --obs-mode ${OBS_MODE} \
    --included-cameras ${INCLUDED_CAMERAS} \
    --log-freq 1000 \
    --eval-freq 5000 \
    --num-eval-episodes 100 \
    --num-eval-envs 10 \
    --max-episode-steps 100 \
    --track \
    --control-mode pd_ee_delta_pose \
    --sim-backend physx_cuda \
    --predict_actor_pose_names "cube" \
    --predict_ee_pose
```