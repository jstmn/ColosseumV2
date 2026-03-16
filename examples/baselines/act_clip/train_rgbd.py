import h5py
ALGO_NAME = 'BC_ACT_CLIP_rgbd'

from collections import defaultdict
import argparse
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from act.evaluate import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro
import wandb
from mani_skill.utils.io_utils import load_json
import json
from collections import Counter
from mani_skill.envs.tasks.tabletop.colosseum_v2.distraction_set import DISTRACTION_SETS
from mani_skill.envs.tasks.tabletop import *

# Note(@jstmn): 'world__T__ee', 'world__T__root' were added to the observation space of the Panda agent as a 
# convenience feature. We remove it here because it isn't included in the default ACT method. Additionally, it 
# causes at torch shape mismatch error, because the configuration space is [batch x ndof], but these values are
# [batch x 4 x 4]
OBS_KEYS_TO_REMOVE = {"world__T__ee", "world__T__root"}

@dataclass
class Args:

    distraction_set: str

    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int | None = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    lang_instruction: Optional[str] = None
    """ language_instruction for clip embedding"""
    internal_instruction: bool = False
    """ use pre-defiend language_instructions for clip embedding"""
    is_multi_task: bool = False
    """ use multi-task dataset"""
    target_num_cams: int = 1
    """ num of cameras"""
    
    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10
    """weight for the kl loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    # Transformer
    # enc_layers: int = 2
    enc_layers: int = 4
    # dec_layers: int = 4
    dec_layers: int = 7
    # dim_feedforward: int = 512
    dim_feedforward: int = 1600
    # hidden_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    #lr_project_lr
    lr_lang: float = 1e-3

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    max_episode_steps_from_lookup: bool = False
    """If toggled, the max episode steps will be looked up from a hardcoded dictionary"""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    real: bool = False


    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    checkpoint_path: str | None = None
    """the path to the checkpoint to load"""
    results_path: str | None = None
    """the path to save results to"""
    is_multi_task: bool | None = None
    """whether the dataset is multi-task. Must be set for evaluation"""
    target_num_cams: int | None = None
    """the number of cameras to use for the evaluation environments. Must be set for evaluation"""

    variation_factors_subset: list[str] = field(default_factory=list)
    """A subset of the variation factors to evaluate on, when running evaluations automatically. Runs on all 
    colosseum v2 variation factors by default"""
    tasks_subset: list[str] = field(default_factory=list)
    """A subset of the tasks to evaluate on, when running evaluations automatically. Runs on all colosseum v2 tasks 
    by default"""


TASK_TEXT_MAP = {
    #Single_arm
    "PickSodaFromCabinet-v1": "pick up the soda can from the cabinet",
    "PlaceBookInShelf-v1": "place the book into the bookshelf",
    "HammerNail-v1": "use the hammer to hit the nail into the wood",
    "ScoopBanana-v1": "scoop the banana and move it to the target location",
    "OpenDrawer-v1": "grasp the handle and pull the drawer open",
    "OpenCabinet-v1": "grasp the handle and open the cabinet door",
    "PlaceCubeInDrawer-v1": "place the cube inside the open drawer",
    "CookItemInPan-v1": "place the food item in the pan for cooking",
    "LiftPegUpright-v1": "lift the peg and stand it upright on the table",
    "PegInsertionSide-v2": "insert the peg into the hole from the side",
    "PickDishFromRack-v1": "pick up a dish from the drying rack",
    "PlaceDishInRack-v1": "place the dish into the drying rack",
    "PlugCharger-v1": "plug the charger into the wall outlet",
    "RaiseCube-v1": "lift the cube up to a certain height",
    "RotateArrow-v1": "rotate the arrow lever to the target direction",
    "StackCube-v1": "stack red cube on top of green cube",
    "StackCubeColosseumV2-v1": "stack red cube on top of green cube",
    "LiftPegUprightColosseumV2-v1": "lift the peg and stand it upright on the table",
    "PegInsertionSideColosseumV2-v1": "insert the peg into the hole from the side",
    "PlugChargerColosseumV2-v1": "plug the charger into the wall outlet",

    #Bimanual
    "DualArmPickCube-v1": "use both arms to pick up the cube",
    "DualArmPickBottle-v1": "use both arms to pick up the bottle",
    "DualArmLiftPot-v1": "grasp the pot with both hands and lift it up",
    "DualArmLiftTray-v1": "grasp the tray with both hands and lift it up",
    "DualArmPushBox-v1": "use both hands to push the box to the target location",
    "DualArmPourPot-v1": "lift the pot with both hands and pour its contents into the target container",
    "DualArmThreading-v1": "use both hands to thread the object through the target opening",
    "DualArmPenCap-v1": "hold the pen with one hand and remove the cap with the other hand",
    "DualArmDrawerPlace-v1": "open the drawer with one hand and place the object inside with the other hand",
    "DualArmDrawerOpen-v1": "use both hands to grasp and pull the drawer open",
    "DualArmStackCube-v1": "use both hands to stack one cube on top of another",
    "DualArmStack3Cube-v1": "use both hands to stack three cubes into a tower",

}

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, is_multi_task, target_num_cams, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.is_multi_task = is_multi_task
        self.target_num_cams = target_num_cams
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # resize the input image
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        for key in OBS_KEYS_TO_REMOVE:
            try:
                del observation["agent"][key]
            except KeyError:
                pass

        images_rgb = []
        images_depth = []
  
        if self.target_num_cams == 1:
            cam_data = sensor_data["base_camera"]
            if self.include_rgb:
                resized_rgb = self.transforms(
                    cam_data["rgb"].permute(0, 3, 1, 2)
                )
                images_rgb.append(resized_rgb)
            
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(
                    depth.permute(0, 3, 1, 2)
                )
                images_depth.append(resized_depth)
        else:   
            for cam_data in sensor_data.values():
                if self.include_rgb:
                    resized_rgb = self.transforms(
                        cam_data["rgb"].permute(0, 3, 1, 2)
                    )  # (1, 3, 224, 224)
                    images_rgb.append(resized_rgb)
                if self.include_depth:
                    depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                    resized_depth = self.transforms(
                        depth.permute(0, 3, 1, 2)
                    )  # (1, 1, 224, 224)
                    images_depth.append(resized_depth)

        rgb = torch.stack(images_rgb, dim=1) # (1, num_cams, C, 224, 224), uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (1, num_cams, C, 224, 224), float16

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)

        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb
        elif self.include_rgb and self.include_depth:
            ret["rgb"] = rgb
            ret["depth"] = depth
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth
        return ret


class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, num_queries, num_traj, internal_instruction, lang_instruction, is_multi_task, include_depth=True, args=None):
        self.args = args
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
            
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # pre-trained models from torchvision.models expect input image to be at least 224x224


        self.internal_instruction = internal_instruction
        self.lang_instruction = lang_instruction
        self.is_multi_task = is_multi_task
        self.target_num_cams = args.target_num_cams

        # Pre-process the observations, make them align with the obs returned by the FlattenRGBDObservationWrapper
        obs_traj_dict_list = []
        for obs_traj_dict in tqdm.tqdm(trajectories['observations'], desc='Pre-processing observations'):
            obs_traj_dict = self.process_obs(obs_traj_dict)
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict.keys())

        #multi-task envs
        if self.is_multi_task:
            self.json_data = load_json(data_path.replace(".h5", ".json"))
            self.episode_env_ids = []
            for ep in self.json_data["episodes"]:
                eid = ep.get("env_id", args.env_id)
                self.episode_env_ids.append(eid)

        # Pre-process the actions
        for i in tqdm.tqdm(range(len(trajectories['actions'])), desc='Pre-processing actions'):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in tqdm.tqdm(range(self.num_traj), desc='Pre-processing trajectories'):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        #language conditioning
        if self.internal_instruction:
            env_id = self.episode_env_ids[traj_idx]
            assert env_id in TASK_TEXT_MAP, f"Task '{env_id}' not in TASK_TEXT_MAP"
            instruction = TASK_TEXT_MAP[env_id]
        elif self.lang_instruction is not None:
            instruction = self.lang_instruction
        else:
            instruction = ""

        # get state at start_ts only
        state = self.trajectories['observations'][traj_idx]['state'][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)
                
        # normalize state and act_seq
        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # get rgb or rgbd data at start_ts and combine with state to form obs
        if self.include_depth:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            depth = self.trajectories['observations'][traj_idx]['depth'][ts]
            obs = dict(state=state, rgb=rgb, depth=depth)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            obs = dict(state=state, rgb=rgb)

        return {
            'observations': obs,
            'actions': act_seq,
            'lang_instruction': instruction
        }

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        # remove keys that shouldn't be included in the observation space
        for key in OBS_KEYS_TO_REMOVE:
            try:
                del obs_dict["agent"][key]
            except KeyError:
                pass

        # get rgbd data
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images_rgb = []
        images_depth = []

        if self.target_num_cams == 1:
            cam_list = [sensor_data["base_camera"]]
        else:
            cam_list = sensor_data.values()

        for cam_data in cam_list:
            rgb = torch.from_numpy(cam_data["rgb"])
            resized_rgb = self.transforms(rgb.permute(0, 3, 1, 2))
            images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.from_numpy(cam_data["depth"].astype(np.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1) # (ep_len, num_cams, 3, 224, 224) # still uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (ep_len, num_cams, 1, 224, 224) # float16

       # flatten the rest of the data which should just be state data
        if 'extra' in obs_dict:
            obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()} # dirty fix for data that has one dimension (e.g. is_grasped)
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)
    
        processed_obs = dict(state=obs_dict, rgb=rgb, depth=depth) if self.include_depth else dict(state=obs_dict, rgb=rgb)

        return processed_obs

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        for traj_idx, ts in self.slices:
            state = self.trajectories['observations'][traj_idx]['state'][ts]
            act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
            action_len = act_seq.shape[0]
            if action_len < self.num_queries:
                target_pos = act_seq[-1]
                act_seq = torch.cat([act_seq, target_pos.repeat(self.num_queries-action_len, 1)], dim=0)
            all_state_data.append(state)
            all_action_data.append(act_seq)

        all_state_data = torch.stack(all_state_data)
        all_action_data = torch.concatenate(all_action_data)

        # normalize obs (state) data
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std,
                 "example_state": state}

        return stats


class Agent(nn.Module):
    def __init__(self, env, args, is_multi_task: bool):
        super().__init__()
        self.args = args
        if env is not None:
            assert len(env.single_observation_space['state'].shape) == 1 # (obs_dim,)
            assert len(env.single_observation_space['rgb'].shape) == 4 # (num_cams, C, H, W)
            assert len(env.single_action_space.shape) == 1
            #assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        #real dataset
        if args.real:
            self.state_dim = 18
            self.act_dim = 9
        else:
            self.state_dim = env.single_observation_space['state'].shape[0]
            self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        if args.lang_instruction is not None or args.internal_instruction:
            use_lang_instruction = True
        else:
            use_lang_instruction = False
        self.internal_instruction = args.internal_instruction
        self.is_multi_task = is_multi_task



        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
            use_lang_instruction=use_lang_instruction,
        )

    def compute_loss(self, obs, action_seq, lang_instruction):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if args.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, (mu, logvar) = self.model(
        obs=obs, 
        actions=action_seq, 
        lang_instruction=lang_instruction)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs, lang_instruction=None):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if self.args.include_depth:
            obs['depth'] = obs['depth'].float()

        # forward pass
        a_hat, (_, _) = self.model(
            obs=obs, 
            lang_instruction=lang_instruction
        )

        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)

    assert args.sim_backend in ("physx_cpu", "physx_cuda")


    # Multi-task policy:
    #   - either use internal_instruction
    #   - or use no language embedding at all (both None/False)
    # Single-task policy:
    #   - either use lang_instruction (fixed sentence)
    #   - or no language embedding at all

    demo_json = load_json(args.demo_path.replace(".h5", ".json"))
    is_multi_task = bool(demo_json.get("multi_env", False))
    if args.is_multi_task:
        assert is_multi_task
    if args.lang_instruction is not None or args.internal_instruction:
        use_lang_instruction = True
    else:
        use_lang_instruction = False
        

    if is_multi_task:
        if args.internal_instruction:
            if args.lang_instruction is not None:
                raise ValueError("do not use internal_instruction and lang_instruction at the same time"
                )
        else:
            if args.lang_instruction is not None:
                raise ValueError(
                    "Multi-task -> internal_instruction=True or "
                    "set lang_instruction=None(=no language embedding)"
                )
    else:
        if args.internal_instruction:
            raise ValueError(
                "do not use internal_instruction for Single-task. "
                "use lang_instruction"
            )


    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array",
        distraction_set=DISTRACTION_SETS[args.distraction_set.upper()],
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None

    envs = None
    envs_by_task = None

    #Real tasks do not need eval_envs
    if not args.real:
        wrappers = [partial(FlattenRGBDObservationWrapper, is_multi_task=is_multi_task, target_num_cams=args.target_num_cams, depth=args.include_depth)]
        envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers)


        #eval envs for each task
        if is_multi_task:
            envs_by_task = {}
            for eid in demo_json.get("env_ids", []):
                envs_by_task[eid] = make_eval_envs(
                    eid, args.num_eval_envs, args.sim_backend,
                    env_kwargs, other_kwargs,
                    video_dir=f'runs/{run_name}/videos' if args.capture_video else None,
                    wrappers=wrappers
                )
    else:
        print("SKIP EVALUATION")


    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, num_traj=args.num_demos, internal_instruction=args.internal_instruction, lang_instruction=args.lang_instruction, is_multi_task = is_multi_task, include_depth=args.include_depth, args=args)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    if args.track:
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT_CLIP",
            tags=["act_clip"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # agent setup
    agent = Agent(envs, args, is_multi_task).to(device)

    if use_lang_instruction:
        param_dicts = [
        {
            "params": [
                p for n, p in agent.named_parameters() 
                if "backbone" not in n and "lang_proj" not in n and p.requires_grad
            ]
        },
        #Backbone
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        #Lang_mlp
        {
            "params": [p for n, p in agent.named_parameters() if "lang_proj" in n and p.requires_grad],
            "lr": args.lr_lang,
        },
        ]

    else:
        param_dicts = [
            {
                "params": [
                    p for n, p in agent.named_parameters()
                    if "backbone" not in n
                    and "lang_proj" not in n
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in agent.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
        ]


    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    # LR drop by a factor of 10 after lr_drop iters
    lr_drop = int((2/3)*args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args, is_multi_task).to(device)

    # Evaluation
    
    eval_stats = None
    if dataset.norm_stats is not None:
        eval_stats = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in dataset.norm_stats.items()}
    
    eval_kwargs = dict(
        stats=eval_stats,
        num_queries=args.num_queries,
        temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps,
        device=device,
        sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for cur_iter, data_batch in tqdm.tqdm(enumerate(train_dataloader), desc='Training', total=args.total_iters):
        last_tick = time.time()
        # copy data from cpu to gpu
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].cuda(non_blocking=True)


        instructions = data_batch['lang_instruction']

        # forward and compute loss
        loss_dict = agent.compute_loss(
            obs=obs_batch_dict, # obs_batch_dict['state'] is (B, obs_dim)
            action_seq=act_batch, # (B, num_queries, act_dim)
            lang_instruction=instructions,
        )
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timings["update"] += time.time() - last_tick

        #Evaluation
        if cur_iter % args.eval_freq == 0 and not args.real:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())

            if not is_multi_task:
                # --- [Single-task Evaluation] ---
                eval_lang = args.lang_instruction
                eval_metrics = evaluate(
                    args.num_eval_episodes, ema_agent, envs, eval_kwargs,
                    lang_instruction=eval_lang,
                    save_name="latest_eval"
                )
            else:
                # --- [Multi-task Evaluation] ---
                all_task_success_rates = []
                combined_metrics = defaultdict(list)

                for eid, task_envs in envs_by_task.items():
                    eval_lang = TASK_TEXT_MAP[eid] if args.internal_instruction else None
                    
                    task_metrics = evaluate(
                        args.num_eval_episodes, ema_agent, task_envs, eval_kwargs,
                        lang_instruction=eval_lang,
                        save_name=f"latest_eval_{eid}"
                    )

                    s_rate = np.mean(task_metrics["success_at_end"])
                    all_task_success_rates.append(s_rate)

                    for k, v in task_metrics.items():
                        m = np.mean(v)
                        writer.add_scalar(f"eval/{eid}/{k}", m, cur_iter)
                        combined_metrics[k].append(m)
                    
                    print(f"[{eid}] success_at_end: {s_rate:.4f}")

                eval_metrics = {}
                for k, v_list in combined_metrics.items():
                    eval_metrics[k] = np.mean(v_list)
                
                avg_success = np.mean(all_task_success_rates)
                writer.add_scalar("eval/overall_avg_success", avg_success, cur_iter)
                print(f"--- [Overall] Average Success Rate: {avg_success:.4f} ---")

            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {args.num_eval_episodes} episodes per task")
            for k in eval_metrics.keys():
                m_val = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", m_val, cur_iter)
                print(f"Total {k}: {m_val:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and np.mean(eval_metrics[k]) > best_eval_metrics[k]:
                    best_eval_metrics[k] = np.mean(eval_metrics[k])
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {best_eval_metrics[k]:.4f}. Saving checkpoint.')

        # Checkpoint
        if args.save_freq is not None and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))


    if envs is not None:
        envs.close()
    if envs_by_task is not None:
        for _eid, _env in envs_by_task.items():
            _env.close()
    writer.close()