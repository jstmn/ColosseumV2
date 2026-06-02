import numpy as np
import torch
from gymnasium import spaces
from h5py import Dataset, File, Group
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path,
    actor_names_to_predict: list[str] | None,
    should_predict_ee_pose: bool,
    keys=["observations", "actions"],
    num_traj=None,
    concat=True,
):
    """

    From the docs (https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/advanced.html#custom-extra-state):

    "Actor state is a flat 13 dimensional composed of 3D position, 4D quaternion, 3D linear velocity, and 3D angular velocity"

    Trajectory structure:
    /traj_8                  Group
    /traj_8/actions          Dataset {79, 8}
    /traj_8/env_states       Group
    /traj_8/env_states/actors Group
    /traj_8/env_states/actors/cube Dataset {80, 13}
    /traj_8/env_states/actors/table Dataset {80, 13}
    /traj_8/env_states/articulations Group
    /traj_8/env_states/articulations/panda_wristcam Dataset {80, 31}
    /traj_8/obs              Group
    /traj_8/obs/agent        Group
    /traj_8/obs/agent/qpos   Dataset {80, 9}
    /traj_8/obs/agent/qvel   Dataset {80, 9}
    /traj_8/obs/agent/world__T__ee Dataset {80, 4, 4}
    /traj_8/obs/agent/world__T__root Dataset {80, 4, 4}
    /traj_8/obs/extra        Group
    /traj_8/obs/extra/tcp_pose Dataset {80, 7}
    /traj_8/obs/sensor_data  Group
    /traj_8/obs/sensor_data/external1_camera Group
    /traj_8/obs/sensor_data/external1_camera/rgb Dataset {80, 224, 224, 3}
    /traj_8/obs/sensor_data/external2_camera Group
    /traj_8/obs/sensor_data/external2_camera/rgb Dataset {80, 224, 224, 3}
    /traj_8/obs/sensor_data/hand_camera Group
    /traj_8/obs/sensor_data/hand_camera/rgb Dataset {80, 128, 128, 3}
    /traj_8/obs/sensor_param Group
    /traj_8/obs/sensor_param/external1_camera Group
    /traj_8/obs/sensor_param/external1_camera/cam2world_gl Dataset {80, 4, 4}
    /traj_8/obs/sensor_param/external1_camera/extrinsic_cv Dataset {80, 3, 4}
    /traj_8/obs/sensor_param/external1_camera/intrinsic_cv Dataset {80, 3, 3}
    /traj_8/obs/sensor_param/external2_camera Group
    /traj_8/obs/sensor_param/external2_camera/cam2world_gl Dataset {80, 4, 4}
    /traj_8/obs/sensor_param/external2_camera/extrinsic_cv Dataset {80, 3, 4}
    /traj_8/obs/sensor_param/external2_camera/intrinsic_cv Dataset {80, 3, 3}
    /traj_8/obs/sensor_param/hand_camera Group
    /traj_8/obs/sensor_param/hand_camera/cam2world_gl Dataset {80, 4, 4}
    /traj_8/obs/sensor_param/hand_camera/extrinsic_cv Dataset {80, 3, 4}
    /traj_8/obs/sensor_param/hand_camera/intrinsic_cv Dataset {80, 3, 3}
    /traj_8/success          Dataset {79}
    /traj_8/terminated       Dataset {79}
    /traj_8/truncated        Dataset {79}
    """
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    actor_names_to_predict = actor_names_to_predict or []

    if "actions" in keys and (actor_names_to_predict or should_predict_ee_pose):
        for traj_name, traj_data in raw_data.items():
            action_seq = traj_data["actions"]
            action_len = action_seq.shape[0]
            pose_targets = []

            for actor_name in actor_names_to_predict:
                actor_states = traj_data["env_states"]["actors"].get(actor_name, None)
                assert actor_states is not None, (
                    f"Actor '{actor_name}' not found in {traj_name}. "
                    f"Available actors: {list(traj_data['env_states']['actors'].keys())}"
                )
                assert (
                    actor_states.shape[-1] == 13
                ), f"Actor state for '{actor_name}' in {traj_name} should be 13, got shape {actor_states.shape}"
                actor_pose = actor_states[:, :7]
                if actor_pose.shape[0] == action_len + 1:
                    actor_pose = actor_pose[:-1]
                else:
                    assert actor_pose.shape[0] == action_len, (
                        f"Unexpected actor pose horizon for '{actor_name}' in {traj_name}. "
                        f"Expected {action_len} or {action_len + 1}, got {actor_pose.shape[0]}"
                    )
                pose_targets.append(actor_pose.astype(np.float32, copy=False))

            if should_predict_ee_pose:
                ee_pose = traj_data["obs"]["extra"]["tcp_pose"]
                if ee_pose.shape[0] == action_len + 1:
                    ee_pose = ee_pose[:-1]
                else:
                    assert ee_pose.shape[0] == action_len, (
                        f"Unexpected tcp_pose horizon for {traj_name}. "
                        f"Expected {action_len} or {action_len + 1}, got {ee_pose.shape[0]}"
                    )
                pose_targets.append(ee_pose.astype(np.float32, copy=False))

            if pose_targets:
                pose_targets_np = np.concatenate(pose_targets, axis=-1)
                traj_data["actions"] = np.concatenate(
                    [action_seq, pose_targets_np.astype(action_seq.dtype, copy=False)], axis=-1
                )
                print(
                    f"Augmented actions for {traj_name}: {action_seq.shape[-1]} -> "
                    f"{traj_data['actions'].shape[-1]} dims"
                )
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data["traj_0"]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(dataset[target_key][0]) > len(
                raw_data["traj_0"]["actions"]
            ):
                dataset[target_key] = np.concatenate([t[:-1] for t in dataset[target_key]], axis=0)
            elif target_key in ["next_observations", "next_states"] and len(dataset[target_key][0]) > len(
                raw_data["traj_0"]["actions"]
            ):
                dataset[target_key] = np.concatenate([t[1:] for t in dataset[target_key]], axis=0)
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


def convert_obs(
    obs,
    concat_fn,
    transpose_fn,
    state_obs_extractor,
    depth=True,
    included_cameras: list[str] | None = None,
):
    img_dict = obs["sensor_data"]
    if included_cameras:
        missing_cameras = [cam for cam in included_cameras if cam not in img_dict]
        assert not missing_cameras, (
            f"Requested cameras not found in observation: {missing_cameras}. "
            f"Available cameras: {list(img_dict.keys())}"
        )
        img_dict = {cam: img_dict[cam] for cam in included_cameras}
    ls = ["rgb"]
    if depth:
        ls = ["rgb", "depth"]

    new_img_dict = {
        key: transpose_fn(concat_fn([v[key] for v in img_dict.values()])) for key in ls  # (C, H, W) or (B, C, H, W)
    }
    if "depth" in new_img_dict and isinstance(
        new_img_dict["depth"], torch.Tensor
    ):  # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict["depth"] = new_img_dict["depth"].to(torch.float16)

    # Remove all 4x4 transform-like values from state inputs entirely.
    keys_to_remove = []
    for group_name in ("agent", "extra"):
        if group_name in obs and isinstance(obs[group_name], dict):
            for key, value in obs[group_name].items():
                value = np.asarray(value)
                if value.ndim >= 2 and value.shape[-2:] == (4, 4):
                    keys_to_remove.append((group_name, key))
    if keys_to_remove:
        obs = dict(obs)
        for group_name in ("agent", "extra"):
            if group_name in obs and isinstance(obs[group_name], dict):
                obs[group_name] = dict(obs[group_name])
        for group_name, key in keys_to_remove:
            obs[group_name].pop(key, None)

    # Unified version
    # Flatten each state component over feature axes for robust concatenation.
    states_to_stack = state_obs_extractor(obs)
    processed_states = []
    for x in states_to_stack:
        x = np.asarray(x)
        if x.dtype == np.float64:
            x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1)
        processed_states.append(x)
    state = np.concatenate(processed_states, axis=-1)

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]

    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(-float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
