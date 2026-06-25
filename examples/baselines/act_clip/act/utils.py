from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from h5py import File, Group, Dataset
from typing import Optional
from random import sample
from tqdm import tqdm

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
    

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
    'states': 'env_states',
    'observations': 'obs',
    'success': 'success',
    'next_observations': 'obs',
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    'actions': 'actions',
}

def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")

def load_hdf5(path, ):
    print(f'load_hdf5({path=}, ) | Loading HDF5 file')
    file = File(path, 'r')
    ret = load_content_from_h5_file(file)
    file.close()
    print('Loaded')
    return ret

def load_traj_hdf5(path: str, max_n_demos_in_dataset : int | None =None) -> tuple[dict, bool]:
    """Load a h5 dataset file
    
    Args:
        path (str): the path to the .h5 dataset
        max_n_demos_in_dataset (int | None, optional): The maximum number of demos to include

    Returns:
        tuple[dict, bool]: 
            - the dataset
            - whether all episodes were loaded from the h5 file. This won't be the case if num-episodes > max_n_demos_in_dataset
    """
    print(f'load_traj_hdf5({path=}, {max_n_demos_in_dataset=}) | Loading HDF5 file')
    file = File(path, 'r')
    keys = list(file.keys())
    all_episodes_loaded = True

    if max_n_demos_in_dataset is not None and len(keys) > max_n_demos_in_dataset:
        keys = sample(keys, max_n_demos_in_dataset) # randomly samples without replacement
        all_episodes_loaded = False

    ret = {}
    for key in tqdm(keys):
        ret = {key: load_content_from_h5_file(file[key])}

    file.close()
    print('Loaded')
    return ret, all_episodes_loaded


def load_demo_dataset(path: str, keys=['observations', 'actions'], max_n_demos_in_dataset: int | None = None, concat: bool =True) -> tuple[dict, bool]:

    # assert num_traj is None
    raw_data, all_episodes_loaded = load_traj_hdf5(path, max_n_demos_in_dataset)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_X'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = raw_data[list(raw_data.keys())[0]]
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [ raw_data[idx][source_key] for idx in raw_data ]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ['observations', 'states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[:-1] for t in dataset[target_key]
                ], axis=0)
            elif target_key in ['next_observations', 'next_states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[1:] for t in dataset[target_key]
                ], axis=0)
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print('Load', target_key, dataset[target_key].shape)
        else:
            print('Load', target_key, len(dataset[target_key]), type(dataset[target_key][0]))
    return dataset, all_episodes_loaded