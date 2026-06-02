from pathlib import Path
import numpy as np
import pytest
import h5py

from diffusion_policy.utils import load_demo_dataset

# python -m pytest examples/baselines/diffusion_policy_poseinv/tests/test_utils.py

def _trim_to_action_horizon(arr: np.ndarray, action_len: int) -> np.ndarray:
    if arr.shape[0] == action_len + 1:
        return arr[:-1]
    assert arr.shape[0] == action_len
    return arr


def _expected_actions(
    base_actions: np.ndarray,
    actor_states: dict[str, np.ndarray],
    tcp_pose: np.ndarray,
    actor_names_to_predict: list[str],
    should_predict_ee_pose: bool,
) -> np.ndarray:
    parts = [base_actions]
    for actor_name in actor_names_to_predict:
        pose = _trim_to_action_horizon(actor_states[actor_name][:, :7], base_actions.shape[0])
        parts.append(pose)
    if should_predict_ee_pose:
        parts.append(_trim_to_action_horizon(tcp_pose, base_actions.shape[0]))
    return np.concatenate(parts, axis=-1).astype(base_actions.dtype, copy=False)


def _assert_dataset_actions_match(
    h5_path: Path,
    actor_names_to_predict: list[str],
    should_predict_ee_pose: bool,
) -> None:
    loaded = load_demo_dataset(
        str(h5_path),
        actor_names_to_predict=actor_names_to_predict,
        should_predict_ee_pose=should_predict_ee_pose,
        keys=["observations", "actions"],
        concat=False,
    )
    assert set(loaded.keys()) == {"observations", "actions"}

    with h5py.File(h5_path, "r") as f:
        traj_names = sorted(f.keys(), key=lambda x: int(x.split("_")[-1]))
        assert len(loaded["actions"]) == len(traj_names)
        assert len(loaded["observations"]) == len(traj_names)
        for i, traj_name in enumerate(traj_names):
            traj = f[traj_name]
            base_actions = traj["actions"][()]
            tcp_pose = traj["obs"]["extra"]["tcp_pose"][()]
            actor_states = {name: traj["env_states"]["actors"][name][()] for name in traj["env_states"]["actors"].keys()}
            expected = _expected_actions(
                base_actions=base_actions,
                actor_states=actor_states,
                tcp_pose=tcp_pose,
                actor_names_to_predict=actor_names_to_predict,
                should_predict_ee_pose=should_predict_ee_pose,
            )
            np.testing.assert_allclose(loaded["actions"][i], expected)
            assert isinstance(loaded["observations"][i], dict)


@pytest.fixture
def synthetic_h5_path(tmp_path: Path) -> Path:
    h5_path = tmp_path / "synthetic_trajectory.h5"
    with h5py.File(h5_path, "w") as f:
        for traj_idx in range(2):
            traj = f.create_group(f"traj_{traj_idx}")
            action_len = 3 + traj_idx
            actions = (
                np.arange(action_len * 2, dtype=np.float32).reshape(action_len, 2)
                + 100 * traj_idx
            )
            traj.create_dataset("actions", data=actions)

            obs = traj.create_group("obs")
            extra = obs.create_group("extra")
            tcp_pose = (
                np.arange((action_len + 1) * 7, dtype=np.float32).reshape(action_len + 1, 7)
                + 1000 * traj_idx
            )
            extra.create_dataset("tcp_pose", data=tcp_pose)

            env_states = traj.create_group("env_states")
            actors = env_states.create_group("actors")
            cube = (
                np.arange((action_len + 1) * 13, dtype=np.float32).reshape(action_len + 1, 13)
                + 10000 * traj_idx
            )
            table = (
                np.arange((action_len + 1) * 13, dtype=np.float32).reshape(action_len + 1, 13)
                + 20000 * traj_idx
            )
            actors.create_dataset("cube", data=cube)
            actors.create_dataset("table", data=table)
    return h5_path


@pytest.mark.parametrize(
    "actor_names_to_predict,should_predict_ee_pose",
    [
        ([], False),
        (["cube"], False),
        ([], True),
        (["cube"], True),
    ],
)
def test_load_demo_dataset_synthetic(
    synthetic_h5_path: Path,
    actor_names_to_predict: list[str],
    should_predict_ee_pose: bool,
) -> None:
    _assert_dataset_actions_match(
        synthetic_h5_path,
        actor_names_to_predict=actor_names_to_predict,
        should_predict_ee_pose=should_predict_ee_pose,
    )


@pytest.mark.parametrize(
    "actor_names_to_predict,should_predict_ee_pose",
    [
        ([], False),
        (["cube"], False),
        ([], True),
        (["cube"], True),
    ],
)
def test_load_demo_dataset_real_fixture(
    actor_names_to_predict: list[str],
    should_predict_ee_pose: bool,
) -> None:
    fixture_path = Path(__file__).resolve().parent / "trajectory__external1_camera__rgb__2.rgb.pd_ee_delta_pose.physx_cpu.h5"
    assert fixture_path.exists(), f"Missing test fixture: {fixture_path}"
    _assert_dataset_actions_match(
        fixture_path,
        actor_names_to_predict=actor_names_to_predict,
        should_predict_ee_pose=should_predict_ee_pose,
    )
