# jstmn/ManiSkill/tools/evaluate_rgbd.py
import torch
import argparse
import gymnasium as gym
import mani_skill.envs  # registers ManiSkill envs
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB-D model on ManiSkill environments"
    )

    parser.add_argument(
        "--env-id", "-e", required=True,
        help="ManiSkill environment ID (e.g., PickCube-v1)"
    )

    parser.add_argument(
        "--model-checkpoint", "-m", required=False,
        help="Path to π₀ model checkpoint"
    )

    parser.add_argument(
        "--num-episodes", "-n", type=int, default=50,
        help="Total episodes to run"
    )

    parser.add_argument(
        "--num-envs", type=int, default=8,
        help="Number of parallel environments"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed (passed to env.reset)"
    )

    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="Device to run policy on ('cpu' or 'cuda')"
    )

    return parser.parse_args()


def build_vector_env(env_id, num_envs):
    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode=None,
    )
    return envs


def preprocess_rgbd(obs, camera_name="base_camera", device="cpu"):
    cam_data = obs["sensor_data"][camera_name]

    # RGB
    rgb = cam_data["rgb"].float() / 255.0
    rgb = rgb.permute(0, 3, 1, 2).contiguous()

    # Depth
    depth = cam_data["depth"].float()
    depth[depth <= 0] = 0.0
    depth = depth / 1000.0  # mm -> meters
    depth = torch.clamp(depth, 0.0, 5.0)
    depth = depth.permute(0, 3, 1, 2).contiguous()

    return {
        "rgb": rgb.to(device),
        "depth": depth.to(device),
    }


def load_pi0_policy(checkpoint_name="pi05_droid", device="cpu"):
    """
    Load a trained Pi0 policy from OpenPI using the modern API.
    """
    try:
        from openpi.training import config as _config
        from openpi.shared import download
        from openpi.policies import policy_config as _policy_config
    except ImportError:
        raise RuntimeError("OpenPI is not installed. Cannot load π₀ policy.")

    # 1. Load config
    config = _config.get_config(checkpoint_name)

    # 2. Download checkpoint
    checkpoint_dir = download.maybe_download(f"gs://openpi-assets/checkpoints/{checkpoint_name}")

    # 3. Create trained policy
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 4. Set device (PyTorch policies)
    if hasattr(policy, "_is_pytorch_model") and policy._is_pytorch_model:
        policy._pytorch_device = device

    return policy


def pi0_policy_placeholder(obs_rgbd, env):
    """
    Returns random actions within the environment's action space.
    """
    batch_size = obs_rgbd["rgb"].shape[0]
    low = env.single_action_space.low
    high = env.single_action_space.high
    actions = np.random.uniform(low, high, size=(batch_size, env.single_action_space.shape[0]))
    return actions


def main():
    args = parse_args()
    device = args.device

    envs = build_vector_env(args.env_id, args.num_envs)
    obs, _ = envs.reset(seed=args.seed)
    rgbd = preprocess_rgbd(obs, camera_name="base_camera", device=device)

    # Load policy if checkpoint provided
    if args.model_checkpoint:
        policy = load_pi0_policy(args.model_checkpoint, device=device)
        use_openpi = True
        print(f"Loaded OpenPI policy from {args.model_checkpoint}")
    else:
        policy = None
        use_openpi = False
        print("No checkpoint provided. Using random placeholder policy.")

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_counts = np.zeros(args.num_envs, dtype=np.int32)

    step = 0
    while episode_counts.sum() < args.num_episodes:
        # Generate actions
        if use_openpi:
            with torch.no_grad():
                actions = policy(rgbd)
                actions = actions.cpu().numpy()
        else:
            actions = pi0_policy_placeholder(rgbd, envs)

        # Step environments
        obs_new, rewards, terminateds, truncateds, infos = envs.step(actions)

        # Convert tensors to NumPy if needed
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(terminateds, torch.Tensor):
            terminateds = terminateds.cpu().numpy()
        if isinstance(truncateds, torch.Tensor):
            truncateds = truncateds.cpu().numpy()

        episode_returns += rewards
        done_envs = np.logical_or(terminateds, truncateds)

        for idx in np.where(done_envs)[0]:
            print(
                f"Env {idx} finished episode {episode_counts[idx] + 1}, "
                f"return = {episode_returns[idx]:.3f}"
            )
            episode_counts[idx] += 1
            episode_returns[idx] = 0.0

        # Reset ONLY finished envs
        if done_envs.any():
            obs_reset, _ = envs.reset(
                seed=args.seed + step,
                options={"reset_indices": np.where(done_envs)[0]},
            )
            obs_new["sensor_data"]["base_camera"]["rgb"][done_envs] = \
                obs_reset["sensor_data"]["base_camera"]["rgb"][done_envs]
            obs_new["sensor_data"]["base_camera"]["depth"][done_envs] = \
                obs_reset["sensor_data"]["base_camera"]["depth"][done_envs]

        obs = obs_new
        rgbd = preprocess_rgbd(obs, camera_name="base_camera", device=device)

        if step % 50 == 0:
            print(f"Step {step}, episodes completed: {episode_counts.sum()}")

        step += 1



if __name__ == "__main__":
    main()
