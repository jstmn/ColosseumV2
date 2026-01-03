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


def load_pi0_policy(checkpoint_path, device="cpu"):
    """
    Load OpenPI π₀ policy from checkpoint using official API.
    """
    try:
        from openpi.policies.policy import Policy
        from openpi.policies.policy_config import PolicyConfig
    except ImportError:
        raise RuntimeError("OpenPI is not installed. Cannot load π₀ policy.")

    cfg = PolicyConfig.load_from_checkpoint(checkpoint_path)
    policy = Policy(cfg)
    policy.load_checkpoint(checkpoint_path)
    policy.to(device)
    policy.eval()
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
                actions = policy(rgbd)  # π₀ policy returns Tensor
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

        # Determine which environments are done
        done_envs = np.logical_or(terminateds, truncateds)
        for idx in np.where(done_envs)[0]:
            print(f"Env {idx} finished episode {episode_counts[idx]+1}, return = {episode_returns[idx]:.3f}")
            episode_counts[idx] += 1
            episode_returns[idx] = 0.0

        # Reset done environments
        obs, _ = envs.reset(seed=args.seed + step)
        rgbd = preprocess_rgbd(obs, camera_name="base_camera", device=device)
        step += 1


if __name__ == "__main__":
    main()
