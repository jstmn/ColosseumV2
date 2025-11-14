#!/usr/bin/env python3
"""Check the actual joint limits."""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv

def check_joint_limits():
    env = gym.make(
        "PlaceDishInRack-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
    )

    obs, info = env.reset()

    agent = env.unwrapped.agent
    robot = agent.robot
    qlimits = robot.get_qlimits()

    print("Panda Joint Limits:")
    joint_names = agent.arm_joint_names if hasattr(agent, 'arm_joint_names') else []
    for i, name in enumerate(joint_names):
        if i < qlimits.shape[1]:
            lower = qlimits[0, i, 0].item()
            upper = qlimits[0, i, 1].item()
            print(f"{name:15s}: [{lower:7.3f}, {upper:7.3f}]")

    env.close()


if __name__ == "__main__":
    check_joint_limits()