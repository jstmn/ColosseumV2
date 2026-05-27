import mplib
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import BaseMotionPlanningSolver
from transforms3d import quaternions


class TwoFingerGripperMotionPlanningSolver(BaseMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        slow_down: bool = False,
        add_sinusoidal_noise: bool = False,
    ):
        super().__init__(env, debug, vis, base_pose, print_env_info, joint_vel_limits, joint_acc_limits)
        self.slow_down = slow_down
        self.add_sinusoidal_noise = add_sinusoidal_noise
        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp_pose)

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _slowdown_path(self, actions, n_intermediate_steps: int = 50):
        """Interpolate between waypoints to create a slower, smoother path."""
        n_step = len(actions)
        actions_slowed = []
        for i in range(n_step):
            action_0 = actions[max(i-1, 0)]
            action_1 = actions[i]
            for j in range(n_intermediate_steps):
                action = action_0 + (action_1 - action_0) * j / (n_intermediate_steps - 1)
                actions_slowed.append(action)
        return np.array(actions_slowed)

    def _add_sinusoidal_noise(self, actions, n_dof_arm):
        """Add sinusoidal noise to the actions for each joint."""
        # Convert list to numpy array if needed
        n_steps = len(actions)
        if n_steps < 15:
            print(f"Warning: Not enough steps to add sinusoidal noise. Only {n_steps} steps.")
            return actions
        sin_scales = np.random.uniform(0.025, 0.05, size=n_dof_arm)
        periods = 0.5 * np.ones_like(sin_scales)
        noise = np.zeros((n_steps, n_dof_arm))
        t_steps = np.linspace(0, n_steps, n_steps)
        for i in range(n_dof_arm):
            noise[:, i] = sin_scales[i] * np.sin((periods[i] * 2 * np.pi * t_steps) / n_steps)
        actions[:, :n_dof_arm] = actions[:, :n_dof_arm] + noise
        return actions

    def follow_path(self, result, refine_steps: int = 0):
        assert self.control_mode == "pd_joint_pos", "Only pd_joint_pos is supported for two finger gripper"
        n_dof_arm = result["position"].shape[1]
        n_step = result["position"].shape[0]
        actions = []
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            actions.append(np.hstack([qpos, self.gripper_state]))
        actions = np.array(actions)

        if self.slow_down:
            actions = self._slowdown_path(actions, n_intermediate_steps=5)
        if self.add_sinusoidal_noise:
            actions = self._add_sinusoidal_noise(actions, n_dof_arm)

        for idx, action in enumerate(actions):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    # def follow_path(self, result, refine_steps: int = 0):
    #     n_step = result["position"].shape[0]
    #     for i in range(n_step + refine_steps):
    #         qpos = result["position"][min(i, n_step - 1)]
    #         if self.control_mode == "pd_joint_pos_vel":
    #             qvel = result["velocity"][min(i, n_step - 1)]
    #             action = np.hstack([qpos, qvel, self.gripper_state])
    #         else:
    #             action = np.hstack([qpos, self.gripper_state])
    #         obs, reward, terminated, truncated, info = self.env.step(action)
    #         self.elapsed_steps += 1
    #         if self.print_env_info:
    #             print(
    #                 f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
    #             )
    #         if self.vis:
    #             self.base_env.render_human()
    #     return obs, reward, terminated, truncated, info

    def open_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, : len(self.planner.joint_vel_limits)].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


def build_two_finger_gripper_grasp_pose_visual(scene: ManiSkillScene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05

    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
