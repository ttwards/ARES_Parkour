from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_conjugate
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence
import isaaclab.utils.math as math_utils
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

# 在你的 rewards.py 或相应的reward manager文件中

import torch
from isaaclab.managers import SceneEntityCfg


class reward_feet_edge(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.parkour_event: ParkourEvent =  env.parkour_manager.get_term(cfg.params["parkour_name"])
        # Get body_name from config, default to 'base_link' if not specified
        body_name = cfg.params.get("body_name", "base_link")
        self.body_id = self.contact_sensor.find_bodies(body_name)[0]
        self.horizontal_scale = env.scene.terrain.cfg.terrain_generator.horizontal_scale
        size_x, size_y = env.scene.terrain.cfg.terrain_generator.size
        self.rows_offset = (size_x * env.scene.terrain.cfg.terrain_generator.num_rows/2)
        self.cols_offset = (size_y * env.scene.terrain.cfg.terrain_generator.num_cols/2)
        total_x_edge_maskes = torch.from_numpy(self.parkour_event.terrain.terrain_generator_class.x_edge_maskes).to(device = self.device)
        self.x_edge_masks_tensor = total_x_edge_maskes.permute(0, 2, 1, 3).reshape(
            env.scene.terrain.terrain_generator_class.total_width_pixels, env.scene.terrain.terrain_generator_class.total_length_pixels
        )

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        body_name: str = "base_link",
        ) -> torch.Tensor:
        feet_pos_x = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,0] + self.rows_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_y = ((self.asset.data.body_state_w[:, self.asset_cfg.body_ids ,1] + self.cols_offset)
                      /self.horizontal_scale).round().long() 
        feet_pos_x = torch.clip(feet_pos_x, 0, self.x_edge_masks_tensor.shape[0]-1)
        feet_pos_y = torch.clip(feet_pos_y, 0, self.x_edge_masks_tensor.shape[1]-1)
        feet_at_edge = self.x_edge_masks_tensor[feet_pos_x, feet_pos_y]
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        contact = torch.norm(contact_forces, dim=-1) > 2.
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        self.feet_at_edge = contact_filt & feet_at_edge
        rew = (self.parkour_event.terrain.terrain_levels > 2) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew


def joint_power(env: ParkourManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def feet_air_time_variance_penalty(env: ParkourManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_torques(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def reward_dof_error(    
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_hip_pos(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] \
                                    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)

def reward_ang_vel_xy(
    env: ParkourManagerBasedRLEnv,        
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:,:2]), dim=1)


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.action_term_name: str = cfg.params.get("action_term_name", "joint_pos")
        self.action_term = env.action_manager.get_term(self.action_term_name)
        action_dim = getattr(self.action_term, "_num_joints", None)
        if action_dim is None:
            # 这里假设创建 reward 时已经至少走过一次 step / reset
            action_dim = int(self.action_term.raw_actions.shape[-1])
        self.previous_actions = torch.zeros(
            env.num_envs, 2, action_dim, dtype=torch.float, device=self.device
        )
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_actions[env_ids, 0, :] = 0.0
        self.previous_actions[env_ids, 1, :] = 0.0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        # 使用与初始化一致的 action term
        self.previous_actions[:, 1, :] = self.action_term.raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:, 0, :], dim=1)


class reward_action_l2(ManagerTermBase):
    """惩罚 action 的 L2 范数(action 大小)。
    
    该惩罚项鼓励机器人使用较小的 action 值，有助于：
    - 减少能耗
    - 产生更平滑的动作
    - 防止极端的关节位置指令
    """
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.action_term_name: str = cfg.params.get("action_term_name", "joint_pos")
        self.action_term = env.action_manager.get_term(self.action_term_name)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        action_term_name: str = "joint_pos",
        ) -> torch.Tensor:
        # 返回 raw_actions 的平方和
        return torch.sum(torch.square(self.action_term.raw_actions), dim=1)

    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_joint_vel[env_ids, 0,:] = 0.
        self.previous_joint_vel[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:,0,:]) / self.dt), dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ParkourManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "mirror_joints_cache") or env.mirror_joints_cache is None:
        env.mirror_joints_cache = [
            asset.find_joints(joint_name) for joint_pair in mirror_joints for joint_name in joint_pair
        ]
    # compute out of limits constraints
    diff1 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[0][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[1][0]]
        ),
        dim=-1,
    )
    diff2 = torch.sum(
        torch.square(
            asset.data.joint_pos[:, env.mirror_joints_cache[2][0]]
            - asset.data.joint_pos[:, env.mirror_joints_cache[3][0]]
        ),
        dim=-1,
    )
    reward = 0.5 * (diff1 + diff2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_lin_vel_z(
    env: ParkourManagerBasedRLEnv,        
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(asset.data.root_lin_vel_b[:, 2])
    rew[(terrain_names !='parkour_flat')[:,-1]] *= 0.5
    return rew

def reward_orientation(
    env: ParkourManagerBasedRLEnv,   
    parkour_name:str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor: 
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return rew

def reward_feet_stumble(
    env: ParkourManagerBasedRLEnv,        
    sensor_cfg: SceneEntityCfg ,
    ) -> torch.Tensor: 
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:,0,sensor_cfg.body_ids]
    rew = torch.any(torch.norm(net_contact_forces[:, :, :2], dim=2) >\
            4 *torch.abs(net_contact_forces[:, :, 2]), dim=1)
    return rew.float()

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_l2(
    env: ParkourManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty only when below target height
    # If height is below target (negative diff), penalize; if above, no penalty
    height_diff = asset.data.root_pos_w[:, 2] - adjusted_target_height
    reward = torch.square(torch.clamp(height_diff, max=0.0))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def reward_tracking_goal_vel(
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    target_pos_rel = parkour_event.target_pos_rel
    target_vel = target_pos_rel / (torch.norm(target_pos_rel, dim=-1, keepdim=True) + 1e-5)
    cur_vel = asset.data.root_vel_w[:, :2]
    proj_vel = torch.sum(target_vel * cur_vel, dim=-1)
    command_vel = env.command_manager.get_command('base_velocity')[:, 0]
    rew_move = torch.minimum(proj_vel, command_vel) / (command_vel + 1e-5)
    return rew_move * (parkour_event.terrain.terrain_levels / 3.0 + 1)

class reward_cumulative_speed_error(ManagerTermBase):
    """
    只有当速度误差连续超过阈值持续一段时间后才累加；低于阈值后快速归零。
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.accumulated_error = torch.zeros(self.num_envs, device=self.device)
        # 连续超阈计数（按步）
        self._over_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 如果你更喜欢用“秒”做时间阈值，会尝试从环境里自动拿 dt
        self._env_dt = None
        for attr in ("dt", "step_dt", "sim_dt"):
            if hasattr(env, attr):
                val = getattr(env, attr)
                # 兼容张量/标量
                try:
                    self._env_dt = float(val)
                except Exception:
                    pass
                break

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.accumulated_error[env_ids] = 0.0
        self._over_steps[env_ids] = 0

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        # ------- 新增/调整参数 -------
        speed_threshold: float = 0.2,     # 速度误差阈值（m/s）
        time_threshold_s: float | None = 0.25,   # 连续超阈需维持的时间（秒）；若为 None 则使用 steps_threshold
        steps_threshold: int | None = None,      # 连续超阈需维持的“步数”；若 time_threshold_s 与 dt 可用则忽略
        fast_clear_decay: float = 0.0,    # 低于阈值时的快速归零衰减系数：0=瞬时清零，0.1=每步保留10%
        normal_decay: float = 1.0,        # 正常累加时的衰减（<=1）；用于“缓慢遗忘”的效果
        accumulation_scale: float = 1.0,
        max_penalty: float = 5.0,
    ) -> torch.Tensor:
        """
        fast_clear_decay ∈ [0,1]，建议 0~0.2；normal_decay ∈ [0,1]。
        """
        asset: Articulation = env.scene[asset_cfg.name]
        desired_vel = env.command_manager.get_command(command_name)[:, :2]
        current_vel = asset.data.root_lin_vel_b[:, :2]

        # 标量误差（m/s）
        error = torch.norm(desired_vel - current_vel, dim=-1)

        # 判定用阈值掩码
        over_mask = error > speed_threshold

        # 连续超阈计数（按步）
        self._over_steps = torch.where(
            over_mask,
            self._over_steps + 1,
            torch.zeros_like(self._over_steps)
        )

        # 计算需要的连续步数阈值
        if steps_threshold is not None:
            need_steps = max(int(steps_threshold), 1)
        else:
            # 用秒 -> 步（需要环境 dt）
            if time_threshold_s is None:
                # 兜底：如果没给任何时间阈值，就当作1步
                need_steps = 1
            else:
                if self._env_dt is None or self._env_dt <= 0:
                    # 没法从环境拿到 dt，就退化为1步
                    need_steps = 1
                else:
                    need_steps = max(int(round(time_threshold_s / self._env_dt)), 1)

        # 是否“已超过阈值并持续到达阈时”
        active_mask = self._over_steps >= need_steps

        # 计算本步要累加的量（仅在 active 状态）
        inc = (error * accumulation_scale)

        # 正常阶段使用 normal_decay 进行“缓慢遗忘 + 累加”
        normal_decay = float(min(max(normal_decay, 0.0), 1.0))
        new_accum = self.accumulated_error * normal_decay + inc

        # 低于阈值时快速归零：acc *= fast_clear_decay
        fast_clear_decay = float(min(max(fast_clear_decay, 0.0), 1.0))
        cleared = self.accumulated_error * fast_clear_decay

        # 组合更新：active 用 new_accum；否则用 cleared
        self.accumulated_error = torch.where(active_mask, new_accum, cleared)

        # 裁剪上限
        if max_penalty > 0:
            self.accumulated_error = torch.clamp(self.accumulated_error, max=max_penalty)

        return self.accumulated_error

def reward_tracking_yaw(     
    env: ParkourManagerBasedRLEnv, 
    parkour_name : str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
                    1 - 2*(q[:,2]**2 + q[:,3]**2))
    return torch.exp(-torch.abs((parkour_event.target_yaw - yaw)))

class reward_delta_torques(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_torque = torch.zeros(env.num_envs, 2,  self.asset.num_joints, dtype= torch.float ,device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_torque[env_ids, 0,:] = 0.
        self.previous_torque[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_torque[:, 0, :] = self.previous_torque[:, 1, :]
        self.previous_torque[:, 1, :] = self.asset.data.applied_torque
        return torch.sum(torch.square((self.previous_torque[:, 1, :] - self.previous_torque[:,0,:])), dim=1)

def reward_collision(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return torch.sum(1.*(torch.norm(net_contact_forces, dim=-1) > 0.1), dim=1)


class reward_symmetric_contact_time(ManagerTermBase):
    """奖励左右脚对称的接地时间，鼓励协调步态
    
    在固定时间窗口内统计左右脚的总接地时间，奖励差异小的情况
    """
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        
        # 获取脚的配对名称，并转换为索引
        foot_pairs_names = cfg.params.get("foot_pairs", [["LF_Knee_link", "RF_Knee_link"], ["LH_Knee_link", "RH_Knee_link"]])
        
        # 获取所有body名称列表
        body_names = self.contact_sensor.body_names
        body_list = [body_names[idx] for idx in self.sensor_cfg.body_ids]
        
        # 将名称对转换为索引对
        self.foot_pairs = []
        for left_name, right_name in foot_pairs_names:
            try:
                left_idx = body_list.index(left_name)
                right_idx = body_list.index(right_name)
                self.foot_pairs.append([left_idx, right_idx])
            except ValueError:
                raise ValueError(f"Foot name not found in sensor bodies. Available: {body_list}, Looking for: {left_name}, {right_name}")
        
        # 时间窗口大小（控制步数）
        # 默认250步，如果控制步长为0.02s，则对应5秒
        self.window_size = cfg.params.get("window_size", 250)
        
        # 使用循环缓冲区记录历史接地状态
        num_feet = len(self.sensor_cfg.body_ids)
        self.contact_history = torch.zeros(
            env.num_envs, num_feet, self.window_size, 
            device=self.device, dtype=torch.bool
        )
        self.current_idx = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.contact_history[env_ids] = False

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        foot_pairs: list = [["LF_Knee_link", "RF_Knee_link"], ["LH_Knee_link", "RH_Knee_link"]],
        window_size: int = 250,
    ) -> torch.Tensor:
        # 检测脚部是否在地面上
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        in_contact = torch.norm(net_contact_forces, dim=-1) > 1.0

        # 更新循环缓冲区
        self.contact_history[:, :, self.current_idx] = in_contact
        self.current_idx = (self.current_idx + 1) % self.window_size

        # 统计时间窗口内的总接地时间
        contact_time = self.contact_history.sum(dim=2).float()  # [num_envs, num_feet]

        # 计算每对脚的对称性奖励
        pair_rewards = []
        for left_idx, right_idx in self.foot_pairs:
            left_time = contact_time[:, left_idx]
            right_time = contact_time[:, right_idx]
            # 计算左右脚接地时间差异（占总窗口的比例）
            diff = torch.abs(left_time - right_time) / self.window_size
            # 奖励差异小的情况，使用指数衰减
            # diff在[0,1]范围，当差异为10%时奖励约为0.9，差异为20%时奖励约为0.82
            pair_reward = torch.exp(-diff * 5.0)
            pair_rewards.append(pair_reward)

        # 总奖励是所有配对的平均
        total_reward = torch.stack(pair_rewards).mean(dim=0)

        return total_reward


class reward_foot_no_contact_time(ManagerTermBase):
    """惩罚某个脚长时间不落地
    
    跟踪每个脚的连续未接地时间，超过阈值则给予惩罚。
    这可以防止机器人出现三足行走或某条腿长时间悬空的不自然步态。
    """
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        
        # 接地力阈值（N），超过此值认为脚在地面上
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)
        
        # 最大允许的连续未接地时间（控制步数）
        # 默认50步，如果控制步长为0.02s，则对应1秒
        self.max_no_contact_steps = cfg.params.get("max_no_contact_steps", 50)
        
        # 记录每个脚的连续未接地计数
        num_feet = len(self.sensor_cfg.body_ids)
        self.no_contact_count = torch.zeros(
            env.num_envs, num_feet,
            device=self.device, dtype=torch.float32
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.no_contact_count[env_ids] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float = 1.0,
        max_no_contact_steps: int = 50,
    ) -> torch.Tensor:
        # 检测脚部是否在地面上
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        in_contact = torch.norm(net_contact_forces, dim=-1) > self.contact_force_threshold
        
        # 更新连续未接地计数
        # 如果接地，计数归零；如果未接地，计数加1
        self.no_contact_count = torch.where(
            in_contact,
            torch.zeros_like(self.no_contact_count),
            self.no_contact_count + 1
        )
        
        # 计算惩罚：当连续未接地时间超过阈值时开始惩罚
        # 超出部分越多，惩罚越大
        excess_time = torch.clamp(self.no_contact_count - self.max_no_contact_steps, min=0.0)
        
        # 对每个环境，取所有脚中最严重的惩罚（最大值）
        # 也可以改为求和，这样会对多个脚同时悬空惩罚更重
        penalty = torch.max(excess_time, dim=-1)[0]
        
        # 归一化惩罚值，避免过大
        # 当超出50步时，惩罚值为1；超出100步时，惩罚值约为1.6
        penalty = torch.tanh(penalty / self.max_no_contact_steps)
        
        return penalty


class reward_link_below_ground(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 1. 初始化 Asset (机器人)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]
        
        # 2. 获取 Parkour Event 以便后续访问 terrain_levels (关键步骤)
        # 确保 yaml config 中传入了 "parkour_name"
        parkour_name = cfg.params.get("parkour_name", "base_parkour")
        self.parkour_event = env.parkour_manager.get_term(parkour_name)
        
        # 3. 获取其他参数
        self.threshold = cfg.params.get("threshold", -0.1)

    def __call__(
        self, 
        env: ParkourManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = -0.1,
        parkour_name: str = "base_parkour",
    ) -> torch.Tensor:
        # 获取指定 body 的高度 (Z轴)
        # 使用 body_state_w (N, num_bodies, 13) -> 取位置 z
        link_heights = self.asset.data.body_state_w[:, self.asset_cfg.body_ids, 2]
        
        # 计算原始惩罚/Reward (低于阈值的部分)
        # 逻辑：当高度 < threshold 时，diff 为正数，产生非零值
        diff = self.threshold - link_heights
        raw_rew = torch.sum(torch.clamp(diff, min=0.0), dim=-1)
        
        # 4. 应用 Level 过滤 (关键逻辑)
        # 只有 terrain_levels > 2 时才给予 reward，否则乘以 0
        rew = (self.parkour_event.terrain.terrain_levels > 0) * raw_rew
        
        return rew


class reward_link_below_goal_height(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Asset 配置
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]

        # Parkour 事件（用于获取 current goal）
        parkour_name = cfg.params.get("parkour_name", "base_parkour")
        self.parkour_event = env.parkour_manager.get_term(parkour_name)

        # 在 current_goal 高度下方留出的容差
        self.goal_margin = cfg.params.get("goal_margin", 0.0)
        self.distance_threshold = cfg.params.get("distance_threshold", 0.5)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        parkour_name: str = "base_parkour",
        goal_margin: float | None = None,
        distance_threshold: float | None = None,
    ) -> torch.Tensor:
        # 当前每个 link 的世界位置
        link_pos = self.asset.data.body_state_w[:, self.asset_cfg.body_ids, :3]
        link_heights = link_pos[..., 2]

        # 当前关卡目标高度（N, 1），让低于 current_goal 的部分产生惩罚
        margin = self.goal_margin if goal_margin is None else goal_margin
        
        # 获取目标位置 (N, 1, 3)
        # cur_goals 是相对坐标 (XY相对, Z绝对)，需要加上 env_origins 转换到世界坐标
        env_origins = env.scene.terrain.env_origins
        cur_goals_w = self.parkour_event.cur_goals.clone()
        cur_goals_w[:, :2] += env_origins[:, :2]
        cur_goals_w = cur_goals_w.unsqueeze(1)
        
        goal_height = cur_goals_w[..., 2] - margin

        # 计算 link 到 goal 的水平距离
        dist_threshold = self.distance_threshold if distance_threshold is None else distance_threshold
        dist_xy = torch.norm(link_pos[..., :2] - cur_goals_w[..., :2], dim=-1)
        
        # 只有距离小于阈值时才计算高度惩罚
        in_range_mask = dist_xy < dist_threshold

        diff = goal_height - link_heights
        # clamp(diff, min=0) 只有当 link 低于 goal 时才有值，且只在范围内生效
        raw_penalty = torch.sum(torch.clamp(diff, min=0.0) * in_range_mask.float(), dim=-1)
        raw_penalty = torch.clamp(raw_penalty, max=1.0)

        # 同样只在关卡激活时生效
        rew = (self.parkour_event.terrain.terrain_levels > 0) * raw_penalty
        return rew


def reward_terrain_level(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str = "base_parkour",
) -> torch.Tensor:
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    return parkour_event.terrain.terrain_levels.float() / parkour_event.terrain.max_terrain_level


class reward_foot_clearance(ManagerTermBase):
    """奖励足端在无接触且离地高度超过阈值时抬腿。
    
    使用4个独立的 RayCaster sensor 从每个足端向下发射光线测量离地高度。
    
    奖励逻辑：
    - 当足端无接触（no contact）且离地高度 > threshold 时给予正奖励
    - 使用 tanh 函数平滑奖励值，避免过大奖励
    
    需要在 scene 中配置 foot_scanner_LF/RF/LR/RR 四个 RayCaster sensor。
    """
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 接触力传感器
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["contact_sensor_cfg"].name]
        self.contact_sensor_cfg = cfg.params["contact_sensor_cfg"]
        
        # 4个足端高度 RayCaster 传感器
        self.foot_scanners = [
            env.scene.sensors["foot_scanner_LF"],
            env.scene.sensors["foot_scanner_RF"],
            env.scene.sensors["foot_scanner_LR"],
            env.scene.sensors["foot_scanner_RR"],
        ]
        
        # 参数
        self.height_threshold = cfg.params.get("height_threshold", 0.01)
        self.tanh_scale = cfg.params.get("tanh_scale", 40.0)
        self.contact_threshold = cfg.params.get("contact_threshold", 1.0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        contact_sensor_cfg: SceneEntityCfg,
        height_threshold: float = 0.01,
        tanh_scale: float = 40.0,
        contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        """计算足端离地奖励
        
        Args:
            height_threshold: 离地高度阈值（米），超过此值开始奖励
            tanh_scale: tanh缩放因子，控制饱和速度
            contact_threshold: 接触力阈值（N），低于此值认为无接触
        
        Returns:
            每个环境的奖励值
        """
        # 1. 从4个 RayCaster 获取足端离地高度
        clearances = []
        for scanner in self.foot_scanners:
            # pos_w: (N, 3), ray_hits_w: (N, 1, 3)
            foot_z = scanner.data.pos_w[:, 2]
            hit_z = scanner.data.ray_hits_w[:, 0, 2]
            clearances.append(foot_z - hit_z)
        
        # 堆叠成 (N, 4)
        clearance = torch.stack(clearances, dim=1)
        
        # 2. 检测足端是否接触地面
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.contact_sensor_cfg.body_ids]
        contact_force_mag = torch.norm(net_contact_forces, dim=-1)  # (N, 4)
        no_contact = contact_force_mag < self.contact_threshold  # (N, 4)
        
        # 3. 计算奖励：no contact 且 height > threshold
        height_above_threshold = clearance - self.height_threshold
        raw_reward = torch.tanh(height_above_threshold * self.tanh_scale)
        raw_reward = torch.clamp(raw_reward, min=0.0)
        
        # 只有在无接触时才给奖励
        masked_reward = raw_reward * no_contact.float()
        
        # 4. 对所有足端求和
        total_reward = torch.sum(masked_reward, dim=-1)
        
        command_vel = env.command_manager.get_command('base_velocity')[:, 0]

        return total_reward * (command_vel > 0.4)


def penalty_both_front_feet_airborne(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    front_feet_names: list[str] = ["LF_Foot_link", "RF_Foot_link"],
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    惩罚前两只脚同时离地的情况。
    
    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        front_feet_names: 前脚的名称列表，默认为左前脚和右前脚
        contact_force_threshold: 接触力阈值（N），低于此值认为脚离地
    
    Returns:
        惩罚值张量 shape: (num_envs,)，当两只前脚同时离地时返回1.0，否则返回0.0
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取前脚的body_ids
    front_feet_indices = contact_sensor.find_bodies(front_feet_names)[0]
    
    # 获取接触力 (N, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, front_feet_indices]
    
    # 计算接触力大小 (N, num_front_feet)
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)
    
    # 判断是否离地：接触力 < 阈值
    is_airborne = contact_force_mag < contact_force_threshold  # (N, 2)
    
    # 判断两只前脚是否同时离地
    both_airborne = torch.all(is_airborne, dim=-1)  # (N,)
    
    # 返回惩罚：同时离地返回1.0，否则返回0.0
    return both_airborne.float()


def penalty_both_rear_feet_airborne(
    env: ParkourManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    rear_feet_names: list[str] = ["LR_Foot_link", "RR_Foot_link"],
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    惩罚后两只脚同时离地的情况。
    
    Args:
        env: 环境实例
        sensor_cfg: 接触传感器配置
        rear_feet_names: 后脚的名称列表，默认为左后脚和右后脚
        contact_force_threshold: 接触力阈值（N），低于此值认为脚离地
    
    Returns:
        惩罚值张量 shape: (num_envs,)，当两只后脚同时离地时返回1.0，否则返回0.0
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取后脚的body_ids
    rear_feet_indices = contact_sensor.find_bodies(rear_feet_names)[0]
    
    # 获取接触力 (N, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, rear_feet_indices]
    
    # 计算接触力大小 (N, num_rear_feet)
    contact_force_mag = torch.norm(net_contact_forces, dim=-1)
    
    # 判断是否离地：接触力 < 阈值
    is_airborne = contact_force_mag < contact_force_threshold  # (N, 2)
    
    # 判断两只后脚是否同时离地
    both_airborne = torch.all(is_airborne, dim=-1)  # (N,)
    
    # 返回惩罚：同时离地返回1.0，否则返回0.0
    return both_airborne.float()


class reward_feet_position_deviation(ManagerTermBase):
    """惩罚脚部位置偏离默认位置（仅计算XY方向）
    
    该奖励函数计算脚部相对于base_link的XY平面位置偏差，鼓励机器人保持自然的站姿。
    通过惩罚脚部位置偏离初始默认位置，可以：
    - 防止脚部过度伸展或收缩
    - 保持稳定的站姿
    - 避免不自然的腿部配置
    
    计算方式：
    1. 获取每个脚在body坐标系下相对于base_link的位置
    2. 与默认位置（初始重置时的位置）进行比较
    3. 只计算XY平面的L2距离并进行惩罚（忽略Z方向）
    """

    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]

        # 获取脚部的body_ids
        feet_pattern = cfg.params.get("feet_pattern", ".*_Foot_link")
        self.feet_body_ids = self.asset.find_bodies(feet_pattern)[0]

        # 存储默认的脚部相对位置（在body坐标系下）
        # 这将在第一次reset时初始化
        self.default_feet_pos_b = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """在reset时记录默认的脚部位置"""
        if env_ids is None:
            env_ids = slice(None)

        # 如果是第一次初始化，记录所有环境的默认脚部位置
        if self.default_feet_pos_b is None:
            # 获取base_link的世界位置和旋转
            base_pos_w = self.asset.data.root_pos_w
            base_quat_w = self.asset.data.root_quat_w

            # 获取脚部的世界位置
            feet_pos_w = self.asset.data.body_pos_w[:, self.feet_body_ids, :]

            # 转换到body坐标系
            feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
            feet_pos_b = torch.zeros_like(feet_pos_rel)
            for i in range(len(self.feet_body_ids)):
                feet_pos_b[:, i, :] = quat_apply(
                    math_utils.quat_conjugate(base_quat_w),
                    feet_pos_rel[:, i, :]
                )

            self.default_feet_pos_b = feet_pos_b.clone()

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        feet_pattern: str = ".*_Foot_link",
    ) -> torch.Tensor:
        """计算脚部位置偏差惩罚（仅XY方向）

        Returns:
            惩罚值张量 shape: (num_envs,)，值越大表示偏差越大
        """
        # 确保已经初始化默认位置
        if self.default_feet_pos_b is None:
            return torch.zeros(env.num_envs, device=self.device)

        # 获取当前base_link的位置和旋转
        base_pos_w = self.asset.data.root_pos_w
        base_quat_w = self.asset.data.root_quat_w

        # 获取当前脚部的世界位置
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_body_ids, :]

        # 转换到body坐标系
        feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
        current_feet_pos_b = torch.zeros_like(feet_pos_rel)
        for i in range(len(self.feet_body_ids)):
            current_feet_pos_b[:, i, :] = quat_apply(
                math_utils.quat_conjugate(base_quat_w),
                feet_pos_rel[:, i, :]
            )

        # 只计算XY方向的偏差（忽略Z方向）
        deviation_xy = torch.norm(
            current_feet_pos_b[:, :, :2] - self.default_feet_pos_b[:, :, :2],
            dim=-1
        )  # (N, num_feet)

        # 对所有脚的偏差求和
        total_deviation = torch.sum(deviation_xy, dim=-1)  # (N,)

        return total_deviation


class joint_power_variance(ManagerTermBase):
    """计算关节间功率方差：先计算每个关节在2秒窗口内的平均功率，然后计算关节之间的方差"""
    
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg = cfg.params["asset_cfg"]
        
        # 时间窗口参数
        self.window_time = cfg.params.get("window_time", 2.0)  # 默认2秒
        self.dt = env.cfg.decimation * env.cfg.sim.dt  # 控制步长
        self.window_size = max(int(self.window_time / self.dt), 1)  # 转换为步数
        
        # 获取关节数量
        num_joints = len(self.asset_cfg.joint_ids)
        
        # 创建循环缓冲区存储每个关节在每个时刻的功率 (N, window_size, num_joints)
        self.power_history = torch.zeros(
            env.num_envs, self.window_size, num_joints,
            device=self.device, dtype=torch.float32
        )
        self.current_idx = 0
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.power_history[env_ids] = 0.0
    
    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        window_time: float = 2.0,
    ) -> torch.Tensor:
        """计算关节间的功率方差（基于窗口内平均功率）
        
        Args:
            env: 环境实例
            asset_cfg: 资产配置
            window_time: 时间窗口大小（秒）
            
        Returns:
            每个环境的关节间功率方差
        """
        # 计算当前时刻每个关节的功率 (N, num_joints)
        current_power = torch.abs(
            self.asset.data.joint_vel[:, self.asset_cfg.joint_ids] * 
            self.asset.data.applied_torque[:, self.asset_cfg.joint_ids]
        )
        
        # 更新循环缓冲区
        self.power_history[:, self.current_idx, :] = current_power
        self.current_idx = (self.current_idx + 1) % self.window_size
        
        # 计算每个关节在时间窗口内的平均功率 (N, num_joints)
        avg_power_per_joint = torch.mean(self.power_history, dim=1)
        
        # 计算关节之间的方差 (N,)
        variance_across_joints = torch.var(avg_power_per_joint, dim=1)
        
        return variance_across_joints
