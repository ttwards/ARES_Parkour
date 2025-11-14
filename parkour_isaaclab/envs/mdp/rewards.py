from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

# 在你的 rewards.py 或相应的reward manager文件中

import torch
from isaaclab.managers import SceneEntityCfg


def hurdle_collision_penalty(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    collision_threshold: float = 0.4,  # 距离目标0.4m内检测碰撞
    hurdle_height_margin: float = 0.05,  # 横杆底部5cm的安全裕度
    penalty_scale: float = -5.0,
    parkour_name: str = "base_parkour",
) -> torch.Tensor:
    """
    惩罚机器狗在接近跨栏目标时从上方通过（碰撞横杆）
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        collision_threshold: 触发检测的距离阈值（米）
        hurdle_height_margin: 横杆下方的安全裕度（米）
        penalty_scale: 惩罚系数（负数）
        
    Returns:
        惩罚值张量 shape: (num_envs,)
    """
    # 获取机器人
    robot: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    
    # 获取机器人base_link的世界坐标高度
    base_height_w = robot.data.root_pos_w[:, 2]
    
    # 获取机器人在地形局部坐标系中的位置
    robot_pos_local = robot.data.root_pos_w[:, :2] - parkour_event.env_origins[:, :2]
    
    # 计算到当前目标的距离
    dist_to_goal = torch.norm(robot_pos_local - parkour_event.cur_goals[:, :2], dim=1)
    
    # 只在接近目标时检测（距离 < collision_threshold）
    near_goal = dist_to_goal < collision_threshold
    
    # 获取当前目标的高度（横杆底部高度）
    # 注意：这里假设 cur_goals[:, 2] 存储的是目标高度
    goal_height = parkour_event.cur_goals[:, 2]
    
    # 检测是否发生碰撞：base_link高度超过横杆底部（考虑安全裕度）
    collision_detected = base_height_w > (goal_height + hurdle_height_margin)
    
    # 组合条件：接近目标 且 发生碰撞
    penalty_mask = near_goal & collision_detected
    
    # 返回惩罚（只有满足条件的环境才有惩罚）
    penalty = torch.where(penalty_mask, penalty_scale, 0.0)
    
    return penalty

def wall_clear_reward(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    parkour_name: str = "base_parkour",
    clearance_threshold: float = 0.45,
    height_margin: float = 0.0,
    saturation_margin: float = 0.2,
    reward_scale: float = 1.0,
) -> torch.Tensor:
    """根据跨越墙体的高度余量给奖励，超出可配置的裕度后奖励不再增加。"""
    robot: Articulation = env.scene[asset_cfg.name]
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)

    base_height_w = robot.data.root_pos_w[:, 2]
    robot_pos_local = robot.data.root_pos_w[:, :2] - parkour_event.env_origins[:, :2]
    dist_to_goal = torch.norm(robot_pos_local - parkour_event.cur_goals[:, :2], dim=1)
    near_goal = dist_to_goal < clearance_threshold

    goal_height = parkour_event.cur_goals[:, 2]
    height_diff = base_height_w - goal_height
    clearance = torch.clamp(height_diff - height_margin, min=0.0)
    denom = saturation_margin if saturation_margin > 1e-6 else 1e-6
    normalized_reward = torch.clamp(clearance / denom, max=1.0) * reward_scale

    return torch.where(near_goal, normalized_reward, torch.zeros_like(normalized_reward))

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
        rew = (self.parkour_event.terrain.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        ## This is for debugging to matching index and x_edge_mask
        # origin = self.x_edge_masks_tensor.detach().cpu().numpy().astype(np.uint8) * 255
        # cv2.imshow('origin',origin)
        # origin[feet_pos_x.detach().cpu().numpy(), feet_pos_y.detach().cpu().numpy()] -= 100
        # cv2.imshow('feet_edge',origin)
        # cv2.waitKey(1)
        return rew

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
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

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


def reward_body_height(
    env: ParkourManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "base_link",
) -> torch.Tensor:
    """基于高度误差的 L2 惩罚（平方误差）。"""
    parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 缓存 body_id 到环境中，避免每次重复查找
    cache_key = f"reward_target_height_body_id_{body_name}"
    if not hasattr(env, cache_key):
        body_ids = asset.find_bodies(body_name)
        if len(body_ids) > 0 and len(body_ids[0]) > 0:
            setattr(env, cache_key, body_ids[0][0])
        else:
            # 如果找不到指定的 body，回退到使用 root
            setattr(env, cache_key, None)
    
    body_id = getattr(env, cache_key)
    
    # 获取当前机器人和目标的高度
    if body_id is not None:
        # 使用指定的 body 高度
        current_robot_height = asset.data.body_pos_w[:, body_id, 2]
    else:
        # 回退到 root 高度
        current_robot_height = asset.data.root_pos_w[:, 2]
    
    current_target_height = parkour_event.cur_goals[:, 2]
    # 返回平方误差（正值），在配置中用负权重进行惩罚
    return torch.square(current_target_height - current_robot_height)


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Use the actual action dimension instead of total joint count
        action_dim = env.action_manager.get_term('joint_pos')._num_joints
        self.previous_actions = torch.zeros(env.num_envs, 2, action_dim, dtype=torch.float, device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_actions[env_ids, 0, :] = 0.
        self.previous_actions[env_ids, 1, :] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:, 0, :], dim=1)
    
class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # Use all joints for velocity tracking (including wheels)
        self.previous_joint_vel = torch.zeros(env.num_envs, 2, asset.num_joints, dtype=torch.float, device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_joint_vel[env_ids, 0, :] = 0.
        self.previous_joint_vel[env_ids, 1, :] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(torch.square((self.previous_joint_vel[:, 1, :] - self.previous_joint_vel[:, 0, :]) / self.dt), dim=1)


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
    return rew_move

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

def track_lin_vel_xy_exp(
    env: ParkourManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

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
