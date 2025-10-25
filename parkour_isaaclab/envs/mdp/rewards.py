from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

import cv2
import numpy as np 

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

class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_actions = torch.zeros(env.num_envs, 2,  asset.num_joints, dtype= torch.float ,device=self.device)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_actions[env_ids, 0,:] = 0.
        self.previous_actions[env_ids, 1,:] = 0.

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        ) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term('joint_pos').raw_actions
        return torch.norm(self.previous_actions[:, 1, :] - self.previous_actions[:,0,:], dim=1)
    
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
    parkour_event: ParkourEvent =  env.parkour_manager.get_term(parkour_name)
    terrain_names = parkour_event.env_per_terrain_name
    asset: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    rew[(terrain_names !='parkour_flat')[:,-1]] = 0.
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


class reward_dual_contact_at_high_speed(ManagerTermBase):
    """惩罚高速时指定配对的两只脚同时触地
    
    当机器人速度大于阈值时，惩罚指定配对的脚同时接地的情况。
    这鼓励机器人在高速运动时采用更动态的步态（如跑步、小跑）而非静态步态（如慢走）。
    """
    def __init__(self, cfg: RewardTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
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
        
        # 速度阈值（m/s），超过此值时启用惩罚
        self.speed_threshold = cfg.params.get("speed_threshold", 0.4)
        
        # 接地力阈值（N），超过此值认为脚在地面上
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        foot_pairs: list = [["LF_Knee_link", "RF_Knee_link"], ["LH_Knee_link", "RH_Knee_link"]],
        speed_threshold: float = 0.4,
        contact_force_threshold: float = 1.0,
    ) -> torch.Tensor:
        # 计算机器人的水平速度
        lin_vel = self.asset.data.root_lin_vel_b[:, :2]  # 取x和y方向速度
        speed = torch.norm(lin_vel, dim=-1)
        
        # 检测脚部是否在地面上
        net_contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids]
        in_contact = torch.norm(net_contact_forces, dim=-1) > self.contact_force_threshold
        
        # 只在速度大于阈值时才惩罚
        high_speed_mask = speed > self.speed_threshold
        
        # 计算每对脚同时接地的惩罚
        pair_penalties = []
        for left_idx, right_idx in self.foot_pairs:
            # 检查该配对的两只脚是否都在地面上
            both_contact = in_contact[:, left_idx] & in_contact[:, right_idx]
            pair_penalties.append(both_contact.float())
        
        # 计算总惩罚：任何一对脚同时接地都会产生惩罚
        # 可以选择求和（多对同时接地惩罚更重）或最大值
        total_penalty = torch.stack(pair_penalties).sum(dim=0)
        
        # 应用速度掩码：只在高速时惩罚
        penalty = total_penalty * high_speed_mask.float()
        
        return penalty


