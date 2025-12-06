# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""
from __future__ import annotations
import torchvision
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.assets import Articulation
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent 
from collections.abc import Sequence
import numpy as np 
import cv2
if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
import re


class ExtremeParkourObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # 1. 获取基础组件
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        # self.depth_camera: RayCasterCamera = env.scene.sensors['depth_camera']
        self.parkour_event = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        
        # 2. 处理 Body Name
        target_body_name = cfg.params.get("body_name", "base")
        self.body_id = self.asset.find_bodies(target_body_name)[0]

        # 3. 解析关节索引 (Pos vs Vel)
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.history_length = cfg.params['history_length']
        
        pos_joint_names = cfg.params.get("pos_joints", [".*"])
        vel_joint_names = cfg.params.get("vel_joints", [".*"])
        
        self.pos_dof_ids, _ = self.asset.find_joints(pos_joint_names)
        self.vel_dof_ids, _ = self.asset.find_joints(vel_joint_names)
        self.pos_dof_ids = torch.tensor(self.pos_dof_ids, device=self.device)
        self.vel_dof_ids = torch.tensor(self.vel_dof_ids, device=self.device)
        
        self.num_pos_joints = len(self.pos_dof_ids)
        self.num_vel_joints = len(self.vel_dof_ids)

        # 4. 收集所有匹配的 Action Terms
        action_pattern = cfg.params.get("action_term_name", "joint_pos")
        self.action_terms = []  # 改用列表存储
        
        # 遍历所有 terms
        for name, term in env.action_manager._terms.items():
            if re.fullmatch(action_pattern, name):
                self.action_terms.append(term)
        
        if not self.action_terms:
            available_terms = list(env.action_manager._terms.keys())
            raise ValueError(f"ExtremeParkourObservations: 找不到匹配 '{action_pattern}' 的 Action Term! 现有的 Terms: {available_terms}")
        
        # --- 累加所有匹配 term 的 action_dim ---
        self.num_actions = sum(term.action_dim for term in self.action_terms)

        # 5. 计算总观测维度
        base_obs_dim = 13
        contact_dim = 4
        
        self.obs_dim = (base_obs_dim + 
                        self.num_pos_joints + 
                        self.num_vel_joints + 
                        self.num_actions +
                        contact_dim)

        # 6. 初始化 Buffer
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, self.obs_dim, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, self.ray_sensor.num_rays, device=self.device)
        self.env = env
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0. 

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        parkour_name: str,
        history_length: int,
        body_name: str = "base",
        pos_joints: list = None,
        vel_joints: list = None,
        action_term_name: str = None, 
        ) -> torch.Tensor:
        
        terrain_names = self.parkour_event.env_per_terrain_name
        env_idx_tensor = torch.tensor((terrain_names != 'parkour_flat')).to(dtype = torch.bool, device=self.device)
        invert_env_idx_tensor = torch.tensor((terrain_names == 'parkour_flat')).to(dtype = torch.bool, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
            
        commands = env.command_manager.get_command('base_velocity')
        
        all_joint_pos = self.asset.data.joint_pos - self.asset.data.default_joint_pos
        all_joint_vel = self.asset.data.joint_vel
        
        filtered_joint_pos = all_joint_pos[:, self.pos_dof_ids] 
        filtered_joint_vel = all_joint_vel[:, self.vel_dof_ids] * 0.05 
        
        # --- 关键修改：拼接所有 Action Terms 的历史 ---
        # 遍历列表，取出每个 term 的 buffer 并 cat 起来
        action_history_list = [term.action_history_buf[:, -1] for term in self.action_terms]
        combined_action_history = torch.cat(action_history_list, dim=-1)
        
        obs_buf = torch.cat((
                            self.asset.data.root_ang_vel_b * 0.25,
                            imu_obs,
                            0 * self.delta_yaw[:, None],
                            self.delta_yaw[:, None],
                            self.delta_next_yaw[:, None],
                            0 * commands[:, 0:2],
                            commands[:, 0:1],
                            env_idx_tensor,
                            invert_env_idx_tensor,
                            filtered_joint_pos,
                            filtered_joint_vel,
                            combined_action_history,
                            self._get_contact_fill(),
                            ), dim=-1)

        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()

        observations = torch.cat([obs_buf,
                                  self.measured_heights,
                                  priv_explicit,
                                  priv_latent,
                                  self._obs_history_buffer.view(self.num_envs, -1)
                                  ], dim=-1)

        obs_buf[:, 6:8] = 0

        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return observations
    
    # ... 下面的辅助函数保持不变 (确保包含了 Epsilon 修复) ...
    def _get_contact_fill(self):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] 
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids]
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)

    def _get_priv_explicit(self):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)
    
    def _get_priv_latent(self):
        epsilon = 1e-6
        joint_stiffness = self.asset.data.joint_stiffness.to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness.to(self.device)
        joint_damping = self.asset.data.joint_damping.to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping.to(self.device)

        stiffness_ratio = (joint_stiffness / (default_joint_stiffness + epsilon)) - 1
        damping_ratio = (joint_damping / (default_joint_damping + epsilon)) - 1
        
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]

        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            stiffness_ratio, 
            damping_ratio
        ), dim=-1).to(self.device)
        
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -2, 2).to(self.device)


class image_features(ManagerTermBase):
    
    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera_sensor: RayCasterCamera = env.scene[cfg.params["sensor_cfg"].name]
        self.clipping_range = self.camera_sensor.cfg.max_distance
        resized = cfg.params["resize"]
        self.buffer_len = cfg.params['buffer_len']
        self.debug_vis = cfg.params['debug_vis']
        self.resize_transform = torchvision.transforms.Resize(
                                    (resized[0], resized[1]), 
                                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(env.device)
        self.depth_buffer = torch.zeros(self.num_envs,  
                                        self.buffer_len, 
                                        resized[0], 
                                        resized[1]).to(self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)[env_ids]
        for depth_image, env_id in zip(depth_images, env_ids):
            processed_image = self._process_depth_image(depth_image)
            self.depth_buffer[env_id] = torch.stack([processed_image]* 2, dim=0)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,        
        sensor_cfg: SceneEntityCfg,
        resize: tuple(int,int), 
        buffer_len: int,
        debug_vis:bool
        ):
        if env.common_step_counter % 5 == 0:
            depth_images = self.camera_sensor.data.output["distance_to_camera"].squeeze(-1)
            for env_id, depth_image in enumerate(depth_images):
                processed_image = self._process_depth_image(depth_image)
                self.depth_buffer[env_id] = torch.cat([self.depth_buffer[env_id, 1:], 
                                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
        if self.debug_vis:
            depth_images_np = self.depth_buffer[:, -2].detach().cpu().numpy()
            depth_images_norm = []
            for img in depth_images_np:
                depth_images_norm.append(img)
            rows = []
            ncols = 4
            for i in range(0, len(depth_images_norm), ncols):
                row = np.hstack(depth_images_norm[i:i+ncols])  
                rows.append(row)

            grid_img = np.vstack(rows)   
            cv2.imshow("depth_images_grid", grid_img)
            cv2.waitKey(1)
        return self.depth_buffer[:, -2].to(env.device)

    def _process_depth_image(self, depth_image):
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self._normalize_depth_image(depth_image)
        return depth_image

    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]

    def _normalize_depth_image(self, depth_image):
        depth_image = depth_image  # make similiar to scandot 
        depth_image = (depth_image) / (self.clipping_range)  - 0.5
        return depth_image
    
class obervation_delta_yaw_ok(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)

    def __call__(
        self,
        env: ParkourManagerBasedRLEnv,    
        parkour_name: str,
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        if env.common_step_counter % 5 == 0:
            parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
            asset: Articulation = env.scene[asset_cfg.name]
            _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
            self.delta_yaw = parkour_event.target_yaw - wrap_to_pi(yaw)
        return self.delta_yaw < threshold


# class observation_target_height(ManagerTermBase):
#     """观测目标高度和下一个目标高度相对于机器人当前位置的差异"""

#     def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self.target_height_rel = torch.zeros(self.num_envs, device=self.device)
#         self.next_target_height_rel = torch.zeros(self.num_envs, device=self.device)
        
#         # 如果需要特定body的高度，可以获取body_id
#         body_name = cfg.params.get("body_name", None)
#         if body_name:
#             asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#             self.body_id = asset.find_bodies(body_name)[0][0]
#         else:
#             self.body_id = None

#     def __call__(
#         self,
#         env: ParkourManagerBasedRLEnv,
#         parkour_name: str,
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         body_name: str = None,  # 可选参数，指定特定的body
#     ):
#         if env.common_step_counter % 5 == 0:
#             parkour_event: ParkourEvent = env.parkour_manager.get_term(parkour_name)
#             asset: Articulation = env.scene[asset_cfg.name]

#             # 根据是否指定body_id选择高度来源
#             if self.body_id is not None:
#                 current_robot_height = asset.data.body_pos_w[:, self.body_id, 2]
#             else:
#                 current_robot_height = asset.data.root_pos_w[:, 2]

#             current_target_height = parkour_event.cur_goals[:, 2]
#             self.target_height_rel = current_target_height - current_robot_height

#             next_target_height = parkour_event.next_goals[:, 2]
#             self.next_target_height_rel = next_target_height - current_robot_height

#         return torch.stack([self.target_height_rel, self.next_target_height_rel], dim=1)
