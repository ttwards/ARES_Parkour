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


class ExtremeParkourObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.parkour_event: ParkourEvent = env.parkour_manager.get_term(cfg.params["parkour_name"])
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.history_length = cfg.params['history_length']
        
        # Get joint configuration from params
        pos_joint_patterns = cfg.params.get("pos_joints", [".*"])  # Default: all joints
        vel_joint_patterns = cfg.params.get("vel_joints", [])  # Default: empty (same as pos)
        
        # Get joint indices for position observation
        self.pos_joint_ids = []
        for pattern in pos_joint_patterns:
            joint_ids, _ = self.asset.find_joints(pattern)
            self.pos_joint_ids.extend(joint_ids)
        self.pos_joint_ids = sorted(list(set(self.pos_joint_ids)))
        
        # Get joint indices for velocity observation
        if len(vel_joint_patterns) == 0:
            # If not specified, use same joints as position
            self.vel_joint_ids = self.pos_joint_ids
        else:
            self.vel_joint_ids = []
            for pattern in vel_joint_patterns:
                joint_ids, _ = self.asset.find_joints(pattern)
                self.vel_joint_ids.extend(joint_ids)
            self.vel_joint_ids = sorted(list(set(self.vel_joint_ids)))
        
        # Calculate observation dimensions
        num_pos_joints = len(self.pos_joint_ids)
        num_vel_joints = len(self.vel_joint_ids)
        num_contacts = len(self.sensor_cfg.body_ids)
        
        # obs_dim = ang_vel(3) + imu(2) + delta_yaws(3) + commands(3) + env_idx(2) + 
        #           joint_pos(num_pos_joints) + joint_vel(num_vel_joints) + actions(num_pos_joints) + contacts(num_contacts)
        obs_dim = 3 + 2 + 3 + 3 + 2 + num_pos_joints + num_vel_joints + num_pos_joints + num_contacts
        
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, obs_dim, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.measured_heights = torch.zeros(self.num_envs, 132, device=self.device)
        self.env = env
        
        # Get body_name from config, default to 'base' if not specified
        body_name = cfg.params.get("body_name", "base")
        self.body_id = self.asset.find_bodies(body_name)[0]
        
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
    ) -> torch.Tensor:

        terrain_names = self.parkour_event.env_per_terrain_name
        env_idx_tensor = torch.tensor((terrain_names != 'parkour_flat')).to(dtype=torch.bool, device=self.device)
        invert_env_idx_tensor = torch.tensor((terrain_names == 'parkour_flat')).to(dtype=torch.bool, device=self.device)
        roll, pitch, yaw = euler_xyz_from_quat(self.asset.data.root_quat_w)
        imu_obs = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1).to(self.device)
        if env.common_step_counter % 5 == 0:
            self.delta_yaw = self.parkour_event.target_yaw - wrap_to_pi(yaw)
            self.delta_next_yaw = self.parkour_event.next_target_yaw - wrap_to_pi(yaw)
            self.measured_heights = self._get_heights()
        commands = env.command_manager.get_command('base_velocity')
        
        # Get joint data using configured indices
        joint_pos = (self.asset.data.joint_pos[:, self.pos_joint_ids] -
                     self.asset.data.default_joint_pos[:, self.pos_joint_ids])
        joint_vel = self.asset.data.joint_vel[:, self.vel_joint_ids] * 0.05
        actions = env.action_manager.get_term('joint_pos').action_history_buf[:, -1]
        
        obs_buf = torch.cat((
            self.asset.data.root_ang_vel_b * 0.25,  # [N, 3]
            imu_obs,  # [N, 2]
            0 * self.delta_yaw[:, None],  # [N, 1]
            self.delta_yaw[:, None],  # [N, 1]
            self.delta_next_yaw[:, None],  # [N, 1]
            0 * commands[:, 0:2],  # [N, 2]
            commands[:, 0:1],  # [N, 1]
            env_idx_tensor,  # [N, 1]
            invert_env_idx_tensor,  # [N, 1]
            joint_pos,  # [N, num_pos_joints]
            joint_vel,  # [N, num_vel_joints]
            actions,  # [N, num_pos_joints]
            self._get_contact_fill(),  # [N, num_contacts]
        ), dim=-1)
        priv_explicit = self._get_priv_explicit()
        priv_latent = self._get_priv_latent()
        # obs_buf: dynamic size based on joint config
        # measured_heights: 132
        # priv_explicit: 9
        # priv_latent: 5 + 2 * num_pos_joints (mass(4) + friction(1) + stiffness + damping)
        # history: obs_dim * history_length
        observations = torch.cat([
            obs_buf,
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

    def _get_contact_fill(
        self,
        ):
        contact_forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.sensor_cfg.body_ids] #(N, 4, 3)
        contact = torch.norm(contact_forces, dim=-1) > 2.
        previous_contact_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.sensor_cfg.body_ids] # N, 4, 3
        last_contacts = torch.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = torch.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)
    
    def _get_priv_explicit(
        self,
        ):
        base_lin_vel = self.asset.data.root_lin_vel_b 
        return torch.cat((base_lin_vel * 2.0,
                        0 * base_lin_vel,
                        0 * base_lin_vel), dim=-1).to(self.device)
    
    def _get_priv_latent(self):
        body_mass = self.asset.root_physx_view.get_masses()[:, self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:, self.body_id, :].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com], dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        
        # Use only position joints for stiffness and damping
        joint_stiffness = self.asset.data.joint_stiffness[:, self.pos_joint_ids].to(self.device)
        default_joint_stiffness = self.asset.data.default_joint_stiffness[:, self.pos_joint_ids].to(self.device)
        joint_damping = self.asset.data.joint_damping[:, self.pos_joint_ids].to(self.device)
        default_joint_damping = self.asset.data.default_joint_damping[:, self.pos_joint_ids].to(self.device)
        
        return torch.cat((
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            (joint_stiffness / default_joint_stiffness) - 1,
            (joint_damping / default_joint_damping) - 1
        ), dim=-1).to(self.device)
    
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)

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
