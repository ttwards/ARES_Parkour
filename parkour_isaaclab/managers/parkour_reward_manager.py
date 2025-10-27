from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import RewardManager

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


class ParkourRewardManager(RewardManager):
    _env: ParkourManagerBasedRLEnv

    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)

    def compute(self, dt: float) -> torch.Tensor:
        """Same to Legged Gym"""
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            # 检查是否有terrain_weight_map配置
            if hasattr(term_cfg, 'terrain_weight_map') and term_cfg.terrain_weight_map is not None:
                # 获取每个环境的地形类型名称
                terrain_names = self._get_terrain_names()
                # 创建权重张量
                weight_tensor = torch.ones(self._env.num_envs, device=self._env.device) * term_cfg.weight

                # 根据地形类型调整权重
                for terrain_type, multiplier in term_cfg.terrain_weight_map.items():
                    mask = terrain_names == terrain_type
                    weight_tensor[mask] = term_cfg.weight * multiplier

                # compute term's value
                value = term_cfg.func(self._env, **term_cfg.params)
                value = value * weight_tensor * dt
            else:
                # 原有的计算方式
                value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt

            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt
        # self._reward_buf[:] = torch.clip(self._reward_buf[:], min=0.)
        return self._reward_buf

    def _get_terrain_names(self):
        """获取每个环境的地形类型名称"""
        # 从parkour_manager中获取terrain names
        if hasattr(self._env, 'parkour_manager'):
            parkour_event = None
            # 找到ParkourEvent实例
            for term in self._env.parkour_manager._terms.values():
                if hasattr(term, 'env_per_terrain_name'):
                    parkour_event = term
                    break

            if parkour_event is not None:
                return parkour_event.env_per_terrain_name.flatten()

        # 如果找不到，返回空字符串数组
        return [''] * self._env.num_envs
