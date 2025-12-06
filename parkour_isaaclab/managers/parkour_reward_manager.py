from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Sequence
from isaaclab.managers import RewardManager

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv


class ParkourRewardManager(RewardManager):
    _env: ParkourManagerBasedRLEnv

    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Cache of terrain labels per environment used for logging.
        self._terrain_name_cache: np.ndarray | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        extras: dict[str, torch.Tensor] = {}
        env_id_tensor = self._materialize_env_ids(env_ids)

        for key in self._episode_sums.keys():
            term_episode_values = self._episode_sums[key][env_ids]
            episodic_sum_avg = torch.mean(term_episode_values)
            log_key = f"Episode_Reward/{key}"
            extras[log_key] = episodic_sum_avg / self._env.max_episode_length_s

            extras.update(
                self._terrain_reward_logs(
                    term_name=key,
                    term_episode_values=term_episode_values,
                    env_id_tensor=env_id_tensor,
                )
            )

            self._episode_sums[key][env_ids] = 0.0

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Same to Legged Gym"""
        # reset computation
        self._reward_buf[:] = 0.0
        self._update_terrain_name_cache()
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

    def _update_terrain_name_cache(self):
        """Store a copy of the current terrain labels for later logging."""
        terrain_names = self._get_terrain_names()
        if terrain_names is None:
            self._terrain_name_cache = None
            return

        if isinstance(terrain_names, torch.Tensor):
            terrain_names_np = terrain_names.detach().cpu().numpy().flatten()
        else:
            terrain_names_np = np.array(terrain_names).flatten()

        if terrain_names_np.size == 0:
            self._terrain_name_cache = None
        else:
            self._terrain_name_cache = terrain_names_np

    def _terrain_reward_logs(
        self,
        term_name: str,
        term_episode_values: torch.Tensor,
        env_id_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Creates per-terrain logging entries for the provided term."""

        if (
            self._terrain_name_cache is None
            or env_id_tensor.numel() == 0
            or term_episode_values.numel() == 0
        ):
            return {}

        env_indices_cpu = env_id_tensor.detach().cpu().numpy()
        if env_indices_cpu.size == 0:
            return {}
        # Guard against stale caches when the terrain array is shorter than num_envs.
        if env_indices_cpu.max() >= len(self._terrain_name_cache):
            return {}

        terrain_subset = self._terrain_name_cache[env_indices_cpu]
        term_values = term_episode_values
        extras: dict[str, torch.Tensor] = {}

        unique_terrains = np.unique(terrain_subset)
        for terrain in unique_terrains:
            if terrain is None or terrain == '':
                continue
            mask = terrain_subset == terrain
            if not np.any(mask):
                continue

            mask_tensor = torch.from_numpy(mask).to(device=term_values.device, dtype=torch.bool)
            masked_values = term_values[mask_tensor]
            if masked_values.numel() == 0:
                continue

            log_key = f"Episode_Reward/{terrain}/{term_name}"
            extras[log_key] = masked_values.mean() / self._env.max_episode_length_s

        return extras

    def _materialize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice) -> torch.Tensor:
        """Converts the provided env id selector to a 1-D tensor of indices."""

        if isinstance(env_ids, slice):
            env_id_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            env_id_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_id_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if env_id_tensor.ndim == 0:
            env_id_tensor = env_id_tensor.unsqueeze(0)

        return env_id_tensor
