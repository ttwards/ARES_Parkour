from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    from .parkour_command_cfg import TargetHeightCommandCfg


class UniformTargetHeightCommand(CommandTerm):
    """Uniformly sample and update target height command.

    The command is a tensor of shape (num_envs, 1) representing desired body height (m).
    Optionally supports terrain-aware ranges via ``terrain_height_map``.
    """

    cfg: TargetHeightCommandCfg

    def __init__(self, cfg: TargetHeightCommandCfg, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

        # metrics
        self.metrics["height_tracking_error"] = torch.zeros(self.num_envs, device=self.device)

        # terrain-aware flag
        self._terrain_aware = cfg.terrain_height_map is not None and cfg.parkour_term_name is not None
        if self._terrain_aware:
            self._parkour_term_name = cfg.parkour_term_name

    def __str__(self) -> str:
        msg = "UniformTargetHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        # integrate absolute error per resampling window (best-effort; uses current body height)
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        current_height = self._env.scene["robot"].data.root_pos_w[:, 2]
        self.metrics["height_tracking_error"] += (
            torch.abs(self.height_command[:, 0] - current_height) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if self._terrain_aware:
            parkour_event = self._env.parkour_manager.get_term(self._parkour_term_name)
            terrain_names = parkour_event.env_per_terrain_name.flatten()
            for env_id in env_ids:
                terrain_name = terrain_names[env_id]
                if terrain_name in self.cfg.terrain_height_map:
                    height_range = self.cfg.terrain_height_map[terrain_name]
                else:
                    height_range = self.cfg.height_range
                self.height_command[env_id, 0] = torch.empty(1, device=self.device).uniform_(*height_range)
        else:
            r = torch.empty(len(env_ids), device=self.device)
            self.height_command[env_ids, 0] = r.uniform_(*self.cfg.height_range)

    def _update_command(self):
        # nothing to do; command directly holds sampled target height
        pass


