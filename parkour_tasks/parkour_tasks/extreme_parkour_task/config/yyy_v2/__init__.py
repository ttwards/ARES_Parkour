# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for velocity-based locomotion environments."""

# We leave this file empty since we don't want to expose any configs in this package directly.
# We still need this file to import the "config" module in the parent package.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv2TeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv2TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv2TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv2ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v2-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv2StudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v2-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv2StudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v2-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv2StudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv2ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
