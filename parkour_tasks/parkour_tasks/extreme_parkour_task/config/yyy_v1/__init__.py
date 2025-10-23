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
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v1-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv1TeacherParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv1ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v1-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv1TeacherParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv1ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Teacher-Ares-yyy-v1-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_teacher_cfg:AresYYYv1TeacherParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_teacher_ppo_cfg:AresYYYv1ParkourTeacherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v1-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv1StudentParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv1ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v1-Play-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv1StudentParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv1ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Extreme-Parkour-Student-Ares-yyy-v1-Eval-v0",
    entry_point="parkour_isaaclab.envs:ParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_student_cfg:AresYYYv1StudentParkourEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_student_ppo_cfg:AresYYYv1ParkourStudentPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_parkour_ppo_cfg.yaml",
    },
)
