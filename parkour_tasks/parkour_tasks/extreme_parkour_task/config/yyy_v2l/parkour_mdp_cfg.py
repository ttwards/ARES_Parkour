from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from parkour_isaaclab.managers import ParkourRewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.events import ( 
    randomize_rigid_body_mass,
    apply_external_force_torque,
    reset_joints_by_scale
)
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg 
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands

@configclass
class CommandsCfg:
    """Command specifications for the MDP.
    
    支持根据sub_terrain类型设置不同的速度范围：
    通过 terrain_ranges_map 参数可以为不同地形指定不同的速度范围。
    
    示例：
        terrain_ranges_map={
            "parkour_flat": {
                "lin_vel_x": (0.5, 1.0),  # 平地可以跑快一点
                "heading": (-1.6, 1.6)
            },
            "parkour_gap": {
                "lin_vel_x": (0.2, 0.5),  # 间隙地形需要慢速通过
                "heading": (-1.0, 1.0)
            },
            "parkour_hurdle": {
                "lin_vel_x": (0.3, 0.6),  # 栏架地形中等速度
                "heading": (-1.6, 1.6)
            }
        }
    """

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        heading_control_stiffness=0.8,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 3.),  # 默认速度范围
            heading=(-1.6, 1.6)
        ),
        clips=parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.4
        ),
        # 可选：根据地形类型设置不同的速度范围
        terrain_ranges_map={
            "parkour_flat": {
                "lin_vel_x": (0.8, 3.4),  # 平地可以跑快一点
                "heading": (-1.6, 1.6)
            },
            "parkour_rough": {
                "lin_vel_x": (0.8, 3.2),  # 平地可以跑快一点
                "heading": (-1.6, 1.6)
            },
            "parkour_gap": {
                "lin_vel_x": (0.9, 1.6),  # 间隙地形需要慢速通过
                "heading": (-1.0, 1.0)
            },
            "parkour_hurdle": {
                "lin_vel_x": (0.5, 0.7),  # 栏架地形中等速度
                "heading": (-0.8, 0.8)
            },
            "parkour_step": {
                "lin_vel_x": (0.6, 1.6),  # 台阶地形较慢速度
                "heading": (-1.6, 1.6)
            },
            "parkour_wall": {
                "lin_vel_x": (0.6, 1.6),  # 墙壁地形中等速度
                "heading": (-1.6, 1.6)
            },
            "parkour_slope": {
                "lin_vel_x": (0.8, 3.),  # 斜坡地形中等速度
                "heading": (-1.6, 1.6)
            }
        },
        parkour_term_name="base_parkour",  # 指定从哪个parkour term获取地形信息
    )

    # 新增：目标高度命令（1D）。注意你已添加对应的 obs，本命令用于下游 reward/控制引用
    # target_height = parkour_commands.TargetHeightCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(6.0,6.0 ),
    #     height_range=(0.26, 0.38),
    #     # terrain_height_map={
    #     #     # 可根据地形调整高度目标范围（示例值，可按需修改）
    #     #     "parkour_flat": (0.245, 0.285),
    #     #     "parkour_gap": (0.255, 0.305),
    #     #     "parkour_hurdle": (0.250, 0.295),
    #     #     "parkour_step": (0.248, 0.290),
    #     # },
    #     parkour_term_name="base_parkour",
    # )


@configclass
class ParkourEventsCfg:
    """Command specifications for the MDP."""
    base_parkour = parkours.ParkourEventsCfg(
        asset_name='robot',
    )

@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
                "parkour_name": 'base_parkour',
                "history_length": 20,
                "body_name": "base_link"
            },
            clip=(-20, 20)
        )
    
    # @configclass
    # class TargetHeightPolicyCfg(ObsGroup):
    #     target_height = ObsTerm(
    #         func=observations.observation_target_height,
    #         params={
    #             "parkour_name": 'base_parkour',
    #             "asset_cfg": SceneEntityCfg("robot"),
    #             "body_name": "base_link",
    #         },
    #         clip=(0.26, 0.38)
    #     )
    
    policy: PolicyCfg = PolicyCfg()
    # target_height: TargetHeightPolicyCfg = TargetHeightPolicyCfg()

@configclass
class StudentObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
                "parkour_name": 'base_parkour',
                "history_length": 20,
                "pos_joints": [".*_HipA_joint", ".*_HipF_joint", ".*_Knee_joint"],  # 12 leg joints
                "vel_joints": [".*_HipA_joint", ".*_HipF_joint", ".*_Knee_joint"],  # 12 joints
            },
            clip=(-20, 20)
        )

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={            
            "sensor_cfg":SceneEntityCfg("depth_camera"),
            "resize": (58, 87),
            "buffer_len": 2,
            "debug_vis":True
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        deta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={   
                "parkour_name": 'base_parkour',
                'threshold': 0.6
            },
        )
    
    # @configclass
    # class TargetHeightPolicyCfg(ObsGroup):
    #     target_height = ObsTerm(
    #         func=observations.observation_target_height,
    #         params={
    #             "parkour_name": 'base_parkour',
    #             "asset_cfg": SceneEntityCfg("robot"),
    #             "body_name": "base_link",
    #         },
    #         clip=(0.26, 0.38)
    #     )
    
    policy: PolicyCfg = PolicyCfg()
    # depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    # delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()
    # target_height: TargetHeightPolicyCfg = TargetHeightPolicyCfg()


@configclass
class StudentRewardsCfg:
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-0., 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*Knee_link"]),
        },
    )
    

@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base_link',
    'LF_HipA_link', 'RF_HipA_link', 'LR_HipA_link', 'RR_HipA_link',
    'LF_HipF_link', 'RF_HipF_link', 'LR_HipF_link', 'RR_HipF_link',
    'LF_Foot_link', 'RF_Foot_link', 'LR_Foot_link', 'RR_Foot_link'
    ']
    
    Terrain-based weight adjustment:
    可以为任何奖励项添加 'terrain_weight_map' 来根据不同sub_terrain调整权重倍数。
    
    Available sub_terrain types (根据您的配置):
    - 'parkour_flat': 平地
    - 'parkour_gap': 间隙/裂缝
    - 'parkour_hurdle': 栏架/障碍
    - 'parkour_step': 台阶
    - 'parkour': 常规跑酷地形
    - 'parkour_demo': 演示地形
    - 'parkour_wall': 墙壁
    
    使用示例:
        reward_collision = RewTerm(
            func=rewards.reward_collision,
            weight=-4.0,  # 基础权重
            params={
                "sensor_cfg": SceneEntityCfg(...),
            },
            terrain_weight_map={  # 注意：terrain_weight_map 在 params 外面！
                "parkour_flat": 0.5,    # 平地上碰撞惩罚减半
                "parkour_gap": 2.0,     # 间隙地形碰撞惩罚翻倍
                "parkour_hurdle": 1.5,  # 栏架地形增加50%惩罚
            }
        )
    """
# Available Body strings: 
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-10.5, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_HipA_link", ".*_HipF_link"]),
        },
        # 可选: 根据地形类型调整权重（在 params 外面！）
        terrain_weight_map={
            "parkour_flat": 1.5,    # 平地减少碰撞惩罚
            "parkour_gap": 2.0,     # 间隙增加碰撞惩罚
            "parkour_hurdle": 1.5,  # 栏架增加碰撞惩罚
            "parkour_step": 1.5,
            "parkour_wall": 1.5,
        }
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-1.,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]),
            "parkour_name": 'base_parkour',
            "body_name": "base_link",
        },
        terrain_weight_map={
            "parkour_gap": 0.5,  # Gap 地形上更强的边缘惩罚
            "parkour_wall": 0.0,
        }
    )
    # reward_link_below_ground = RewTerm(
    #     func=rewards.reward_link_below_ground,
    #     weight=-12.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "threshold": -0.02,
    #         "parkour_name": 'base_parkour',
    #     },
    #     terrain_weight_map={
    #         "parkour_slope": 0.0,
    #         "parkour_gap": 1.0,
    #     }
    # )
    reward_link_below_goal_height = RewTerm(
        func=rewards.reward_link_below_goal_height,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "parkour_name": 'base_parkour',
            "goal_margin": 0.0,
        },
        terrain_weight_map={
            "parkour_gap": 1.0,
            "parkour_flat": 0.0,
            "parkour_rough": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_step": 0.0,
            "parkour_wall": 0.0,
            "parkour_slope": 0.0,
            "parkour": 0.0,
            "parkour_demo": 0.0,
        }
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques, 
        weight=-0.000016,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.25,
            "parkour_rough": 1.0,
            "parkour_gap": 0.6,
            "parkour_hurdle": 0.8,
            "parkour_step": 0.8,
            "parkour_wall": 0.7,
        }
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.085,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_wall": 0.4,   # 墙壁上允许更大的关节偏差
            "parkour_gap": 0.6,
            "parkour_step": 0.5,   # 台阶上允许更大的关节偏差
        }
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipA_joint"]),
        },
        terrain_weight_map={
            "parkour_hurdle": 1.25,
        }
    )
    reward_hipf_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.075,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipF_joint"]),
        },
        # terrain_weight_map={
        #     "parkour_flat": 1.5,  # 平地上更强的惩罚，防止无意义抬腿
        #     "parkour_rough": 1.3,
        #     "parkour_gap": 0.9,   # Gap 上允许抬腿跨越
        #     "parkour_hurdle": 0.5,
        #     "parkour_step": 0.35,  # 台阶上允许更大幅度抬腿
        #     "parkour_wall": 0.3,   # 墙壁上需要大幅抬腿
        # }
    )
    joint_power = RewTerm(
        func=rewards.joint_power,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
        terrain_weight_map={
            "parkour_gap": 0.5,
        }
    )
    feet_air_time_variance = RewTerm(
        func=rewards.feet_air_time_variance_penalty,
        weight=-1.,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link")},
        terrain_weight_map={
            "parkour_gap": 0.4,
        }
    )
    reward_knee_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_joint"]),
        },
        # 也可以只在平地/rough 上权重大一点
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.0,
            "parkour_gap": 0.6,
            "parkour_step": 0.7,   # 台阶上膝盖需要更大弯曲
            "parkour_wall": 0.6,   # 墙壁上膝盖需要大幅弯曲抬腿
        }
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_wall": 0.5,
            "parkour_step": 0.6,
            "parkour_gap": 0.5,
        }
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.09,  # 基础权重，适当降低
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.5,    # 平地上：加重惩罚，要求丝滑
            "parkour_rough": 1.2,
            "parkour_gap": 0.6,     # Gap上：大幅减小惩罚，允许剧烈动作
            "parkour_hurdle": 0.6,  # 跨栏上：允许剧烈动作
            "parkour_step": 0.8,
            "parkour_wall": 0.6,
        }
    )
    reward_action_l2 = RewTerm(
        func=rewards.reward_action_l2,
        weight=-0.09,  # 惩罚 action 大小，鼓励较小的动作指令
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_term_name": "joint_pos",  # 指定要惩罚的 action term
        },
        terrain_weight_map={
            "parkour_flat": 1.2,   # 平地上适当加重惩罚
            "parkour_rough": 1.0,
            "parkour_gap": 0.7,    # 复杂地形上降低惩罚
            "parkour_hurdle": 0.6,
            "parkour_step": 0.8,
            "parkour_wall": 0.6,
        }
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_gap": 0.6,
        }
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-1.,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        # 可选：根据地形类型调整权重
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.2,
            "parkour_gap": 0.4,
            "parkour_step": 0.4,
            "parkour_wall": 0.4,
        }
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        # 可选：根据地形类型调整权重
        terrain_weight_map={
            "parkour_flat": 1.7,
            "parkour_rough": 1.0,
            "parkour_gap": 0.4,
            "parkour_hurdle": 0.5,
            "parkour_step": 0.5,
            "parkour_wall": 0.4,
            "parkour_slope": 1.2,
        }
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
        },
    )

    feet_air_time = RewTerm(
        func=rewards.feet_air_time,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
        },
        terrain_weight_map={
            "parkour_gap": 2.2,    # Gap 地形上鼓励更长的腾空时间
            "parkour_hurdle": 1.,
            "parkour_step": 1.,    # 台阶上鼓励抬脚
            "parkour_wall": 1.0,    # 墙壁上更强烈鼓励抬脚
        }
    )
    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_link"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=4.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        # 可选：根据地形类型调整权重
        terrain_weight_map={
            "parkour_flat": 2.6,
            "parkour_rough": 2.2,
            "parkour_gap": 1.2,
            "parkour_hurdle": 1.9,
            "parkour_step": 1.8,
            "parkour_wall": 1.8,
            "parkour_slope": 2.8,
        }
    )

    reward_cumulative_speed_error = RewTerm(
        func=rewards.reward_cumulative_speed_error,
        weight=-1.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "speed_threshold": 0.5,
            "time_threshold_s": 2.0,
            "fast_clear_decay": 0.6,
            "normal_decay": 0.97,
            "accumulation_scale": 0.5,
            "max_penalty": 9.0,
        },
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.0,
            "parkour_gap": 0.8,  # Gap 地形上卡住的惩罚更重
            "parkour_hurdle": 1.0,
            "parkour_step": 1.0,
            "parkour_wall": 1.0,
        }
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.45,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
    )
    # reward_body_height = RewTerm(
    #     func=rewards.reward_body_height,
    #     weight=-1.8,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "parkour_name": 'base_parkour',
    #         "body_name": 'base_link',
    #     },
    #     terrain_weight_map={
    #         "parkour_flat": 200.0,
    #         "parkour_gap": 0.0,
    #         "parkour_hurdle": 0.0,
    #         "parkour_step": 0.0,
    #     }
    # )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques, 
        weight=-1.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.2,
            "parkour_gap": 0.3,
            "parkour_hurdle": 0.4,
            "parkour_step": 0.6,
            "parkour_wall": 0.5,
        }
    )
    # reward_symmetric_contact_time = RewTerm(
    #     func=rewards.reward_symmetric_contact_time,
    #     weight=0.05,
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
    #         # foot_pairs: [[左脚名, 右脚名], ...]
    #         "foot_pairs": [
    #             ["LF_Foot_link", "RF_Foot_link"],  # 前腿对称
    #             ["LR_Foot_link", "RR_Foot_link"],  # 后腿对称
    #         ],
    #         "window_size": 1000,
    #     },
    # )
    # reward_foot_no_contact_time = RewTerm(
    #     func=rewards.reward_foot_no_contact_time,
    #     weight=-0.02,
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
    #         "contact_force_threshold": 0.5,  # 接地力阈值（N）
    #         "max_no_contact_steps": 250,  # 允许的最大连续未接地步数（约3秒）
    #     },
    # )
    # reward_dual_contact_at_high_speed = RewTerm(
    #     func=rewards.reward_dual_contact_at_high_speed,
    #     weight=-0.3,  # 负权重表示惩罚
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         # foot_pairs: [[脚1, 脚2], ...] 指定需要检查同时接地的脚对
    #         "foot_pairs": [
    #             ["LF_Foot_link", "RF_Foot_link"],  # 前腿左右配对
    #             ["LR_Foot_link", "RR_Foot_link"],  # 后腿左右配对
    #         ],
    #         "speed_threshold": 0.4,  # 速度阈值（m/s）
    #         "contact_force_threshold": 1.0,  # 接地力阈值（N）
    #     },
    # )
    # joint_mirror = RewTerm(
    #     func=rewards.joint_mirror,
    #     weight=-0.375,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [
    #             ["LF_Knee_joint", "RF_Knee_joint"],
    #             ["LR_Knee_joint", "RR_Knee_joint"],
    #             ["LF_HipF_joint", "RF_HipF_joint"],
    #             ["LR_HipF_joint", "RR_HipF_joint"],
    #         ],
    #     },
    # )

    gait_reward = RewTerm(
        func=rewards.GaitReward,
        weight=0.45,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.3,
            "command_threshold": 0.2,
            "synced_feet_pair_names": [
                ["LF_Foot_link", "RR_Foot_link"],  # 对角线配对（trot步态）
                ["RF_Foot_link", "LR_Foot_link"],  # 对角线配对（trot步态）
            ],
        },
        terrain_weight_map={
            "parkour_flat": 1.15,
            "parkour_gap": 1.0,
            "parkour_hurdle": 1.0,
            "parkour_step": 1.0,
            "parkour_wall": 1.2,
        }
    )

    # reward_terrain_level = RewTerm(
    #     func=rewards.reward_terrain_level,
    #     weight=1.0,
    #     params={
    #         "parkour_name": 'base_parkour',
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
@configclass
class EventCfg:
    ## Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func=events.reset_root_state,
        params={'offset': 3.},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale,
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = EventTerm( # Okay
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )

    ## we don't use this event, If you use this, you will get a bad result
    # randomize_actuator_gains = EventTerm(
    #     func= events.randomize_actuator_gains,
    #     params={
    #         "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.975, 1.025),  
    #         "damping_distribution_params": (0.975, 1.025),
    #         "operation": "scale",
    #         },
    #     mode="startup",
    # )
    randomize_rigid_body_mass = EventTerm(
        func= randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1., 3.0),
            "operation": "add",
            },
    )
    randomize_rigid_body_com = EventTerm(
        func= events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {'x':(-0.02, 0.02),'y':(-0.02, 0.02),'z':(-0.02, 0.02)}
            },
    )
    random_camera_position = EventTerm(
        func= events.random_camera_position,
        mode="startup",
        params={'sensor_cfg':SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch':(-5, 5)},
                'convention':'ros',
                },
    )
    push_by_setting_velocity = EventTerm( # Okay
        func = events.push_by_setting_velocity, 
        params={'velocity_range':{"x":(-0.5, 0.5), "y":(-0.5, 0.5)}},
        interval_range_s = (8. ,8. ),
        is_global_time= False, 
        mode="interval",
    )
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class ActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "RF_HipA_joint", "RF_HipF_joint", "RF_Knee_joint",
            "LF_HipA_joint", "LF_HipF_joint", "LF_Knee_joint",
            "RR_HipA_joint", "RR_HipF_joint", "RR_Knee_joint",
            "LR_HipA_joint", "LR_HipF_joint", "LR_Knee_joint",
        ],
        scale=0.25,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=20,
        use_delay=True,
        clip={'.*': (-4.8, 4.8)}
    )
