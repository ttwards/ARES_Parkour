# 重新设计的奖励配置 - 更简洁、更平衡
# 复制此配置到原文件 line 214-715 位置

@configclass
class TeacherRewardsCfg:
    """重新设计的奖励系统
    
    设计哲学:
    ========
    1. 简化奖励项数量 (30个 -> 18个)
    2. 明确奖励层级和权重尺度
    3. 消除冲突的奖励项
    4. 增强核心任务信号
    
    奖励架构:
    ========
    Tier 1 - 任务奖励 (8-15):  驱动机器人完成目标
    Tier 2 - 安全惩罚 (3-10):  防止危险行为
    Tier 3 - 运动质量 (0.5-3): 优化步态和动作
    Tier 4 - 能效优化 (<0.5):  降低能耗
    
    主要改动:
    ========
    - 大幅提升速度跟踪奖励 (5.2 -> 12.0)
    - 合并前后脚腾空时间奖励
    - 移除冲突的前后脚同时离地惩罚
    - 移除冗余的action_l2和dof_acc
    - 简化关节位置惩罚
    - 强化步态奖励权重
    """
    
    # ========================================================================
    # Tier 1: 核心任务奖励 - 这是机器人的主要目标
    # ========================================================================
    
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=12.0,  # 5.2 -> 12.0，成为最主要的奖励信号
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        terrain_weight_map={
            "parkour_flat": 1.6,      # 平地可以跑得更快更稳
            "parkour_rough": 1.5,
            "parkour_fixed_gap": 2.0,  # Gap成功通过价值更高
            "parkour_hurdle": 1.4,
            "parkour_step": 1.5,
            "parkour_wall": 1.4,
            "parkour_slope": 1.6,
        }
    )
    
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=2.5,  # 0.55 -> 2.5，偏航控制很重要
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        terrain_weight_map={
            "parkour_fixed_gap": 1.6,  # Gap需要精确对准
            "parkour_hurdle": 1.4,
            "parkour_step": 1.2,
        }
    )
    
    # ========================================================================
    # Tier 2: 安全约束 - 防止机器人做危险动作
    # ========================================================================
    
    reward_cumulative_speed_error = RewTerm(
        func=rewards.reward_cumulative_speed_error,
        weight=-5.0,  # -2.3 -> -5.0，强力惩罚卡住行为
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "speed_threshold": 0.3,      # 0.4 -> 0.3，更早检测卡住
            "time_threshold_s": 1.2,     # 2.0 -> 1.2，更快触发惩罚
            "fast_clear_decay": 0.75,    # 恢复快速时快速清除累积
            "normal_decay": 0.95,        # 0.97 -> 0.95，稍快衰减
            "accumulation_scale": 0.7,   # 0.5 -> 0.7，更重的累积
            "max_penalty": 12.0,         # 7.0 -> 12.0，最大惩罚加倍
        },
        terrain_weight_map={
            "parkour_flat": 0.8,         # 平地卡住不应该发生
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 2.5,    # Gap卡住是大问题
            "parkour_hurdle": 2.2,
            "parkour_step": 1.4,
            "parkour_wall": 1.6,
        }
    )
    
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-6.0,  # -10.5 -> -6.0，稍微降低以允许探索
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_HipA_link", ".*_HipF_link"]),
        },
        terrain_weight_map={
            "parkour_flat": 1.5,         # 平地碰撞不应该
            "parkour_fixed_gap": 1.3,
            "parkour_hurdle": 1.6,       # 栏架碰撞需要强调
            "parkour_step": 1.4,
            "parkour_wall": 1.5,
        }
    )
    
    reward_link_below_goal_height = RewTerm(
        func=rewards.reward_link_below_goal_height,
        weight=-15.0,  # -12.0 -> -15.0，高度安全非常重要
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "parkour_name": 'base_parkour',
            "goal_margin": 0.0,
        },
        terrain_weight_map={
            "parkour_fixed_gap": 1.0,    # 只在Gap地形启用
            "parkour_flat": 0.0,
            "parkour_rough": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_step": 0.0,
            "parkour_wall": 0.0,
            "parkour_slope": 0.0,
        }
    )
    
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-6.0,  # -3.2 -> -6.0，姿态控制加强
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        terrain_weight_map={
            "parkour_flat": 1.8,         # 平地要求更正的姿态
            "parkour_rough": 1.2,
            "parkour_fixed_gap": 0.7,    # Gap允许姿态调整
            "parkour_hurdle": 0.9,
            "parkour_step": 1.0,
            "parkour_wall": 0.7,
            "parkour_slope": 1.6,
        }
    )
    
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-3.0,  # -1.0 -> -3.0，脚踩边缘很危险
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]),
            "parkour_name": 'base_parkour',
            "body_name": "base_link",
        },
        terrain_weight_map={
            "parkour_fixed_gap": 0.5,    # Gap上允许一定边缘接触
            "parkour_step": 0.7,
            "parkour_wall": 0.0,         # 墙壁不考虑
        }
    )
    
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-2.0,  # -1.0 -> -2.0，绊倒很危险
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
        },
    )
    
    # ========================================================================
    # Tier 3: 运动质量 - 优化步态和动作流畅度
    # ========================================================================
    
    gait_reward = RewTerm(
        func=rewards.GaitReward,
        weight=4.0,  # 1.0 -> 4.0，步态质量非常重要
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.6,              # 0.7 -> 0.6，更严格的相位要求
            "command_name": "base_velocity",
            "max_err": 0.15,         # 0.2 -> 0.15，降低容差
            "velocity_threshold": 0.35,  # 0.5 -> 0.35，更早强制步态
            "command_threshold": 0.08,   # 0.1 -> 0.08
            "synced_feet_pair_names": [
                ["LF_Knee_link", "RR_Knee_link"],  # Trot步态对角线
                ["RF_Knee_link", "LR_Knee_link"],
            ],
        },
        terrain_weight_map={
            "parkour_flat": 2.0,         # 平地强制良好步态
            "parkour_rough": 1.7,
            "parkour_fixed_gap": 0.4,    # Gap放松步态要求
            "parkour_hurdle": 0.5,
            "parkour_step": 0.6,
            "parkour_wall": 0.6,
            "parkour_slope": 1.8,
        }
    )
    
    feet_air_time_variance = RewTerm(
        func=rewards.feet_air_time_variance_penalty,
        weight=-2.5,  # -1.0 -> -2.5，步态一致性很重要
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link")},
        terrain_weight_map={
            "parkour_fixed_gap": 0.2,    # Gap允许不对称步态
            "parkour_hurdle": 0.4,
            "parkour_slope": 2.0,
            "parkour_flat": 2.2,         # 平地要求高度一致
            "parkour_rough": 1.8,
            "parkour_step": 0.5,
            "parkour_wall": 0.4,
        }
    )
    
    feet_air_time = RewTerm(
        func=rewards.feet_air_time,
        weight=1.2,  # 合并前后脚，0.15*2 -> 1.2，适当增强
        params={
            "command_name": "base_velocity",
            "threshold": 0.35,       # 0.5 -> 0.35，更早奖励抬脚
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 2.8,    # Gap强烈鼓励腾空
            "parkour_hurdle": 2.4,
            "parkour_step": 1.8,
            "parkour_wall": 2.0,
            "parkour_flat": 0.6,         # 平地不需要太高
            "parkour_rough": 0.7,
        }
    )
    
    foot_clearance_reward = RewTerm(
        func=rewards.reward_foot_clearance,
        weight=2.0,  # 1.3 -> 2.0，增强抬腿高度奖励
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "height_threshold": 0.015,   # 0.005 -> 0.015，提高门槛
            "tanh_scale": 35.0,          # 50.0 -> 35.0，稍微平缓
            "contact_threshold": 1.0,
        },
        terrain_weight_map={
            "parkour_flat": 0.4,
            "parkour_rough": 0.6,
            "parkour_fixed_gap": 1.8,
            "parkour_hurdle": 0.2,       # 栏架有专门处理
            "parkour_step": 1.4,
            "parkour_wall": 1.5,
        }
    )
    
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.8,  # -0.2 -> -0.8，强调动作平滑
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 2.5,         # 平地要求非常平滑
            "parkour_rough": 2.0,
            "parkour_fixed_gap": 0.3,    # 复杂地形允许激烈动作
            "parkour_hurdle": 0.4,
            "parkour_step": 0.6,
            "parkour_wall": 0.4,
        }
    )
    
    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.8,  # -0.1 -> -0.8，滑动不利于控制
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Knee_link"),
        },
        terrain_weight_map={
            "parkour_flat": 1.8,
            "parkour_rough": 1.4,
            "parkour_fixed_gap": 0.5,
        }
    )
    
    # ========================================================================
    # Tier 4: 能效与姿态优化 - 微调细节
    # ========================================================================
    
    reward_torques = RewTerm(
        func=rewards.reward_torques, 
        weight=-0.00004,  # -0.000016 -> -0.00004，适度增强
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.8,
            "parkour_rough": 1.4,
            "parkour_fixed_gap": 0.3,
            "parkour_hurdle": 0.4,
            "parkour_step": 0.5,
            "parkour_wall": 0.4,
        }
    )
    
    joint_power = RewTerm(
        func=rewards.joint_power,
        weight=-0.000015,  # -5e-6 -> -1.5e-5，增强功率限制
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
        terrain_weight_map={
            "parkour_flat": 2.0,
            "parkour_fixed_gap": 0.2,
            "parkour_hurdle": 0.3,
        }
    )
    
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.2,  # -0.085 -> -0.2，加强默认姿态约束
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_wall": 0.25,
            "parkour_fixed_gap": 0.3,
            "parkour_step": 0.3,
            "parkour_hurdle": 0.4,
        }
    )
    
    reward_joint_limits = RewTerm(
        # 合并原来的hip/hipf/knee pos惩罚为统一的关节限位惩罚
        func=rewards.reward_hip_pos,  # 复用这个函数
        weight=-0.6,  # 统一权重
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 所有关节
        },
        terrain_weight_map={
            "parkour_flat": 1.5,
            "parkour_rough": 1.2,
            "parkour_fixed_gap": 0.3,
            "parkour_hurdle": 0.2,
            "parkour_step": 0.3,
            "parkour_wall": 0.2,
        }
    )
    
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-2.5,  # -1.0 -> -2.5，减少上下跳动
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        terrain_weight_map={
            "parkour_flat": 2.0,
            "parkour_rough": 1.5,
            "parkour_fixed_gap": 0.2,    # Gap允许垂直运动
            "parkour_step": 0.4,
            "parkour_wall": 0.3,
        }
    )
    
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.3,  # -0.08 -> -0.3，加强roll/pitch稳定
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.8,
            "parkour_wall": 0.3,
            "parkour_step": 0.4,
            "parkour_fixed_gap": 0.3,
        }
    )
    
    # ========================================================================
    # 对称性与协调性
    # ========================================================================
    
    joint_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=1.5,  # 0.5 -> 1.5，加强左右对称
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["LF_HipA_joint", "RF_HipA_joint"],
                ["LR_HipA_joint", "RR_HipA_joint"],
            ],
        },
        terrain_weight_map={
            "parkour_flat": 1.5,
            "parkour_rough": 1.3,
        }
    )



@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base_link',
    'LF_HipA_link', 'RF_HipA_link', 'LR_HipA_link', 'RR_HipA_link',
    'LF_HipF_link', 'RF_HipF_link', 'LR_HipF_link', 'RR_HipF_link',
    'LF_Knee_link', 'RF_Knee_link', 'LR_Knee_link', 'RR_Knee_link'
    ']
    
    Terrain-based weight adjustment:
    可以为任何奖励项添加 'terrain_weight_map' 来根据不同sub_terrain调整权重倍数。
    
    Available sub_terrain types (根据您的配置):
    - 'parkour_flat': 平地
    - 'parkour_fixed_gap': 间隙/裂缝
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
                "parkour_fixed_gap": 2.0,     # 间隙地形碰撞惩罚翻倍
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
            "parkour_fixed_gap": 1.2,     # 间隙增加碰撞惩罚
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
            "parkour_fixed_gap": 0.2,  # Gap 地形上更强的边缘惩罚
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
    #         "parkour_fixed_gap": 1.0,
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
            "parkour_fixed_gap": 1.0,
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
            "parkour_fixed_gap": 0.6,
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
            "parkour_wall": 0.5,   # 墙壁上允许更大的关节偏差
            "parkour_fixed_gap": 0.6,
            "parkour_step": 0.5,   # 台阶上允许更大的关节偏差
        }
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipA_joint"]),
        },
        terrain_weight_map={
            "parkour_hurdle": 0.65,
        }
    )
    reward_hipf_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.075,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipF_joint"]),
        },
        terrain_weight_map={
            # "parkour_flat": 1.5,  # 平地上更强的惩罚，防止无意义抬腿
            # "parkour_rough": 1.3,
            "parkour_fixed_gap": 0.6,   # Gap 上允许抬腿跨越
            "parkour_hurdle": 0.5,
            # "parkour_step": 0.35,  # 台阶上允许更大幅度抬腿
            # "parkour_wall": 0.3,   # 墙壁上需要大幅抬腿
        }
    )
    joint_power = RewTerm(
        func=rewards.joint_power,
        weight=-5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 0.5,
        }
    )
    feet_air_time_variance = RewTerm(
        func=rewards.feet_air_time_variance_penalty,
        weight=-1.,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link")},
        terrain_weight_map={
            "parkour_fixed_gap": 0.6,
            "parkour_slope": 2.0,
            "parkour_flat": 2.0,
            "parkour_rough": 2.0,
            "parkour_hurdle": 2.0,
            "parkour_step": 1.0,
            "parkour_wall": 1.0,
        }
    )
    reward_knee_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.11,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_joint"]),
        },
        # 也可以只在平地/rough 上权重大一点
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 0.6,
            "parkour_hurdle": 0.5,
            "parkour_step": 0.5,   # 台阶上膝盖需要更大弯曲
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
            "parkour_fixed_gap": 0.5,
        }
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.2,  # 基础权重，适当降低
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.5,    # 平地上：加重惩罚，要求丝滑
            "parkour_rough": 1.2,
            "parkour_fixed_gap": 1.2,     # Gap上：大幅减小惩罚，允许剧烈动作
            "parkour_hurdle": 0.6,  # 跨栏上：允许剧烈动作
            "parkour_step": 0.8,
            "parkour_wall": 0.6,
        }
    )
    reward_action_l2 = RewTerm(
        func=rewards.reward_action_l2,
        weight=-0.06,  # 惩罚 action 大小，鼓励较小的动作指令
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_term_name": "joint_pos",  # 指定要惩罚的 action term
        },
        terrain_weight_map={
            "parkour_flat": 1.2,   # 平地上适当加重惩罚
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 0.7,    # 复杂地形上降低惩罚
            "parkour_hurdle": 0.7,
            "parkour_step": 0.8,
            "parkour_wall": 0.6,
        }
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 0.6,
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
            "parkour_fixed_gap": 0.6,
            "parkour_step": 0.6,
            "parkour_wall": 0.6,
        }
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-3.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        # 可选：根据地形类型调整权重
        terrain_weight_map={
            "parkour_flat": 1.8,
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 1.2,
            "parkour_hurdle": 1.4,
            "parkour_step": 1.1,
            "parkour_wall": 1.0,
            "parkour_slope": 1.8,
        }
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
        },
    )

    feet_air_time_F = RewTerm(
        func=rewards.feet_air_time,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*F_Knee_link"),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 2.2,    # Gap 地形上鼓励更长的腾空时间
            "parkour_hurdle": 1.,
            "parkour_step": 1.,    # 台阶上鼓励抬脚
            "parkour_wall": 1.0,    # 墙壁上更强烈鼓励抬脚
        }
    )
    feet_air_time_R = RewTerm(
        func=rewards.feet_air_time,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*R_Knee_link"),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 2.2,    # Gap 地形上鼓励更长的腾空时间
            "parkour_hurdle": 1.,
            "parkour_step": 1.,    # 台阶上鼓励抬脚
            "parkour_wall": 1.0,    # 墙壁上更强烈鼓励抬脚
        }
    )
    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Knee_link"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=5.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        # 可选：根据地形类型调整权重
        terrain_weight_map={
            "parkour_flat": 2.1,
            "parkour_rough": 2.1,
            "parkour_fixed_gap": 2.4,
            "parkour_hurdle": 1.6,
            "parkour_step": 1.8,
            "parkour_wall": 1.8,
            "parkour_slope": 2.1,
        }
    )

    reward_cumulative_speed_error = RewTerm(
        func=rewards.reward_cumulative_speed_error,
        weight=-2.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "speed_threshold": 0.4,
            "time_threshold_s": 2.0,
            "fast_clear_decay": 0.6,
            "normal_decay": 0.97,
            "accumulation_scale": 0.5,
            "max_penalty": 7.0,
        },
        terrain_weight_map={
            "parkour_flat": 1.25,
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 2.7,  # Gap 地形上卡住的惩罚更重
            "parkour_hurdle": 2.4,
            "parkour_step": 1.3,
            "parkour_wall": 1.2,
        }
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.55,
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
    #         "parkour_fixed_gap": 0.0,
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
            "parkour_fixed_gap": 0.3,
            "parkour_hurdle": 0.4,
            "parkour_step": 0.6,
            "parkour_wall": 0.5,
        }
    )
    joint_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["LF_HipA_joint", "RF_HipA_joint"],
                ["LR_HipA_joint", "RR_HipA_joint"],
            ],
        },
    )
    undesired_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["LF_(HipF|Knee).*", "RR_(HipF|Knee).*"],
                ["LR_(HipF|Knee).*", "LR_(HipF|Knee).*"],
            ],
        },
    )

    gait_reward = RewTerm(
        func=rewards.GaitReward,
        weight=1.,  # 增加权重：0.7 -> 1.5，让步态奖励更重要
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.7,
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,  # 降低速度阈值：0.3 -> 0.2，更早开始强制步态
            "command_threshold": 0.1,  # 降低命令阈值：0.2 -> 0.15
            "synced_feet_pair_names": [
                ["LF_Knee_link", "RR_Knee_link"],  # 对角线配对（trot步态）
                ["RF_Knee_link", "LR_Knee_link"],  # 对角线配对（trot步态）
            ],
        },
        terrain_weight_map={
            "parkour_flat": 1.5,  # 平地上进一步提高步态要求
            "parkour_rough": 1.3,  # 粗糙地形也要求良好步态
            "parkour_fixed_gap": 1.0,
            "parkour_hurdle": 1.0,
            "parkour_step": 1.0,
            "parkour_wall": 1.2,
            "parkour_slope": 1.4,  # 斜坡上也要求良好步态
        }
    )

    # 足端离地高度奖励：奖励脚在无接触且离地高度超过阈值时抬腿
    # 使用4个独立的 RayCaster sensor (foot_scanner_LF/RF/LR/RR)
    foot_clearance_reward = RewTerm(
        func=rewards.reward_foot_clearance,
        weight=1.3,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "height_threshold": 0.005,  # 0.01m以上开始奖励
            "tanh_scale": 50.0,  # 0.06m时饱和 (tanh(0.05*40)≈0.96)
            "contact_threshold": 1.0,
        },
        terrain_weight_map={
            "parkour_flat": 1.0,
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 1.0,
            "parkour_hurdle": 0.1,
            "parkour_step": 1.2,
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

    # 惩罚前脚同时离地
    penalty_both_front_feet_airborne = RewTerm(
        func=rewards.penalty_both_front_feet_airborne,
        weight=-2.0,  # 负权重表示惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*F_Knee_link"),
            "front_feet_names": ["LF_Knee_link", "RF_Knee_link"],
            "contact_force_threshold": 1.0,
        },
    )

    # 惩罚后脚同时离地
    penalty_both_rear_feet_airborne = RewTerm(
        func=rewards.penalty_both_rear_feet_airborne,
        weight=-2.0,  # 负权重表示惩罚
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*R_Knee_link"),
            "rear_feet_names": ["LR_Knee_link", "RR_Knee_link"],
            "contact_force_threshold": 1.0,
        },
    )