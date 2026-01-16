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

from isaaclab.envs import mdp

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
            "parkour_fixed_gap": {
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
            lin_vel_x=(0.3, 2.),  # 默认速度范围
            heading=(-1.6, 1.6)
        ),
        clips=parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip=0.2,
            ang_vel_clip=0.4
        ),
        # 可选：根据地形类型设置不同的速度范围
        terrain_ranges_map={
            "parkour_flat": {
                "lin_vel_x": (0.5, 1.9),  # 平地可以跑快一点
                "heading": (-1.6, 1.6)
            },
            "parkour_rough": {
                "lin_vel_x": (0.5, 1.6),  # 平地可以跑快一点
                "heading": (-1.6, 1.6)
            },
            "parkour_fixed_gap": {
                "lin_vel_x": (0.5, 1.5),  # 间隙地形需要慢速通过
                "heading": (-1.0, 1.0)
            },
            "parkour_hurdle": {
                "lin_vel_x": (0.3, 0.9),  # 栏架地形中等速度
                "heading": (-0.5, 0.5)
            },
            "parkour_step": {
                "lin_vel_x": (0.4, 1.5),  # 台阶地形较慢速度
                "heading": (-1.6, 1.6)
            },
            "parkour_wall": {
                "lin_vel_x": (0.4, 1.3),  # 墙壁地形中等速度
                "heading": (-1.6, 1.6)
            },
            "parkour_slope": {
                "lin_vel_x": (0.6, 1.8),  # 斜坡地形中等速度
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
    #     #     "parkour_fixed_gap": (0.255, 0.305),
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
        next_goal_threshold=0.32,
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
                "body_name": "base_link",
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
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
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": True
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


# 定义常量：机器人link名称模式
FOOT_LINK_NAME = ".*_Foot_link"
BASE_LINK_NAME = "base_link"

@configclass
class StudentRewardsCfg:
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-0.,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_Knee_link"]),
        },
    )

@configclass
class TeacherRewardsCfg:
    base_height_l2 = RewTerm(
        func=rewards.base_height_l2,
        weight=-80.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.3,
        },
        terrain_weight_map={
            "parkour_flat": 2.0,
            "parkour_rough": 2.0,
            "parkour_fixed_gap": 1.0,
            "parkour_hurdle": 0.1,
            "parkour_step": 1.0,
            "parkour_wall": 0.7,
            "parkour_slope": 1.8,
        }
    )
    # ========================================================================
    # Tier 1: 核心任务奖励 - 这是机器人的主要目标
    # ========================================================================
    
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=20.0,  # 5.2 -> 12.0，成为最主要的奖励信号
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        terrain_weight_map={
            "parkour_flat": 1.6,      # 平地可以跑得更快更稳
            "parkour_rough": 1.5,
            "parkour_fixed_gap": 4.0,  # Gap成功通过价值更高
            "parkour_hurdle": 1.4,
            "parkour_step": 1.5,
            "parkour_wall": 1.4,
            "parkour_slope": 1.6,
        }
    )

    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.7,  # 0.55 -> 2.5，偏航控制很重要
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour'
        },
        terrain_weight_map={
            "parkour_fixed_gap": 1.6,  # Gap需要精确对准
            "parkour_hurdle": 3.1,
            "parkour_step": 1.2,
        }
    )
    
    # ========================================================================
    # Tier 2: 安全约束 - 防止机器人做危险动作
    # ========================================================================
    
    reward_cumulative_speed_error = RewTerm(
        func=rewards.reward_cumulative_speed_error,
        weight=-4.5,  # -2.3 -> -5.0，强力惩罚卡住行为
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "speed_threshold": 0.4,      # 0.4 -> 0.3，更早检测卡住
            "time_threshold_s": 1.2,     # 2.0 -> 1.2，更快触发惩罚
            "fast_clear_decay": 0.75,    # 恢复快速时快速清除累积
            "normal_decay": 0.95,        # 0.97 -> 0.95，稍快衰减
            "accumulation_scale": 0.4,   # 0.5 -> 0.7，更重的累积
            "max_penalty": 4.0,         # 7.0 -> 12.0，最大惩罚加倍
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
        weight=-8.0,  # -10.5 -> -6.0，稍微降低以允许探索
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK_NAME, ".*_HipA_link", ".*_HipF_link"]),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_link"),
            "parkour_name": 'base_parkour',
            "goal_margin": 0.0,
            "distance_threshold": 10.4,
        },
        terrain_weight_map={
            "parkour_fixed_gap": 1.0,    # 只在Gap地形启用
            "parkour_flat": 0.0,
            "parkour_rough": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_step": 0.8,
            "parkour_wall": 0.0,
            "parkour_slope": 0.0,
        }
    )

    reward_climb_up = RewTerm(
        func=rewards.reward_link_below_goal_height,
        weight=-12.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot_link"),
            "parkour_name": 'base_parkour',
            "goal_margin": -0.1,
            "distance_threshold": 0.2,
        },
        terrain_weight_map={
            "parkour_fixed_gap": 0.0,
            "parkour_flat": 0.0,
            "parkour_rough": 0.0,
            "parkour_hurdle": 0.0,
            "parkour_step": 0.0,
            "parkour_wall": 1.0,
            "parkour_slope": 0.0,
        },
    )
    
    reward_orientation = RewTerm(
        func=rewards.reward_orientation,
        weight=-10.0,  # 增加权重以防止磕头
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        terrain_weight_map={
            "parkour_flat": 1.8,         # 平地要求更正的姿态
            "parkour_rough": 1.2,
            "parkour_fixed_gap": 1.2,    # 提高Gap地形姿态控制，防止磕头
            "parkour_hurdle": 1.5,
            "parkour_step": 1.0,
            "parkour_wall": 0.7,
            "parkour_slope": 1.6,
        }
    )

    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=[".*"]),
            "parkour_name": 'base_parkour',
            "body_name": BASE_LINK_NAME,
        },
        terrain_weight_map={
            "parkour_fixed_gap": 0.5,    # Gap上允许一定边缘接触
            "parkour_step": 0.7,
            "parkour_wall": 0.0,         # 墙壁不考虑
        }
    )

    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
        },
    )
    
    # ========================================================================
    # Tier 3: 运动质量 - 优化步态和动作流畅度
    # ========================================================================
    
    gait_reward = RewTerm(
        func=rewards.GaitReward,
        weight=2.0,  # 1.0 -> 4.0，步态质量非常重要
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.6,              # 0.7 -> 0.6，更严格的相位要求
            "command_name": "base_velocity",
            "max_err": 0.2,         # 0.2 -> 0.15，降低容差
            "velocity_threshold": 0.25,  # 0.5 -> 0.35，更早强制步态
            "command_threshold": 0.2,   # 0.1 -> 0.08
            "synced_feet_pair_names": [
                ["LF_Foot_link", "RR_Foot_link"],  # Trot步态对角线
                ["RF_Foot_link", "LR_Foot_link"],
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
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
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
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "threshold": 0.6,       # 0.5 -> 0.35，更早奖励抬脚
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
        },
        terrain_weight_map={
            "parkour_fixed_gap": 2.1,    # Gap强烈鼓励腾空
            "parkour_hurdle": 1.8,
            "parkour_step": 1.8,
            "parkour_wall": 2.0,
            "parkour_flat": 0.6,         # 平地不需要太高
            "parkour_rough": 0.7,
        }
    )
    
    foot_clearance_reward = RewTerm(
        func=rewards.reward_foot_clearance,
        weight=1.2,  # 1.3 -> 2.0，增强抬腿高度奖励
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "height_threshold": 0.01,
            "tanh_scale": 50.0,          # 50.0 -> 35.0，稍微平缓
            "contact_threshold": 1.0,
        },
        terrain_weight_map={
            "parkour_flat": 1.2,
            "parkour_rough": 1.2,
            "parkour_fixed_gap": 1.8,
            "parkour_hurdle": 0.2,       # 栏架有专门处理
            "parkour_step": 1.4,
            "parkour_wall": 1.5,
        }
    )
    
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 2.0,         # 平地要求非常平滑
            "parkour_rough": 2.0,
            "parkour_slope": 2.0,
            "parkour_fixed_gap": 0.3,    # 复杂地形允许激烈动作
            "parkour_hurdle": 1.0,
            "parkour_step": 0.6,
            "parkour_wall": 0.4,
        }
    )

    reward_action_l2 = RewTerm(
        func=rewards.reward_action_l2,
        weight=-0.55,  # -0.06 -> -0.15，惩罚 action 大小
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_term_name": "joint_pos",
        },
    )
    
    feet_slide = RewTerm(
        func=rewards.feet_slide,
        weight=-0.6,  # -0.1 -> -0.8，滑动不利于控制
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
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
        weight=-0.0000185,  # -0.000016 -> -0.00004，适度增强
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
        weight=-5e-6,  # -5e-6 -> -1.5e-5，增强功率限制
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
        terrain_weight_map={
            "parkour_flat": 2.0,
            "parkour_fixed_gap": 0.2,
            "parkour_hurdle": 0.3,
        }
    )

    reward_hipa_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-5.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipA_joint"]),
        },
        terrain_weight_map={
            "parkour_hurdle": 1.8,
        }
    )
    reward_hipf_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-2.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipF_joint"]),
        },
    )

    reward_knee_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-1.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_joint"]),
        },
    )

    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z,
        weight=-8.5,  # 进一步增加，减少垂直方向运动
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
        },
        terrain_weight_map={
            "parkour_flat": 2.0,
            "parkour_rough": 1.5,
            "parkour_fixed_gap": 0.9,    # 适当提高Gap地形约束
            "parkour_step": 0.6,
            "parkour_wall": 0.5,
            "parkour_hurdle": 0.5,
        }
    )
    
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-1.2,  # 大幅增加，有效抑制pitch角速度（磕头动作）
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
        terrain_weight_map={
            "parkour_flat": 1.8,
            "parkour_wall": 0.5,
            "parkour_step": 0.6,
            "parkour_fixed_gap": 1.0,    # Gap地形也要保持姿态稳定
        }
    )
    
    # ========================================================================
    # 对称性与协调性
    # ========================================================================
    
    joint_mirror = RewTerm(
        func=rewards.joint_mirror,
        weight=0.9,  # 0.5 -> 1.5，加强左右对称
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
            "parkour_hurdle": 3.0,
        }
    )
    
    # ========================================================================
    # 脚部位置约束 - 防止脚部过度偏离默认位置
    # ========================================================================
    
    reward_feet_position_deviation = RewTerm(
        func=rewards.reward_feet_position_deviation,
        weight=-0.05,  # 负权重表示惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_pattern": FOOT_LINK_NAME,  # 匹配所有脚部link
        },
        terrain_weight_map={
            "parkour_flat": 1.2,         # 平地要求保持自然站姿
            "parkour_rough": 1.0,
            "parkour_fixed_gap": 0.3,    # 复杂地形允许更大偏差
            "parkour_hurdle": 0.4,
            "parkour_step": 0.5,
            "parkour_wall": 0.5,
            "parkour_slope": 1.2,
        }
    )

    is_terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )


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
