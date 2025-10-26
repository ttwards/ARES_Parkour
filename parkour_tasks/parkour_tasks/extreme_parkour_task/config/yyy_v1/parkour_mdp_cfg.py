from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.events import ( 
randomize_rigid_body_mass,
apply_external_force_torque,
reset_joints_by_scale

)
from isaaclab.envs.mdp.rewards import undesired_contacts
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg 
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0,6.0 ),
        heading_control_stiffness=0.8,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8), 
            heading=(-1.6, 1.6)
        ),
        clips= parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip = 0.2,
            ang_vel_clip = 0.4
        )
    )

@configclass
class ParkourEventsCfg:
    """Command specifications for the MDP."""
    base_parkour = parkours.ParkourEventsCfg(
        asset_name = 'robot',
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
                "parkour_name": 'base_parkour',
                "history_length": 10,
                "body_name": "base_link"
            },
            clip=(-100, 100)
        )
    
    @configclass
    class TargetHeightPolicyCfg(ObsGroup):
        target_height = ObsTerm(
            func=observations.observation_target_height,
            params={
                "parkour_name": 'base_parkour',
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "base_link",
            },
            clip=(0.22, 0.3)
        )
    
    policy: PolicyCfg = PolicyCfg()
    target_height: TargetHeightPolicyCfg = TargetHeightPolicyCfg()

@configclass
class StudentObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg":SceneEntityCfg("robot"),
                "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
                "parkour_name":'base_parkour',
                "history_length": 10,
            },
            clip=(-100, 100)
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
        deta_yaw_ok =  ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={            
            "parkour_name":'base_parkour',
            'threshold': 0.6
            },
        )
    
    @configclass
    class TargetHeightPolicyCfg(ObsGroup):
        target_height = ObsTerm(
            func=observations.observation_target_height,
            params={
                "parkour_name": 'base_parkour',
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "base_link",
            },
            clip=(0.22, 0.3)
        )
    
    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()
    target_height: TargetHeightPolicyCfg = TargetHeightPolicyCfg()


@configclass
class StudentRewardsCfg:
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-0., 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link",".*Knee_link"]),
        },
    )
    

@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base_link',
    'LF_HipA_link',
    'RF_HipA_link',
    'LH_HipA_link',
    'RH_HipA_link',
    'LF_HipF_link',
    'RF_HipF_link',
    'LH_HipF_link',
    'RH_HipF_link',
    'LF_Knee_link',
    'RF_Knee_link',
    'LH_Knee_link',
    'RH_Knee_link']
    """
# Available Body strings: 
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-4., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base_link", ".*_HipA_link"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg(name="robot", body_names=[".*_Knee_link"]),
            "sensor_cfg":SceneEntityCfg(name="contact_forces", body_names=".*_Knee_link"),
            "parkour_name":'base_parkour',
            "body_name": "base_link",
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques, 
        weight=-0.00001, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error, 
        weight=-0.02, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.3, 
        params={
            "asset_cfg":SceneEntityCfg("robot", joint_names=[".*_HipA_joint"]),
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy, 
        weight=-0.05, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate, 
        weight=-0.1, 
        params={
          "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-1.0e-6, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z, 
        weight=-1.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation, 
        weight=-0.6, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-0.12, 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=1.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_target_height = RewTerm(
        func=rewards.reward_target_height,
        weight=0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": 'base_parkour',
            "body_name": 'base_link',
        },
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques, 
        weight=-1.0e-7, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    # reward_symmetric_contact_time = RewTerm(
    #     func=rewards.reward_symmetric_contact_time,
    #     weight=0.05,
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
    #         # foot_pairs: [[左脚名, 右脚名], ...]
    #         "foot_pairs": [
    #             ["LF_Knee_link", "RF_Knee_link"],  # 前腿对称
    #             ["LH_Knee_link", "RH_Knee_link"],  # 后腿对称
    #         ],
    #         "window_size": 1000,
    #     },
    # )
    reward_foot_no_contact_time = RewTerm(
        func=rewards.reward_foot_no_contact_time,
        weight=-0.2,
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "contact_force_threshold": 0.5,  # 接地力阈值（N）
            "max_no_contact_steps": 250,  # 允许的最大连续未接地步数（约3秒）
        },
    )
    # reward_dual_contact_at_high_speed = RewTerm(
    #     func=rewards.reward_dual_contact_at_high_speed,
    #     weight=-0.3,  # 负权重表示惩罚
    #     params={
    #         "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
    #         "asset_cfg":SceneEntityCfg("robot"),
    #         # foot_pairs: [[脚1, 脚2], ...] 指定需要检查同时接地的脚对
    #         "foot_pairs": [
    #             ["LF_Knee_link", "RF_Knee_link"],  # 前腿左右配对
    #             ["LH_Knee_link", "RH_Knee_link"],  # 后腿左右配对
    #         ],
    #         "speed_threshold": 0.4,  # 速度阈值（m/s）
    #         "contact_force_threshold": 1.0,  # 接地力阈值（N）
    #     },
    # )
    # joint_mirror = RewTerm(
    #     func=rewards.joint_mirror,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mirror_joints": [
    #             ["LF_HipA_joint", "RF_HipA_joint"],
    #             ["LH_HipA_joint", "RH_HipA_joint"],
    #         ],
    #     },
    # )
    gait_reward = RewTerm(
        func=rewards.GaitReward,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Knee_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.3,
            "command_threshold": 0.2,
            "synced_feet_pair_names": [
                ["LF_Knee_link", "RH_Knee_link"],  # 对角线配对（trot步态）
                ["RF_Knee_link", "LH_Knee_link"],  # 对角线配对（trot步态）
            ],
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,
        params= {
            "asset_cfg":SceneEntityCfg("robot")
        },
    )
    
@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func= events.reset_root_state,
        params = {'offset': 3.},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func= reset_joints_by_scale, 
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
        is_global_time= True, 
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
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
        action_delay_steps = [1, 1],
        delay_update_global_steps = 24 * 8000,
        history_length = 8,
        use_delay = True,
        clip = {'.*': (-4.8,4.8)}
        )
