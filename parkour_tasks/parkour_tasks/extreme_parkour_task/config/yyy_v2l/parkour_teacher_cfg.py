from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, RayCasterCameraCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from isaaclab.utils import configclass
##
# Pre-defined configs
##
from parkour_isaaclab.terrains.extreme_parkour.config.parkour import (
    EXTREME_PARKOUR_TERRAINS_CFG,
    EXTREME_PARKOUR_TERRAINS_PLAY_CFG
)  # isort: skip
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from .parkour_mdp_cfg import * 
from parkour_tasks.yyy_v2l_cfg import ParkourDefaultSceneCfg, VIEWER
import torch
import math


def quat_from_euler_xyz_tuple(roll: float, pitch: float, yaw: float) -> tuple:
    roll_rad = torch.tensor(math.radians(roll), dtype=torch.float32)
    pitch_rad = torch.tensor(math.radians(pitch), dtype=torch.float32)
    yaw_rad = torch.tensor(math.radians(yaw), dtype=torch.float32)

    cy = torch.cos(yaw_rad * 0.5)
    sy = torch.sin(yaw_rad * 0.5)
    cr = torch.cos(roll_rad * 0.5)
    sr = torch.sin(roll_rad * 0.5)
    cp = torch.cos(pitch_rad * 0.5)
    sp = torch.sin(pitch_rad * 0.5)

    # 计算四元数 (ZYX 外旋, 等效于 XYZ 内旋)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # 按 (w, x, y, z) 顺序堆叠
    # 注意：您原始代码中乘以 [1, 1, 1, -1] 是一个特定的操作，
    # 这里予以保留，因为它可能用于特定的坐标系或约定。
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1., 1., 1., -1.])

    # 转换为元组并返回
    return tuple(convert.numpy().tolist())


@configclass
class ParkourTeacherSceneCfg(ParkourDefaultSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.25),
            rot=quat_from_euler_xyz_tuple(0, 0, 0),
        ),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    front_scanner = RayCasterCameraCfg(
        prim_path='{ENV_REGEX_NS}/Robot/base_link',
        data_types=["distance_to_camera"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.31505, 0.0175, 0.023),
            rot=quat_from_euler_xyz_tuple(*tuple(torch.tensor([0, 90, 0]))),
            convention="ros"
        ),
        depth_clipping_behavior='max',
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=11.041,
            horizontal_aperture=20.955,
            vertical_aperture=12.240,
            height=8,
            width=12,
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=1.,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis=False,
                                      force_threshold=1.
                                      )
    # 足端离地高度测量 RayCaster - 每个足端一个，世界坐标系下垂直向下发射光线
    foot_scanner_LF = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_Knee_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.112, 0.213, 0.0)),
        ray_alignment='yaw',  # 只跟随yaw，保持世界坐标系下垂直向下
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=[0.001, 0.001]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=1.0,
    )
    foot_scanner_RF = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_Knee_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.112, 0.213, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=[0.001, 0.001]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=1.0,
    )
    foot_scanner_LR = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LR_Knee_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.112, 0.213, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=[0.001, 0.001]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=1.0,
    )
    foot_scanner_RR = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_Knee_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.112, 0.213, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=[0.001, 0.001]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_CFG
        
@configclass
class AresYYYv2lTeacherParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    scene: ParkourTeacherSceneCfg = ParkourTeacherSceneCfg(num_envs=6144, env_spacing=1.)
    # Basic settings
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        # update sensor update periods
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_LF.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_RF.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_LR.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_RR.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True

@configclass
class AresYYYv2lTeacherParkourEnvCfg_EVAL(AresYYYv2lTeacherParkourEnvCfg):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0,1.0)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        self.events.push_by_setting_velocity.interval_range_s = (6.,6.)
        self.commands.base_velocity.resampling_time_range = (60.,60.)
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key ==['parkour','parkour_hurdle','parkour_step','parkour_gap']:
                sub_terrain.noise_range = (0.02, 0.02)
                sub_terrain.proportion = 0.25
                
@configclass
class AresYYYv2lTeacherParkourEnvCfg_PLAY(AresYYYv2lTeacherParkourEnvCfg_EVAL):
    viewer = VIEWER 

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 60.
        self.scene.num_envs = 16
        self.parkours.base_parkour.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        # 使用 PLAY 专用的地形配置
        self.scene.terrain.terrain_generator = EXTREME_PARKOUR_TERRAINS_PLAY_CFG
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.2, 1.0)
        self.events.push_by_setting_velocity = None


