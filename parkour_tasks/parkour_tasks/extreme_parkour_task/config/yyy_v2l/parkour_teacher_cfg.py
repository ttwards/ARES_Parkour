
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
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
            pos=(-0.5, 0.0, 0.25),
            rot=quat_from_euler_xyz_tuple(0, 300, 0),
        ),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.125, size=[2.0, 1.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                      history_length=2, 
                                      track_air_time=True, 
                                      debug_vis=False,
                                      force_threshold=1.
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
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = False
        self.actions.joint_pos.history_length = 1
        self.events.random_camera_position = None

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
            self.scene.terrain.terrain_generator.difficulty_range = (0.7,1.0)
        self.events.push_by_setting_velocity = None


