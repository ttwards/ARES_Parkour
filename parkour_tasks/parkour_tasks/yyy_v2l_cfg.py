from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.ares_yyy_v2 import ARES_YYY2L_CFG  # isort: skip
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from parkour_isaaclab.terrains.parkour_terrain_importer import ParkourTerrainImporter
from parkour_tasks.extreme_parkour_task.config.go2 import agents
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.envs import ViewerCfg
import os, torch
from parkour_isaaclab.actuators.parkour_actuator_cfg import ParkourDCMotorCfg


def quat_from_euler_xyz_tuple(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> tuple:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # compute quaternion
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    convert = torch.stack([qw, qx, qy, qz], dim=-1) * torch.tensor([1.,1.,1.,-1])
    return tuple(convert.numpy().tolist())


@configclass
class ParkourDefaultSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ARES_YYY2L_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    terrain = TerrainImporterCfg(
        class_type=ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=None,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = False
        self.robot.actuators['base_legs'] = ParkourDCMotorCfg(
            joint_names_expr=[".*_HipA_joint", ".*_HipF_joint", ".*_Knee_joint"],
            effort_limit={
                '.*_HipA_joint': 17,
                '.*_HipF_joint': 17,
                '.*_Knee_joint': 25,
            },
            saturation_effort={
                '.*_HipA_joint': 17,
                '.*_HipF_joint': 17,
                '.*_Knee_joint': 25,
            },
            peak_torque_speed={
                '.*_HipA_joint': 8.4,
                '.*_HipF_joint': 8.4,
                '.*_Knee_joint': 3.7,
            },
            velocity_limit={
                '.*_HipA_joint': 22.0,
                '.*_HipF_joint': 22.0,
                '.*_Knee_joint': 13.0,
            },
            stiffness=30.0,
            damping=0.5,
            friction=0.0,
        )
        # actuators={
        #     "base_legs": DCMotorCfg(
        #         joint_names_expr=[".*_HipA_joint", ".*_HipF_joint"],
        #         effort_limit=17,
        #         saturation_effort=17,
        #         peak_torque_speed=8.4,
        #         velocity_limit=22.0,
        #         stiffness=25.0,
        #         damping=0.5,
        #         friction=0.0,
        #     ),
        #     "double_legs": DCMotorCfg(
        #         joint_names_expr=[".*_Knee_joint"],
        #         effort_limit=34,
        #         saturation_effort=34,
        #         peak_torque_speed=4.1,
        #         velocity_limit=11.0,
        #         stiffness=25.0,
        #         damping=0.5,
        #         friction=0.0,
        #     ),
        # },


# we are now using a raycaster based camera, not a pinhole camera. see tail issue https://github.com/isaac-sim/IsaacLab/issues/719
CAMERA_CFG = RayCasterCameraCfg(
    prim_path='{ENV_REGEX_NS}/Robot/base_link',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.24, 0.0, 0.0),
        rot=quat_from_euler_xyz_tuple(*tuple(torch.tensor([0, 0, 0]))),
        convention="ros"
    ),
    depth_clipping_behavior='max',
    pattern_cfg=PinholeCameraPatternCfg(
        focal_length=11.041,
        horizontal_aperture=20.955,
        vertical_aperture=12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance=2.,
)

CAMERA_USD_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link/d435",
    spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(agents.__path__[0], 'd435.usd')),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.22, 0.002, 0.023),
        rot=quat_from_euler_xyz_tuple(*tuple(torch.tensor([0, 0, 0]))),
    )
)
VIEWER = ViewerCfg(
    eye=(-0., 2.6, 1.6),
    asset_name="robot",
    origin_type='asset_root',
)
