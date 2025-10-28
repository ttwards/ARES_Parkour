from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

import math
from dataclasses import MISSING
from .uniform_parkour_command import UniformParkourCommand
from .target_height_command import UniformTargetHeightCommand

@configclass
class ParkourCommandCfg(CommandTermCfg):
    class_type: type = UniformParkourCommand
    asset_name: str = MISSING
    heading_control_stiffness: float = 1.0
    small_commands_to_zero: bool = True 

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = MISSING

    @configclass 
    class Clips:
        lin_vel_clip: float = MISSING 
        ang_vel_clip: float = MISSING 

    ranges: Ranges = MISSING
    clips: Clips = MISSING
    
    # 根据地形类型调整速度范围（可选）
    # 格式: {"terrain_name": {"lin_vel_x": (min, max), "heading": (min, max)}}
    terrain_ranges_map: dict[str, dict] | None = None
    # 指定parkour_manager中的哪个term提供地形信息
    parkour_term_name: str | None = None

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class TargetHeightCommandCfg(CommandTermCfg):
    """Configuration for a 1D target height command.

    height_range: (min, max) height in meters.
    terrain_height_map: optional mapping from terrain name to (min, max).
    parkour_term_name: name of the parkour term that exposes per-env terrain names.
    """
    class_type: type = UniformTargetHeightCommand
    asset_name: str = "robot"
    height_range: tuple[float, float] = (0.24, 0.32)
    # 可选: 根据地形类型为不同地形设置不同的高度范围
    terrain_height_map: dict[str, tuple[float, float]] | None = None
    # 指定parkour_manager中的哪个term提供地形信息
    parkour_term_name: str | None = None
