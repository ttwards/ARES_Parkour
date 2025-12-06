from copy import deepcopy

from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import * 

EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.08,  # original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_gap": ExtremeParkourGapTerrainCfg(
            proportion=0.23,
            apply_roughness=True,
            x_range=(0.8, 1.5),
            half_valid_width=(0.6, 1.2),
            gap_width='0.18 + 0.22*difficulty',
            # height_drop_per_gap='0.2 - 0.5*difficulty',
            height_drop_per_gap='0.',
        ),
        "parkour_hurdle": ExtremeParkourHurdleTriMeshTerrainCfg(
            proportion=0.21,
            apply_roughness=False,
            pole_size='0.10 - 0.05 * difficulty',
            bar_size='0.10 - 0.05 * difficulty',
            hurdle_height_range='0.45 - 0.2 * difficulty, 0.4 - 0.2 * difficulty',
            x_range=(1.5, 2.5),
            y_range=(-0.3, 0.3),
            half_valid_width=(0.85, 1.15),
        ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.08,
            apply_roughness=False,
            apply_flat=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range='0.1+0.1*difficulty, 0.15+0.15*difficulty'
        ),
        # "parkour_rough": ExtremeParkourHurdleTerrainCfg(
        #     proportion=0.16,
        #     apply_roughness=True,
        #     apply_flat=True,
        #     x_range=(1.2, 2.2),
        #     half_valid_width=(0.4, 0.8),
        #     hurdle_height_range='0.1+0.1*difficulty, 0.15+0.15*difficulty'
        # ),
        "parkour_step": ExtremeParkourStepTerrainCfg(
            proportion=0.22,
            apply_roughness=True,
            x_range=(0.3, 1.5),
            half_valid_width=(0.5, 1),
            step_height='0.15 + 0.2*difficulty'
        ),
        # "parkour": ExtremeParkourTerrainCfg(
        #                 proportion=0.16,
        #                 apply_roughness=True,
        #                 x_range  = '-0.1, 0.1+0.3*difficulty',
        #                 y_range  = '0.2, 0.3+0.1*difficulty',
        #                 stone_len  = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty',
        #                 incline_height = '0.25*difficulty',
        #                 last_incline_height = 'incline_height + 0.1 - 0.1*difficulty'
        #                 ),
        # "parkour_demo": ExtremeParkourDemoTerrainCfg(
        #                 proportion=0.0,
        #                 apply_roughness=True,
        #                 ),
        "parkour_wall": ExtremeParkourWallTerrainCfg(
            proportion=0.21,  # 设为 0.2 来启用这个地形
            apply_roughness=False,
            x_range=(1.5, 2.0),
            half_valid_width=(0.4, 0.8),
            wall_thickness='0.12 - 0.1*difficulty',
            wall_height_range='0.12 + 0.24*difficulty, 0.17 + 0.24*difficulty'
        ),
        "parkour_slope": ExtremeParkourSlopeTerrainCfg(
            proportion=0.15,           # 比例你可以之后自己调
            apply_roughness=True,      # 可以关掉来纯看斜坡效果
            # 斜率范围（沿 y 方向的高度变化，相对“世界坐标”的 m/m）
            slope_range='-0.1 - 0.2 * difficulty, 0.1 + 0.2 * difficulty',
            # 每一段斜坡在 x 方向（机器人行走方向）的长度范围
            # segment_width_range='0.8 + 0.2 * difficulty, 1.6 + 0.4 * difficulty',
            segment_width_range='2.0, 2.0',
        ),
    },
)


# Copy base terrain settings and only tweak the arena scale/resolution for play mode.
EXTREME_PARKOUR_TERRAINS_PLAY_CFG = deepcopy(EXTREME_PARKOUR_TERRAINS_CFG)
EXTREME_PARKOUR_TERRAINS_PLAY_CFG.size = (16.0, 4.0)
EXTREME_PARKOUR_TERRAINS_PLAY_CFG.num_rows = 4
EXTREME_PARKOUR_TERRAINS_PLAY_CFG.num_cols = 8
