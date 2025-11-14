
from __future__ import annotations

import numpy as np
import random
import scipy.interpolate as interpolate
import trimesh
from typing import TYPE_CHECKING, Tuple
from ..utils import parkour_field_to_mesh
if TYPE_CHECKING:
    from . import extreme_parkour_terrains_cfg

"""
Reference from https://arxiv.org/pdf/2309.14341
"""

def padding_height_field_raw(
    height_field_raw:np.ndarray, 
    cfg:extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg
    )->np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw

def random_uniform_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
    ):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled 
    return height_field_raw 

@parkour_field_to_mesh
def parkour_gap_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourGapTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        mid_y = length_pixels // 2  # length is actually y width
        gap_size = eval(cfg.gap_size,{"difficulty":difficulty})
        gap_size = round(gap_size / cfg.horizontal_scale)

        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale) + gap_size
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) + gap_size

        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        gap_depth = -round(np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1]) / cfg.vertical_scale)
        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len - 1, mid_y]
        dis_x = platform_len
        last_dis_x = dis_x
        for i in range(num_goals - 2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x += rand_x
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if not cfg.apply_flat:
                height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        # 记录通道两侧的高度突变，用于生成 edge mask
        edge_mask = np.zeros_like(height_field_raw, dtype=bool)
        min_height_step = 1  # 由于高度已经离散化到网格单位，1格即可视为台阶
        height_diff_y = np.diff(height_field_raw, axis=1)
        edge_mask[:, :-1] |= height_diff_y <= -min_height_step
        edge_mask[:, 1:] |= height_diff_y >= min_height_step
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale, edge_mask

@parkour_field_to_mesh
def parkour_hurdle_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = round(stone_len / cfg.horizontal_scale)

        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
        hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
        hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
        hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        dis_x = platform_len
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height

        goals[0] = [platform_len - 1, mid_y]

        for i in range(num_goals-2):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            dis_x += rand_x
            if not cfg.apply_flat:
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, ] = np.random.randint(hurdle_height_min, hurdle_height_max)
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, :mid_y+rand_y-half_valid_width] = 0
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, mid_y+rand_y+half_valid_width:] = 0
            goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y]
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_step_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourStepTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_height = eval(cfg.step_height,{'difficulty':difficulty} )
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))

        mid_y = length_pixels // 2  # length is actually y width
        dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
        dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) 
        dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

        step_height = round(step_height / cfg.vertical_scale)

        half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height

        dis_x = platform_len
        last_dis_x = dis_x
        stair_height = 0
        goals = np.zeros((num_goals, 2))
        goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]
        goal_heights = np.ones((num_goals)) * platform_height

        num_stones = num_goals - 2
        for i in range(num_stones):
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            rand_y = np.random.randint(dis_y_min, dis_y_max)
            if i < num_stones // 2:
                stair_height += step_height
            elif i > num_stones // 2:
                stair_height -= step_height
            height_field_raw[dis_x:dis_x+rand_x, ] = stair_height
            dis_x += rand_x
            height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = 0
            height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = 0
            
            last_dis_x = dis_x
            goals[i+1] = [dis_x-rand_x//2, mid_y+rand_y]
            goal_heights[i+1] = stair_height
        final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
        # import ipdb; ipdb.set_trace()
        if final_dis_x > width_pixels:
            final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights 

@parkour_field_to_mesh
def parkour_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        height_field_raw = np.zeros((width_pixels, length_pixels))
        height_field_raw[:] = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
        mid_y = length_pixels // 2  # length is actually y width
        stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
        stone_len = np.random.uniform(*stone_len)
        stone_len = 2 * round(stone_len / 2.0, 1)
        stone_len = round(stone_len / cfg.horizontal_scale)
        x_range = eval(cfg.x_range, {"difficulty": difficulty})
        y_range = eval(cfg.y_range, {"difficulty": difficulty})
        dis_x_min = stone_len + round(x_range[0] / cfg.horizontal_scale)
        dis_x_max = stone_len + round(x_range[1] / cfg.horizontal_scale)
        dis_y_min = round(y_range[0] / cfg.horizontal_scale)
        dis_y_max = round(y_range[1] / cfg.horizontal_scale)

        platform_len = round(cfg.platform_len / cfg.horizontal_scale)
        platform_height = round(cfg.platform_height / cfg.vertical_scale)
        height_field_raw[0:platform_len, :] = platform_height
        
        stone_width = round(cfg.stone_width / cfg.horizontal_scale)
        last_stone_len = round(cfg.last_stone_len / cfg.horizontal_scale)

        incline_height = eval(cfg.incline_height, {"difficulty": difficulty})
        last_incline_height = eval(cfg.last_incline_height, {"difficulty": difficulty, "incline_height":incline_height})
        last_incline_height = round(last_incline_height / cfg.vertical_scale)
        incline_height = round(incline_height / cfg.vertical_scale)

        dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
        goals = np.zeros((num_goals, 2))
        goal_heights = np.ones((num_goals)) * platform_height
        goals[0] = [platform_len -  stone_len // 2, mid_y]
        left_right_flag = np.random.randint(0, 2)
        dis_z = 0
        num_stones = num_goals - 2
        for i in range(num_stones):
            dis_x += np.random.randint(dis_x_min, dis_x_max)
            pos_neg = round(2*(left_right_flag - 0.5))
            dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
            if i == num_stones - 1:
                dis_x += last_stone_len // 4
                heights = np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1)) * pos_neg
                height_field_raw[dis_x-last_stone_len//2:dis_x+last_stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            else:
                heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
                height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2, dis_y-stone_width//2: dis_y+stone_width//2] = heights.astype(int) + dis_z
            
            goals[i+1] = [dis_x, dis_y]
            goal_heights[i+1] = np.mean(heights.astype(int))

            left_right_flag = 1 - left_right_flag
        final_dis_x = dis_x + 2*np.random.randint(dis_x_min, dis_x_max)
        final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // cfg.horizontal_scale)
        height_field_raw[final_platform_start:, :] = platform_height
        goals[-1] = [final_dis_x, mid_y]
        height_field_raw = padding_height_field_raw(height_field_raw,cfg)
        if cfg.apply_roughness:
            height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
        
        return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_demo_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourDemoTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals = np.zeros((num_goals, 2))
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    mid_y = length_pixels // 2  # length is actually y width

    height_field_raw = np.zeros((width_pixels, length_pixels))
    goal_heights = np.ones((num_goals)) * round(cfg.platform_height / cfg.vertical_scale)
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+hurdle_depth, round(mid_y-hurdle_width/2):round(mid_y+hurdle_width/2)] = hurdle_height

    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / cfg.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[1] = [platform_length+first_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+first_step_depth, round(mid_y-first_step_width/2):round(mid_y+first_step_width/2)] = first_step_height
    goal_heights[1] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length+second_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+second_step_depth, round(mid_y-second_step_width/2):round(mid_y+second_step_width/2)] = second_step_height
    goal_heights[2] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)
    
    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[3] = [platform_length+third_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+third_step_depth, round(mid_y-third_step_width/2):round(mid_y+third_step_width/2)] = third_step_height
    goal_heights[3] = third_step_height
    
    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length+forth_step_depth/2, mid_y]
    height_field_raw[platform_length:platform_length+forth_step_depth, round(mid_y-forth_step_width/2):round(mid_y+forth_step_width/2)] = forth_step_height
    goal_heights[4] = forth_step_height
    
    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size
    
    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    
    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)
    
    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    goals[5] = [platform_length+slope_depth/2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    height_field_raw[platform_length:platform_length+slope_depth, left_y-slope_width//2: left_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[5] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size
    goals[6] = [platform_length+slope_depth/2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    height_field_raw[platform_length:platform_length+slope_depth, right_y-slope_width//2: right_y+slope_width//2] = heights.astype(int) + platform_height
    goal_heights[6] = np.mean(heights.astype(int) + platform_height)
    
    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)
    goals[-1] = [platform_length, left_y]

    height_field_raw = padding_height_field_raw(height_field_raw,cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_wall_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourWallTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建墙跳地形 - 机器人需要跳过垂直的墙
    Wall jumping terrain - robot needs to jump over vertical walls
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    mid_y = length_pixels // 2
    
    # 解析配置参数
    wall_thickness = eval(cfg.wall_thickness, {"difficulty": difficulty})
    wall_thickness = round(wall_thickness / cfg.horizontal_scale)
    
    wall_height_range = eval(cfg.wall_height_range, {"difficulty": difficulty})
    wall_height_min = round(wall_height_range[0] / cfg.vertical_scale)
    wall_height_max = round(wall_height_range[1] / cfg.vertical_scale)
    
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)
    
    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    
    # 初始平台
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    
    dis_x = platform_len
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - 1, mid_y]
    
    # 生成多个墙
    for i in range(num_goals - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        
        # 墙前的平台
        height_field_raw[dis_x:dis_x+rand_x-wall_thickness, :] = platform_height
        
        # 创建墙
        wall_height = np.random.randint(wall_height_min, wall_height_max)
        wall_start = dis_x + rand_x - wall_thickness
        wall_end = dis_x + rand_x
        
        # 整个墙的高度（先设置为墙高）
        height_field_raw[wall_start:wall_end, :] = wall_height + platform_height
        
        if cfg.allow_bypass:
            # 允许绕过：在墙上留出可通过的区域（中间有通道）
            height_field_raw[wall_start:wall_end, :mid_y+rand_y-half_valid_width] = platform_height
            height_field_raw[wall_start:wall_end, mid_y+rand_y+half_valid_width:] = platform_height
        else:
            # 不允许绕过：墙占据整个宽度
            if cfg.add_side_pits:
                # 在墙两侧添加深坑，防止机器人从边缘绕过
                pit_depth_value = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
                # 墙前的两侧区域变成深坑
                height_field_raw[dis_x:wall_start, :mid_y+rand_y-half_valid_width] = pit_depth_value
                height_field_raw[dis_x:wall_start, mid_y+rand_y+half_valid_width:] = pit_depth_value
                # 墙后的两侧区域也是深坑
                height_field_raw[wall_end:dis_x+rand_x+round(0.5/cfg.horizontal_scale), :mid_y+rand_y-half_valid_width] = pit_depth_value
                height_field_raw[wall_end:dis_x+rand_x+round(0.5/cfg.horizontal_scale), mid_y+rand_y+half_valid_width:] = pit_depth_value
            # 否则墙就是全宽的，机器人必须跳过
        
        dis_x += rand_x
        # 目标点设置在墙后面
        goals[i+1] = [dis_x + wall_thickness//2, mid_y + rand_y]
    
    # 最后的平台
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - round(0.5 / cfg.horizontal_scale)
    
    height_field_raw[dis_x:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    
    # 添加边界和粗糙度
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


@parkour_field_to_mesh
def parkour_slope_terrain(
    difficulty: float, 
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourSlopeTerrainCfg,
    num_goals: int, 
    )->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建斜坡地形 - 机器人横向穿越斜坡，始终保持一边高一边低
    Slope terrain - robot traverses across a tilted surface, always with one side higher than the other
    
    斜坡沿Y轴方向倾斜，机器人沿X轴方向前进
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    mid_y = length_pixels // 2
    
    # 解析配置参数
    slope_angle = eval(cfg.slope_angle, {"difficulty": difficulty})
    slope_length = eval(cfg.slope_length, {"difficulty": difficulty})
    
    # 计算斜坡参数
    # 斜坡角度转换为弧度
    slope_angle_rad = np.radians(slope_angle)
    
    # 计算斜坡的高度差（从一边到另一边）
    # 地形的总宽度是 cfg.size[1]
    total_width = cfg.size[1]
    height_difference = total_width * np.tan(slope_angle_rad)
    
    # 转换为像素单位
    slope_length_pixels = round(slope_length / cfg.horizontal_scale)
    height_diff_pixels = round(height_difference / cfg.vertical_scale)
    
    # 平台参数
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    
    # 创建起始平台（水平）
    height_field_raw[0:platform_len, :] = platform_height
    
    # X方向的目标点间距
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    
    # 初始化目标点数组
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - 1, mid_y]
    
    # 计算斜坡的起始和结束位置
    slope_start_x = platform_len
    slope_end_x = min(slope_start_x + slope_length_pixels, width_pixels - platform_len)
    
    # 创建斜坡区域
    # 沿Y轴方向创建高度梯度（从一边低到另一边高）
    for x in range(slope_start_x, slope_end_x):
        for y in range(length_pixels):
            # 计算该点相对于中心线的Y偏移
            y_offset_from_center = y - mid_y
            # 根据Y位置计算高度（线性渐变）
            # y = 0 (最低), y = length_pixels (最高)
            slope_height = platform_height + (y / length_pixels) * height_diff_pixels - height_diff_pixels / 2
            height_field_raw[x, y] = slope_height
    
    # 创建结束平台（与斜坡中心高度相同）
    height_field_raw[slope_end_x:, :] = platform_height
    
    # 在斜坡区域分布目标点
    dis_x = slope_start_x
    for i in range(1, num_goals - 1):
        if dis_x < slope_end_x:
            rand_x = np.random.randint(dis_x_min, dis_x_max)
            dis_x = min(dis_x + rand_x, slope_end_x - 1)
            
            # 目标点设置在中心线上
            goals[i] = [dis_x, mid_y]
            # 目标高度是该X位置的中心Y位置的高度
            goal_heights[i] = height_field_raw[dis_x, mid_y]
        else:
            break
    
    # 最后一个目标点在结束平台上
    final_dis_x = slope_end_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - round(0.5 / cfg.horizontal_scale)
    goals[-1] = [final_dis_x, mid_y]
    goal_heights[-1] = platform_height
    
    # 添加边界
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    
    # 可选：添加粗糙度（但这会破坏斜坡的平滑性，建议关闭）
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale


def create_box_mesh(center, size):
    """
    创建一个长方体mesh
    center: [x, y, z] 中心位置
    size: [width, depth, height] 尺寸 (x, y, z方向)
    """
    w, d, h = size
    cx, cy, cz = center
    
    # 定义8个顶点
    vertices = np.array([
        [cx - w/2, cy - d/2, cz - h/2],  # 0: 左下后
        [cx + w/2, cy - d/2, cz - h/2],  # 1: 右下后
        [cx + w/2, cy + d/2, cz - h/2],  # 2: 右下前
        [cx - w/2, cy + d/2, cz - h/2],  # 3: 左下前
        [cx - w/2, cy - d/2, cz + h/2],  # 4: 左上后
        [cx + w/2, cy - d/2, cz + h/2],  # 5: 右上后
        [cx + w/2, cy + d/2, cz + h/2],  # 6: 右上前
        [cx - w/2, cy + d/2, cz + h/2],  # 7: 左上前
    ])
    
    # 定义12个三角形面（每个面2个三角形）
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 7, 6], [4, 6, 5],  # 顶面
        [0, 4, 5], [0, 5, 1],  # 后面
        [2, 6, 7], [2, 7, 3],  # 前面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 5, 6], [1, 6, 2],  # 右面
    ])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def parkour_hurdle_terrain_trimesh(
    difficulty: float,
    cfg,  # extreme_parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg
    num_goals: int,
) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建跨栏地形 - 直接生成trimesh
    柱子和横杆都是正方形横截面
    
    返回:
        meshes: trimesh对象列表
        origin: 原点坐标
        goals: 目标点坐标
        goal_heights: 目标点高度
    """
    width = cfg.size[0]
    length = cfg.size[1]
    mid_y = length / 2
    
    # 解析配置参数 - 所有横截面都是正方形
    pole_size = eval(cfg.pole_size, {"difficulty": difficulty})  # 柱子边长
    bar_size = eval(cfg.bar_size, {"difficulty": difficulty})    # 横杆边长
    
    hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
    hurdle_height = np.random.uniform(hurdle_height_range[0], hurdle_height_range[1])
    
    dis_x_min = cfg.x_range[0]
    dis_x_max = cfg.x_range[1]
    dis_y_min = cfg.y_range[0]
    dis_y_max = cfg.y_range[1]
    
    half_valid_width = np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1])
    
    platform_len = cfg.platform_len
    platform_height = cfg.platform_height
    
    # 初始化
    meshes = []
    dis_x = platform_len
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones(num_goals) * platform_height
    goals[0] = [platform_len - 1, mid_y]
    
    # 创建地面平台
    ground_mesh = create_box_mesh(
        center=[width/2, length/2, platform_height/2],
        size=[width, length, platform_height]
    )
    meshes.append(ground_mesh)
    
    # 生成多个跨栏
    for i in range(num_goals - 2):
        rand_x = np.random.uniform(dis_x_min, dis_x_max)
        rand_y = np.random.uniform(dis_y_min, dis_y_max)
        dis_x += rand_x
        
        hurdle_center_y = mid_y + rand_y
        
        # 左侧柱子 - 正方形横截面
        left_pole_y = hurdle_center_y - half_valid_width - pole_size/2
        left_pole_center = [dis_x, left_pole_y, platform_height + hurdle_height/2]
        left_pole_size = [pole_size, pole_size, hurdle_height]  # 正方形横截面
        left_pole_mesh = create_box_mesh(left_pole_center, left_pole_size)
        meshes.append(left_pole_mesh)
        
        # 右侧柱子 - 正方形横截面
        right_pole_y = hurdle_center_y + half_valid_width + pole_size/2
        right_pole_center = [dis_x, right_pole_y, platform_height + hurdle_height/2]
        right_pole_size = [pole_size, pole_size, hurdle_height]  # 正方形横截面
        right_pole_mesh = create_box_mesh(right_pole_center, right_pole_size)
        meshes.append(right_pole_mesh)
        
        # 横杆 - 正方形横截面（长条形）
        bar_center_z = platform_height + hurdle_height
        bar_length = right_pole_y - left_pole_y  # 两柱子之间的距离
        bar_center = [dis_x, hurdle_center_y, bar_center_z]
        bar_mesh_size = [bar_size, bar_length, bar_size]  # 正方形横截面，沿Y方向延伸
        bar_mesh = create_box_mesh(bar_center, bar_mesh_size)
        meshes.append(bar_mesh)
        
        # 设置目标点（在跨栏中间的地面）
        goals[i + 1] = [dis_x - rand_x/2, hurdle_center_y]
        goal_heights[i + 1] = platform_height
    
    # 最终目标
    final_dis_x = dis_x + np.random.uniform(dis_x_min, dis_x_max)
    if final_dis_x > width:
        final_dis_x = width - 0.5
    goals[-1] = [final_dis_x, mid_y]
    goal_heights[-1] = platform_height
    
    # 调整goals坐标系（相对于中心）
    goals -= np.array([width/2, length/2])
    
    # 计算origin
    origin = np.array([width/2, length/2, platform_height])
    
    # 创建 x_edge_mask (用于trimesh不需要边缘检测，返回全False数组)
    width_pixels = int(width / cfg.horizontal_scale) + 1
    length_pixels = int(length / cfg.horizontal_scale) + 1
    x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=bool)
    
    return meshes, origin, goals, goal_heights, x_edge_mask
