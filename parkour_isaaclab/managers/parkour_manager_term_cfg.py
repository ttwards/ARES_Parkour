"""Custom term configurations for parkour tasks."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

@configclass
class ParkourRewardTermCfg(RewardTermCfg):
    """Configuration for reward terms with terrain-based weight adjustment.

    This extends the standard RewardTermCfg to support dynamic weight adjustment
    based on terrain types in parkour environments.

    Attributes:
        terrain_weight_map: Optional mapping of terrain names to weight multipliers.
            Format: {"terrain_name": multiplier, ...}
            Example: {"parkour_flat": 0.5, "parkour_gap": 2.0}
    """

    terrain_weight_map: dict[str, float] | None = None


if TYPE_CHECKING:
    from .parkour_manager import ParkourTerm

@configclass
class ParkourTermCfg:

    class_type: type[ParkourTerm] = MISSING

    debug_vis: bool = False
