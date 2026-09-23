# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import (
    get_unnormalized_joint_position,
    normalize_value,
    set_unnormalized_joint_position,
)


class Turnable(AffordanceBase):
    """Interface for turnable objects with discrete levels."""

    def __init__(
        self,
        turnable_joint_name: str,
        min_level_angle_deg: float,
        max_level_angle_deg: float,
        num_levels: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.turnable_joint_name = turnable_joint_name
        assert min_level_angle_deg >= 0.0, "min_level_angle must be non-negative"
        assert min_level_angle_deg < max_level_angle_deg, "min_level_angle must be less than max_level_angle"
        self.min_level_angle_rad = min_level_angle_deg * math.pi / 180.0
        self.max_level_angle_rad = max_level_angle_deg * math.pi / 180.0
        self.num_levels = num_levels
        assert self.num_levels >= 1, "num_levels must be at least 1"

    def get_turning_level(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        """Get the current turning level in range ``[-1, num_levels - 1]``."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        theta = get_unnormalized_joint_position(env, asset_cfg)

        in_dead_zone = theta < self.min_level_angle_rad
        theta_clamped = torch.clamp(theta, self.min_level_angle_rad, self.max_level_angle_rad)
        normalized_position = normalize_value(theta_clamped, self.min_level_angle_rad, self.max_level_angle_rad)
        level = torch.floor(normalized_position * (self.num_levels - 1))
        level = torch.where(in_dead_zone, torch.full_like(level, -1.0), level)
        return level.to(int)

    def turn_to_level(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        target_level: int = -1,
    ):
        """Set the turning level of the object in all requested environments."""
        assert (
            target_level >= -1 and target_level < self.num_levels
        ), f"target_level must be between -1 and {self.num_levels - 1}"

        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        target_level = int(target_level)
        if target_level == -1:
            theta = 0.0
        else:
            step_size = (self.max_level_angle_rad - self.min_level_angle_rad) / (self.num_levels - 1)
            theta = self.min_level_angle_rad + step_size * target_level + step_size / 2.0

        set_unnormalized_joint_position(env, asset_cfg, theta, env_ids)

    def is_at_level(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg | None = None,
        target_level: int = -1,
    ) -> torch.Tensor:
        """Check if the object is at the requested discrete level."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        current_level = self.get_turning_level(env, asset_cfg)
        return torch.eq(current_level, int(target_level))

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.turnable_joint_name]
        return asset_cfg
