# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import get_normalized_joint_position, set_normalized_joint_position


class Openable(AffordanceBase):
    """Interface for openable objects."""

    def __init__(
        self,
        openable_joint_name: str,
        openable_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.openable_joint_name = openable_joint_name
        self.openable_threshold = openable_threshold

    def get_openness(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        """Returns the percentage open that the object is."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        return get_normalized_joint_position(env, asset_cfg)

    def is_open(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is open."""
        openable_threshold = threshold if threshold is not None else self.openable_threshold
        openness = self.get_openness(env, asset_cfg)
        return openness > openable_threshold

    def is_closed(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is closed."""
        openable_threshold = threshold if threshold is not None else self.openable_threshold
        openness = self.get_openness(env, asset_cfg)
        return openness <= openable_threshold

    def rotate_revolute_joint(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 0.0,
    ):
        """Rotate the revolute joint of the object to the given percentage."""
        assert percentage >= 0.0 and percentage <= 1.0, "Percentage must be between 0.0 and 1.0"
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        set_normalized_joint_position(env, asset_cfg, percentage, env_ids)

    def open(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 1.0,
    ):
        """Open the object in all requested environments."""
        self.rotate_revolute_joint(env, env_ids, asset_cfg, percentage)

    def close(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 0.0,
    ):
        """Close the object in all requested environments."""
        self.rotate_revolute_joint(env, env_ids, asset_cfg, percentage)

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.openable_joint_name]
        return asset_cfg
