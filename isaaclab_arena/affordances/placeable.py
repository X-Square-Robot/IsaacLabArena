# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase


class Placeable(AffordanceBase):
    """Interface for objects that can be placed upright in a scene."""

    def __init__(
        self,
        upright_axis_name: Literal["x", "y", "z"] = "z",
        orientation_threshold: float = math.pi / 18.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert upright_axis_name in ["x", "y", "z"], "Upright axis must be one of x, y, or z"
        self.upright_axis_name = upright_axis_name
        self.orientation_threshold = orientation_threshold

    def is_placed_upright(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg | None = None,
        orientation_threshold: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return whether the object is within the upright orientation threshold."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        if orientation_threshold is None:
            orientation_threshold = self.orientation_threshold

        object_entity: RigidObject = env.scene[asset_cfg.name]
        object_quat = (object_entity.data.root_quat_w).torch
        upright_axis_world = get_object_axis_in_world_frame(object_quat, self.upright_axis_name)

        world_up = torch.zeros_like(upright_axis_world)
        world_up[..., 2] = 1.0

        cos_angle = torch.sum(upright_axis_world * world_up, dim=-1).clamp(-1.0, 1.0)
        angle_error = torch.acos(cos_angle)
        threshold = torch.as_tensor(orientation_threshold, device=object_quat.device, dtype=object_quat.dtype)
        return angle_error < threshold

    def place_upright(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        upright_percentage: float | torch.Tensor = 1.0,
    ):
        """Place the object upright in all requested environments."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        set_normalized_object_pose(
            env=env,
            asset_cfg=asset_cfg,
            upright_percentage=upright_percentage,
            env_ids=env_ids,
            upright_axis_name=self.upright_axis_name,
        )


def get_object_axis_in_world_frame(object_quat: torch.Tensor, upright_axis_name: str) -> torch.Tensor:
    """Get one local object axis in the world frame from WXYZ quaternions."""
    axis_index = {"x": 0, "y": 1, "z": 2}[upright_axis_name]
    rotation_mats = math_utils.matrix_from_quat(object_quat)
    return rotation_mats[:, :, axis_index]


def set_normalized_object_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    upright_percentage: float | torch.Tensor = 1.0,
    env_ids: torch.Tensor | None = None,
    upright_axis_name: str = "z",
) -> None:
    """Rotate a rigid object towards the upright orientation through its root pose."""
    object_entity: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    dtype = (object_entity.data.root_quat_w).torch.dtype

    if env_ids is not None:
        env_ids = env_ids.to(env.device)
    else:
        env_ids = torch.arange((object_entity.data.root_quat_w).torch.shape[0], device=env.device)

    if isinstance(upright_percentage, torch.Tensor):
        num_envs = len(env_ids)
        if upright_percentage.numel() != num_envs:
            raise ValueError(
                f"upright_percentage tensor must have {num_envs} elements to match env_ids, "
                f"but got {upright_percentage.numel()} elements"
            )

    object_quat = (object_entity.data.root_quat_w).torch[env_ids]
    object_pos = (object_entity.data.root_pos_w).torch[env_ids]
    target_quat = _compute_target_quaternions(
        object_quat=object_quat,
        upright_percentage=upright_percentage,
        upright_axis_name=upright_axis_name,
    )

    pose_tensor = torch.cat([object_pos, target_quat], dim=-1)
    object_entity.write_root_pose_to_sim_index(root_pose=pose_tensor, env_ids=env_ids)
    zero_velocity = torch.zeros((env_ids.numel(), 6), device=device, dtype=dtype)
    object_entity.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)


def _compute_target_quaternions(
    object_quat: torch.Tensor,
    upright_percentage: float | torch.Tensor,
    upright_axis_name: str,
) -> torch.Tensor:
    """Compute target WXYZ quaternions for a requested upright percentage."""
    upright_percentage_t = torch.as_tensor(upright_percentage, device=object_quat.device, dtype=object_quat.dtype)
    if upright_percentage_t.dim() == 0:
        upright_percentage_t = upright_percentage_t.expand(object_quat.shape[0])

    current_axis = get_object_axis_in_world_frame(object_quat, upright_axis_name)
    world_up = torch.zeros_like(current_axis)
    world_up[:, 2] = 1.0

    horizontal = current_axis - torch.sum(current_axis * world_up, dim=-1, keepdim=True) * world_up
    horizontal_norm = horizontal.norm(dim=-1, keepdim=True)
    fallback_axis = torch.zeros_like(horizontal)
    fallback_axis[:, 0] = 1.0

    needs_fallback = horizontal_norm.squeeze(-1) < 1e-6
    if needs_fallback.any():
        horizontal[needs_fallback] = fallback_axis[needs_fallback]
        horizontal_norm[needs_fallback] = horizontal[needs_fallback].norm(dim=-1, keepdim=True)
    horizontal_dir = horizontal / horizontal_norm.clamp(min=1e-6)

    target_angles = (1.0 - upright_percentage_t).view(-1, 1) * (math.pi / 2.0)
    target_axis = torch.sin(target_angles) * horizontal_dir + torch.cos(target_angles) * world_up
    target_axis = torch.nn.functional.normalize(target_axis, dim=-1)

    dot_product = torch.sum(current_axis * target_axis, dim=-1).clamp(-1.0, 1.0)
    rotation_angle = torch.acos(dot_product)
    rotation_axis = torch.cross(current_axis, target_axis, dim=-1)
    axis_norm = rotation_axis.norm(dim=-1, keepdim=True)

    needs_fallback = axis_norm.squeeze(-1) < 1e-6
    if needs_fallback.any():
        current_axis_fb = current_axis[needs_fallback]
        fallback_axis_fb = fallback_axis[needs_fallback]
        near_parallel_x = torch.abs(current_axis_fb[:, 0]) > 0.9
        if near_parallel_x.any():
            fallback_axis_fb[near_parallel_x] = torch.tensor(
                [0.0, 1.0, 0.0],
                device=object_quat.device,
                dtype=object_quat.dtype,
            )
        rotation_axis[needs_fallback] = torch.cross(current_axis_fb, fallback_axis_fb, dim=-1)
        axis_norm[needs_fallback] = rotation_axis[needs_fallback].norm(dim=-1, keepdim=True)

    rotation_axis_unit = rotation_axis / axis_norm.clamp(min=1e-6)
    align_quat = math_utils.quat_from_angle_axis(rotation_angle, rotation_axis_unit)
    new_quat = math_utils.quat_mul(align_quat, object_quat)
    return torch.nn.functional.normalize(new_quat, dim=-1)
