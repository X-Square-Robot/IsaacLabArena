# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.placeable import Placeable
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import ManaTask
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class PlaceUprightTask(ManaTask):
    def __init__(
        self,
        placeable_object: Placeable,
        orientation_threshold: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(placeable_object, Placeable), "Placeable object must be an instance of Placeable"
        self.placeable_object = placeable_object
        self.orientation_threshold = (
            orientation_threshold if orientation_threshold is not None else placeable_object.orientation_threshold
        )
        self.scene_config = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Place the {placeable_object.name} upright" if task_description is None else task_description
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        params = {}
        if self.orientation_threshold is not None:
            params["orientation_threshold"] = self.orientation_threshold
        success = TerminationTermCfg(
            func=self.placeable_object.is_placed_upright,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return PlaceUprightMimicEnvCfg(
            embodiment_name=embodiment_name,
            placeable_object_name=self.placeable_object.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.placeable_object)]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.placeable_object, offset=np.array([1.5, 1.5, 1.5]))


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class PlaceUprightMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic environment config class for Place Upright env."""

    embodiment_name: str = "franka"
    placeable_object_name: str = "placeable_object"

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_placeupright_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
            SubTaskConfig(
                object_ref=self.placeable_object_name,
                subtask_term_signal="grasp",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.placeable_object_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        if self.embodiment_name == "franka":
            self.subtask_configs["robot"] = subtask_configs
        elif self.embodiment_name == "gr1_pink":
            self.subtask_configs["right"] = subtask_configs
            self.subtask_configs["left"] = [
                SubTaskConfig(
                    object_ref=self.placeable_object_name,
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ]
        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
