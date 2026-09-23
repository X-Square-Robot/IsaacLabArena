# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import object_on_destination
from isaaclab_arena.tasks.task_base import ManaTask


class LocomanipPickAndPlaceTask(ManaTask):
    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        force_threshold: float = 1.0,
        velocity_threshold: float = 0.1,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object = pick_up_object
        self.destination_location = destination_location
        self.background_scene = background_scene
        self.force_threshold = force_threshold
        self.velocity_threshold = velocity_threshold
        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Pick up the {pick_up_object.name}, and place it into the {destination_location.name}"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        success = TerminationTermCfg(
            func=object_on_destination,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
                "force_threshold": self.force_threshold,
                "velocity_threshold": self.velocity_threshold,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(success=success, object_dropped=object_dropped)

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return LocomanipPickAndPlaceMimicEnvCfg(
            pick_up_object_name=self.pick_up_object.name,
            destination_name=self.destination_location.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class LocomanipPickAndPlaceMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic environment config class for locomanip pick-and-place."""

    pick_up_object_name: str = MISSING
    destination_name: str = MISSING

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = f"locomanip_pick_and_place_{self.pick_up_object_name}_to_{self.destination_name}_D0"
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

        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="idle_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="grasp_and_idle_right",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="idle_left",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="grasp_and_idle_left",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["body"] = [
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="navigate_to_table",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="navigate_turn_inplace",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                subtask_term_signal="navigate_to_bin",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
            SubTaskConfig(
                object_ref=self.pick_up_object_name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
