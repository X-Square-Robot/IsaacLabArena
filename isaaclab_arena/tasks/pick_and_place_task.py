# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING, dataclass
from functools import partial

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.object_settling import objects_settled
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height, object_on_destination
from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


class PickAndPlaceTask(ManaTask):
    """Staged pick-and-place task.

    ``__init__`` keeps cfg-only values. The originating
    ``TaskGenerator`` binds symbolic slots against the scene, then
    calls :meth:`finalize` to attach the resolved assets and build
    scene / event / termination runtime cfgs.
    """

    def __init__(self, cfg: "PickAndPlaceTaskCFG"):
        super().__init__(cfg=cfg)
        self.pick_up_object = None
        self.destination_location = None
        self.background_scene = None
        self.contact_sensor_name = None
        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = None

    def finalize(self, scene=None, robot=None):
        del scene, robot
        self.pick_up_object = self._require_asset(self.cfg.pick_up_object, "pick_up_object")
        self.destination_location = self._require_asset(self.cfg.destination_location, "destination_location")
        self.background_scene = self._require_asset(self.cfg.background_scene, "background_scene")
        self.cfg.pick_up_object = self.pick_up_object
        self.cfg.destination_location = self.destination_location
        self.cfg.background_scene = self.background_scene
        self.contact_sensor_name = f"contact_sensor_{self.pick_up_object.name}"
        self.scene_config = self.make_scene_cfg()
        self.events_cfg = EventsCfg(pick_up_object=self.pick_up_object)
        self.termination_cfg = self._make_termination_cfg()
        return self

    @staticmethod
    def _require_asset(ref: Asset | str | None, field_name: str) -> Asset:
        if isinstance(ref, Asset):
            return ref
        raise TypeError(
            f"PickAndPlaceTask requires a resolved Asset for {field_name}, got {type(ref).__name__}: {ref!r}. "
            "Resolve symbolic references in the originating TaskGenerator.finalize stage before constructing the task."
        )

    def apply_reachability_constraints(self) -> None:
        """The robot must reach the object it picks up and the location it places onto."""
        self._apply_reachability_constraints([self.pick_up_object, self.destination_location])

    def make_scene_cfg(self):
        contact_sensor_cfg = self.pick_up_object.get_contact_sensor_cfg(
            contact_against_object=self.destination_location,
        )
        scene_cfg_type = make_configclass(
            "SceneCfg",
            [(self.contact_sensor_name, type(contact_sensor_cfg), contact_sensor_cfg)],
        )
        return scene_cfg_type()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def _make_termination_cfg(self):
        success = TerminationTermCfg(
            func=object_on_destination,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "contact_sensor_cfg": SceneEntityCfg(self.contact_sensor_name),
                "force_threshold": self.cfg.force_threshold,
                "velocity_threshold": self.cfg.velocity_threshold,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        if self.cfg is not None and self.cfg.prompt:
            return self.cfg.prompt
        if isinstance(self.pick_up_object, Asset) and isinstance(self.destination_location, Asset):
            return f"Pick up {self.pick_up_object.name} and place it at {self.destination_location.name}."
        return "Pick up the object and place it at the destination."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return PickPlaceMimicEnvCfg(
            embodiment_name=embodiment_name,
            pick_up_object_name=self.pick_up_object.name,
            destination_location_name=self.destination_location.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

    def get_progress_objectives(self) -> list[ProgressObjective]:
        return [
            ProgressObjective(
                name="pick_and_place",
                predicate_groups=[
                    partial(
                        objects_settled,
                        object_names=[self.pick_up_object.name],
                    ),
                    partial(
                        object_is_above_height,
                        object_name=self.pick_up_object.name,
                        use_settled_state=True,
                    ),
                    partial(
                        object_on_destination,
                        object_cfg=SceneEntityCfg(self.pick_up_object.name),
                        contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
                        force_threshold=self.cfg.force_threshold,
                        velocity_threshold=self.cfg.velocity_threshold,
                    ),
                ],
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@dataclass
class PickAndPlaceTaskCFG(TaskCFG):
    class_type: type = PickAndPlaceTask
    pick_up_object: Asset | str = "@pickable"
    destination_location: Asset | str = "@destination"
    background_scene: Asset | str = "@background"
    force_threshold: float = 1.0
    velocity_threshold: float = 0.1


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """Configuration for Pick and Place."""

    reset_pick_up_object_pose: EventTermCfg = MISSING

    def __init__(self, pick_up_object: Asset):
        initial_pose = pick_up_object.get_initial_pose()
        if initial_pose is not None:
            self.reset_pick_up_object_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(pick_up_object.name),
                },
            )
        else:
            print(
                f"Pick up object {pick_up_object.name} has no initial pose. Not setting reset pick up object pose"
                " event."
            )
            self.reset_pick_up_object_pose = None


@configclass
class PickPlaceMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Pick and Place env.
    """

    embodiment_name: str = "franka"

    pick_up_object_name: str = "pick_up_object"

    destination_location_name: str = "destination_location"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_pickplace_isaac_lab_task_D0"
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

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.pick_up_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                # TODO(alexmillane, 2025.09.02): This is currently broken. FIX.
                # We need a way to pass in a reference to an object that exists in the
                # scene.
                object_ref=self.destination_location_name,
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        if self.embodiment_name == "franka":
            self.subtask_configs["robot"] = subtask_configs
        # We need to add the left and right subtasks for GR1.
        elif self.embodiment_name == "gr1_pink":
            self.subtask_configs["right"] = subtask_configs
            # EEF on opposite side (arm is static)
            subtask_configs = []
            subtask_configs.append(
                SubTaskConfig(
                    # Each subtask involves manipulation with respect to a single object frame.
                    object_ref=self.pick_up_object_name,
                    # Corresponding key for the binary indicator in "datagen_info" for completion
                    subtask_term_signal=None,
                    # Time offsets for data generation when splitting a trajectory
                    subtask_term_offset_range=(0, 0),
                    # Selection strategy for source subtask segment
                    selection_strategy="nearest_neighbor_object",
                    # Optional parameters for the selection strategy function
                    selection_strategy_kwargs={"nn_k": 3},
                    # Amount of action noise to apply during this subtask
                    action_noise=0.005,
                    # Number of interpolation steps to bridge to this subtask segment
                    num_interpolation_steps=0,
                    # Additional fixed steps for the robot to reach the necessary pose
                    num_fixed_steps=0,
                    # If True, apply action noise during the interpolation phase and execution
                    apply_noise_during_interpolation=False,
                )
            )
            self.subtask_configs["left"] = subtask_configs

        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
