# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING, dataclass

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.metrics.door_moved_rate import DoorMovedRateMetric
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


class OpenDoorTask(ManaTask):
    """Staged open-door task.

    ``__init__`` keeps cfg-only values. The originating
    ``TaskGenerator`` binds symbolic slots to scene assets, then calls
    :meth:`finalize` to attach the resolved object and build runtime
    event / termination cfgs.
    """

    def __init__(self, cfg: "OpenDoorTaskCFG"):
        super().__init__(cfg=cfg)
        self.openable_object = None
        self.target_joint_percentage_threshold = cfg.openness_threshold
        self.reset_joint_percentage = cfg.reset_openness
        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = None

    @staticmethod
    def _require_openable(ref) -> Openable:
        if isinstance(ref, Openable):
            return ref
        raise TypeError(
            "OpenDoorTask requires a resolved Openable for cfg.openable_object, got "
            f"{type(ref).__name__}: {ref!r}. Resolve symbolic references in the "
            "originating TaskGenerator.finalize stage before constructing the task."
        )

    def finalize(self, scene=None, robot=None):
        del scene, robot
        self.openable_object = self._require_openable(self.cfg.openable_object)
        self.cfg.openable_object = self.openable_object
        self.events_cfg = OpenDoorEventCfg(
            self.openable_object,
            reset_openable_object_revolute_joint_percentage=self.reset_joint_percentage,
        )
        self.termination_cfg = self.make_termination_cfg()
        return self

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_open,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        if self.cfg is not None and self.cfg.prompt:
            return self.cfg.prompt
        if isinstance(self.openable_object, Openable):
            return f"Open {self.openable_object.name}."
        return "Open the target object."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return OpenDoorMimicEnvCfg(
            embodiment_name=embodiment_name,
            openable_object_name=self.openable_object.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            DoorMovedRateMetric(
                self.openable_object,
                reset_openness=self.reset_joint_percentage,
            ),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))


@dataclass
class OpenDoorTaskCFG(TaskCFG):
    class_type: type = OpenDoorTask
    openable_object: Openable | str = "@Openable"
    openness_threshold: float | None = None
    reset_openness: float | None = None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING


@configclass
class OpenDoorEventCfg:
    """Configuration for Open Door."""

    reset_openable_object_revolute_joint_percentage: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openable_object_revolute_joint_percentage: float | None):
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openable_object_revolute_joint_percentage is not None:
            params["percentage"] = reset_openable_object_revolute_joint_percentage
        self.reset_openable_object_revolute_joint_percentage = EventTermCfg(
            func=openable_object.rotate_revolute_joint,
            mode="reset",
            params=params,
        )


@configclass
class OpenDoorMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Open Door env.
    """

    embodiment_name: str = "franka"

    openable_object_name: str = "openable_object"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_opendoor_isaac_lab_task_D0"
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
                object_ref=self.openable_object_name,
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
                object_ref=self.openable_object_name,
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
                    object_ref=self.openable_object_name,
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
