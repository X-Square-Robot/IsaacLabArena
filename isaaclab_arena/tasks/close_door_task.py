# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.tasks.open_door_task import OpenDoorMimicEnvCfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask


class CloseDoorTask(RotateRevoluteJointTask):
    def __init__(
        self,
        openable_object: Openable,
        closedness_threshold: float | None = None,
        reset_openness: float = 1.0,
        episode_length_s: float | None = None,
        task_description: str | None = None,
    ):
        super().__init__(
            openable_object=openable_object,
            target_joint_percentage_threshold=closedness_threshold,
            reset_joint_percentage=reset_openness,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )
        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Reach out to the {openable_object.name} and close it." if task_description is None else task_description
        )

    def make_termination_cfg(self):
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        success = TerminationTermCfg(
            func=self.openable_object.is_closed,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_mimic_env_cfg(self, embodiment_name: str):
        return OpenDoorMimicEnvCfg(
            embodiment_name=embodiment_name,
            openable_object_name=self.openable_object.name,
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
