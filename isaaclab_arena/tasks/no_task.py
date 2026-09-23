# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.common import ViewerCfg

from isaaclab_arena.tasks.task_base import ManaTask


class NoTask(ManaTask):
    """Null object for environments without a task."""

    name = "no_task"

    def __init__(self):
        super().__init__()

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, embodiment_name: str):
        return None

    def get_metrics(self):
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))
