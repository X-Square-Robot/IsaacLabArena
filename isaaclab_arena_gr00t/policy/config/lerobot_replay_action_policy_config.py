# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path

from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode


@dataclass
class LerobotReplayActionPolicyConfig:
    """Configuration for replaying recorded actions from a LeRobot dataset."""

    dataset_path: str = field(default="", metadata={"description": "Full path to the LeRobot dataset directory."})
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions fetched per chunk from the dataset."}
    )
    embodiment_tag: str = field(
        default="NEW_EMBODIMENT",
        metadata={
            "description": (
                "Identifier for the robot embodiment used for joint name mapping and modality config lookup"
                " (e.g. 'GR1' or 'NEW_EMBODIMENT')."
            )
        },
    )
    video_backend: str = field(
        default="torchcodec", metadata={"description": "Video decoding backend for the LeRobot dataset loader."}
    )
    modality_config_path: str = field(
        default="",
        metadata={
            "description": (
                "Optional path to a python module registering modality configs in GR00T's MODALITY_CONFIGS"
                " registry. Leave empty for embodiments whose configs are pre-registered."
            )
        },
    )
    policy_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    task_mode_name: str = field(
        default=TaskMode.G1_LOCOMANIPULATION.value,
        metadata={"description": "Task option name of the policy inference."},
    )
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": "Path to the YAML file specifying the joint ordering configuration for the sim action space."
        },
    )
    action_chunk_length: int = field(
        default=1,  # Replay actions from every recorded timestamp in the dataset
        metadata={"description": "Number of actions to execute per fetched chunk (can be less than action_horizon)."},
    )

    def __post_init__(self):
        assert (
            self.action_chunk_length <= self.action_horizon
        ), "action_chunk_length must be less than or equal to action_horizon"
        assert Path(
            self.policy_joints_config_path
        ).exists(), f"policy_joints_config_path does not exist: {self.policy_joints_config_path}"
        assert Path(
            self.action_joints_config_path
        ).exists(), f"action_joints_config_path does not exist: {self.action_joints_config_path}"
        if self.modality_config_path:
            assert Path(
                self.modality_config_path
            ).exists(), f"modality_config_path does not exist: {self.modality_config_path}"
        # The LeRobot episode loader does not take relative paths
        self.dataset_path = str(Path(self.dataset_path).resolve())
        assert Path(self.dataset_path).exists(), f"dataset_path does not exist: {self.dataset_path}"
        # embodiment_tag (normalize case so e.g. 'gr1' and 'GR1' are both accepted)
        self.embodiment_tag = self.embodiment_tag.upper()
        # task_mode
        self.task_mode = TaskMode(self.task_mode_name)
