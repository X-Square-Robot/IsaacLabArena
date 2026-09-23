# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay action policy backed by a LeRobot dataset.

Replays recorded actions from a LeRobot-format dataset episode by episode,
translating them to the sim action space with the same pipeline as the GR00T
closed-loop policies (``gr00t_core.build_gr00t_action_tensor``).
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.lerobot_replay_action_policy_config import LerobotReplayActionPolicyConfig
from isaaclab_arena_gr00t.policy.config.task_mode import TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import Gr00tBasePolicyArgs, build_gr00t_action_tensor, compute_action_dim
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
    load_robot_joints_config_from_yaml,
)


@dataclass
class ReplayLerobotActionPolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for ReplayLerobotActionPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs.
    """

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate"})
    trajectory_index: int = field(default=0, metadata={"help": "Dataset episode index to start replaying from"})

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> ReplayLerobotActionPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
            trajectory_index=getattr(args, "trajectory_index", 0),
        )


class ReplayLerobotActionPolicy(PolicyBase):
    """Replays actions recorded in a LeRobot dataset.

    Loads episodes through GR00T's ``LeRobotEpisodeLoader`` and replays the
    recorded action stream chunk by chunk, remapped to the sim action space.
    All envs replay the same trajectory (actions are tiled across envs).
    """

    name = "replay_lerobot"
    config_class = ReplayLerobotActionPolicyArgs

    def __init__(
        self,
        config: ReplayLerobotActionPolicyArgs,
        policy_config: LerobotReplayActionPolicyConfig | None = None,
    ):
        super().__init__(config)

        # A pre-built config takes precedence over the YAML path; this lets
        # embedding frameworks construct the config programmatically.
        self.policy_config: LerobotReplayActionPolicyConfig = policy_config or create_config_from_yaml(
            config.policy_config_yaml_path, LerobotReplayActionPolicyConfig
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (policy joint order -> sim action joint order)
        self.policy_joints_config = load_robot_joints_config_from_yaml(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = load_robot_joints_config_from_yaml(
            self.policy_config.action_joints_config_path
        )
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)

        # Action modality keys come from GR00T's modality config registry
        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )
        self._action_keys: list[str] = list(self.modality_configs["action"].modality_keys)

        # Episode loader over the action modality only (no video decoding needed)
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self._loader = LeRobotEpisodeLoader(
            dataset_path=self.policy_config.dataset_path,
            modality_configs={"action": self.modality_configs["action"]},
            video_backend=self.policy_config.video_backend,
        )
        assert len(self._loader) > 0, f"Dataset {self.policy_config.dataset_path} contains no episodes"

        # Replay cursor state
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.trajectory_index = config.trajectory_index
        self._reset_cursor(self.trajectory_index)

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Replay Lerobot Action Policy",
            "Arguments for replaying actions from a LeRobot dataset.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the LeRobot replay policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for Arena-side tensor operations (default: cuda)",
        )
        group.add_argument(
            "--trajectory_index",
            type=int,
            default=0,
            help="Dataset episode index to start replaying from",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> ReplayLerobotActionPolicy:
        config = ReplayLerobotActionPolicyArgs.from_cli_args(args)
        return ReplayLerobotActionPolicy(config)

    @classmethod
    def from_policy_config(
        cls,
        policy_config: LerobotReplayActionPolicyConfig,
        num_envs: int = 1,
        policy_device: str = "cuda",
        trajectory_index: int = 0,
    ) -> ReplayLerobotActionPolicy:
        """Create the policy from an already-built ``LerobotReplayActionPolicyConfig``.

        Entry point for embedding frameworks (e.g. manaenv) that assemble the
        policy config programmatically instead of loading it from a YAML file.
        """
        args = ReplayLerobotActionPolicyArgs(
            policy_config_yaml_path="",
            policy_device=policy_device,
            num_envs=num_envs,
            trajectory_index=trajectory_index,
        )
        return cls(args, policy_config=policy_config)

    # ---------------------- Trajectory cursor -------------------

    def _reset_cursor(self, trajectory_index: int) -> None:
        """Point the replay cursor at the start of ``trajectory_index``."""
        num_trajectories = len(self._loader)
        if trajectory_index >= num_trajectories:
            raise ValueError(f"Trajectory index {trajectory_index} exceeds available trajectories {num_trajectories}")
        self._episode_index = trajectory_index
        self._frame_index = 0
        self._episode_df = None
        self.current_action_chunk: torch.Tensor | None = None
        self.current_action_index = 0

    def _current_episode_df(self):
        """Lazily load the current episode's DataFrame, advancing past empty episodes."""
        while self._episode_df is None:
            assert self._episode_index < len(self._loader), (
                f"LeRobot dataset exhausted (episode {self._episode_index} of {len(self._loader)});"
                " no more actions to replay"
            )
            df = self._loader[self._episode_index]
            if len(df) > 0:
                self._episode_df = df
            else:
                self._episode_index += 1
        return self._episode_df

    def get_trajectory_length(self, trajectory_index: int) -> int:
        """Get the number of frames in one trajectory in the dataset."""
        assert trajectory_index < len(self._loader.episode_lengths)
        return int(self._loader.episode_lengths[trajectory_index])

    def set_trajectory_index(self, trajectory_index: int) -> None:
        """Set the policy to start replaying from a specific trajectory index."""
        self.trajectory_index = trajectory_index
        self._reset_cursor(trajectory_index)

    def get_trajectory_index(self) -> int:
        return self.trajectory_index

    # ---------------------- Policy interface -------------------

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Return the next replayed action. Shape: (num_envs, action_dim)."""
        if self.current_action_chunk is None:
            self.current_action_chunk = self.get_action_chunk()
            self.current_action_index = 0
            assert self.current_action_chunk.shape[1] >= self.action_chunk_length

        action = self.current_action_chunk[:, self.current_action_index]

        self.current_action_index += 1
        if self.current_action_index == self.action_chunk_length:
            self.current_action_chunk = None
            self.current_action_index = 0
        return action

    def get_action_chunk(self) -> torch.Tensor:
        """Fetch the next ``action_horizon`` recorded actions as a chunk.

        Reads frames from the current episode (padding by repeating the last
        frame when the episode is shorter than the horizon), advances the
        cursor by ``action_chunk_length`` frames, and rolls over to the next
        episode when the current one is exhausted.

        Returns:
            Action tensor of shape (num_envs, action_horizon, action_dim).
        """
        df = self._current_episode_df()
        horizon = self.policy_config.action_horizon
        num_frames = len(df)
        frame_indices = [min(self._frame_index + offset, num_frames - 1) for offset in range(horizon)]

        # Group name -> (num_envs, horizon, group_dim), same dict shape as GR00T policy output
        robot_action_policy: dict[str, np.ndarray] = {}
        for key in self._action_keys:
            frames = np.stack([np.asarray(df[f"action.{key}"].iloc[t], dtype=np.float64) for t in frame_indices])
            robot_action_policy[key] = np.tile(frames[None, ...], (self.num_envs, 1, 1))

        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        # Advance the cursor; roll over to the next episode at the end
        self._frame_index += self.action_chunk_length
        if self._frame_index >= num_frames:
            self._episode_index += 1
            self._frame_index = 0
            self._episode_df = None

        assert action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor.to(dtype=torch.float)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Rewind the replay to the start of the configured trajectory."""
        self._reset_cursor(self.trajectory_index)

    def close(self) -> None:
        """Release the dataset loader."""
        self._loader = None
        self._episode_df = None
        self.current_action_chunk = None
