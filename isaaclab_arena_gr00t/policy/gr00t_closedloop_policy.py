# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T local closed-loop policy running in-process inference.

Counterpart of :mod:`isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy`
that loads the GR00T model in the current process (no policy server needed) and
shares the same observation/action translation pipeline from ``gr00t_core``.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import torch
from dataclasses import dataclass, field
from typing import Any

from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler, ActionScheduler, SyncedBatchActionScheduler
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg, TaskMode
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
    load_gr00t_policy_from_config,
)
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml, load_gr00t_modality_config_from_file


@dataclass
class Gr00tClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for Gr00tClosedloopPolicy (local in-process inference).

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs.
    The model checkpoint to load comes from ``model_path`` in the policy config
    YAML (see ``Gr00tClosedloopPolicyCfg``).
    """

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate"})

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tClosedloopPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
        )


class Gr00tClosedloopPolicy(PolicyBase):
    """GR00T closed-loop policy running model inference in the current process.

    Loads ``Gr00tPolicy`` from the checkpoint configured in the policy config
    YAML (``model_path``) and reuses the same observation/action translation as
    the remote policy.
    """

    name = "gr00t_closedloop"
    config_class = Gr00tClosedloopPolicyArgs

    def __init__(
        self,
        config: Gr00tClosedloopPolicyArgs,
        action_scheduler_cls: type[ActionScheduler] = ActionChunkScheduler,
        policy_config: Gr00tClosedloopPolicyCfg | None = None,
    ):
        super().__init__(config)

        # Policy config (obs/action translation + local model loading).
        # A pre-built config takes precedence over the YAML path; this lets
        # embedding frameworks construct the config programmatically.
        self.policy_config: Gr00tClosedloopPolicyCfg = policy_config or create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyCfg
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim from/to policy joint space remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(self.task_mode, self.robot_action_joints_config)
        self.action_chunk_length = self.policy_config.action_chunk_length

        self._chunking_state: ActionScheduler | None = action_scheduler_cls(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float,
        )

        # Load the GR00T model in-process
        self._policy = load_gr00t_policy_from_config(self.policy_config)

        self.task_description: str | None = None

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t Closedloop Policy",
            "Arguments for GR00T local closed-loop policy evaluation.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for policy inference and Arena-side tensor operations (default: cuda)",
        )
        group.add_argument(
            "--scheduler",
            type=str,
            default="chunk",
            choices=["chunk", "synced_batch"],
            help=(
                "Action scheduler: 'chunk' fetches a new chunk for any env that needs one;"
                " 'synced_batch' waits until ALL envs need a new chunk and then issues a single"
                " full-batch inference call (envs that finish early hold their current robot state)."
            ),
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tClosedloopPolicy:
        config = Gr00tClosedloopPolicyArgs.from_cli_args(args)
        scheduler_cls: type[ActionScheduler] = (
            SyncedBatchActionScheduler
            if getattr(args, "scheduler", "chunk") == "synced_batch"
            else ActionChunkScheduler
        )
        return Gr00tClosedloopPolicy(config, action_scheduler_cls=scheduler_cls)

    @classmethod
    def from_policy_config(
        cls,
        policy_config: Gr00tClosedloopPolicyCfg,
        num_envs: int = 1,
        policy_device: str = "cuda",
        action_scheduler_cls: type[ActionScheduler] = ActionChunkScheduler,
    ) -> Gr00tClosedloopPolicy:
        """Create the policy from an already-built ``Gr00tClosedloopPolicyCfg``.

        Entry point for embedding frameworks (e.g. manaenv) that assemble the
        policy config programmatically instead of loading it from a YAML file.
        """
        args = Gr00tClosedloopPolicyArgs(
            policy_config_yaml_path="",
            policy_device=policy_device,
            num_envs=num_envs,
        )
        return cls(args, action_scheduler_cls=action_scheduler_cls, policy_config=policy_config)

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._chunking_state is not None, "GR00T local policy has been closed"

        def fetch_chunk() -> torch.Tensor:
            return self._get_action_chunk(observation, self.policy_config.pov_cam_name_sim)

        return self._chunking_state.get_action(
            fetch_chunk,
            hold_action=self._extract_hold_action(observation),
        )

    def _extract_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Build the action vector that waiting envs should hold: their current sim joint positions
        copied into the action slots that share a joint name with the state config."""
        joint_pos_sim = observation["policy"]["robot_joint_pos"].to(device=self.device, dtype=torch.float)
        hold_action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float, device=self.device)
        for joint_name, action_idx in self.robot_action_joints_config.items():
            state_idx = self.robot_state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Public batched entry point: one forward pass returning the full action chunk.

        Used by external batch drivers (e.g. manaenv's global-sync executor)
        that schedule chunks themselves instead of going through get_action().
        """
        return self._get_action_chunk(observation, camera_names)

    def _get_action_chunk(
        self, observation: dict[str, Any], camera_names: list[str] | str = "robot_head_cam_rgb"
    ) -> torch.Tensor:
        """Get an action chunk from the in-process GR00T model."""
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        # 1. Same obs translation as the remote policy
        assert self.task_description is not None, "Task description is not set"
        assert self._policy is not None, "GR00T local policy has been closed"
        rgb_list_np, joint_pos_sim_np = extract_obs_numpy_from_torch(nested_obs=observation, camera_names=camera_names)
        policy_observations = build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self.robot_state_joints_config,
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
        )

        # 2. Run local inference
        robot_action_policy, _ = self._policy.get_action(policy_observations)

        # 3. Action translation from policy output to sim action tensor
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        assert self._policy is not None, "GR00T local policy has been closed"
        assert self._chunking_state is not None, "GR00T local policy has been closed"
        self._policy.reset()
        self._chunking_state.reset(env_ids)

    def close(self) -> None:
        """Release the in-process GR00T model and Arena-side resources."""
        self._policy = None
        self._chunking_state = None
        self.modality_configs = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
