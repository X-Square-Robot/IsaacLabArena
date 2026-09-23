# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena_gr00t.policy.config.lerobot_replay_action_policy_config import LerobotReplayActionPolicyConfig
from isaaclab_arena_gr00t.policy.replay_lerobot_action_policy import ReplayLerobotActionPolicy
from isaaclab_arena_mana.policy import ManaPolicyHelperCFG, as_config


class ArenaReplayLerobotActionPolicy(ReplayLerobotActionPolicy):
    """Alias wrapper for replay Lerobot action policy in mana module."""

    def __init__(self, cfg: ManaPolicyHelperCFG, num_envs: int = 1, device: str = "cuda", trajectory_index: int = 0):
        policy_config = as_config(cfg.translate(), LerobotReplayActionPolicyConfig)
        args = ReplayLerobotActionPolicy.config_class(
            policy_config_yaml_path="",
            policy_device=device,
            num_envs=num_envs,
            trajectory_index=trajectory_index,
        )
        super().__init__(args, policy_config=policy_config)
