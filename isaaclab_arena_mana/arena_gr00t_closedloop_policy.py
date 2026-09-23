# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import Gr00tClosedloopPolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_closedloop_policy import Gr00tClosedloopPolicy
from isaaclab_arena_mana.policy import ManaPolicyHelperCFG, as_config


class ArenaGr00tClosedloopPolicy(Gr00tClosedloopPolicy):
    """Alias wrapper for the local in-process GR00T closed-loop policy in mana module."""

    def __init__(self, cfg: ManaPolicyHelperCFG, num_envs: int = 1, device: str = "cuda"):
        policy_config = as_config(cfg.translate(), Gr00tClosedloopPolicyCfg)
        args = Gr00tClosedloopPolicy.config_class(
            policy_config_yaml_path="",
            policy_device=device,
            num_envs=num_envs,
        )
        super().__init__(args, policy_config=policy_config)
