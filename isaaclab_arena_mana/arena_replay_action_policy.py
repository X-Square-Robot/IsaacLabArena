# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy, ReplayActionPolicyArgs
from isaaclab_arena_mana.policy import ManaPolicyHelperCFG, as_config


class ArenaReplayActionPolicy(ReplayActionPolicy):
    """Alias wrapper for replay action policy in mana module."""

    def __init__(self, cfg: ManaPolicyHelperCFG):
        super().__init__(as_config(cfg.translate(), ReplayActionPolicyArgs))
