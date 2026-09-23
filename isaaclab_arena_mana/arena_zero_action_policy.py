# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyArgs
from isaaclab_arena_mana.policy import ManaPolicyHelperCFG


class ArenaZeroActionPolicy(ZeroActionPolicy):
    """Alias wrapper for zero action policy in mana module."""

    def __init__(self, cfg: ManaPolicyHelperCFG):
        # ZeroActionPolicy takes no meaningful configuration; translate() is
        # still invoked so subclasses can validate/consume their raw payload.
        cfg.translate()
        super().__init__(ZeroActionPolicyArgs())
