# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ArmMode(str, Enum):
    """Arm mode used by Arena mimic task templates."""

    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"
    LEFT = "left"
    RIGHT = "right"

    def get_other_arm(self) -> "ArmMode":
        assert self in [ArmMode.LEFT, ArmMode.RIGHT], f"Arm mode {self} is not a bimanual arm mode"
        return ArmMode.RIGHT if self == ArmMode.LEFT else ArmMode.LEFT
