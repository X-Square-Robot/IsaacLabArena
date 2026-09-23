# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.metrics.revolute_joint_moved_rate import RevoluteJointMovedRateMetric


class DoorMovedRateMetric(RevoluteJointMovedRateMetric):
    """Backward-compatible door moved metric backed by revolute-joint motion."""

    name = "door_moved_rate"
    recorder_term_name = "door_joint_state"

    def __init__(self, object: Openable, reset_openness: float | None, openness_delta_threshold: float = 0.05):
        super().__init__(
            object=object,
            reset_joint_percentage=0.0 if reset_openness is None else reset_openness,
            joint_percentage_delta_threshold=openness_delta_threshold,
        )
