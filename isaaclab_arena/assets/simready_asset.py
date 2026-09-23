# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: This module requires the `omni.simready.explorer` extension. It is imported
# lazily by `AssetRegistry._try_get_simready_asset_builder()`, which enables the
# extension first. Do not import this module before the extension is enabled.
import omni.simready.explorer as sre

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

simready_to_arena = {
    sre.AssetType.PROP: ObjectType.RIGID,
    # TODO: alignment types, add more types
}


class GeneralSimreadyAsset(Object):
    """
    A general SimReady asset, wrapping an asset found by the SimReady Explorer.
    """

    def __init__(
        self,
        simready_asset: sre.SimreadyAsset,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            simready_asset.name,
            usd_path=simready_asset.main_url,
            tags=simready_asset.tags,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )
