# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg
from pxr import Usd

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections
from isaaclab_arena.utils.usd_helpers import get_prim_depth, is_articulation_root, is_rigid_body


def detect_object_type(
    usd_path: str | None = None,
    stage: Usd.Stage | None = None,
    variants: dict[str, str] | None = None,
) -> ObjectType:
    """Detect the object type of the asset

    Goes through the USD tree and detects the object type. The detection is based
    on the presence of a RigidBodyAPI or ArticulationRootAPI at the shallowest depth
    in which one of these APIs is present.

    When multiple physics roots coexist at the shallowest depth,
    ArticulationRootAPI takes priority over RigidBodyAPI (an articulation
    typically contains rigid-body children, so the presence of both is
    expected for articulated assets).

    Args:
        usd_path: The path to the USD file to inspect. Either this or stage must be provided.
        stage: The stage to inspect. Either this or usd_path must be provided.
        variants: USD variants to select before looking at the asset. SimReady props need
            ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        The object type of the asset.
    """
    assert usd_path is not None or stage is not None, "Either usd_path or stage must be provided"
    assert usd_path is None or stage is None, "Either usd_path or stage must be provided"
    if usd_path is not None:
        # Open a stage to inspect the USD.
        stage = Usd.Stage.Open(usd_path)
    if variants:
        apply_usd_variant_selections(stage, variants)
    # BFS through prims. Collect *all* physics-annotated prims at the
    # shallowest depth where at least one appears, then decide the type.
    open_prims = [stage.GetPseudoRoot()]
    found_depth = -1
    has_articulation = False
    has_rigid = False
    while len(open_prims) > 0:
        prim = open_prims.pop(0)
        depth = get_prim_depth(prim)
        # We already collected everything at the shallowest depth — stop.
        if found_depth >= 0 and depth > found_depth:
            break
        open_prims.extend(prim.GetChildren())
        if is_articulation_root(prim):
            has_articulation = True
            found_depth = depth
        elif is_rigid_body(prim):
            has_rigid = True
            found_depth = depth
    if not has_articulation and not has_rigid:
        return ObjectType.BASE
    # Articulation takes priority: an articulated asset will often have
    # both ArticulationRootAPI and RigidBodyAPI on sibling prims.
    if has_articulation:
        return ObjectType.ARTICULATION
    return ObjectType.RIGID


# Predefined rigid body property configurations for assembly tasks
# High iteration count for precision tasks (peg/hole insertion)
RIGID_BODY_PROPS_HIGH_PRECISION = PhysxRigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)

# Standard iteration count for gear mesh tasks
RIGID_BODY_PROPS_MEDIUM_PRECISION = PhysxRigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=32,
    max_contact_impulse=1e32,
)

# Initial state configuration for articulations without joints (e.g., rigid bodies treated as articulations).
# We explicitly set joint_pos and joint_vel to empty dicts to avoid the default pattern {".*": 0.0} in ArticulationCfg.InitialStateCfg,
# which would fail to match when there are no joints in the articulation.
EMPTY_ARTICULATION_INIT_STATE_CFG = ArticulationCfg.InitialStateCfg(
    joint_pos={},
    joint_vel={},
)
