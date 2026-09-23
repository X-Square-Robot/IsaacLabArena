# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID embodiment — Franka Panda + Robotiq 2F-85 gripper.

Ported from RoboLab ``robolab/robots/droid.py`` to work within the
IsaacLabArena ``EmbodimentBase`` framework. Uses the same robot USD,
joint PD gains, camera placement, and action/observation configs as
RoboLab so that Cosmos3 / DreamZero policies evaluate identically.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

_ASSET_ROOT = Path(__file__).resolve().parents[5] / "single_arm_assets"
_ROBOT_USD = str(_ASSET_ROOT / "robots" / "franka_robotiq_2f_85_flattened.usd")

EEF_OFFSET_POS = (0.0, 0.0, 0.0)
EEF_OFFSET_ROT = (0.5, -0.5, 0.5, -0.5)

_frame_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TF")
_frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)


# ── Observation helpers (match RoboLab droid.py exactly) ──────────────


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in joint_names]
    return (robot.data.joint_pos).torch[:, joint_indices]


def droid_gripper_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name == "finger_joint"]
    return (robot.data.joint_pos).torch[:, joint_indices] / (np.pi / 4)


def droid_ee_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("base_link")
    return (robot.data.body_pos_w).torch[:, body_idx, :] - env.scene.env_origins[:, 0:3]


def droid_ee_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    body_idx = robot.data.body_names.index("base_link")
    return (robot.data.body_quat_w).torch[:, body_idx, :]


def droid_eef_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return (frames.data.target_pos_w).torch[:, idx, :] - env.scene.env_origins[:, 0:3]


def droid_eef_quat(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("frames")):
    frames = env.scene[asset_cfg.name]
    idx = frames.data.target_frame_names.index("eef_frame")
    return (frames.data.target_quat_w).torch[:, idx, :]


# ── Custom gripper action (threshold > 0.5 = close) ─────────────────


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            binary_mask = actions == 0
        else:
            binary_mask = actions > 0.5
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    class_type = BinaryJointPositionZeroToOneAction


# ── Configs ──────────────────────────────────────────────────────────


DROID_ARTICULATION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(0, 0, 0, 1),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1 / 5 * np.pi,
            "panda_joint3": 0.0,
            "panda_joint4": -4 / 5 * np.pi,
            "panda_joint5": 0.0,
            "panda_joint6": 3 / 5 * np.pi,
            "panda_joint7": 0.0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=None,
            damping=None,
            velocity_limit=5.0,
        ),
    },
)


@configclass
class DroidSceneCfg:
    robot: ArticulationCfg = DROID_ARTICULATION_CFG

    frames: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_frame_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/panda_link{i}",
                name=f"panda_link{i}",
            )
            for i in range(8)
        ]
        + [
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                name="gripper_base",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link",
                name="eef_frame",
                offset=OffsetCfg(pos=EEF_OFFSET_POS, rot=EEF_OFFSET_ROT),
            ),
        ],
    )


@configclass
class DroidJointPositionActionCfg:
    body: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class DroidRelIKActionCfg:
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    finger_joint: ActionTermCfg = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class DroidObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=droid_gripper_pos)
        eef_pos = ObsTerm(func=droid_eef_pos)
        eef_quat = ObsTerm(func=droid_eef_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class DroidEventCfg:
    reset = EventTerm(func=mdp_isaac_lab.reset_scene_to_default, mode="reset")


# ── Embodiment ───────────────────────────────────────────────────────


@register_asset
class DroidEmbodiment(EmbodimentBase):
    """DROID embodiment — Franka Panda + Robotiq 2F-85, matching RoboLab."""

    name = "droid"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = DroidSceneCfg()
        self.action_config = DroidJointPositionActionCfg()
        self.observation_config = DroidObservationsCfg()
        self.event_config = DroidEventCfg()
        self.JointActionsCfg = DroidJointPositionActionCfg
        self.ActionsCfg = DroidRelIKActionCfg

    def get_recorder_term_cfg(self):
        """Record state/actions and camera observations for benchmark sidecars."""
        from manaenv.pipeline.robot import build_embodiment_recorder_cfg

        return build_embodiment_recorder_cfg(self.enable_cameras)
