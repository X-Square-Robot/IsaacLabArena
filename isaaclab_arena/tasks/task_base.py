# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.relations.relations import RequiresReachability


@dataclass
class TaskCFG:
    class_type: type["ManaTask"] | None = None
    episode_length_s: float | None = None
    prompt: str = ""
    runtime_cfgs: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    definition: dict[str, Any] = field(default_factory=dict)
    # Optional task-relationship graph (manaenv ``ManaGraph``), held opaquely as
    # ``Any`` so this vendored arena module keeps zero dependency on manaenv.
    # ``None`` for atomic / legacy tasks; populated for graph-composed tasks.
    # Consumed only by the manaenv scheduler / GNN / phase projection — never by
    # the Arena env builder, which reads only the ``get_*`` getters.
    graph: Any = None


class TaskBase:
    """Arena's runtime task base class.

    Exposes the final runtime getters consumed by the Arena environment
    builder, all of which read from ``cfg.runtime_cfgs`` by default.
    Concrete subclasses (e.g. ``PickAndPlaceTask``) may override these
    getters to surface task-specific cfg objects. All lifecycle staging
    (compile / instancing / finalize) is owned by the originating
    ``TaskGenerator`` and ``TaskFactory.finalize_task``.

    ``ManaTask`` (defined below) is the manaenv canonical task type that
    extends this Arena contract with the optional task-relationship graph.
    """

    DEFAULT_EPISODE_LENGTH_S: float = 20.0

    def __init__(self, cfg: TaskCFG | None = None, episode_length_s: float | None = None):
        self.cfg: TaskCFG = cfg or TaskCFG(episode_length_s=episode_length_s)

    @property
    def episode_length_s(self) -> float | None:
        return self.cfg.episode_length_s

    @episode_length_s.setter
    def episode_length_s(self, value: float | None) -> None:
        self.cfg.episode_length_s = value

    @property
    def runtime_cfgs(self) -> dict[str, Any]:
        return self.cfg.runtime_cfgs

    @runtime_cfgs.setter
    def runtime_cfgs(self, value: dict[str, Any]) -> None:
        self.cfg.runtime_cfgs = value

    @property
    def roles(self) -> dict[str, Any]:
        return self.cfg.roles

    @roles.setter
    def roles(self, value: dict[str, Any]) -> None:
        self.cfg.roles = value

    @property
    def definition(self) -> dict[str, Any]:
        return self.cfg.definition

    @definition.setter
    def definition(self, value: dict[str, Any]) -> None:
        self.cfg.definition = value

    @property
    def task_type(self) -> str:
        return self.cfg.definition.get("task_type", "")

    @property
    def scene_constraints(self) -> Any:
        return self.cfg.runtime_cfgs.get("scene_constraints", [])

    @property
    def robot_constraints(self) -> Any:
        return self.cfg.definition.get("robot_constraints", [])

    def get_scene_cfg(self) -> Any:
        return self.runtime_cfgs.get("scene_cfg")

    def get_termination_cfg(self) -> Any:
        return self.runtime_cfgs.get("termination_terms")

    def get_events_cfg(self) -> Any:
        return self.runtime_cfgs.get("event_terms")

    def get_prompt(self) -> str:
        if self.cfg is not None and self.cfg.prompt:
            return self.cfg.prompt
        return self.runtime_cfgs.get("instruction", "")

    def get_task_description(self) -> str | None:
        # env_builder feeds this into env_cfg.task_description (policy language instruction); reuse the prompt source.
        return self.get_prompt() or None

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return self.runtime_cfgs.get("mimic_env_cfg")

    def get_metrics(self) -> list[MetricBase]:
        return self.runtime_cfgs.get("metrics", [])

    def get_progress_objectives(self) -> list[ProgressObjective]:
        # env_builder calls this unconditionally; atomic tasks default to none.
        # CompositeTaskBase overrides to concatenate child objectives.
        return self.runtime_cfgs.get("progress_objectives", [])

    def get_success_term_cfg(self) -> Any:
        return self.runtime_cfgs.get("success_term_cfg")

    def get_evaluation_term_cfg(self) -> Any:
        return self.runtime_cfgs.get("evaluation_term_cfg")

    def get_observation_cfg(self) -> Any:
        return self.runtime_cfgs.get("observation_cfg")

    def get_rewards_cfg(self) -> Any:
        return self.runtime_cfgs.get("rewards_cfg")

    def get_curriculum_cfg(self) -> Any:
        return self.runtime_cfgs.get("curriculum_cfg")

    def get_commands_cfg(self) -> Any:
        return self.runtime_cfgs.get("commands_cfg")

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg:
        return self.runtime_cfgs.get("recorder_term_cfg")

    def get_domain_randomization_cfg(self) -> Any:
        return self.runtime_cfgs.get("domain_randomization_cfg")

    def get_default_visualizer_cfg(self) -> Any | None:
        """Return a materialized native visualizer cfg when one is available."""
        visualizer_cfg = self.runtime_cfgs.get("default_visualizer_cfg")
        return None if isinstance(visualizer_cfg, dict) else visualizer_cfg

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        terms = self.runtime_cfgs.get("viewer_cfg")
        if not terms:
            return ViewerCfg()
        if isinstance(terms, dict) and isinstance(terms.get("terms"), dict):
            return ViewerCfg()
        if isinstance(terms, dict):
            for _name, cfg in terms.items():
                if isinstance(cfg, ViewerCfg):
                    return cfg
        return ViewerCfg()

    def get_episode_length_s(self) -> float:
        # Default at the getter (not in cfg): resolve_episode_length still needs the
        # raw None to derive length from max_steps_per_episode when unset.
        return self.episode_length_s if self.episode_length_s is not None else self.DEFAULT_EPISODE_LENGTH_S

    def resolve_episode_length(self, env_cfg) -> None:
        max_steps = self.definition.get("max_steps_per_episode")
        if max_steps is None:
            return
        if self.episode_length_s is not None:
            return
        env_cfg.episode_length_s = float(max_steps) * env_cfg.sim.dt * env_cfg.decimation

    def apply_reachability_constraints(self) -> None:
        """Stamp RequiresReachability on the objects the robot must be able to reach for this task."""
        pass

    def _apply_reachability_constraints(self, targets: list[Asset]) -> None:
        """Stamp RequiresReachability on each placed object in targets.

        An object reference or a background location is a static, non-placed asset and is skipped;
        only placed objects are IK-checked during layout validation.
        """
        for target in targets:
            if isinstance(target, Object) and not target.has_relation(RequiresReachability):
                target.add_relation(RequiresReachability())


class ManaTask(TaskBase):
    """Manaenv canonical task type: the Arena ``TaskBase`` runtime contract
    plus the optional task-relationship graph (``ManaGraph``) consumed by the
    manaenv scheduler / GNN / phase projection.

    Every manaenv task variant extends this class so manaenv-side
    ``isinstance(task, ManaTask)`` checks hold. The Arena environment builder
    only reads the ``TaskBase`` getters and never touches ``graph``.
    """

    @property
    def graph(self) -> Any:
        """Optional task-relationship graph (manaenv ``ManaGraph`` or ``None``)."""
        return getattr(self.cfg, "graph", None)

    @graph.setter
    def graph(self, value: Any) -> None:
        self.cfg.graph = value


TaskCFG.class_type = ManaTask
