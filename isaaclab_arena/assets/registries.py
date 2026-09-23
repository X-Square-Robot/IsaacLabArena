# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from isaaclab_arena.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from isaaclab.devices.device_base import DeviceCfg
    from isaaclab_teleop import IsaacTeleopCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.hdr_image import HDRImage
    from isaaclab_arena.assets.teleop_device_base import TeleopDeviceBase
    from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
    from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
    from isaaclab_arena.relations.relations import RelationBase
    from isaaclab_arena.tasks.task_base import TaskBase
    from isaaclab_arena.utils.pose import Pose


# Have to define all classes here in order to avoid circular import.
class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self._components = {}

    def register(self, component: Any, key: str | None = None):
        """Register an asset with a name.

        Args:
            key (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert key not in self._components, f"component {key} already registered"
        assert key is not None, "component name is not set"
        self._components[key] = component

    def is_registered(self, key: str, ensure_loaded: bool = True) -> bool:
        """Check whether a component is already registered under ``key``.

        Args:
            key: The name to look up.
            ensure_loaded: Whether to load every component before answering.

                Components register themselves lazily: nothing is in the registry until
                ``ensure_assets_registered()`` imports all the library modules. So a plain
                membership check can say "not registered" simply because the libraries
                haven't been imported yet. With ``ensure_loaded=True`` (the default) we
                import them first, so the answer reflects everything that exists.

                The ``register_*`` decorators pass ``False``. They run *while* those
                library modules are being imported, and all they need is to spot a
                duplicate key. Forcing a full load at that moment would re-enter the
                import that's already in progress and pull in the task/environment modules
                — which import Isaac Sim's ``pxr``/USD packages. If that happens during
                pytest collection (before ``SimulationApp()`` starts) the simulator
                segfaults, because those packages must be imported only after it starts.
        """
        if ensure_loaded and isinstance(self, REGISTRIES):
            ensure_registry_registered(type(self))
        return key in self._components

    def get_component_by_name(self, key: str) -> Any:
        """Get an component by name.

        Args:
            key (str): The name of the component.

        Returns:
            Any: The component.
        """
        if isinstance(self, REGISTRIES):
            ensure_registry_registered(type(self))
        assert key in self._components, f"component {key} not found, please check if requested component is registered"
        return self._components[key]

    def get_all_keys(self) -> list[str]:
        """Get all the keys of the components.

        Returns:
            list[str]: The list of keys.
        """
        if isinstance(self, REGISTRIES):
            ensure_registry_registered(type(self))
        return list(self._components.keys())


class AssetRegistry(Registry):

    def __init__(self):
        super().__init__()

    def get_asset_by_name(self, name: str) -> type["Asset"]:
        """Gets an asset by name.

        Falls back to a SimReady Explorer lookup when the name is not in the
        registry: if a SimReady asset matches, a builder callable constructing a
        ``GeneralSimreadyAsset`` is returned. Note that ``is_registered()`` keeps
        answering ``False`` for such names. If neither lookup matches, the usual
        "component not found" assertion from the registry is raised.

        Args:
            name (str): The name of the asset.
        """
        ensure_asset_providers_registered(include_embodiments=False)
        if name not in self._components:
            ensure_asset_providers_registered(include_embodiments=True)
        if name not in self._components:
            builder = self._try_get_simready_asset_builder(name)
            if builder is not None:
                return builder
        assert (
            name in self._components
        ), f"component {name} not found, please check if requested component is registered"
        return self._components[name]

    def get_assets_by_tag(self, tag: str | list[str]) -> list[type["Asset"]]:
        """Gets a list of assets matching the given tag(s).

        When *tag* is a single string it may contain comma-separated values
        (e.g. ``"glasses,rigid"``). When a list is passed each element is one
        required tag. An asset must have **all** requested tags to be included
        in the result.

        Args:
            tag: One tag, a comma-separated string, or a list of tags.

        Returns:
            list[Asset]: The list of matching assets.
        """
        ensure_asset_providers_registered(include_embodiments=True)
        if isinstance(tag, str):
            required = {t.strip() for t in tag.split(",")}
        else:
            required = set(tag)
        required.discard("")
        return [asset for asset in self._components.values() if asset.tags and required.issubset(asset.tags)]

    def get_assets_with_all_tags(self, tags: list[str]) -> list[str]:
        """Return asset names whose ``tags`` include every tag in ``tags``.

        Args:
            tags: Tags that must all be present on a candidate asset.
                When empty, every registered asset name is returned.

        Returns:
            Sorted asset names matching every tag, or all asset names when
            ``tags`` is empty.
        """
        ensure_asset_providers_registered(include_embodiments=True)
        return sorted(asset.name for asset in self._components.values() if all(tag in asset.tags for tag in tags))

    def get_random_asset_by_tag(self, tag: str) -> type["Asset"]:
        """Gets a random asset which has the given tag.

        Args:
            tag: One tag, a comma-separated string, or a list of tags.

        Returns:
            Asset: The random asset.
        """
        ensure_asset_providers_registered(include_embodiments=True)
        assets = self.get_assets_by_tag(tag)
        if len(assets) == 0:
            raise ValueError(f"No assets found with tag {tag}")
        return random.choice(assets)

    def _try_get_simready_asset_builder(self, name: str) -> Callable[..., Any] | None:
        """Try to resolve ``name`` against the SimReady Explorer asset database.

        Lazily enables the ``omni.simready.explorer`` extension (and its browser
        model) on first use, then runs an asynchronous asset search, pumping the
        Kit update loop until it completes.

        Args:
            name (str): The asset name to search for.

        Returns:
            A builder callable creating a ``GeneralSimreadyAsset`` for the first
            matching SimReady asset, or ``None`` if the name is empty or no
            SimReady asset matches.
        """
        if not name:
            return None
        import omni.kit.app

        app = omni.kit.app.get_app()
        ext_manager = app.get_extension_manager()
        if not ext_manager.is_extension_enabled("omni.simready.explorer"):
            ext_manager.set_extension_enabled_immediate("omni.simready.explorer", True)

        import omni.simready.explorer as sre

        if sre.get_instance().browser_model is None:
            import omni.kit.actions.core as actions

            actions.execute_action("omni.simready.explorer", "toggle_window")

        from isaaclab_arena.assets.simready_asset import GeneralSimreadyAsset

        search_task = asyncio.ensure_future(sre.find_assets(search_words=[name]))
        while not search_task.done():
            app.update()
        assets = search_task.result()
        if not assets:
            return None

        def builder(prim_path: str | None = None, initial_pose: "Pose | None" = None, **kwargs):
            return GeneralSimreadyAsset(
                assets[0],
                prim_path=prim_path,
                initial_pose=initial_pose,
                **kwargs,
            )

        return builder


class DeviceRegistry(Registry):

    def __init__(self):
        super().__init__()

    def get_device_by_name(self, name: str) -> type["TeleopDeviceBase"]:
        """Gets a device by name.

        Args:
            name (str): The name of the device.
        """
        ensure_registry_registered(DeviceRegistry)
        return self.get_component_by_name(name)

    def get_teleop_device_cfg(
        self, device: type["TeleopDeviceBase"], embodiment: object
    ) -> "DeviceCfg | IsaacTeleopCfg":
        retargeter_registry = RetargeterRegistry()
        retargeter_key = (device.name, embodiment.name)
        retargeter_key_str = retargeter_registry.convert_tuple_to_str(retargeter_key)
        retargeter = retargeter_registry.get_component_by_name(retargeter_key_str)()
        pipeline_builder = retargeter.get_pipeline_builder(embodiment)
        return device.get_device_cfg(pipeline_builder=pipeline_builder, embodiment=embodiment)


class RetargeterRegistry(Registry):
    def __init__(self):
        super().__init__()

    def convert_tuple_to_str(self, key: tuple[str, str]) -> str:
        # Double underscore is used to separate device and embodiment names.
        return f"{key[0]}__{key[1]}"

    def convert_str_to_tuple(self, key: str) -> tuple[str, str]:
        # Double underscore is used to separate device and embodiment names.
        return (key.split("__")[0], key.split("__")[1])


class PolicyRegistry(Registry):
    def __init__(self):
        super().__init__()
        self._cfg_types: dict[type["PolicyBase"], type["PolicyCfg"]] = {}
        self._policy_types_by_cfg_type: dict[type["PolicyCfg"], type["PolicyBase"]] = {}

    def register_policy(self, policy_type: type["PolicyBase"], cfg_type: type["PolicyCfg"]) -> None:
        """Register a policy and its typed configuration."""
        assert cfg_type not in self._policy_types_by_cfg_type, f"Policy config {cfg_type.__name__} already registered"
        self.register(policy_type, policy_type.name)
        self._cfg_types[policy_type] = cfg_type
        self._policy_types_by_cfg_type[cfg_type] = policy_type

    # TODO(cvolk, 2026-07-06): [typed-config-migration] Remove this policy-to-config
    # lookup when policy_runner and experiment_runner receive PolicyCfg instances directly.
    def get_policy_cfg_type(self, policy_type: type["PolicyBase"]) -> type["PolicyCfg"]:
        """Get the config type used by the temporary policy frontend adapters."""
        ensure_assets_registered()
        assert policy_type in self._cfg_types, f"Policy {policy_type.__name__} must register a PolicyCfg"
        return self._cfg_types[policy_type]

    # NOTE(cvolk, 2026-07-07): [typed-config-migration] This reverse lookup belongs
    # to typed run execution and remains after the legacy adapters are removed.
    def get_policy_type_for_cfg(self, cfg: "PolicyCfg") -> type["PolicyBase"]:
        """Get the registered policy that consumes a concrete configuration."""
        ensure_assets_registered()
        cfg_type = type(cfg)
        assert cfg_type in self._policy_types_by_cfg_type, f"Policy config {cfg_type.__name__} is not registered"
        return self._policy_types_by_cfg_type[cfg_type]

    def get_policy(self, name: str) -> type["PolicyBase"]:
        """Gets a policy by name.

        Args:
            name (str): The name of the policy.
        """
        ensure_registry_registered(PolicyRegistry)
        return self.get_component_by_name(name)


class HDRImageRegistry(Registry):
    """Registry for HDR/EXR environment map textures."""

    def __init__(self):
        super().__init__()

    def get_hdr_by_name(self, name: str) -> type["HDRImage"]:
        """Gets an HDRImage class by name.

        Args:
            name (str): The name of the HDRImage.
        """
        ensure_registry_registered(HDRImageRegistry)
        return self.get_component_by_name(name)

    def get_hdrs_by_tag(self, tag: str) -> list[type["HDRImage"]]:
        """Gets a list of HDRImage classes that have the given tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            list[type[HDRImage]]: The matching HDRImage classes.
        """
        ensure_registry_registered(HDRImageRegistry)
        return [hdr for hdr in self._components.values() if tag in hdr.tags]

    def get_random_hdr_by_tag(self, tag: str) -> type["HDRImage"]:
        """Gets a random HDRImage class which has the given tag.

        Args:
            tag (str): The tag to filter by.

        Returns:
            type[HDRImage]: A random HDRImage class.
        """
        ensure_registry_registered(HDRImageRegistry)
        hdrs = self.get_hdrs_by_tag(tag)
        if len(hdrs) == 0:
            raise ValueError(f"No HDRs found with tag {tag}")
        return random.choice(hdrs)


class EnvironmentRegistry(Registry):
    """Registry for Arena environment factories and their configs."""

    def __init__(self):
        super().__init__()
        self._cfg_types_by_factory_type: dict[type["ArenaEnvironmentFactory"], type["ArenaEnvironmentCfg"]] = {}
        self._factory_types_by_cfg_type: dict[type["ArenaEnvironmentCfg"], type["ArenaEnvironmentFactory"]] = {}

    def register_environment(
        self,
        factory_type: type["ArenaEnvironmentFactory"],
        cfg_type: type["ArenaEnvironmentCfg"],
    ) -> None:
        """Register an environment factory and the config it consumes."""
        assert (
            cfg_type not in self._factory_types_by_cfg_type
        ), f"Environment config {cfg_type.__name__} already registered"
        self.register(factory_type, factory_type.name)
        self._cfg_types_by_factory_type[factory_type] = cfg_type
        self._factory_types_by_cfg_type[cfg_type] = factory_type

    # TODO(cvolk, 2026-07-07): [typed-config-migration] Remove this factory-to-config
    # lookup and _cfg_types_by_factory_type when the legacy JSON adapter no longer
    # resolves an environment name into its config type.
    def get_environment_cfg_type(
        self,
        factory_type: type["ArenaEnvironmentFactory"],
    ) -> type["ArenaEnvironmentCfg"]:
        """Get the config type needed to translate a legacy named environment."""
        ensure_assets_registered()
        assert (
            factory_type in self._cfg_types_by_factory_type
        ), f"Environment {factory_type.__name__} must register a config"
        return self._cfg_types_by_factory_type[factory_type]

    # NOTE(cvolk, 2026-07-07): [typed-config-migration] This reverse lookup belongs
    # to typed run execution and remains after the legacy adapters are removed.
    def get_factory_type_for_cfg(self, cfg: "ArenaEnvironmentCfg") -> type["ArenaEnvironmentFactory"]:
        """Resolve the registered factory used to execute a typed environment config."""
        ensure_assets_registered()
        cfg_type = type(cfg)
        assert cfg_type in self._factory_types_by_cfg_type, f"Environment config {cfg_type.__name__} is not registered"
        return self._factory_types_by_cfg_type[cfg_type]


class ObjectRelationLibraryRegistry(Registry):
    """Registry for object relation classes."""

    def __init__(self):
        super().__init__()

    def get_object_relation_by_name(self, name: str) -> type["RelationBase"]:
        """Gets an object relation by name.

        Args:
            name (str): The name of the object relation.
        """
        ensure_registry_registered(ObjectRelationLibraryRegistry)
        return self.get_component_by_name(name)


class TaskRegistry(Registry):
    """Registry for TaskBase subclasses."""

    def __init__(self):
        super().__init__()

    def get_task_by_name(self, name: str) -> type["TaskBase"]:
        """Gets a task class by name.

        Args:
            name (str): The name of the task class (typically the class __name__).
        """
        ensure_registry_registered(TaskRegistry)
        return self.get_component_by_name(name)


# Registries populated lazily by ensure_assets_registered(). EnvironmentRegistry is
# excluded: triggering the cascade during env registration causes an env<->tasks cycle.
REGISTRIES = (
    AssetRegistry,
    DeviceRegistry,
    RetargeterRegistry,
    PolicyRegistry,
    HDRImageRegistry,
    ObjectRelationLibraryRegistry,
    TaskRegistry,
)


# Lazy registration to avoid circular imports. Registration is tracked per
# registry type so scene/robot asset lookup does not import optional policy
# dependencies such as RSL-RL. ``ensure_assets_registered()`` remains as the
# compatibility full-load entrypoint for callers that intentionally need every
# registry populated.
_SCENE_ASSET_IMPORTS = (
    "isaaclab_arena.assets.background_library",
    "isaaclab_arena.assets.object_library",
    "isaaclab_arena.assets.simready_object_library",
)
_EMBODIMENT_ASSET_IMPORTS = ("isaaclab_arena.embodiments",)

_REGISTRY_IMPORTS: dict[type[Registry], tuple[str, ...]] = {
    AssetRegistry: (*_SCENE_ASSET_IMPORTS, *_EMBODIMENT_ASSET_IMPORTS),
    DeviceRegistry: ("isaaclab_arena.assets.device_library",),
    RetargeterRegistry: ("isaaclab_arena.assets.retargeter_library",),
    PolicyRegistry: ("isaaclab_arena.policy",),
    HDRImageRegistry: ("isaaclab_arena.assets.hdr_image_library",),
    ObjectRelationLibraryRegistry: ("isaaclab_arena.relations.relations",),
    TaskRegistry: ("isaaclab_arena.tasks.task_library",),
}

_registered_registries: set[type[Registry]] = set()
_registered_asset_provider_groups: set[str] = set()
# Blocks re-entry: registration decorators call is_registered() while a provider
# module is importing. A re-entrant full load can import partially initialized
# modules and trigger circular imports.
_registry_registration_in_progress: set[type[Registry]] = set()
_asset_provider_registration_in_progress: set[str] = set()

# Backward-compatible aliases kept for tests/external callers that reset the old
# module-level flags directly.
_assets_registered = False
_registration_in_progress = False


def _refresh_registration_flags():
    global _assets_registered, _registration_in_progress
    _registered_asset_registry = {"scene", "embodiment"}.issubset(_registered_asset_provider_groups)
    if _registered_asset_registry:
        _registered_registries.add(AssetRegistry)
    _assets_registered = all(registry in _registered_registries for registry in REGISTRIES)
    _registration_in_progress = bool(_registry_registration_in_progress or _asset_provider_registration_in_progress)


def _import_provider_group(group_name: str, module_names: tuple[str, ...]):
    if group_name in _registered_asset_provider_groups or group_name in _asset_provider_registration_in_progress:
        return
    _asset_provider_registration_in_progress.add(group_name)
    _refresh_registration_flags()
    try:
        for module_name in module_names:
            importlib.import_module(module_name)
        _registered_asset_provider_groups.add(group_name)
    finally:
        _asset_provider_registration_in_progress.discard(group_name)
        _refresh_registration_flags()


def ensure_asset_providers_registered(include_embodiments: bool = True):
    """Ensure asset providers are imported, optionally skipping robot embodiments."""
    _import_provider_group("scene", _SCENE_ASSET_IMPORTS)
    if include_embodiments:
        _import_provider_group("embodiment", _EMBODIMENT_ASSET_IMPORTS)


def ensure_registry_registered(registry_type: type[Registry]):
    """Ensure provider modules for one registry type are imported."""
    if registry_type is AssetRegistry:
        ensure_asset_providers_registered(include_embodiments=True)
        return
    if registry_type in _registered_registries or registry_type in _registry_registration_in_progress:
        return
    _registry_registration_in_progress.add(registry_type)
    _refresh_registration_flags()
    try:
        for module_name in _REGISTRY_IMPORTS[registry_type]:
            importlib.import_module(module_name)
        _registered_registries.add(registry_type)
    finally:
        _registry_registration_in_progress.discard(registry_type)
        _refresh_registration_flags()


def ensure_assets_registered():
    """Ensure all Arena registries are registered.

    Prefer registry-specific public methods for normal lookups. This full-load
    compatibility helper intentionally imports every registry provider, including
    optional policy providers.
    """
    for registry_type in REGISTRIES:
        ensure_registry_registered(registry_type)
