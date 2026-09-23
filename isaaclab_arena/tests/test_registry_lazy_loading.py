# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib
import types


def test_scene_asset_lookup_does_not_import_embodiments_or_policy(monkeypatch):
    """Scene asset lookup must not pull in robot/policy dependencies.

    Policy registration can import optional RL stacks. Scene construction should
    resolve ordinary scene assets from scene providers before loading robot
    embodiments or policy providers.
    """
    from isaaclab_arena.assets import registries

    imported_modules: list[str] = []

    def fake_registration_import(name: str):
        imported_modules.append(name)
        if name.startswith("isaaclab_arena.policy"):
            raise AssertionError("AssetRegistry loading must not import isaaclab_arena.policy")
        return types.ModuleType(name)

    def fake_import_module(name: str, package: str | None = None):
        if package is not None:
            name = importlib.util.resolve_name(name, package)
        return fake_registration_import(name)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.startswith("isaaclab_arena."):
            return fake_registration_import(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    if hasattr(registries, "_assets_registered"):
        monkeypatch.setattr(registries, "_assets_registered", False)
    if hasattr(registries, "_registration_in_progress"):
        monkeypatch.setattr(registries, "_registration_in_progress", False)
    if hasattr(registries, "_registered_registries"):
        monkeypatch.setattr(registries, "_registered_registries", set())
    if hasattr(registries, "_registered_asset_provider_groups"):
        monkeypatch.setattr(registries, "_registered_asset_provider_groups", set())
    if hasattr(registries, "_registry_registration_in_progress"):
        monkeypatch.setattr(registries, "_registry_registration_in_progress", set())
    if hasattr(registries, "_asset_provider_registration_in_progress"):
        monkeypatch.setattr(registries, "_asset_provider_registration_in_progress", set())

    class FakeSceneAsset:
        name = "scene_asset"
        tags = ["object"]

    registry = registries.AssetRegistry()
    original_components = dict(registry._components)
    registry._components.clear()
    registry.register(FakeSceneAsset, FakeSceneAsset.name)

    try:
        assert registry.get_asset_by_name("scene_asset") is FakeSceneAsset
    finally:
        registry._components.clear()
        registry._components.update(original_components)

    assert "isaaclab_arena.assets.object_library" in imported_modules
    assert "isaaclab_arena.embodiments" not in imported_modules
    assert not any(name.startswith("isaaclab_arena.policy") for name in imported_modules)
