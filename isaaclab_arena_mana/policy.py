# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ManaPolicyHelperCFG(ABC):
    """Helper class for Mana policies.

    Subclasses (defined on the manaenv side) implement :meth:`translate` to
    convert their raw config payload into whatever the wrapped Arena policy
    expects — either the policy's config dataclass instance or a plain dict of
    its constructor fields.
    """

    raw_data: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def translate(self):
        # SHPY Comment: using raw_data if translate is not needed
        return self.raw_data


def as_config(translated: Any, config_class: type) -> Any:
    """Coerce a :meth:`ManaPolicyHelperCFG.translate` result to ``config_class``.

    Accepts either an already-built ``config_class`` instance (returned as-is)
    or a dict of its constructor fields.
    """
    if isinstance(translated, config_class):
        return translated
    assert isinstance(
        translated, dict
    ), f"translate() must return a {config_class.__name__} or a dict, got {type(translated).__name__}"
    return config_class(**translated)
