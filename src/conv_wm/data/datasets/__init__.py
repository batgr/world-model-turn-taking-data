"""Registry of dataset specifications.

Built-in datasets register themselves on import. A dataset that is present on
disk but not registered is still audited generically; ``get`` then returns an
empty specification so generic code never branches on dataset names.
"""

from __future__ import annotations

from collections.abc import Iterator

from conv_wm.data.datasets.ego4d import EGO4D
from conv_wm.data.datasets.egocom import EGOCOM
from conv_wm.data.datasets.spec import AudioInterpretation, DatasetSpec

_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec, *, replace: bool = False) -> DatasetSpec:
    """Register ``spec`` under its name; refuse silent overwrites unless ``replace``."""
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"Dataset {spec.name!r} is already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def unregister(name: str) -> None:
    """Remove a registration (used by tests that register fixtures)."""
    _REGISTRY.pop(name, None)


def get(name: str) -> DatasetSpec:
    """Return the registered spec, or an empty spec for an unregistered dataset."""
    return _REGISTRY.get(
        name, DatasetSpec(name=name, description="unregistered dataset")
    )


def is_registered(name: str) -> bool:
    """Whether ``name`` has an explicit specification."""
    return name in _REGISTRY


def names() -> list[str]:
    """Registered dataset names, sorted."""
    return sorted(_REGISTRY)


def iter_specs() -> Iterator[DatasetSpec]:
    """Registered specs in name order."""
    for name in names():
        yield _REGISTRY[name]


for _builtin in (EGO4D, EGOCOM):
    register(_builtin)

__all__ = [
    "AudioInterpretation",
    "DatasetSpec",
    "get",
    "is_registered",
    "iter_specs",
    "names",
    "register",
    "unregister",
]
