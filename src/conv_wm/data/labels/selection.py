"""Which labels a consumer asks for: exact names, ``family.*`` or ``all``.

The whole selection language:

```yaml
labels:
  enabled: false        # the default: nothing is resolved, nothing is read
  include: [all]        # or exact names ("timing.time_to_next_ego_onset")
                        # or families ("timing.*"); entries are unioned
  modalities: []        # empty = no filter; otherwise keep a label only when
                        # every modality it requires is listed
```

``modalities: [audio, video]`` therefore keeps audio-only, video-only and
audio+video labels, and drops text or metadata ones.

Unknown names and unknown families are errors. A label that exists but cannot
be materialized (``unsupported``) is an error when named exactly and is
reported in :attr:`ResolvedSelection.skipped` when reached through ``all`` or
a wildcard — the same rule applies later, per dataset, to labels a store has
not materialized.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conv_wm.data.labels.registry import (
    FAMILIES,
    Availability,
    LabelSpec,
    Modality,
)

ALL = "all"
WILDCARD = ".*"


class LabelSelectionError(ValueError):
    """The selection names a label, family or modality that does not exist."""


class LabelUnavailableError(LookupError):
    """An explicitly requested label exists but cannot be provided."""


@dataclass(frozen=True)
class LabelSelection:
    """A consumer's request. The default requests nothing."""

    enabled: bool = False
    include: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> LabelSelection:
        """Build from a config mapping (``None`` or ``{}`` means disabled)."""
        if not value:
            return cls()
        unknown = set(value) - {"enabled", "include", "modalities"}
        if unknown:
            raise LabelSelectionError(f"unknown selection keys {sorted(unknown)}")
        return cls(
            enabled=bool(value.get("enabled", False)),
            include=_strings(value.get("include") or (), "include"),
            modalities=_strings(value.get("modalities") or (), "modalities"),
        )

    @classmethod
    def everything(cls, *, modalities: Sequence[str] = ()) -> LabelSelection:
        """Every available label (optionally filtered by modality)."""
        return cls(enabled=True, include=(ALL,), modalities=tuple(modalities))


def _strings(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    items = tuple(value)
    if not all(isinstance(item, str) for item in items):
        raise LabelSelectionError(f"{key} must be a list of strings")
    return items


@dataclass(frozen=True)
class ResolvedSelection:
    """The labels a selection resolves to, in registry order."""

    labels: tuple[LabelSpec, ...]
    skipped: Mapping[str, str] = field(default_factory=dict)
    """Labels reached through ``all``/a wildcard that were left out, and why."""

    @property
    def names(self) -> tuple[str, ...]:
        """Resolved label names."""
        return tuple(spec.name for spec in self.labels)

    def __bool__(self) -> bool:
        return bool(self.labels)


def _suggest(name: str, candidates: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(candidates), n=3, cutoff=0.6)
    return f" Did you mean {close}?" if close else ""


def resolve(
    selection: LabelSelection, registry: Sequence[LabelSpec]
) -> ResolvedSelection:
    """Resolve ``selection`` against ``registry``; see the module docstring."""
    if not selection.enabled:
        return ResolvedSelection(())
    modalities = _modalities(selection.modalities)
    names = {spec.name: spec for spec in registry}
    families = {spec.family for spec in registry} | set(FAMILIES)
    if not selection.include:
        raise LabelSelectionError(
            "labels.enabled is true but labels.include is empty; use [all], "
            "'family.*' patterns or exact names"
        )
    explicit: set[str] = set()
    wanted: set[str] = set()
    for item in selection.include:
        if item == ALL:
            wanted.update(names)
        elif item.endswith(WILDCARD):
            family = item[: -len(WILDCARD)]
            if family not in families:
                raise LabelSelectionError(
                    f"unknown label family {family!r}." + _suggest(family, families)
                )
            wanted.update(spec.name for spec in registry if spec.family == family)
        elif item in names:
            wanted.add(item)
            explicit.add(item)
        else:
            raise LabelSelectionError(
                f"unknown label {item!r}." + _suggest(item, names)
            )
    chosen: list[LabelSpec] = []
    skipped: dict[str, str] = {}
    for spec in registry:
        if spec.name not in wanted:
            continue
        if modalities is not None and not set(spec.modalities) <= modalities:
            if spec.name in explicit:
                raise LabelSelectionError(
                    f"{spec.name} requires modalities "
                    f"{[str(m) for m in spec.modalities]}, not all of which are in "
                    f"labels.modalities {sorted(str(m) for m in modalities)}"
                )
            continue
        if spec.availability is Availability.UNSUPPORTED:
            reason = f"unsupported ({spec.source_kind}): {spec.unsupported_reason}"
            if spec.name in explicit:
                raise LabelUnavailableError(f"{spec.name} is {reason}")
            skipped[spec.name] = reason
            continue
        chosen.append(spec)
    return ResolvedSelection(tuple(chosen), skipped)


def _modalities(values: Sequence[str]) -> set[Modality] | None:
    if not values:
        return None
    valid = {str(item) for item in Modality}
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise LabelSelectionError(
            f"unknown modalities {unknown}; expected a subset of {sorted(valid)}"
        )
    return {Modality(value) for value in values}


__all__ = [
    "ALL",
    "LabelSelection",
    "LabelSelectionError",
    "LabelUnavailableError",
    "ResolvedSelection",
    "resolve",
]
