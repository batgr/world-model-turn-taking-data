"""The label registry: one machine-readable description per label.

Every label the data layer knows about — built, buildable on request, or known
but not extractable — has exactly one :class:`LabelSpec`. The catalog of specs
lives in :mod:`conv_wm.data.labels.catalog`; this module holds the vocabulary
and the checks every spec must pass.

A spec says *what* a label means (description, validity, time reference,
modalities, source kind) and *where* it is stored (extractor, table, columns).
It never says which dataset supports it: support is computed from the facts a
dataset adapter declares it provides (:attr:`LabelSpec.requires`), so a new
dataset gains every label whose requirements it satisfies without touching the
registry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

REGISTRY_VERSION = 1
"""Bumped when a label is added, removed, renamed or changes meaning."""

LABEL_SCHEMA_VERSION = 2
"""Bumped when the physical layout of the label artifacts changes.

Version 2 stores vectors that may be null as a whole (joint-state occupancy,
face boxes, frame colours) as variable-size lists: pyarrow cannot read a null
fixed-size list back from Parquet.
"""


class Level(StrEnum):
    """The unit one value of a label describes."""

    GRID = "grid"
    """One 100 ms decision cell of the action grid."""
    SUBFRAME = "subframe"
    """One of the ``subframes_per_step`` equal parts of a decision cell."""
    EVENT = "event"
    """A point in native time (one row of an events table)."""
    SEGMENT = "segment"
    """A native-time interval (one row of a segments table)."""
    PARTICIPANT = "participant"
    """One participant of one recording."""
    RECORDING = "recording"
    """One recording."""


class Modality(StrEnum):
    """The informational nature of a label, not the file it was read from."""

    AUDIO = "audio"
    TEXT = "text"
    VIDEO = "video"
    METADATA = "metadata"


class SourceKind(StrEnum):
    """How the values of a label come to exist."""

    NATIVE_ANNOTATION = "native_annotation"
    """Provided by the corpus; at most resampled onto the grid."""
    DETERMINISTIC = "deterministic"
    """Derived by a deterministic rule from canonical facts."""
    EXTERNAL_MODEL = "external_model"
    """Estimated by an external model or tool; never ground truth."""
    HUMAN_ANNOTATION = "human_annotation"
    """Needs a human annotation that does not exist in the supported corpora."""


class Role(StrEnum):
    """How a consumer is expected to use a label."""

    CONTEXT = "context"
    """Observable up to the reference time; usable as an input."""
    TARGET = "target"
    """Refers to the future of the reference time; supervision only."""
    DIAGNOSTIC = "diagnostic"
    """Audit, probing and nuisance control; not a social truth."""
    METADATA = "metadata"
    """Identity and native descriptive fields."""


class Availability(StrEnum):
    """Whether the data layer can materialize a label at all."""

    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"


class Extractor(StrEnum):
    """The unit of building: one extractor writes one directory of tables.

    Annotation-only extractors run by default; media extractors decode audio
    or video and run only when requested.
    """

    SPEECH = "speech"
    SOCIAL = "social"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"

    @property
    def decodes_media(self) -> bool:
        """Whether building this extractor needs the raw media files."""
        return self in (Extractor.AUDIO, Extractor.VIDEO)


class Table(StrEnum):
    """The physical table a label's columns live in, inside its extractor directory."""

    GRID = "grid"
    """Keyed by ``(recording_id, decision_index)``; the action grid's rows."""
    EVENTS = "events"
    """One row per native-time point event, keyed by ``(recording_id, event_type, time_s)``."""
    SEGMENTS = "segments"
    """One row per native-time interval, keyed by ``(recording_id, segment_type, segment_id)``."""
    PARTICIPANTS = "participants"
    """Keyed by ``(recording_id, participant_index)``."""
    RECORDINGS = "recordings"
    """Keyed by ``recording_id``."""


TABLE_KEYS: Mapping[Table, tuple[str, ...]] = {
    Table.GRID: ("recording_id", "decision_index", "decision_time_s"),
    Table.EVENTS: ("recording_id", "event_type", "time_s"),
    Table.SEGMENTS: ("recording_id", "segment_type", "segment_id", "start_s", "end_s"),
    Table.PARTICIPANTS: ("recording_id", "participant_index", "participant_id"),
    Table.RECORDINGS: ("recording_id",),
}
"""Columns always returned with any label of a table (never selected, never skipped)."""

LEVEL_TABLES: Mapping[Level, tuple[Table, ...]] = {
    Level.GRID: (Table.GRID,),
    Level.SUBFRAME: (Table.GRID,),
    Level.EVENT: (Table.EVENTS,),
    Level.SEGMENT: (Table.SEGMENTS,),
    Level.PARTICIPANT: (Table.PARTICIPANTS,),
    Level.RECORDING: (Table.RECORDINGS,),
}
"""Which tables may store a label of each level."""

PARTICIPANT_AXIS_COLUMN = "participant_ids"
"""Grid column naming the participants a ``[participant, ...]`` value is indexed by."""


@dataclass(frozen=True)
class ExternalTool:
    """What an ``external_model`` label must record about the tool that made it."""

    tool: str
    """Tool or model family, e.g. ``"Praat (via Parselmouth)"``."""
    package: str | None
    """Python distribution whose installed version is recorded at build time."""
    extra: str | None
    """``uv sync --extra <extra>`` that installs it."""
    checkpoint: str | None
    """Model identifier or checkpoint; ``None`` for algorithmic tools."""
    config: Mapping[str, Any]
    """Parameters that change the output (also recorded at build time)."""
    license: str
    """The tool's licence and what it allows for derived outputs."""
    reference: str
    """Citation or URL."""
    determinism: str
    """Whether identical inputs give identical outputs, and why."""


@dataclass(frozen=True)
class LabelSpec:
    """Registry entry of one label."""

    name: str
    """``<family>.<label>``: the name used to request it."""
    family: str
    description: str
    level: Level
    modalities: tuple[Modality, ...]
    """Modalities whose information the label carries; all are required to use it."""
    dtype: str
    """Arrow type of the value column, e.g. ``list<fixed_size_list<bool, S>>``."""
    shape: str
    """What the axes of one value index, e.g. ``[participant, subframe]``."""
    time_reference: str
    """Which instant or interval a value describes."""
    source_kind: SourceKind
    role: Role
    validity: str
    """What a null, a mask or a sentinel means for this label."""
    version: int = 1
    requires: tuple[str, ...] = ()
    """Facts a dataset adapter must declare (``LabelFactsSpec.provides``) to support it."""
    extractor: Extractor | None = None
    """``None`` exactly when the label is unsupported."""
    table: Table | None = None
    columns: tuple[str, ...] = ()
    """Physical columns (value first, then validity/companion columns)."""
    row_filter: tuple[str, str] | None = None
    """``(column, value)`` selecting this label's rows in an events/segments table."""
    external: ExternalTool | None = None
    unsupported_reason: str | None = None
    """Why the label cannot be materialized; required when unsupported."""
    notes: str = ""

    @property
    def availability(self) -> Availability:
        """``available`` when an extractor materializes the label."""
        return (
            Availability.AVAILABLE
            if self.extractor is not None
            else Availability.UNSUPPORTED
        )

    @property
    def participant_axis(self) -> bool:
        """Whether a grid value is indexed by the recording's participants."""
        return self.table is Table.GRID and "participant" in self.shape

    def supported_by(self, provides: Iterable[str]) -> bool:
        """Whether a dataset declaring ``provides`` can materialize this label."""
        return self.availability is Availability.AVAILABLE and set(
            self.requires
        ).issubset(set(provides))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable description, stable key order."""
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "level": str(self.level),
            "modalities": [str(item) for item in self.modalities],
            "dtype": self.dtype,
            "shape": self.shape,
            "time_reference": self.time_reference,
            "source_kind": str(self.source_kind),
            "role": str(self.role),
            "validity": self.validity,
            "version": self.version,
            "availability": str(self.availability),
            "requires": list(self.requires),
            "extractor": str(self.extractor) if self.extractor else None,
            "table": str(self.table) if self.table else None,
            "columns": list(self.columns),
            "row_filter": list(self.row_filter) if self.row_filter else None,
            "external": _external_dict(self.external),
            "unsupported_reason": self.unsupported_reason,
            "notes": self.notes,
        }


def _external_dict(tool: ExternalTool | None) -> dict[str, Any] | None:
    if tool is None:
        return None
    return {
        "tool": tool.tool,
        "package": tool.package,
        "extra": tool.extra,
        "checkpoint": tool.checkpoint,
        "config": dict(tool.config),
        "license": tool.license,
        "reference": tool.reference,
        "determinism": tool.determinism,
    }


FAMILIES: Mapping[str, str] = {
    "instantaneous": "Who is speaking now: per-participant activity, joint state, floor.",
    "events": "Onsets, offsets and floor changes, on the grid and in native time.",
    "onset_context": "The primitives describing the situation of each wearer onset.",
    "overlap": "Simultaneous speech: native overlap events and their grid view.",
    "timing": "Time since / time to the next event, with explicit censoring.",
    "turns": "Speech runs, pause-closed turns, silences and floor transfers.",
    "next_speaker": "Who speaks or takes the floor next, from the reference time on.",
    "future": "Supervision over configurable future horizons.",
    "prosody": "Per-speaker acoustic features under a reliable speaker mask.",
    "text": "Native transcript tokens and the cues derivable from them.",
    "video": "Physical visual observations (faces, bodies, motion).",
    "social_native": "Native Ego4D Looking-At-Me / Talking-To-Me and face tracks.",
    "addressee": "Who an utterance is addressed to.",
    "backchannel": "Short feedback responses.",
    "profiles": "Per-participant interaction statistics over a recording.",
    "social_states": "Higher-level social constructs (engagement, dominance, ...).",
    "nuisance": "Audio/video diagnostic controls; not social truths.",
    "metadata": "Native identity and descriptive fields.",
}
"""Every family the registry may use, with a one-line purpose (docs are generated from it)."""

_NAME = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class RegistryError(ValueError):
    """A registry entry violates the contract."""


def validate(specs: Sequence[LabelSpec]) -> None:
    """Raise :class:`RegistryError` on the first contract violation found.

    Checks: unique well-formed names; the family prefix matches a declared
    family; at least one modality; unsupported entries carry a reason and no
    storage; available entries name an extractor, a table allowed for their
    level and at least one column; external entries document their tool;
    row filters only on events/segments tables; two labels never own the same
    physical column of the same extractor table unless they select rows.
    """
    seen: set[str] = set()
    owners: dict[tuple[str, str, str], str] = {}
    for spec in specs:
        where = f"label {spec.name!r}"
        if not _NAME.match(spec.name):
            raise RegistryError(f"{where}: name must be '<family>.<label>'")
        if spec.name in seen:
            raise RegistryError(f"{where}: duplicate name")
        seen.add(spec.name)
        if spec.name.split(".", 1)[0] != spec.family:
            raise RegistryError(f"{where}: name prefix differs from family")
        if spec.family not in FAMILIES:
            raise RegistryError(f"{where}: unknown family {spec.family!r}")
        if not spec.modalities or len(set(spec.modalities)) != len(spec.modalities):
            raise RegistryError(f"{where}: needs distinct modalities")
        for text, label in (
            (spec.description, "description"),
            (spec.validity, "validity"),
            (spec.time_reference, "time_reference"),
            (spec.dtype, "dtype"),
            (spec.shape, "shape"),
        ):
            if not text.strip():
                raise RegistryError(f"{where}: empty {label}")
        if spec.source_kind is SourceKind.EXTERNAL_MODEL and spec.external is None:
            raise RegistryError(f"{where}: external_model label without tool record")
        if spec.availability is Availability.UNSUPPORTED:
            if not spec.unsupported_reason:
                raise RegistryError(f"{where}: unsupported label without a reason")
            if spec.table is not None or spec.columns or spec.row_filter:
                raise RegistryError(f"{where}: unsupported label with storage")
            continue
        if spec.unsupported_reason:
            raise RegistryError(f"{where}: available label with unsupported_reason")
        if spec.source_kind is SourceKind.HUMAN_ANNOTATION:
            raise RegistryError(f"{where}: human_annotation labels cannot be built")
        if spec.table is None or spec.table not in LEVEL_TABLES[spec.level]:
            raise RegistryError(f"{where}: table not allowed for level {spec.level}")
        if not spec.columns:
            raise RegistryError(f"{where}: available label without columns")
        if spec.row_filter and spec.table not in (Table.EVENTS, Table.SEGMENTS):
            raise RegistryError(f"{where}: row_filter on a {spec.table} table")
        if set(spec.columns) & set(TABLE_KEYS[spec.table]):
            raise RegistryError(f"{where}: key columns are implicit")
        if spec.row_filter is None:
            assert spec.extractor is not None
            for column in spec.columns:
                key = (str(spec.extractor), str(spec.table), column)
                if key in owners:
                    raise RegistryError(
                        f"{where}: column {column!r} already owned by {owners[key]!r}"
                    )
                owners[key] = spec.name


def by_name(specs: Sequence[LabelSpec]) -> dict[str, LabelSpec]:
    """Index specs by name."""
    return {spec.name: spec for spec in specs}


def registry_document(specs: Sequence[LabelSpec]) -> dict[str, Any]:
    """The ``registry.json`` document: versions, families and every spec."""
    return {
        "registry_version": REGISTRY_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "families": dict(FAMILIES),
        "labels": [spec.to_dict() for spec in specs],
    }


@dataclass(frozen=True)
class SupportRow:
    """Whether one dataset can materialize one label, and why not."""

    label: str
    dataset: str
    supported: bool
    missing_facts: tuple[str, ...] = field(default=())


def support_matrix(
    specs: Sequence[LabelSpec], provides: Mapping[str, Iterable[str]]
) -> list[SupportRow]:
    """Support of every label for every dataset, from declared facts only."""
    rows: list[SupportRow] = []
    for spec in specs:
        for dataset, facts in sorted(provides.items()):
            facts = set(facts)
            rows.append(
                SupportRow(
                    label=spec.name,
                    dataset=dataset,
                    supported=spec.supported_by(facts),
                    missing_facts=tuple(sorted(set(spec.requires) - facts)),
                )
            )
    return rows


__all__ = [
    "FAMILIES",
    "LABEL_SCHEMA_VERSION",
    "LEVEL_TABLES",
    "PARTICIPANT_AXIS_COLUMN",
    "REGISTRY_VERSION",
    "TABLE_KEYS",
    "Availability",
    "ExternalTool",
    "Extractor",
    "LabelSpec",
    "Level",
    "Modality",
    "RegistryError",
    "Role",
    "SourceKind",
    "SupportRow",
    "Table",
    "by_name",
    "registry_document",
    "support_matrix",
    "validate",
]
