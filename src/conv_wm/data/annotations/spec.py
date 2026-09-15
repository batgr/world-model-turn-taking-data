"""Typed description of an annotation source and of what may be checked on it.

An :class:`AnnotationSourceSpec` says what a table *means*: its scope, temporal
coordinate system, which columns point at media, clips, interactions or
participants, where its values come from and what is known to be wrong with
it. Generic integrity checks read these declarations; they never guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

TableLoader = Callable[[DictConfig], pd.DataFrame]


class AnnotationScope(StrEnum):
    """What one row of the source describes."""

    POINT_EVENT = "point_event"
    TEMPORAL_INTERVAL = "temporal_interval"
    CLIP = "clip"
    VIDEO = "video"
    INTERACTION = "interaction"
    PARTICIPANT = "participant"
    PARTICIPANT_INTERACTION = "participant_interaction"
    SEQUENCE_GLOBAL = "sequence_global"


class TimeUnit(StrEnum):
    """Unit of the temporal columns."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    FRAMES = "frames"
    SAMPLES = "samples"


class TemporalOrigin(StrEnum):
    """Zero point of the temporal columns."""

    MEDIA_START = "media_start"
    CLIP_START = "clip_start"
    INTERACTION_START = "interaction_start"
    ABSOLUTE = "absolute"


class AnnotationProvenance(StrEnum):
    """How the values of a source came to exist. Carried forward, never inferred."""

    HUMAN_OBSERVED = "human_observed"
    DETERMINISTIC_DERIVED = "deterministic_derived"
    TRANSFERRED = "transferred"
    PSEUDO_LABEL = "pseudo_label"
    CLUSTER_DERIVED = "cluster_derived"
    MODEL_INFERRED = "model_inferred"


class ReferenceKind(StrEnum):
    """What an entity reference points at."""

    MEDIA = "media"
    CLIP = "clip"
    INTERACTION = "interaction"
    PARTICIPANT = "participant"


class Severity(StrEnum):
    """Whether a violated constraint invalidates the source for downstream use."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class TemporalFields:
    """Temporal columns of a source and their coordinate system.

    ``end`` is ``None`` for point events. ``nullable`` declares that missing
    timestamps are expected (e.g. untimed transcript tokens) and are reported as
    information rather than as an anomaly.
    """

    start: str
    end: str | None
    unit: TimeUnit
    origin: TemporalOrigin
    nullable: bool = False
    out_of_bounds_tolerance: float = 0.0
    """Slack, in ``unit``, allowed beyond the reference duration."""


@dataclass(frozen=True)
class EntityReference:
    """Columns of this source that must exist in another source of the dataset."""

    kind: ReferenceKind
    columns: tuple[str, ...]
    target_source: str
    target_columns: tuple[str, ...]
    unknown_values: tuple[str, ...] = ()
    """Sentinel values (e.g. ``"-1"``) that mean 'unknown', not a dangling key."""
    severity: Severity = Severity.ERROR

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.target_columns):
            raise ValueError("EntityReference column tuples must have the same length.")


@dataclass(frozen=True)
class MediaReference:
    """Columns of this source that name a media file of the dataset.

    ``media_key`` maps a media file's relative path to the key stored in the
    source (for instance the file stem). Media rows come from the media
    metadata audit, so bounds can use measured stream durations.
    """

    columns: tuple[str, ...]
    media_key: Callable[[str], str] = lambda relative_path: Path(relative_path).stem
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class DurationBounds:
    """Where the reference duration that bounds this source's timestamps lives.

    Timestamps are compared with the duration of the row's referenced entity:
    ``source`` names the table holding it, ``key_columns`` how to join, and
    either ``duration_column`` or ``start_column``/``end_column`` give the extent.
    ``"media"`` as ``source`` uses the media metadata table (``audio_duration_sec``
    and ``video_duration_sec``, the smaller of the two).
    """

    source: str
    key_columns: tuple[str, ...]
    target_key_columns: tuple[str, ...]
    duration_column: str | None = None
    start_column: str | None = None
    end_column: str | None = None
    severity: Severity = Severity.WARNING


@dataclass(frozen=True)
class AnnotationSourceSpec:
    """Semantics of one annotation table of a dataset."""

    dataset: str
    name: str
    description: str
    scope: AnnotationScope
    load: TableLoader
    dimensions: tuple[str, ...]
    """Annotation dimensions contributed (``"speech"``, ``"gaze"``, ``"affect"`` ...)."""
    provenance: AnnotationProvenance
    identity: tuple[str, ...] = ()
    """Columns that identify a row; duplicates are reported when non-empty."""
    temporal: TemporalFields | None = None
    media_reference: MediaReference | None = None
    entity_references: tuple[EntityReference, ...] = ()
    bounds: DurationBounds | None = None
    value_fields: tuple[str, ...] = ()
    """Payload columns preserved for downstream use, with their native meaning."""
    known_limitations: tuple[str, ...] = ()
    confidence_field: str | None = None
    """Column carrying a per-row confidence when the source provides one."""

    def __post_init__(self) -> None:
        if self.temporal is None and self.scope in (
            AnnotationScope.POINT_EVENT,
            AnnotationScope.TEMPORAL_INTERVAL,
        ):
            raise ValueError(f"{self.name}: temporal scopes need TemporalFields.")
        if (
            self.temporal is not None
            and self.temporal.end is None
            and (self.scope == AnnotationScope.TEMPORAL_INTERVAL)
        ):
            raise ValueError(f"{self.name}: TEMPORAL_INTERVAL needs an end column.")


@dataclass(frozen=True)
class CrossSourceComparison:
    """Two temporal sources whose intervals are expected to overlap per entity.

    Coverage is the fraction of intervals of one source that overlap at least
    one interval of the other source with the same ``key_columns``. Overlap is
    evidence of agreement, not identity; disagreement is reported, not judged.
    """

    name: str
    left_source: str
    right_source: str
    left_key_columns: tuple[str, ...]
    right_key_columns: tuple[str, ...]
    description: str = ""
    overlap_tolerance: float = 0.0
    """Diagnostic expansion around each interval in the sources' shared time unit."""
    excluded_key_values: tuple[str, ...] = ()
    """Sentinel values omitted from either side, for example an unknown identity."""
    left_required_non_null: tuple[str, ...] = ()
    """Optional source fields that must be populated on selected left rows."""
    right_required_non_null: tuple[str, ...] = ()
    """Optional source fields that must be populated on selected right rows."""

    def __post_init__(self) -> None:
        if self.overlap_tolerance < 0:
            raise ValueError("Cross-source overlap tolerance cannot be negative.")


@dataclass(frozen=True)
class AnnotationSpec:
    """Everything a dataset declares about its annotations."""

    sources: tuple[AnnotationSourceSpec, ...]
    comparisons: tuple[CrossSourceComparison, ...] = ()

    def __post_init__(self) -> None:
        names = {source.name for source in self.sources}
        if len(names) != len(self.sources):
            raise ValueError("AnnotationSpec source names must be unique.")
        for source in self.sources:
            for reference in source.entity_references:
                if reference.target_source not in names:
                    raise ValueError(
                        f"{source.name}: reference target {reference.target_source!r} unknown."
                    )
            if source.bounds is not None and source.bounds.source not in names | {
                "media"
            }:
                raise ValueError(
                    f"{source.name}: bounds source {source.bounds.source!r} unknown."
                )
        for comparison in self.comparisons:
            if {comparison.left_source, comparison.right_source} - names:
                raise ValueError(
                    f"Comparison {comparison.name} references unknown sources."
                )

    def source(self, name: str) -> AnnotationSourceSpec:
        """Look up a source by name."""
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(name)


EMPTY_ANNOTATIONS = AnnotationSpec(sources=())
