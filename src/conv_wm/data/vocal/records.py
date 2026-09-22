"""Typed records of the vocal annotation coverage audit.

Field names are the report schema: ``to_row`` flattens a record into one Parquet
row, so renaming a field renames a column.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from conv_wm.data.vocal.intervals import Intervals, as_intervals


class SpeakerAttribution(StrEnum):
    """Who a detected segment is attributed to."""

    FOCAL = "focal"
    OTHER = "other"
    UNRESOLVED = "unresolved"


class CoverageStatus(StrEnum):
    """Relation between one detected segment and the focal annotation.

    ``NOT_FOCAL`` marks segments attributed to another participant; they are
    never candidates for a missing focal annotation. ``ANNOTATION_UNAVAILABLE``
    marks segments of recordings whose annotation coverage is unknown: absence of
    annotation there is not silence and not a missing annotation.
    """

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    UNCOVERED = "uncovered"
    UNRESOLVED_IDENTITY = "unresolved_identity"
    NOT_FOCAL = "not_focal"
    ANNOTATION_UNAVAILABLE = "annotation_unavailable"


class FocalVoiceActivityStatus(StrEnum):
    """Whether unannotated acoustic activity can be attributed to the wearer."""

    MEASURABLE = "measurable_for_resolved_segments"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class AudioSource:
    """One decodable audio window: a media file and a canonical window inside it.

    ``start_s`` and ``duration_s`` are on the file's native (PTS) timeline;
    ``canonical_offset_s`` maps decoded time ``t`` to the canonical timeline as
    ``canonical_offset_s + t``.
    """

    path: str
    start_s: float
    duration_s: float
    canonical_offset_s: float
    view_id: str


@dataclass(frozen=True)
class FocalRecording:
    """One audited (recording, wearer) pair with everything the audit needs.

    Canonical times are seconds on the annotation authority's timeline (Ego4D:
    clip-relative; EgoCom: conversation-part-relative). ``AudioSource`` retains
    the mapping to native media PTS. ``focal_annotation`` and
    ``other_annotation`` are ``(n, 2)`` arrays on the canonical timeline.
    """

    dataset: str
    recording_id: str
    view_id: str
    wearer_id: str | None
    sync_group_id: str | None
    audio: AudioSource
    canonical_start_s: float
    canonical_end_s: float
    focal_annotation: Intervals
    other_annotation: Intervals
    annotation_available: bool
    annotation_source: str
    """Name and provenance of the annotation table(s), for the report."""
    other_views: tuple[AudioSource, ...] = ()
    """Synchronized audio of the other participants, when the dataset has it."""
    focal_annotation_raw: Intervals | None = None
    """Annotation intervals before merging, for validation of identity methods."""
    other_annotation_by_view: dict[str, Intervals] = field(default_factory=dict)
    """Per other view, the annotation intervals of that view's wearer."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "focal_annotation", as_intervals(self.focal_annotation)
        )
        object.__setattr__(
            self, "other_annotation", as_intervals(self.other_annotation)
        )
        if self.canonical_end_s <= self.canonical_start_s:
            raise ValueError(f"{self.recording_id}: empty canonical window")

    @property
    def duration_s(self) -> float:
        """Length of the audited window."""
        return self.canonical_end_s - self.canonical_start_s


@dataclass(frozen=True)
class DetectedSegment:
    """One acoustic voice activity segment on the canonical timeline."""

    start_s: float
    end_s: float
    detection_confidence: float
    """Mean speech probability over the segment's detector chunks."""

    @property
    def duration_s(self) -> float:
        """Segment length in seconds."""
        return self.end_s - self.start_s


@dataclass(frozen=True)
class IdentityDecision:
    """Speaker attribution of one detected segment."""

    attribution: SpeakerAttribution
    method: str
    confidence: float
    """Method-specific confidence in ``[0, 1]``; NaN when not measurable."""
    evidence: dict[str, float] = field(default_factory=dict)
    """Method-specific numbers (e.g. dominance in dB) kept for inspection."""


@dataclass(frozen=True)
class SegmentAssessment:
    """One detected segment after identity attribution and coverage assessment."""

    dataset: str
    recording_id: str
    sync_group_id: str | None
    view_id: str
    wearer_id: str | None
    canonical_start_s: float
    canonical_end_s: float
    duration_s: float
    annotation_overlap_ratio: float
    """Fraction of the segment inside the (tolerance-dilated) focal annotation."""
    annotation_overlap_ratio_exact: float
    """Same without tolerance; the alignment-free measurement."""
    other_annotation_overlap_ratio: float
    detection_method: str
    detection_confidence: float
    identity_method: str
    identity_confidence: float
    speaker_attribution: SpeakerAttribution
    coverage_status: CoverageStatus
    duration_bucket: str
    onset_offset_s: float
    """``segment start - nearest focal annotation start`` (NaN without annotation)."""
    offset_offset_s: float
    uncovered_duration_s: float
    """Part of the segment outside the dilated focal annotation."""
    identity_evidence: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten into the Parquet column layout."""
        row = asdict(self)
        row["speaker_attribution"] = str(self.speaker_attribution)
        row["coverage_status"] = str(self.coverage_status)
        evidence = row.pop("identity_evidence")
        for key, value in sorted(evidence.items()):
            row[f"identity_{key}"] = value
        return row


@dataclass(frozen=True)
class RecordingCoverageSummary:
    """Coverage of one (recording, wearer) pair (one row of ``summary.parquet``)."""

    dataset: str
    recording_id: str
    sync_group_id: str | None
    view_id: str
    wearer_id: str | None
    canonical_start_s: float
    canonical_end_s: float
    audited_duration_s: float
    annotation_available: bool
    annotation_source: str
    detection_method: str
    identity_method: str
    identity_confidence: float
    """Recording-level confidence of the identity method (NaN if not measured)."""
    focal_voice_activity_status: FocalVoiceActivityStatus
    detected_acoustic_voice_duration_s: float
    """All acoustic voice activity, any speaker."""
    detected_segment_count: int
    annotated_focal_voice_duration_s: float
    detected_focal_voice_duration_s: float
    """Detected segments attributed to the focal participant."""
    focal_annotation_coverage_ratio: float
    """Focal detected duration inside the dilated annotation / focal detected duration."""
    covered_focal_duration_s: float
    uncovered_focal_voice_duration_s: float
    uncovered_focal_segment_count: int
    partially_covered_focal_segment_count: int
    uncovered_short_segment_ratio: float
    """Share of uncovered focal segments shorter than ``short_segment_max_duration_s``."""
    uncovered_segment_count_by_bucket: dict[str, int]
    uncovered_duration_s_by_bucket: dict[str, float]
    other_attributed_duration_s: float
    unresolved_identity_duration_s: float
    unresolved_identity_segment_count: int
    unresolved_identity_ratio: float
    """Unresolved detected duration / all detected duration."""
    unannotated_any_speaker_duration_s: float
    """Detected speech overlapping nobody's annotation (identity aside)."""
    annotated_focal_without_detection_s: float
    """Focal annotation duration where the detector found no speech."""
    onset_offset_s_median: float
    offset_offset_s_median: float
    onset_offset_s_abs_p95: float
    identity_precision_focal: float = math.nan
    identity_precision_other: float = math.nan
    identity_recall_focal: float = math.nan
    identity_undecided_ratio: float = math.nan
    identity_validation_focal_count: int = 0
    identity_validation_other_count: int = 0
    identity_synchronized_other_view_count: int = 0
    identity_excluded_other_view_count: int = 0
    audit_ok: bool = True
    audit_error: str | None = None
    decoded_duration_error_s: float = math.nan
    """Decoded length minus requested window length; large values mean PTS gaps."""

    def to_row(self) -> dict[str, Any]:
        """Flatten into the Parquet column layout (bucket mappings as columns)."""
        row = asdict(self)
        row["focal_voice_activity_status"] = str(self.focal_voice_activity_status)
        counts = row.pop("uncovered_segment_count_by_bucket")
        durations = row.pop("uncovered_duration_s_by_bucket")
        for label, count in counts.items():
            row[f"uncovered_count_{_column_safe(label)}"] = count
        for label, duration in durations.items():
            row[f"uncovered_duration_s_{_column_safe(label)}"] = duration
        return row

    @classmethod
    def failed(
        cls, recording: FocalRecording, error: str, *, detection_method: str
    ) -> RecordingCoverageSummary:
        """Row for a recording whose decoding or detection failed."""
        nan = math.nan
        return cls(
            dataset=recording.dataset,
            recording_id=recording.recording_id,
            sync_group_id=recording.sync_group_id,
            view_id=recording.view_id,
            wearer_id=recording.wearer_id,
            canonical_start_s=recording.canonical_start_s,
            canonical_end_s=recording.canonical_end_s,
            audited_duration_s=0.0,
            annotation_available=recording.annotation_available,
            annotation_source=recording.annotation_source,
            detection_method=detection_method,
            identity_method="none",
            identity_confidence=nan,
            focal_voice_activity_status=FocalVoiceActivityStatus.UNRESOLVED,
            detected_acoustic_voice_duration_s=nan,
            detected_segment_count=0,
            annotated_focal_voice_duration_s=nan,
            detected_focal_voice_duration_s=nan,
            focal_annotation_coverage_ratio=nan,
            covered_focal_duration_s=nan,
            uncovered_focal_voice_duration_s=nan,
            uncovered_focal_segment_count=0,
            partially_covered_focal_segment_count=0,
            uncovered_short_segment_ratio=nan,
            uncovered_segment_count_by_bucket={},
            uncovered_duration_s_by_bucket={},
            other_attributed_duration_s=nan,
            unresolved_identity_duration_s=nan,
            unresolved_identity_segment_count=0,
            unresolved_identity_ratio=nan,
            unannotated_any_speaker_duration_s=nan,
            annotated_focal_without_detection_s=nan,
            onset_offset_s_median=nan,
            offset_offset_s_median=nan,
            onset_offset_s_abs_p95=nan,
            audit_ok=False,
            audit_error=error,
        )


def _column_safe(label: str) -> str:
    return (
        label.replace("<", "lt_")
        .replace(">=", "ge_")
        .replace("-", "_to_")
        .replace(".", "p")
        .replace("s", "s")
    )


def intervals_from_segments(segments: list[DetectedSegment]) -> Intervals:
    """``(n, 2)`` array of segment bounds."""
    return as_intervals(
        np.array([[s.start_s, s.end_s] for s in segments], dtype=float).reshape(-1, 2)
    )
