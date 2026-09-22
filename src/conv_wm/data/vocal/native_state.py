"""Native annotations -> canonical focal vocal-state intervals.

The v0 contract is ego-only and annotation-only: for one recording and its
wearer, native annotated speech is ``SPEAKING``, every valid instant outside
it is ``SILENT``, and instants without trustworthy evidence (invalid media,
explicitly missing annotation) are ``UNKNOWN``. Nothing acoustic is involved:
no detector, no speaker model, no completion. Timestamps stay continuous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

NATIVE_STATE_SCHEMA_VERSION = 1
"""Bumped when the columns or the state semantics of the timeline change."""


class VoiceState(StrEnum):
    """Wearer vocal state derived from native annotations."""

    SPEAKING = "SPEAKING"
    SILENT = "SILENT"
    UNKNOWN = "UNKNOWN"


class SourceKind(StrEnum):
    """Which native evidence an interval is derived from."""

    EGO4D_VOICE_SEGMENTS = "ego4d_voice_segments"
    EGOCOM_TRANSCRIPT = "egocom_transcript"
    INVALID_OR_MISSING = "invalid_or_missing"


@dataclass(frozen=True)
class NativeAnnotation:
    """One native interval on the canonical timeline and its stable identifier."""

    start_s: float
    end_s: float
    annotation_id: str

    def __post_init__(self) -> None:
        if not self.end_s > self.start_s:
            raise ValueError(f"{self.annotation_id}: empty or reversed interval")


@dataclass(frozen=True)
class NativeFocalRecording:
    """Everything one dataset adapter provides for one (recording, wearer) pair.

    ``start_s``/``end_s`` bound the nominal annotated window on the canonical
    timeline (Ego4D: clip-relative; EgoCom: conversation-part-relative).
    ``speaking`` are the wearer's native annotation intervals; ``unknown`` are
    regions the adapter declares untrustworthy (media absent or not covering
    the window, explicitly missing annotation, invalid clip). Intervals may
    extend beyond the window; they are clipped.
    """

    dataset: str
    recording_id: str
    sync_group_id: str | None
    view_id: str
    wearer_id: str
    start_s: float
    end_s: float
    speaking: tuple[NativeAnnotation, ...]
    unknown: tuple[NativeAnnotation, ...]
    source_kind: SourceKind
    annotation_schema_version: str

    def __post_init__(self) -> None:
        if not self.end_s > self.start_s:
            raise ValueError(f"{self.recording_id}: empty canonical window")

    @property
    def duration_s(self) -> float:
        """Length of the nominal window."""
        return self.end_s - self.start_s


@dataclass(frozen=True)
class NativeStateInterval:
    """One row of the exhaustive, non-overlapping focal voice-state timeline."""

    dataset: str
    recording_id: str
    sync_group_id: str | None
    view_id: str
    wearer_id: str
    canonical_start_s: float
    canonical_end_s: float
    voice_state: VoiceState
    source_kind: SourceKind
    source_annotation_id: str | None
    annotation_schema_version: str
    native_state_schema_version: int = NATIVE_STATE_SCHEMA_VERSION

    @property
    def duration_s(self) -> float:
        """Interval length in seconds."""
        return self.canonical_end_s - self.canonical_start_s

    def to_row(self) -> dict[str, Any]:
        """Stable Parquet row: enums as strings, one column per field."""
        row = asdict(self)
        row["voice_state"] = str(self.voice_state)
        row["source_kind"] = str(self.source_kind)
        return row


def _overlapping_ids(
    start_s: float, end_s: float, annotations: tuple[NativeAnnotation, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.annotation_id
                for item in annotations
                if item.start_s < end_s and item.end_s > start_s
            }
        )
    )


def build_native_timeline(recording: NativeFocalRecording) -> list[NativeStateInterval]:
    """Exhaustive timeline of ``recording``'s window, sorted and non-overlapping.

    Precedence: ``UNKNOWN`` (declared untrustworthy regions) over ``SPEAKING``
    (native annotation) over ``SILENT`` (the remaining valid time). Adjacent
    intervals merge only when state, source kind and source annotation ids are
    identical, so a region covered by two overlapping native segments keeps
    both ids (joined with ``|``) rather than losing provenance.
    """
    boundaries = {recording.start_s, recording.end_s}
    for item in (*recording.speaking, *recording.unknown):
        boundaries.add(min(max(item.start_s, recording.start_s), recording.end_s))
        boundaries.add(min(max(item.end_s, recording.start_s), recording.end_s))
    rows: list[NativeStateInterval] = []
    for start_s, end_s in pairwise(sorted(boundaries)):
        if end_s <= start_s:
            continue
        unknown_ids = _overlapping_ids(start_s, end_s, recording.unknown)
        speaking_ids = _overlapping_ids(start_s, end_s, recording.speaking)
        if unknown_ids:
            state, kind, ids = (
                VoiceState.UNKNOWN,
                SourceKind.INVALID_OR_MISSING,
                unknown_ids,
            )
        elif speaking_ids:
            state, kind, ids = VoiceState.SPEAKING, recording.source_kind, speaking_ids
        else:
            state, kind, ids = VoiceState.SILENT, recording.source_kind, ()
        row = NativeStateInterval(
            dataset=recording.dataset,
            recording_id=recording.recording_id,
            sync_group_id=recording.sync_group_id,
            view_id=recording.view_id,
            wearer_id=recording.wearer_id,
            canonical_start_s=start_s,
            canonical_end_s=end_s,
            voice_state=state,
            source_kind=kind,
            source_annotation_id="|".join(ids) if ids else None,
            annotation_schema_version=recording.annotation_schema_version,
        )
        if rows and _mergeable(rows[-1], row):
            previous = rows.pop()
            row = NativeStateInterval(**{**asdict(previous), "canonical_end_s": end_s})
        rows.append(row)
    return rows


def _mergeable(previous: NativeStateInterval, current: NativeStateInterval) -> bool:
    return (
        previous.canonical_end_s == current.canonical_start_s
        and previous.voice_state == current.voice_state
        and previous.source_kind == current.source_kind
        and previous.source_annotation_id == current.source_annotation_id
    )


def state_durations(rows: list[NativeStateInterval]) -> dict[str, float]:
    """Total seconds per state over ``rows``."""
    totals = {str(state): 0.0 for state in VoiceState}
    for row in rows:
        totals[str(row.voice_state)] += row.duration_s
    return totals


__all__ = [
    "NATIVE_STATE_SCHEMA_VERSION",
    "NativeAnnotation",
    "NativeFocalRecording",
    "NativeStateInterval",
    "SourceKind",
    "VoiceState",
    "build_native_timeline",
    "state_durations",
]
