"""Canonical source facts: what a dataset adapter hands to the label derivations.

A dataset contributes a :class:`LabelFactsSpec` on its ``DatasetSpec``: the
set of facts it can provide (declared statically, so label support is known
without reading data) and a loader returning a :class:`LabelSource`. Every
label derivation downstream reads only these structures; none knows which
corpus produced them.

Conventions shared by every fact:

* times are seconds on the recording's canonical clock — the clock of its
  native focal voice state and of its action grid;
* intervals are half-open ``[start_s, end_s)``;
* absence of a speech interval inside the recording window, outside every
  UNKNOWN region, means SILENT (the native-state contract), never unknown;
* ``unknown`` regions are declared by the adapter (invalid clip, media not
  covering the window, explicitly missing annotation) and win over speech.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from conv_wm.data.vocal.native_state import NativeAnnotation

if TYPE_CHECKING:
    from omegaconf import DictConfig

SPEECH = "speech"
"""Native speech intervals of every participant of a recording, wearer included."""
WORDS = "words"
"""Word-level timed transcript tokens with speaker attribution."""
TRANSCRIPT = "transcript"
"""Speaker-attributed transcript text with at least utterance-level timing."""
SOCIAL_LOOKING = "social.looking"
"""Native Looking-At-Me segments (Ego4D Social semantics)."""
SOCIAL_TALKING = "social.talking"
"""Native Talking-To-Me segments (Ego4D Social semantics)."""
FACE_TRACKS = "face_tracks"
"""Native per-frame face bounding boxes attributed to participants."""
MEDIA_AUDIO = "media.audio"
"""Raw audio resolvable through the media manifest."""
MEDIA_VIDEO = "media.video"
"""Raw video resolvable through the media manifest."""
META_BACKGROUND = "meta.background"
"""Native background-condition flags (fan, music) per recording."""
META_NATIVE_SPEAKER = "meta.native_speaker"
"""Native per-participant first-language flag."""
META_HOST = "meta.host"
"""Native per-participant host/role flag."""
META_SOURCE_OFFSET = "meta.source_offset"
"""Position of the recording inside a longer source video."""

KNOWN_FACTS = frozenset(
    {
        SPEECH,
        WORDS,
        TRANSCRIPT,
        SOCIAL_LOOKING,
        SOCIAL_TALKING,
        FACE_TRACKS,
        MEDIA_AUDIO,
        MEDIA_VIDEO,
        META_BACKGROUND,
        META_NATIVE_SPEAKER,
        META_HOST,
        META_SOURCE_OFFSET,
    }
)

UNRESOLVED_PARTICIPANT_ID = "<unresolved>"
"""Pseudo-participant carrying native speech whose speaker identity is unknown."""

TOKEN_COLUMNS = (
    "recording_id",
    "participant_id",
    "unit",
    "text",
    "start_s",
    "end_s",
    "timing_valid",
    "source_row",
)
"""Canonical transcript rows: ``unit`` is ``word`` or ``utterance``."""

SOCIAL_COLUMNS = (
    "recording_id",
    "participant_id",
    "person",
    "kind",
    "start_s",
    "end_s",
    "start_frame",
    "end_frame",
    "is_at_me",
    "annotation_target",
    "source_row",
)
"""Canonical social segments: ``kind`` is ``looking`` or ``talking``;
``participant_id`` is null when ``person`` does not resolve to a participant."""

TRACK_COLUMNS = (
    "recording_id",
    "participant_id",
    "track_id",
    "frame",
    "time_s",
    "x",
    "y",
    "width",
    "height",
)
"""Canonical face-track boxes: one row per (participant, track, frame)."""


@dataclass(frozen=True)
class ParticipantFacts:
    """One participant of one recording."""

    participant_id: str
    is_wearer: bool
    speech: tuple[NativeAnnotation, ...] = ()
    """Native speech intervals; may overlap each other (they are unioned)."""
    unknown: tuple[NativeAnnotation, ...] = ()
    """Regions where this participant's speech is not annotated."""
    resolved: bool = True
    """False only for :data:`UNRESOLVED_PARTICIPANT_ID`."""
    metadata: Mapping[str, object] = field(default_factory=dict)
    """Native per-participant fields (keys documented by the registry)."""


@dataclass(frozen=True)
class RecordingFacts:
    """Every canonical fact about one recording (one point-of-view session)."""

    dataset: str
    recording_id: str
    conversation_id: str
    view_id: str
    wearer_id: str
    start_s: float
    end_s: float
    participants: tuple[ParticipantFacts, ...]
    unknown: tuple[NativeAnnotation, ...] = ()
    """Recording-level UNKNOWN regions (apply to every participant)."""
    frame_rate_hz: float | None = None
    """Native video frame rate, when frames index any fact."""
    metadata: Mapping[str, object] = field(default_factory=dict)
    """Native per-recording fields (keys documented by the registry)."""

    def __post_init__(self) -> None:
        if not self.end_s > self.start_s:
            raise ValueError(f"{self.recording_id}: empty recording window")
        ids = [participant.participant_id for participant in self.participants]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.recording_id}: duplicate participant ids")
        wearers = [p for p in self.participants if p.is_wearer]
        if len(wearers) != 1 or wearers[0].participant_id != self.wearer_id:
            raise ValueError(
                f"{self.recording_id}: exactly one participant must be the wearer "
                f"{self.wearer_id!r}"
            )


@dataclass(frozen=True)
class LabelSource:
    """Canonical facts of one dataset plus the provenance its reports need."""

    dataset: str
    recordings: list[RecordingFacts]
    annotation_paths: tuple[Path, ...]
    """Every interim table the facts were read from (checksummed by the build)."""
    annotation_schema_version: str
    cleaning_rule_version: str
    tokens: pd.DataFrame | None = None
    """:data:`TOKEN_COLUMNS` rows, or ``None`` without transcripts."""
    social: pd.DataFrame | None = None
    """:data:`SOCIAL_COLUMNS` rows, or ``None`` without social annotations."""
    tracks: pd.DataFrame | None = None
    """:data:`TRACK_COLUMNS` rows, or ``None`` without face tracks."""
    statistics: dict[str, int] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()


LabelFactsLoader = Callable[["DictConfig", pd.DataFrame], LabelSource]
"""Given the configuration and the media metadata table, load the canonical facts."""


@dataclass(frozen=True)
class LabelFactsSpec:
    """What a dataset declares on its ``DatasetSpec`` to take part in labels."""

    provides: frozenset[str]
    """Facts the loader delivers; label support is computed from them."""
    load: LabelFactsLoader

    def __post_init__(self) -> None:
        unknown = set(self.provides) - KNOWN_FACTS
        if unknown:
            raise ValueError(f"unknown label facts {sorted(unknown)}")


def participant_order(participants: tuple[ParticipantFacts, ...]) -> list[int]:
    """Canonical participant order: the wearer first, then natural id order.

    Natural order sorts purely numeric ids numerically (``"2" < "10"``) and
    keeps the unresolved pseudo-participant last, so ``participant_index`` 0
    is always the wearer.
    """

    def key(position: int) -> tuple[int, int, str]:
        participant = participants[position]
        if participant.is_wearer:
            return (0, 0, "")
        if not participant.resolved:
            return (3, 0, participant.participant_id)
        identifier = participant.participant_id
        if identifier.isdigit():
            return (1, int(identifier), identifier)
        return (2, 0, identifier)

    return sorted(range(len(participants)), key=key)


def validate_frame(table: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """Require the canonical columns of an optional facts table."""
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} facts miss columns {sorted(missing)}")


__all__ = [
    "FACE_TRACKS",
    "KNOWN_FACTS",
    "MEDIA_AUDIO",
    "MEDIA_VIDEO",
    "META_BACKGROUND",
    "META_HOST",
    "META_NATIVE_SPEAKER",
    "META_SOURCE_OFFSET",
    "SOCIAL_COLUMNS",
    "SOCIAL_LOOKING",
    "SOCIAL_TALKING",
    "SPEECH",
    "TOKEN_COLUMNS",
    "TRACK_COLUMNS",
    "TRANSCRIPT",
    "UNRESOLVED_PARTICIPANT_ID",
    "WORDS",
    "LabelFactsLoader",
    "LabelFactsSpec",
    "LabelSource",
    "ParticipantFacts",
    "RecordingFacts",
    "participant_order",
    "validate_frame",
]
