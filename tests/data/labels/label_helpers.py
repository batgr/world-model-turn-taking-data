"""Tiny builders for synthetic multi-participant recordings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from conv_wm.data.labels.facts import ParticipantFacts, RecordingFacts
from conv_wm.data.vocal.native_state import NativeAnnotation


def spans(
    pairs: Sequence[tuple[float, float]], prefix: str
) -> tuple[NativeAnnotation, ...]:
    return tuple(
        NativeAnnotation(s, e, f"{prefix}#{i}") for i, (s, e) in enumerate(pairs)
    )


def recording(
    speech: Mapping[str, Sequence[tuple[float, float]]],
    *,
    wearer: str = "w",
    start: float = 0.0,
    end: float = 10.0,
    unknown: Sequence[tuple[float, float]] = (),
    participant_unknown: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    recording_id: str = "r1",
    dataset: str = "toy",
    unresolved: Sequence[str] = (),
    metadata: Mapping[str, object] | None = None,
    participant_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> RecordingFacts:
    participant_unknown = participant_unknown or {}
    participant_metadata = participant_metadata or {}
    ids = dict.fromkeys([wearer, *speech])
    participants = tuple(
        ParticipantFacts(
            participant_id=pid,
            is_wearer=pid == wearer,
            speech=spans(speech.get(pid, ()), f"speech-{pid}"),
            unknown=spans(participant_unknown.get(pid, ()), f"unknown-{pid}"),
            resolved=pid not in unresolved,
            metadata=participant_metadata.get(pid, {}),
        )
        for pid in ids
    )
    return RecordingFacts(
        dataset=dataset,
        recording_id=recording_id,
        conversation_id=f"conv-{recording_id}",
        view_id=f"view-{recording_id}",
        wearer_id=wearer,
        start_s=start,
        end_s=end,
        participants=participants,
        unknown=spans(unknown, "unknown"),
        metadata=metadata or {},
    )
