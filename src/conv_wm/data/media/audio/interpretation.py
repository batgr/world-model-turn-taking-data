"""Interpretation of generic audio events with codec and dataset knowledge.

Generic analysis knows nothing about a dataset. This module applies two kinds of
context afterwards: the AAC priming region (codec knowledge, derived from
``skip_samples``) and an optional grid of known boundaries that a dataset
declares (for example the 300 s stitch grid observed in Ego4D recordings).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import pandas as pd

from conv_wm.data.media.audio.records import (
    EVENT_COLUMNS,
    AudioFileInterpretation,
    AudioFileTimelineRecord,
    AudioTimelineEvent,
    EventCategory,
    EventType,
)

EARLY_BOUNDARY_EXTRA_FRAMES = 2
"""Decoded frames after the skipped priming region still counted as 'early'."""


@dataclass(frozen=True)
class KnownBoundaryGrid:
    """Periodic positions where a dataset is known to join recordings.

    Events within ``tolerance_packets`` packet durations of ``k * period_sec``
    (``k >= 1``) are relabelled with ``category``.
    """

    period_sec: float
    tolerance_packets: float = 2.0
    category: EventCategory = EventCategory.STITCH_RELATED

    def __post_init__(self) -> None:
        if self.period_sec <= 0:
            raise ValueError("period_sec must be positive.")


@dataclass(frozen=True)
class InterpretedAudioEvent:
    """A generic event plus the context used to relabel it (one Parquet row)."""

    event: AudioTimelineEvent
    nearest_known_boundary_index: int | None
    nearest_known_boundary_time_sec: float
    distance_to_known_boundary_sec: float
    known_boundary_tolerance_sec: float
    near_known_boundary: bool
    early_boundary_tolerance_sec: float
    has_early_boundary_context: bool

    def to_row(self) -> dict[str, object]:
        """Flatten into the Parquet column layout (event columns first)."""
        row = self.event.to_row()
        row.update(
            {key: value for key, value in asdict(self).items() if key != "event"}
        )
        return row


INTERPRETED_EVENT_COLUMNS: tuple[str, ...] = EVENT_COLUMNS + (
    "nearest_known_boundary_index",
    "nearest_known_boundary_time_sec",
    "distance_to_known_boundary_sec",
    "known_boundary_tolerance_sec",
    "near_known_boundary",
    "early_boundary_tolerance_sec",
    "has_early_boundary_context",
)


def interpret_events(
    events: list[AudioTimelineEvent],
    *,
    sample_rate_hz: int,
    reference_decoded_samples: int,
    skip_samples: int,
    boundary_grid: KnownBoundaryGrid | None,
) -> list[InterpretedAudioEvent]:
    """Relabel generic events with codec-boundary and known-boundary context.

    Precedence: a PCM event inside the priming context (``skip_samples`` plus two
    frames after the origin) is a ``CODEC_BOUNDARY`` event even if it satisfied
    the dropout persistence rule, because its cause is known. Otherwise a
    dropout keeps its label, and a non-dropout episode near a known boundary
    takes the grid's category. Packet-clock errors are never relabelled.
    """
    packet_duration_sec = reference_decoded_samples / sample_rate_hz
    early_tolerance = (
        skip_samples + EARLY_BOUNDARY_EXTRA_FRAMES * reference_decoded_samples
    ) / sample_rate_hz
    interpreted: list[InterpretedAudioEvent] = []
    for event in events:
        if boundary_grid is not None:
            index = max(1, round(event.event_time_sec / boundary_grid.period_sec))
            boundary_time = index * boundary_grid.period_sec
            distance = abs(event.event_time_sec - boundary_time)
            tolerance = boundary_grid.tolerance_packets * packet_duration_sec
            near = distance <= tolerance
            grid_index: int | None = index
        else:
            boundary_time = distance = tolerance = math.nan
            near = False
            grid_index = None

        is_pcm_event = event.event_type in (
            EventType.PCM_STEP_EPISODE,
            EventType.AUDIO_DROPOUT,
        )
        early_context = 0.0 <= event.event_time_sec <= early_tolerance
        is_early = (
            is_pcm_event
            and early_context
            and event.event_end_time_sec <= early_tolerance
            and skip_samples > 0
        )
        category = event.event_category
        if is_early:
            category = EventCategory.CODEC_BOUNDARY
        elif (
            event.event_type == EventType.PCM_STEP_EPISODE
            and near
            and boundary_grid is not None
        ):
            category = boundary_grid.category
        interpreted.append(
            InterpretedAudioEvent(
                event=event.relabel(category),
                nearest_known_boundary_index=grid_index,
                nearest_known_boundary_time_sec=boundary_time,
                distance_to_known_boundary_sec=distance,
                known_boundary_tolerance_sec=tolerance,
                near_known_boundary=near,
                early_boundary_tolerance_sec=early_tolerance,
                has_early_boundary_context=early_context,
            )
        )
    return interpreted


def final_drift_flags(
    final_drift_ms: float, *, video_fps: float
) -> dict[str, float | bool]:
    """Strict (``>``) final-drift thresholds, including half a video frame."""
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    absolute = abs(final_drift_ms)
    half_frame = 500.0 / video_fps
    return {
        "abs_final_cumulative_pcm_drift_ms": absolute,
        "half_video_frame_ms": half_frame,
        "final_pcm_drift_exceeds_1_ms": absolute > 1.0,
        "final_pcm_drift_exceeds_half_video_frame": absolute > half_frame,
        "final_pcm_drift_exceeds_100_ms": absolute > 100.0,
        "final_pcm_drift_exceeds_500_ms": absolute > 500.0,
    }


def interpret_file(
    record: AudioFileTimelineRecord,
    events: list[InterpretedAudioEvent],
) -> AudioFileTimelineRecord:
    """Attach file-level interpretation flags derived from ``events``."""
    if record.pcm_clock is None:
        return replace(
            record,
            interpretation=AudioFileInterpretation.empty(
                video_fps=record.video_avg_frame_rate
            ),
        )
    categories = [item.event.event_category for item in events]
    dropouts = [
        item.event
        for item in events
        if item.event.event_category == EventCategory.AUDIO_DROPOUT
    ]
    flags = final_drift_flags(
        record.pcm_clock.final_cumulative_pcm_drift_ms,
        video_fps=record.video_avg_frame_rate,
    )
    return replace(
        record,
        interpretation=AudioFileInterpretation(
            has_stitch_related_event=EventCategory.STITCH_RELATED in categories,
            has_early_boundary_event=EventCategory.CODEC_BOUNDARY in categories,
            has_audio_dropout=bool(dropouts),
            audio_dropout_event_count=len(dropouts),
            max_audio_dropout_duration_ms=(
                max(event.dropout_duration_ms for event in dropouts)
                if dropouts
                else 0.0
            ),
            total_audio_dropout_excess_ms=sum(
                event.dropout_duration_ms for event in dropouts
            ),
            has_compensated_timestamp_cadence=(
                EventCategory.COMPENSATED_TIMESTAMP_CADENCE in categories
            ),
            **flags,  # type: ignore[arg-type]
        ),
    )


def interpreted_events_frame(events: list[InterpretedAudioEvent]) -> pd.DataFrame:
    """Interpreted events as a DataFrame with the canonical column order."""
    if not events:
        return pd.DataFrame(columns=list(INTERPRETED_EVENT_COLUMNS))
    frame = pd.DataFrame.from_records([item.to_row() for item in events])
    return frame.reindex(columns=list(INTERPRETED_EVENT_COLUMNS))
