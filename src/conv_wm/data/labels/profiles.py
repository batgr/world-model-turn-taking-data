"""Per-participant interaction statistics over one recording.

Profiles summarize the *whole* recording, its future included: they are
non-causal diagnostics for analysis and probing, never an input for
predicting inside the same recording. Nothing here is fitted (no scaler, no
clustering); any fitted transform built on top of them must be fitted on the
training split only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from conv_wm.data.labels.timeline import SPEAKING, RecordingStructure

PROFILE_FEATURES = (
    "speaking_time_s",
    "observed_duration_s",
    "speaking_ratio",
    "speech_run_count",
    "run_duration_mean_s",
    "run_duration_median_s",
    "turn_count",
    "turns_per_minute",
    "turn_duration_mean_s",
    "turn_duration_median_s",
    "turn_duration_std_s",
    "pause_count",
    "pause_duration_mean_s",
    "pause_duration_median_s",
    "onset_count",
    "offset_count",
    "onset_after_silence_count",
    "onset_floor_transfer_count",
    "onset_overlap_count",
    "overlaps_entered",
    "overlaps_received",
    "within_overlaps_entered",
    "between_overlaps_entered",
    "interruption_like_count",
    "overlap_time_s",
    "floor_takes",
    "floor_yields",
    "fto_take_mean_s",
    "fto_take_median_s",
    "fto_take_q10_s",
    "fto_take_q90_s",
    "response_latency_mean_s",
    "response_latency_median_s",
)
"""Profile columns, in the order of ``interaction_profile_features``."""


def _stat(values: Sequence[float], how: str) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return math.nan
    if how == "mean":
        return float(array.mean())
    if how == "median":
        return float(np.median(array))
    if how == "std":
        return float(array.std())
    return float(np.quantile(array, float(how)))


def participant_profiles(structure: RecordingStructure) -> list[dict[str, float]]:
    """One mapping of :data:`PROFILE_FEATURES` per participant, in canonical order."""
    pieces = structure.pieces
    durations = pieces.ends - pieces.starts
    observed = float(durations[pieces.all_known].sum())
    events = structure.events
    overlap_pieces = pieces.all_known & (pieces.speaking_count >= 2)
    profiles: list[dict[str, float]] = []
    for participant in range(pieces.n_participants):
        speaking = pieces.states[:, participant] == SPEAKING
        speaking_time = float(durations[speaking & pieces.all_known].sum())
        runs = [r.duration_s for r in structure.runs if r.participant == participant]
        turns = [t.duration_s for t in structure.turns if t.participant == participant]
        pauses = [
            s.end_s - s.start_s
            for s in structure.silences
            if s.silence_type == "pause" and s.previous_speaker == participant
        ]
        own = events.participant == participant
        types = [
            context.onset_type
            for index, context in structure.contexts.items()
            if events.participant[index] == participant
        ]
        entered = [o for o in structure.overlaps if o.entering == participant]
        received = [o for o in structure.overlaps if o.initial == participant]
        takes = [c for c in structure.floor_changes if c.next_holder == participant]
        yields = [
            c for c in structure.floor_changes if c.previous_holder == participant
        ]
        ftos = [c.fto_s for c in takes if math.isfinite(c.fto_s)]
        latencies = [value for value in ftos if value > 0]
        minutes = observed / 60.0
        between = sum(o.overlap_type == "between" for o in entered)
        profiles.append(
            {
                "speaking_time_s": speaking_time,
                "observed_duration_s": observed,
                "speaking_ratio": speaking_time / observed if observed else math.nan,
                "speech_run_count": float(len(runs)),
                "run_duration_mean_s": _stat(runs, "mean"),
                "run_duration_median_s": _stat(runs, "median"),
                "turn_count": float(len(turns)),
                "turns_per_minute": len(turns) / minutes if minutes else math.nan,
                "turn_duration_mean_s": _stat(turns, "mean"),
                "turn_duration_median_s": _stat(turns, "median"),
                "turn_duration_std_s": _stat(turns, "std"),
                "pause_count": float(len(pauses)),
                "pause_duration_mean_s": _stat(pauses, "mean"),
                "pause_duration_median_s": _stat(pauses, "median"),
                "onset_count": float((own & (events.kind == "onset")).sum()),
                "offset_count": float((own & (events.kind == "offset")).sum()),
                "onset_after_silence_count": float(types.count("after_silence")),
                "onset_floor_transfer_count": float(types.count("floor_transfer")),
                "onset_overlap_count": float(types.count("overlap")),
                "overlaps_entered": float(len(entered)),
                "overlaps_received": float(len(received)),
                "within_overlaps_entered": float(
                    sum(o.overlap_type == "within" for o in entered)
                ),
                "between_overlaps_entered": float(between),
                "interruption_like_count": float(between),
                "overlap_time_s": float(durations[speaking & overlap_pieces].sum()),
                "floor_takes": float(len(takes)),
                "floor_yields": float(len(yields)),
                "fto_take_mean_s": _stat(ftos, "mean"),
                "fto_take_median_s": _stat(ftos, "median"),
                "fto_take_q10_s": _stat(ftos, "0.1"),
                "fto_take_q90_s": _stat(ftos, "0.9"),
                "response_latency_mean_s": _stat(latencies, "mean"),
                "response_latency_median_s": _stat(latencies, "median"),
            }
        )
    return profiles


__all__ = ["PROFILE_FEATURES", "participant_profiles"]
