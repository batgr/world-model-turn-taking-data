"""The speech extractor: canonical speech facts -> its five label tables.

Pure: takes the recordings' canonical facts and the action grid's keys, and
returns Arrow tables. The build stage (:mod:`conv_wm.data.pipeline.labels`)
does the I/O and the provenance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from conv_wm.data.labels.facts import RecordingFacts
from conv_wm.data.labels.gridding import GridFrame, speech_grid
from conv_wm.data.labels.profiles import PROFILE_FEATURES, participant_profiles
from conv_wm.data.labels.registry import Table
from conv_wm.data.labels.timeline import (
    NO_SPEAKER,
    LabelConfig,
    RecordingStructure,
    derive,
    transfer_kind,
    transfer_locality,
)


@dataclass(frozen=True)
class GridKeys:
    """The action grid's rows of one recording: the grid every label table shares."""

    decision_index: np.ndarray
    decision_time_s: np.ndarray


def grid_keys_by_recording(grid: pa.Table | Any) -> dict[str, GridKeys]:
    """Split an action grid (pandas or Arrow) into sorted per-recording keys."""
    frame = grid.to_pandas() if isinstance(grid, pa.Table) else grid
    frame = frame.sort_values(["recording_id", "decision_index"], kind="stable")
    keys: dict[str, GridKeys] = {}
    for recording_id, group in frame.groupby("recording_id", sort=True):
        keys[str(recording_id)] = GridKeys(
            group["decision_index"].to_numpy(dtype=np.int64),
            group["decision_time_s"].to_numpy(dtype=float),
        )
    return keys


EVENT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("event_type", pa.string()),
        ("time_s", pa.float64()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("is_ego", pa.bool_()),
        ("previous_unique_speaker_index", pa.int16()),
        ("was_last_unique_speaker", pa.bool_()),
        ("others_active_before", pa.bool_()),
        ("others_active_at_onset", pa.bool_()),
        ("silence_before_s", pa.float64()),
        ("simultaneous_other_onset", pa.bool_()),
        ("onset_type", pa.string()),
        ("context_valid", pa.bool_()),
        ("previous_holder_index", pa.int16()),
        ("next_holder_index", pa.int16()),
        ("fto_s", pa.float64()),
        ("transfer_kind", pa.string()),
        ("locality", pa.string()),
        ("previous_run_end_s", pa.float64()),
        ("next_run_start_s", pa.float64()),
    ]
)

SEGMENT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("segment_type", pa.string()),
        ("segment_id", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("is_ego", pa.bool_()),
        ("start_censored", pa.bool_()),
        ("end_censored", pa.bool_()),
        ("valid", pa.bool_()),
        ("run_count", pa.int32()),
        ("pause_count", pa.int32()),
        ("pause_total_s", pa.float64()),
        ("pause_max_s", pa.float64()),
        ("previous_turn_participant_index", pa.int16()),
        ("next_turn_participant_index", pa.int16()),
        ("fto_from_previous_s", pa.float64()),
        ("initial_participant_index", pa.int16()),
        ("entering_participant_index", pa.int16()),
        ("overlap_type", pa.string()),
        ("previous_floor_holder_index", pa.int16()),
        ("silence_type", pa.string()),
        ("previous_speaker_index", pa.int16()),
        ("next_speaker_index", pa.int16()),
    ]
)

RECORDING_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("dataset", pa.string()),
        ("conversation_id", pa.string()),
        ("view_id", pa.string()),
        ("wearer_id", pa.string()),
        ("participant_ids", pa.list_(pa.string())),
        ("participant_count", pa.int16()),
        ("wearer_index", pa.int16()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("source_offset_s", pa.float64()),
        ("background_fan", pa.bool_()),
        ("background_music", pa.bool_()),
    ]
)
RECORDING_METADATA = ("source_offset_s", "background_fan", "background_music")
PARTICIPANT_METADATA = ("native_speaker", "is_host")

PARTICIPANT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("is_ego", pa.bool_()),
        ("is_resolved", pa.bool_()),
        ("native_speaker", pa.bool_()),
        ("is_host", pa.bool_()),
        *[(name, pa.float64()) for name in PROFILE_FEATURES],
        ("interaction_profile_features", pa.list_(pa.float64())),
        ("interaction_profile_feature_names", pa.list_(pa.string())),
    ]
)


def _index(value: int) -> int | None:
    return None if value == NO_SPEAKER or value < 0 else int(value)


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _event_rows(structure: RecordingStructure) -> list[dict[str, Any]]:
    recording = structure.recording
    ids = structure.pieces.participant_ids
    events = structure.events
    rows: list[dict[str, Any]] = []
    for index in range(len(events.time_s)):
        participant = int(events.participant[index])
        row: dict[str, Any] = {
            "recording_id": recording.recording_id,
            "event_type": str(events.kind[index]),
            "time_s": float(events.time_s[index]),
            "participant_index": participant,
            "participant_id": ids[participant],
            "is_ego": participant == 0,
        }
        context = structure.contexts.get(index)
        if context is not None:
            previous = _index(context.previous_unique_speaker)
            row.update(
                previous_unique_speaker_index=previous,
                was_last_unique_speaker=None
                if previous is None
                else previous == participant,
                others_active_before=context.others_active_before,
                others_active_at_onset=context.others_active_at_onset,
                silence_before_s=context.silence_before_s,
                simultaneous_other_onset=context.simultaneous_other_onset,
                onset_type=context.onset_type,
                context_valid=context.context_valid,
            )
        rows.append(row)
    for change in structure.floor_changes:
        rows.append(
            {
                "recording_id": recording.recording_id,
                "event_type": "floor_change",
                "time_s": change.time_s,
                "participant_index": change.next_holder,
                "participant_id": ids[change.next_holder],
                "is_ego": change.next_holder == 0,
                "previous_holder_index": change.previous_holder,
                "next_holder_index": change.next_holder,
                "fto_s": _finite(change.fto_s),
                "transfer_kind": transfer_kind(change.fto_s),
                "locality": None,
                "previous_run_end_s": _finite(change.previous_run_end_s),
                "next_run_start_s": change.next_run_start_s,
            }
        )
    return rows


def _segment_rows(structure: RecordingStructure) -> list[dict[str, Any]]:
    recording_id = structure.recording.recording_id
    ids = structure.pieces.participant_ids
    rows: list[dict[str, Any]] = []
    for ordinal, run in enumerate(structure.runs):
        rows.append(
            {
                "segment_type": "speech_run",
                "segment_id": ordinal,
                "start_s": run.start_s,
                "end_s": run.end_s,
                "participant_index": run.participant,
                "participant_id": ids[run.participant],
                "is_ego": run.participant == 0,
                "start_censored": run.start_censored,
                "end_censored": run.end_censored,
                "valid": not (run.start_censored or run.end_censored),
            }
        )
    for ordinal, turn in enumerate(structure.turns):
        rows.append(
            {
                "segment_type": "turn",
                "segment_id": ordinal,
                "start_s": turn.start_s,
                "end_s": turn.end_s,
                "participant_index": turn.participant,
                "participant_id": ids[turn.participant],
                "is_ego": turn.participant == 0,
                "start_censored": turn.start_censored,
                "end_censored": turn.end_censored,
                "valid": not (turn.start_censored or turn.end_censored),
                "run_count": turn.run_count,
                "pause_count": len(turn.pauses),
                "pause_total_s": float(sum(turn.pauses)),
                "pause_max_s": float(max(turn.pauses)) if turn.pauses else None,
                "previous_turn_participant_index": _index(turn.previous_participant),
                "next_turn_participant_index": _index(turn.next_participant),
                "fto_from_previous_s": _finite(turn.fto_from_previous_s),
            }
        )
    for ordinal, overlap in enumerate(structure.overlaps):
        rows.append(
            {
                "segment_type": "overlap",
                "segment_id": ordinal,
                "start_s": overlap.start_s,
                "end_s": overlap.end_s,
                "valid": overlap.valid,
                "initial_participant_index": overlap.initial,
                "entering_participant_index": overlap.entering,
                "overlap_type": overlap.overlap_type,
                "previous_floor_holder_index": _index(overlap.previous_floor_holder),
            }
        )
    for ordinal, silence in enumerate(structure.silences):
        rows.append(
            {
                "segment_type": "silence",
                "segment_id": ordinal,
                "start_s": silence.start_s,
                "end_s": silence.end_s,
                "valid": silence.valid,
                "silence_type": silence.silence_type,
                "previous_speaker_index": _index(silence.previous_speaker),
                "next_speaker_index": _index(silence.next_speaker),
            }
        )
    for row in rows:
        row["recording_id"] = recording_id
        row["duration_s"] = row["end_s"] - row["start_s"]
    return rows


def _recording_row(structure: RecordingStructure) -> dict[str, Any]:
    recording = structure.recording
    ids = list(structure.pieces.participant_ids)
    row: dict[str, Any] = {
        "recording_id": recording.recording_id,
        "dataset": recording.dataset,
        "conversation_id": recording.conversation_id,
        "view_id": recording.view_id,
        "wearer_id": recording.wearer_id,
        "participant_ids": ids,
        "participant_count": len(ids),
        "wearer_index": 0,
        "start_s": recording.start_s,
        "end_s": recording.end_s,
        "duration_s": recording.end_s - recording.start_s,
    }
    for key in RECORDING_METADATA:
        row[key] = recording.metadata.get(key)
    return row


def _participant_rows(structure: RecordingStructure) -> list[dict[str, Any]]:
    recording = structure.recording
    by_id = {p.participant_id: p for p in recording.participants}
    rows: list[dict[str, Any]] = []
    for index, (pid, profile) in enumerate(
        zip(
            structure.pieces.participant_ids,
            participant_profiles(structure),
            strict=True,
        )
    ):
        facts = by_id[pid]
        row: dict[str, Any] = {
            "recording_id": recording.recording_id,
            "participant_index": index,
            "participant_id": pid,
            "is_ego": index == 0,
            "is_resolved": facts.resolved,
        }
        for key in PARTICIPANT_METADATA:
            row[key] = facts.metadata.get(key)
        for name in PROFILE_FEATURES:
            row[name] = _finite(profile[name])
        row["interaction_profile_features"] = [
            _finite(profile[n]) for n in PROFILE_FEATURES
        ]
        row["interaction_profile_feature_names"] = list(PROFILE_FEATURES)
        rows.append(row)
    return rows


def rows_table(rows: Iterable[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    """An Arrow table of ``schema`` from row mappings (absent keys are null)."""
    rows = list(rows)
    return pa.Table.from_pylist(
        [{name: row.get(name) for name in schema.names} for row in rows], schema=schema
    )


def _with_locality(
    rows: list[dict[str, Any]], config: LabelConfig
) -> list[dict[str, Any]]:
    for row in rows:
        if row["event_type"] == "floor_change" and row.get("fto_s") is not None:
            row["locality"] = transfer_locality(row["fto_s"], config)
    return rows


@dataclass(frozen=True)
class SpeechTables:
    """The speech extractor's tables and the counts its manifest reports."""

    tables: dict[Table, pa.Table]
    statistics: dict[str, Any]


def derive_structures(
    recordings: Sequence[RecordingFacts],
    grid_keys: Mapping[str, GridKeys],
    config: LabelConfig,
) -> dict[str, RecordingStructure]:
    """Derive every grid recording once; refuse a grid recording without facts."""
    missing = sorted(set(grid_keys) - {r.recording_id for r in recordings})
    if missing:
        raise ValueError(
            f"{len(missing)} action-grid recordings have no label facts, e.g. "
            f"{missing[:3]}: the facts and the grid come from different annotations"
        )
    return {
        recording.recording_id: derive(recording, config)
        for recording in sorted(recordings, key=lambda item: item.recording_id)
        if recording.recording_id in grid_keys
    }


def speech_tables(
    structures: Mapping[str, RecordingStructure],
    grid_keys: Mapping[str, GridKeys],
    config: LabelConfig,
    *,
    step_s: float,
) -> SpeechTables:
    """Every speech-extractor table for the derived recordings, in recording order."""
    grid_parts: list[pa.Table] = []
    events: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    recording_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for recording_id in sorted(structures):
        structure = structures[recording_id]
        keys = grid_keys[recording_id]
        frame = GridFrame(keys.decision_index, step_s, config.subframes_per_step)
        grid_parts.append(
            pa.table(
                {
                    **key_columns(recording_id, keys),
                    **speech_grid(structure, frame, config),
                }
            )
        )
        events.extend(_with_locality(_event_rows(structure), config))
        segments.extend(_segment_rows(structure))
        recording_rows.append(_recording_row(structure))
        participant_rows.extend(_participant_rows(structure))
    events.sort(
        key=lambda r: (
            r["recording_id"],
            r["time_s"],
            r["event_type"],
            r["participant_index"],
        )
    )
    segments.sort(key=lambda r: (r["recording_id"], r["segment_type"], r["segment_id"]))
    tables = {
        Table.GRID: pa.concat_tables(grid_parts) if grid_parts else _empty_grid(),
        Table.EVENTS: rows_table(events, EVENT_SCHEMA),
        Table.SEGMENTS: rows_table(segments, SEGMENT_SCHEMA),
        Table.RECORDINGS: rows_table(recording_rows, RECORDING_SCHEMA),
        Table.PARTICIPANTS: rows_table(participant_rows, PARTICIPANT_SCHEMA),
    }
    return SpeechTables(
        tables=tables,
        statistics={
            "recordings": len(recording_rows),
            "grid_rows": tables[Table.GRID].num_rows,
            "events": tables[Table.EVENTS].num_rows,
            "segments": tables[Table.SEGMENTS].num_rows,
            "participants": tables[Table.PARTICIPANTS].num_rows,
        },
    )


def key_columns(recording_id: str, keys: GridKeys) -> dict[str, pa.Array]:
    """The grid key columns of one recording, in key order."""
    return {
        "recording_id": pa.array(
            [recording_id] * len(keys.decision_index), pa.string()
        ),
        "decision_index": pa.array(keys.decision_index, pa.int64()),
        "decision_time_s": pa.array(keys.decision_time_s, pa.float64()),
    }


def _empty_grid() -> pa.Table:
    return pa.table(
        {
            "recording_id": pa.array([], pa.string()),
            "decision_index": pa.array([], pa.int64()),
            "decision_time_s": pa.array([], pa.float64()),
        }
    )


__all__ = [
    "EVENT_SCHEMA",
    "PARTICIPANT_SCHEMA",
    "RECORDING_SCHEMA",
    "SEGMENT_SCHEMA",
    "GridKeys",
    "SpeechTables",
    "derive_structures",
    "grid_keys_by_recording",
    "key_columns",
    "rows_table",
    "speech_tables",
]
