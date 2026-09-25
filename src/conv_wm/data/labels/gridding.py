"""Continuous-time structure -> the action grid's cells and subframes.

The grid is the action grid's: cell ``k`` is ``[t_k, t_k + Δ)`` with
``t_k = k·Δ`` and exactly the ``decision_index`` values the action grid
emitted for the recording. Each cell is split into ``S`` equal subframes.
Native events are placed in the subframe containing their exact time (with
the action grid's nanosecond boundary tolerance); intervals mark every unit
they intersect with positive length.

Three-valued logic throughout: a unit whose required participants are
UNKNOWN for any positive time is ``null``, never ``False``.

Point-in-time quantities (timing, next speaker, future targets) are evaluated
at the cell end ``r_k = t_k + Δ``: everything strictly before ``r_k`` is
observed, an event at exactly ``r_k`` belongs to the future. Censoring is
explicit: a value that would need time beyond the observed span containing
``r_k`` is null, and ``timing.observation_bounds`` gives the bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from conv_wm.data.labels.timeline import (
    EPSILON_S,
    MULTIPLE,
    NO_SPEAKER,
    NONE,
    ONSET_TYPES,
    SPEAKING,
    UNKNOWN,
    LabelConfig,
    Pieces,
    RecordingStructure,
)
from conv_wm.data.vocal.action_grid import slot_index

SILENT_CODE, SPEAKING_CODE, UNKNOWN_CODE = 0, 1, 2


@dataclass(frozen=True)
class GridFrame:
    """The cells of one recording and their subframe boundaries."""

    decision_index: np.ndarray
    step_s: float
    subframes: int

    @property
    def cell_start(self) -> np.ndarray:
        """``t_k`` of every cell."""
        return self.decision_index * self.step_s

    @property
    def cell_end(self) -> np.ndarray:
        """``r_k = t_k + Δ``: the reference instant of point-in-time labels."""
        return self.cell_start + self.step_s

    @property
    def subframe_s(self) -> float:
        """Subframe duration ``Δ / S``."""
        return self.step_s / self.subframes

    @property
    def subframe_start(self) -> np.ndarray:
        """``(n_cells, S)`` subframe start times."""
        offsets = np.arange(self.subframes) * self.subframe_s
        return self.cell_start[:, None] + offsets[None, :]

    def locate(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(row, subframe, inside)`` of the subframe containing each time."""
        global_index = slot_index(np.asarray(times, dtype=float), self.subframe_s)
        cell = global_index // self.subframes
        row = np.searchsorted(self.decision_index, cell)
        row_clipped = np.clip(row, 0, max(len(self.decision_index) - 1, 0))
        inside = (row < len(self.decision_index)) & (
            self.decision_index[row_clipped] == cell
            if len(self.decision_index)
            else False
        )
        return row_clipped, global_index - cell * self.subframes, inside


class Cumulative:
    """Time spent in a condition before ``t``, evaluable at any array of times."""

    def __init__(self, pieces: Pieces, mask: np.ndarray) -> None:
        durations = (pieces.ends - pieces.starts) * mask
        self.edges = np.concatenate([pieces.starts, pieces.ends[-1:]])
        self.values = np.concatenate([[0.0], np.cumsum(durations)])

    def between(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Time inside ``[start, end)`` spent in the condition."""
        return np.interp(end, self.edges, self.values) - np.interp(
            start, self.edges, self.values
        )


class IntervalCoverage:
    """Same as :class:`Cumulative` for a union of arbitrary intervals."""

    def __init__(self, intervals: list[tuple[float, float]]) -> None:
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        if not merged:
            self.edges = np.array([0.0, 1.0])
            self.values = np.array([0.0, 0.0])
            return
        edges: list[float] = []
        values: list[float] = []
        total = 0.0
        for start, end in merged:
            edges += [start, end]
            values += [total, total + end - start]
            total += end - start
        self.edges = np.asarray(edges)
        self.values = np.asarray(values)

    def between(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Time inside ``[start, end)`` covered by the intervals."""
        return np.interp(end, self.edges, self.values) - np.interp(
            start, self.edges, self.values
        )


def _tri_state(
    speaking: Cumulative, unknown: Cumulative, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """SILENT / SPEAKING / UNKNOWN code of each unit ``[start, end)``."""
    codes = np.where(
        speaking.between(start, end) > EPSILON_S, SPEAKING_CODE, SILENT_CODE
    )
    return np.where(
        unknown.between(start, end) > EPSILON_S, UNKNOWN_CODE, codes
    ).astype(np.int8)


def _any_three_valued(codes: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """``(value, null)`` of 'any participant speaks' along ``axis``."""
    if codes.shape[axis] == 0:
        shape = list(codes.shape)
        shape.pop(axis)
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    value = (codes == SPEAKING_CODE).any(axis=axis)
    null = ~value & (codes == UNKNOWN_CODE).any(axis=axis)
    return value, null


def _left_piece(pieces: Pieces, times: np.ndarray) -> np.ndarray:
    """Piece containing ``t - ε`` (the state just before ``t``); -1 at the start."""
    index = (
        np.searchsorted(pieces.starts, np.asarray(times) - EPSILON_S, side="left") - 1
    )
    return np.where(index >= 0, index, -1)


# -- Arrow builders ---------------------------------------------------------


def fixed(
    values: np.ndarray, null: np.ndarray | None, size: int, kind: pa.DataType
) -> pa.Array:
    """``fixed_size_list<kind, size>`` from a ``(rows, size)`` array."""
    flat = pa.array(
        values.reshape(-1),
        type=kind,
        mask=None if null is None else null.reshape(-1),
    )
    return pa.FixedSizeListArray.from_arrays(flat, size)


def nullable_vectors(
    values: np.ndarray, null: np.ndarray, kind: pa.DataType
) -> pa.Array:
    """``list<kind>`` of ``(rows, size)`` vectors, a whole row null where ``null``.

    A variable-size list, not a fixed-size one: pyarrow writes a null
    fixed-size list to Parquet but cannot read it back.
    """
    rows, size = values.shape
    flat = pa.array(values.reshape(-1), type=kind)
    offsets = pa.array(np.arange(rows + 1, dtype=np.int32) * size)
    return pa.ListArray.from_arrays(offsets, flat, mask=pa.array(null))


def per_participant(
    values: np.ndarray, null: np.ndarray | None, kind: pa.DataType
) -> pa.Array:
    """``list<fixed_size_list<kind, S>>`` from ``(rows, participants, S)``."""
    rows, participants, size = values.shape
    inner = fixed(
        values.reshape(-1, size),
        None if null is None else null.reshape(-1, size),
        size,
        kind,
    )
    offsets = pa.array(np.arange(rows + 1, dtype=np.int32) * participants)
    return pa.ListArray.from_arrays(offsets, inner)


def participant_list(
    values: np.ndarray, null: np.ndarray | None, kind: pa.DataType
) -> pa.Array:
    """``list<kind>`` from ``(rows, participants)``."""
    rows, participants = values.shape
    flat = pa.array(
        values.reshape(-1), type=kind, mask=None if null is None else null.reshape(-1)
    )
    offsets = pa.array(np.arange(rows + 1, dtype=np.int32) * participants)
    return pa.ListArray.from_arrays(offsets, flat)


def horizon_participants(
    values: np.ndarray, null: np.ndarray, kind: pa.DataType
) -> pa.Array:
    """``fixed_size_list<list<kind>, H>`` from ``(rows, H, participants)``."""
    rows, horizons, participants = values.shape
    flat = pa.array(values.reshape(-1), type=kind, mask=null.reshape(-1))
    offsets = pa.array(np.arange(rows * horizons + 1, dtype=np.int32) * participants)
    inner = pa.ListArray.from_arrays(offsets, flat)
    return pa.FixedSizeListArray.from_arrays(inner, horizons)


def scalar(values: np.ndarray, null: np.ndarray | None, kind: pa.DataType) -> pa.Array:
    """A plain column with an optional null mask."""
    return pa.array(values, type=kind, mask=null)


# -- The speech extractor's grid --------------------------------------------


def speech_grid(
    structure: RecordingStructure, frame: GridFrame, config: LabelConfig
) -> dict[str, pa.Array]:
    """Every grid column of the speech extractor for one recording."""
    pieces = structure.pieces
    count = len(frame.decision_index)
    participants = pieces.n_participants
    columns: dict[str, pa.Array] = {}
    speaking = [
        Cumulative(pieces, pieces.states[:, p] == SPEAKING) for p in range(participants)
    ]
    unknown = [
        Cumulative(pieces, pieces.states[:, p] == UNKNOWN) for p in range(participants)
    ]

    sub_start = frame.subframe_start
    sub_end = sub_start + frame.subframe_s
    sub_codes = np.stack(
        [
            _tri_state(speaking[p], unknown[p], sub_start, sub_end)
            for p in range(participants)
        ],
        axis=2,
    )  # (n, S, P)
    cell_codes = np.stack(
        [
            _tri_state(speaking[p], unknown[p], frame.cell_start, frame.cell_end)
            for p in range(participants)
        ],
        axis=1,
    )  # (n, P)

    by_participant = sub_codes.transpose(0, 2, 1)
    columns["speaker_activity_subframes"] = per_participant(
        by_participant == SPEAKING_CODE, by_participant == UNKNOWN_CODE, pa.bool_()
    )
    columns["speaker_activity"] = participant_list(
        cell_codes == SPEAKING_CODE, cell_codes == UNKNOWN_CODE, pa.bool_()
    )
    ego_sub, ego_cell = sub_codes[:, :, 0], cell_codes[:, 0]
    columns["ego_speaking_subframes"] = fixed(
        ego_sub == SPEAKING_CODE, ego_sub == UNKNOWN_CODE, frame.subframes, pa.bool_()
    )
    columns["ego_speaking"] = scalar(
        ego_cell == SPEAKING_CODE, ego_cell == UNKNOWN_CODE, pa.bool_()
    )
    others_sub, others_sub_null = _any_three_valued(sub_codes[:, :, 1:], axis=2)
    others_cell, others_cell_null = _any_three_valued(cell_codes[:, 1:], axis=1)
    columns["others_active_subframes"] = fixed(
        others_sub, others_sub_null, frame.subframes, pa.bool_()
    )
    columns["others_active"] = scalar(others_cell, others_cell_null, pa.bool_())

    any_unknown = (sub_codes == UNKNOWN_CODE).any(axis=2)
    joint = (ego_sub == SPEAKING_CODE).astype(np.int8) + 2 * others_sub.astype(np.int8)
    joint_null = (ego_sub == UNKNOWN_CODE) | others_sub_null
    columns["joint_speech_state_subframes"] = fixed(
        joint, joint_null, frame.subframes, pa.int8()
    )
    speakers = (sub_codes == SPEAKING_CODE).sum(axis=2).astype(np.int8)
    columns["active_speaker_count_subframes"] = fixed(
        speakers, any_unknown, frame.subframes, pa.int8()
    )
    holder = np.where(speakers == 0, NONE, MULTIPLE).astype(np.int16)
    single = speakers == 1
    holder[single] = np.argmax(sub_codes == SPEAKING_CODE, axis=2)[single]
    columns["floor_holder_subframes"] = fixed(
        holder, any_unknown, frame.subframes, pa.int16()
    )
    columns["joint_speech_state_occupancy"] = _joint_occupancy(pieces, frame)

    columns.update(_event_subframes(structure, frame, speaking, unknown))
    columns.update(_onset_context_subframes(structure, frame))
    columns.update(_overlap_subframes(structure, frame, any_unknown))
    columns.update(_timing(structure, frame))
    columns.update(_next_speaker(structure, frame))
    columns.update(_future(structure, frame, config, speaking))
    columns["participant_ids"] = pa.array(
        [list(pieces.participant_ids)] * count, type=pa.list_(pa.string())
    )
    return columns


def _joint_occupancy(pieces: Pieces, frame: GridFrame) -> pa.Array:
    ego = pieces.states[:, 0]
    others = pieces.states[:, 1:]
    others_speaking = (others == SPEAKING).any(axis=1)
    known = pieces.all_known
    code = (ego == SPEAKING).astype(np.int8) + 2 * others_speaking.astype(np.int8)
    fractions = (
        np.stack(
            [
                Cumulative(pieces, known & (code == value)).between(
                    frame.cell_start, frame.cell_end
                )
                for value in range(4)
            ],
            axis=1,
        )
        / frame.step_s
    )
    hidden = (
        Cumulative(pieces, ~known).between(frame.cell_start, frame.cell_end) > EPSILON_S
    )
    return nullable_vectors(fractions.astype(np.float32), hidden, pa.float32())


def _event_subframes(
    structure: RecordingStructure,
    frame: GridFrame,
    speaking: list[Cumulative],
    unknown: list[Cumulative],
) -> dict[str, pa.Array]:
    pieces = structure.pieces
    events = structure.events
    count, size, participants = (
        len(frame.decision_index),
        frame.subframes,
        pieces.n_participants,
    )
    sub_start = frame.subframe_start
    sub_end = sub_start + frame.subframe_s
    left = _left_piece(pieces, sub_start)
    at_start = sub_start <= pieces.starts[0] + EPSILON_S
    valid = np.zeros((count, size, participants), dtype=bool)
    for p in range(participants):
        known_inside = unknown[p].between(sub_start, sub_end) <= EPSILON_S
        known_before = (left >= 0) & (
            pieces.states[np.clip(left, 0, None), p] != UNKNOWN
        )
        valid[:, :, p] = known_inside & known_before & ~at_start
    onset = np.zeros_like(valid)
    offset = np.zeros_like(valid)
    row, sub, inside = frame.locate(events.time_s)
    for flags, kind in ((onset, "onset"), (offset, "offset")):
        chosen = inside & (events.kind == kind)
        flags[row[chosen], sub[chosen], events.participant[chosen]] = True
    columns: dict[str, pa.Array] = {}
    for name, flags in (("onset", onset), ("offset", offset)):
        null = ~valid & ~flags
        columns[f"speaker_{name}_subframes"] = per_participant(
            flags.transpose(0, 2, 1), null.transpose(0, 2, 1), pa.bool_()
        )
        columns[f"ego_{name}_subframes"] = fixed(
            flags[:, :, 0], null[:, :, 0], size, pa.bool_()
        )
        other = flags[:, :, 1:].any(axis=2)
        other_null = (
            ~other & (~valid[:, :, 1:]).any(axis=2)
            if participants > 1
            else np.zeros_like(other)
        )
        columns[f"other_{name}_subframes"] = fixed(other, other_null, size, pa.bool_())
    columns["speaker_transition_valid_subframes"] = per_participant(
        valid.transpose(0, 2, 1), None, pa.bool_()
    )

    change = np.zeros((count, size), dtype=bool)
    previous = np.zeros((count, size), dtype=np.int16)
    following = np.zeros((count, size), dtype=np.int16)
    seen = np.zeros((count, size), dtype=bool)
    for item in structure.floor_changes:
        (r,), (s,), (ok,) = frame.locate(np.asarray([item.time_s]))
        if not ok:
            continue
        if not seen[r, s]:
            previous[r, s] = item.previous_holder
            seen[r, s] = True
        following[r, s] = item.next_holder
        change[r, s] = True
    all_known_inside = (
        Cumulative(pieces, ~pieces.all_known).between(sub_start, sub_end) <= EPSILON_S
    )
    floor_known = (left >= 0) & (structure.last_unique[np.clip(left, 0, None)] >= 0)
    change_null = ~change & ~(all_known_inside & floor_known & ~at_start)
    columns["floor_change_subframes"] = fixed(change, change_null, size, pa.bool_())
    columns["previous_floor_holder_subframes"] = fixed(
        previous, ~change, size, pa.int16()
    )
    columns["next_floor_holder_subframes"] = fixed(following, ~change, size, pa.int16())
    return columns


def _onset_context_subframes(
    structure: RecordingStructure, frame: GridFrame
) -> dict[str, pa.Array]:
    count, size = len(frame.decision_index), frame.subframes
    events = structure.events
    shape = (count, size)
    has = np.zeros(shape, dtype=bool)
    previous = np.zeros(shape, dtype=np.int16)
    previous_null = np.ones(shape, dtype=bool)
    was_last = np.zeros(shape, dtype=bool)
    was_last_null = np.ones(shape, dtype=bool)
    others_at = np.zeros(shape, dtype=bool)
    others_at_null = np.ones(shape, dtype=bool)
    silence = np.zeros(shape, dtype=np.float32)
    silence_null = np.ones(shape, dtype=bool)
    simultaneous = np.zeros(shape, dtype=bool)
    context_valid = np.zeros(shape, dtype=bool)
    onset_type = np.zeros(shape, dtype=np.int8)
    ego_onsets = sorted(
        (index for index in structure.contexts if events.participant[index] == 0),
        key=lambda index: events.time_s[index],
    )
    rows, subs, inside = (
        frame.locate(events.time_s[ego_onsets])
        if ego_onsets
        else (
            np.empty(0, int),
            np.empty(0, int),
            np.empty(0, bool),
        )
    )
    for index, r, s, ok in zip(ego_onsets, rows, subs, inside, strict=True):
        if not ok or has[r, s]:
            continue
        context = structure.contexts[index]
        has[r, s] = True
        if context.previous_unique_speaker != NO_SPEAKER:
            previous[r, s] = context.previous_unique_speaker
            previous_null[r, s] = False
            was_last[r, s] = context.previous_unique_speaker == 0
            was_last_null[r, s] = False
        if context.others_active_at_onset is not None:
            others_at[r, s] = context.others_active_at_onset
            others_at_null[r, s] = False
        if context.silence_before_s is not None:
            silence[r, s] = context.silence_before_s
            silence_null[r, s] = False
        simultaneous[r, s] = context.simultaneous_other_onset
        context_valid[r, s] = context.context_valid
        onset_type[r, s] = ONSET_TYPES.index(context.onset_type)
    return {
        "previous_unique_speaker_subframes": fixed(
            previous, previous_null, size, pa.int16()
        ),
        "ego_was_last_unique_speaker_subframes": fixed(
            was_last, was_last_null, size, pa.bool_()
        ),
        "others_active_at_ego_onset_subframes": fixed(
            others_at, others_at_null, size, pa.bool_()
        ),
        "silence_duration_before_ego_onset_subframes": fixed(
            silence, silence_null, size, pa.float32()
        ),
        "simultaneous_other_onset_subframes": fixed(
            simultaneous, ~has, size, pa.bool_()
        ),
        "ego_onset_context_valid_subframes": fixed(
            context_valid, None, size, pa.bool_()
        ),
        "ego_onset_type_subframes": fixed(onset_type, ~has, size, pa.int8()),
    }


def _overlap_subframes(
    structure: RecordingStructure, frame: GridFrame, any_unknown: np.ndarray
) -> dict[str, pa.Array]:
    sub_start = frame.subframe_start
    sub_end = sub_start + frame.subframe_s

    def covered(kinds: set[str]) -> np.ndarray:
        coverage = IntervalCoverage(
            [
                (o.start_s, o.end_s)
                for o in structure.overlaps
                if o.overlap_type in kinds
            ]
        )
        return coverage.between(sub_start, sub_end) > EPSILON_S

    undetermined = covered({"undetermined"})
    columns: dict[str, pa.Array] = {}
    for name, kind in (("within", "within"), ("between", "between")):
        value = covered({kind})
        null = ~value & (any_unknown | undetermined)
        columns[f"{name}_overlap_subframes"] = fixed(
            value, null, frame.subframes, pa.bool_()
        )
    value = covered({"simultaneous_onset"})
    columns["simultaneous_onset_overlap_subframes"] = fixed(
        value, ~value & any_unknown, frame.subframes, pa.bool_()
    )
    return columns


@dataclass(frozen=True)
class _Reference:
    """Where each cell end sits relative to the observed spans."""

    time: np.ndarray
    span: np.ndarray
    span_start: np.ndarray
    span_end: np.ndarray

    @property
    def observed(self) -> np.ndarray:
        """Whether the state just before ``r_k`` is observed."""
        return self.span >= 0


def _reference(structure: RecordingStructure, frame: GridFrame) -> _Reference:
    spans = structure.spans
    time = frame.cell_end
    span = spans.containing(time, left_limit=True)
    safe = np.clip(span, 0, None)
    starts = spans.starts[safe] if len(spans.starts) else np.full(len(time), np.nan)
    ends = spans.ends[safe] if len(spans.ends) else np.full(len(time), np.nan)
    return _Reference(
        time,
        span,
        np.where(span >= 0, starts, np.nan),
        np.where(span >= 0, ends, np.nan),
    )


def _since(times: np.ndarray, ref: _Reference) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the last event strictly before each reference (value, valid)."""
    times = np.sort(times)
    index = np.searchsorted(times, ref.time - EPSILON_S, side="left") - 1
    last = np.where(
        index >= 0, times[np.clip(index, 0, None)] if len(times) else 0.0, np.nan
    )
    valid = ref.observed & (index >= 0) & (last >= ref.span_start - EPSILON_S)
    return np.where(valid, ref.time - last, 0.0), valid


def _until(times: np.ndarray, ref: _Reference) -> tuple[np.ndarray, np.ndarray]:
    """Seconds to the first event at or after each reference (value, valid)."""
    times = np.sort(times)
    index = np.searchsorted(times, ref.time - EPSILON_S, side="left")
    ok = index < len(times)
    following = np.where(
        ok,
        times[np.clip(index, 0, max(len(times) - 1, 0))] if len(times) else 0.0,
        np.nan,
    )
    valid = ref.observed & ok & (following <= ref.span_end + EPSILON_S)
    return np.where(valid, np.maximum(following - ref.time, 0.0), 0.0), valid


def _timing(structure: RecordingStructure, frame: GridFrame) -> dict[str, pa.Array]:
    events = structure.events
    ref = _reference(structure, frame)
    onset = events.kind == "onset"
    offset = events.kind == "offset"
    ego = events.participant == 0
    streams = {
        "ego_onset": events.time_s[onset & ego],
        "ego_offset": events.time_s[offset & ego],
        "other_onset": events.time_s[onset & ~ego],
        "other_offset": events.time_s[offset & ~ego],
        "floor_change": np.asarray(
            [c.time_s for c in structure.floor_changes], dtype=float
        ),
        "speaker_onset": events.time_s[onset],
    }
    columns: dict[str, pa.Array] = {}
    for name, times in streams.items():
        if name != "speaker_onset":
            value, valid = _since(times, ref)
            columns[f"time_since_{name}"] = scalar(
                value.astype(np.float32), ~valid, pa.float32()
            )
            columns[f"time_since_{name}_valid"] = pa.array(valid)
        value, valid = _until(times, ref)
        columns[f"time_to_next_{name}"] = scalar(
            value.astype(np.float32), ~valid, pa.float32()
        )
        columns[f"time_to_next_{name}_valid"] = pa.array(valid)

    pieces = structure.pieces
    left = _left_piece(pieces, ref.time)
    safe = np.clip(left, 0, None)
    active_now = (
        (left >= 0) & (pieces.speaking_count[safe] > 0) & pieces.all_known[safe]
    )
    speech_ends = pieces.ends[(pieces.speaking_count > 0) & pieces.all_known]
    since_speech, since_valid = _since_end(speech_ends, ref)
    value = np.where(active_now, 0.0, since_speech)
    valid = ref.observed & (active_now | since_valid)
    columns["time_since_speaker_activity"] = scalar(
        value.astype(np.float32), ~valid, pa.float32()
    )
    columns["time_since_speaker_activity_valid"] = pa.array(valid)

    silence_start = np.full(len(ref.time), np.nan)
    for item in structure.silences:
        inside = (ref.time > item.start_s + EPSILON_S) & (
            ref.time <= item.end_s + EPSILON_S
        )
        silence_start[inside] = item.start_s
    silent_valid = (
        ref.observed
        & np.isfinite(silence_start)
        & (silence_start > ref.span_start + EPSILON_S)
    )
    columns["silence_duration"] = scalar(
        np.where(silent_valid, ref.time - silence_start, 0.0).astype(np.float32),
        ~silent_valid,
        pa.float32(),
    )
    columns["silence_duration_valid"] = pa.array(silent_valid)
    columns["time_since_observation_start"] = scalar(
        np.where(ref.observed, ref.time - ref.span_start, 0.0).astype(np.float32),
        ~ref.observed,
        pa.float32(),
    )
    columns["time_to_observation_end"] = scalar(
        np.where(ref.observed, ref.span_end - ref.time, 0.0).astype(np.float32),
        ~ref.observed,
        pa.float32(),
    )
    return columns


def _since_end(ends: np.ndarray, ref: _Reference) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the last interval end at or before each reference."""
    ends = np.sort(ends)
    index = np.searchsorted(ends, ref.time + EPSILON_S, side="right") - 1
    last = np.where(
        index >= 0, ends[np.clip(index, 0, None)] if len(ends) else 0.0, np.nan
    )
    valid = (index >= 0) & (last >= ref.span_start - EPSILON_S)
    return np.where(valid, np.maximum(ref.time - last, 0.0), 0.0), valid


def _next_onset(structure: RecordingStructure, times: np.ndarray, limit: np.ndarray):
    """Participant of the first onset in ``[t, limit)``; -1 none; ties flagged."""
    events = structure.events
    onset = events.kind == "onset"
    onset_times = events.time_s[onset]
    who = events.participant[onset]
    order = np.argsort(onset_times, kind="stable")
    onset_times, who = onset_times[order], who[order]
    index = np.searchsorted(onset_times, times - EPSILON_S, side="left")
    found = index < len(onset_times)
    safe = np.clip(index, 0, max(len(onset_times) - 1, 0))
    first = np.where(found, onset_times[safe] if len(onset_times) else np.inf, np.inf)
    within = found & (first < limit - EPSILON_S)
    participant = np.where(within, who[safe] if len(who) else -1, -1).astype(np.int16)
    tie = np.zeros(len(times), dtype=bool)
    if len(onset_times) > 1:
        nxt = np.clip(safe + 1, 0, len(onset_times) - 1)
        tie = (
            within
            & (safe + 1 < len(onset_times))
            & (np.abs(onset_times[nxt] - first) <= EPSILON_S)
            & (who[nxt] != who[safe])
        )
    return participant, within, tie


def _next_speaker(
    structure: RecordingStructure, frame: GridFrame
) -> dict[str, pa.Array]:
    ref = _reference(structure, frame)
    limit = np.where(ref.observed, ref.span_end + 2 * EPSILON_S, -np.inf)
    who, within, tie = _next_onset(structure, ref.time, limit)
    valid = ref.observed & within & ~tie
    columns = {
        "next_speaker": scalar(who, ~valid, pa.int16()),
        "next_speaker_valid": pa.array(valid),
    }
    pieces = structure.pieces
    holder = structure.holder
    solo = (holder >= 0) & structure.holder_known
    new_stretch = solo.copy()
    new_stretch[1:] &= ~(solo[:-1] & (holder[:-1] == holder[1:]))
    starts = pieces.starts[new_stretch]
    speaker = holder[new_stretch]
    index = np.searchsorted(starts, ref.time - EPSILON_S, side="left")
    found = index < len(starts)
    safe = np.clip(index, 0, max(len(starts) - 1, 0))
    at = np.where(found, starts[safe] if len(starts) else np.inf, np.inf)
    unique_valid = ref.observed & found & (at <= ref.span_end + EPSILON_S)
    unique = np.where(unique_valid, speaker[safe] if len(speaker) else 0, 0).astype(
        np.int16
    )
    columns["next_unique_speaker"] = scalar(unique, ~unique_valid, pa.int16())
    columns["next_unique_speaker_valid"] = pa.array(unique_valid)
    left = _left_piece(pieces, ref.time)
    current = np.where(
        left >= 0, structure.last_unique[np.clip(left, 0, None)], NO_SPEAKER
    )
    continues_valid = unique_valid & (current >= 0)
    columns["current_speaker_continues"] = scalar(
        unique == current, ~continues_valid, pa.bool_()
    )
    columns["current_speaker_continues_valid"] = pa.array(continues_valid)
    change_times = np.asarray([c.time_s for c in structure.floor_changes], dtype=float)
    change_target = np.asarray(
        [c.next_holder for c in structure.floor_changes], dtype=np.int16
    )
    index = np.searchsorted(change_times, ref.time - EPSILON_S, side="left")
    found = index < len(change_times)
    safe = np.clip(index, 0, max(len(change_times) - 1, 0))
    at = np.where(found, change_times[safe] if len(change_times) else np.inf, np.inf)
    target_valid = ref.observed & found & (at <= ref.span_end + EPSILON_S)
    target = np.where(target_valid, change_target[safe] if len(change_target) else 0, 0)
    columns["floor_transfer_target"] = scalar(
        target.astype(np.int16), ~target_valid, pa.int16()
    )
    columns["floor_transfer_target_valid"] = pa.array(target_valid)
    return columns


def _future(
    structure: RecordingStructure,
    frame: GridFrame,
    config: LabelConfig,
    speaking: list[Cumulative],
) -> dict[str, pa.Array]:
    pieces = structure.pieces
    events = structure.events
    ref = _reference(structure, frame)
    horizons = np.asarray(config.future_horizons_s, dtype=float)
    count, width, participants = len(ref.time), len(horizons), pieces.n_participants
    start = np.repeat(ref.time[:, None], width, axis=1)
    end = start + horizons[None, :]
    valid = ref.observed[:, None] & (end <= ref.span_end[:, None] + EPSILON_S)
    null = ~valid

    activity = np.stack(
        [speaking[p].between(start, end) > EPSILON_S for p in range(participants)],
        axis=2,
    )
    others = (
        activity[:, :, 1:].any(axis=2)
        if participants > 1
        else np.zeros((count, width), bool)
    )
    overlap = Cumulative(
        pieces, pieces.all_known & (pieces.speaking_count >= 2)
    ).between(start, end)

    before_end = _left_piece(pieces, end.reshape(-1)).reshape(count, width)
    safe = np.clip(before_end, 0, None)
    ego_state = pieces.states[safe, 0] == SPEAKING
    others_state = (
        (pieces.states[safe][:, :, 1:] == SPEAKING).any(axis=2)
        if participants > 1
        else np.zeros((count, width), bool)
    )
    joint = ego_state.astype(np.int8) + 2 * others_state.astype(np.int8)
    holder = structure.holder[safe].astype(np.int16)

    def any_event(kind: str, ego: bool | None) -> np.ndarray:
        chosen = events.kind == kind
        if ego is True:
            chosen &= events.participant == 0
        elif ego is False:
            chosen &= events.participant != 0
        times = np.sort(events.time_s[chosen])
        low = np.searchsorted(times, start.reshape(-1) - EPSILON_S, side="left")
        high = np.searchsorted(times, end.reshape(-1) - EPSILON_S, side="left")
        return (high > low).reshape(count, width)

    columns = {
        "future_speaker_activity": horizon_participants(
            activity, np.repeat(null[:, :, None], participants, axis=2), pa.bool_()
        ),
        "future_ego_activity": fixed(activity[:, :, 0], null, width, pa.bool_()),
        "future_others_activity": fixed(others, null, width, pa.bool_()),
        "future_joint_speech_state": fixed(joint, null, width, pa.int8()),
        "future_floor_holder": fixed(holder, null, width, pa.int16()),
        "future_overlap": fixed(overlap > EPSILON_S, null, width, pa.bool_()),
        "future_ego_onset": fixed(any_event("onset", True), null, width, pa.bool_()),
        "future_ego_offset": fixed(any_event("offset", True), null, width, pa.bool_()),
        "future_other_onset": fixed(any_event("onset", False), null, width, pa.bool_()),
        "future_other_offset": fixed(
            any_event("offset", False), null, width, pa.bool_()
        ),
    }
    who, within, tie = _next_onset(structure, start.reshape(-1), end.reshape(-1))
    who = np.where(within, who, -1).reshape(count, width).astype(np.int16)
    columns["future_next_speaker"] = fixed(
        who, null | tie.reshape(count, width), width, pa.int16()
    )
    return columns


__all__ = [
    "Cumulative",
    "GridFrame",
    "IntervalCoverage",
    "fixed",
    "horizon_participants",
    "nullable_vectors",
    "participant_list",
    "per_participant",
    "scalar",
    "speech_grid",
]
