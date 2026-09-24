"""Multi-participant speech facts -> continuous-time conversational structure.

Everything here works on native timestamps; nothing is quantized. The grid
views (:mod:`conv_wm.data.labels.gridding`) are projections of what this
module computes, so an event keeps its exact time and a grid label never
invents one.

The recording window is cut at every interval endpoint into elementary
*pieces*; inside a piece each participant is constant ``SILENT`` (0),
``SPEAKING`` (1) or ``UNKNOWN`` (2). Everything else is read off the pieces:

* **floor holder** — the *instantaneous* unique speaker of a piece: an index,
  ``NONE`` (-1, nobody speaks) or ``MULTIPLE`` (-2); UNKNOWN when any
  participant is unknown.
* **last unique speaker** — the participant who most recently spoke alone.
  It carries over silences and overlaps, and is unknown from the recording
  start (the conversation may have begun earlier) and after any UNKNOWN piece
  until somebody speaks alone again. A **floor change** is a change of the
  last unique speaker between two known participants.
* **events** — onsets and offsets at piece boundaries where a participant goes
  ``SILENT <-> SPEAKING``; transitions to or from UNKNOWN, and speech already
  in progress at the recording start, are not events.
* **overlaps** — pairwise, between a run of an *initial* participant and a run
  of another participant starting inside it (Heldner & Edlund 2010, as in the
  earlier ``mpc-wm`` labels): ``simultaneous_onset`` when both starts are
  within ``simultaneous_onset_tolerance_s``, else ``within`` when the entering
  run ends first, else ``between``.
* **silences** — maximal joint silences, typed ``pause`` (the same participant
  speaks before and after) or ``gap`` (another one does).
* **turns** — a participant's runs joined across internal pauses of at most
  ``turn_max_internal_pause_s`` that are fully observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from conv_wm.data.labels.facts import RecordingFacts, participant_order
from conv_wm.data.vocal.intervals import merge

SILENT, SPEAKING, UNKNOWN = 0, 1, 2
NONE, MULTIPLE = -1, -2
"""Floor-holder categories; UNKNOWN is carried separately (``holder_known``)."""
NO_SPEAKER = -3
"""Internal marker of an unknown last unique speaker; never written to a table."""

ONSET_TYPES = ("undetermined", "after_silence", "floor_transfer", "overlap")
"""Onset-type names; the grid stores their index (0-3)."""

EPSILON_S = 1e-9
"""Tolerance for comparing native timestamps (the action grid's boundary tolerance)."""

LABEL_RULES_VERSION = 1
"""Bumped when a derivation rule of this module changes meaning."""


@dataclass(frozen=True)
class LabelConfig:
    """Every parameter a label derivation depends on (recorded in manifests)."""

    subframes_per_step: int = 3
    """Subframes per Δ; 3 gives 30 Hz, one subframe per frame of 30 fps video."""
    future_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0)
    simultaneous_onset_tolerance_s: float = 1.0 / 30.0
    """Two onsets this close are simultaneous (one 30 Hz subframe, as in mpc-wm)."""
    turn_max_internal_pause_s: float = 0.3
    """Pause closure of the turn rule (mpc-wm default)."""
    max_local_gap_s: float = 2.0
    """Floor transfers with a longer gap are flagged ``long_gap`` (kept)."""
    max_local_overlap_s: float = 2.0
    """Floor transfers with a longer overlap are flagged ``long_overlap`` (kept)."""
    audio_sample_rate_hz: int = 16_000
    audio_min_mask_fraction: float = 0.5
    """Ego prosody is defined only where the ego-solo mask covers this much of a cell."""
    video_frame_width: int = 160
    """Frames are decoded at this width for the video nuisance controls."""

    def __post_init__(self) -> None:
        if self.subframes_per_step < 1:
            raise ValueError("subframes_per_step must be >= 1")
        if not self.future_horizons_s or any(h <= 0 for h in self.future_horizons_s):
            raise ValueError("future_horizons_s must be positive and non-empty")
        if list(self.future_horizons_s) != sorted(set(self.future_horizons_s)):
            raise ValueError("future_horizons_s must be strictly increasing")
        for name in (
            "simultaneous_onset_tolerance_s",
            "turn_max_internal_pause_s",
            "max_local_gap_s",
            "max_local_overlap_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < self.audio_min_mask_fraction <= 1.0:
            raise ValueError("audio_min_mask_fraction must be in (0, 1]")

    @classmethod
    def from_mapping(cls, values: Any) -> LabelConfig:
        """Build from the ``labels`` config section (missing keys take defaults)."""
        if not values:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown labels settings {sorted(unknown)}")
        kwargs = {key: values[key] for key in values}
        if "future_horizons_s" in kwargs:
            kwargs["future_horizons_s"] = tuple(
                float(h) for h in kwargs["future_horizons_s"]
            )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class Pieces:
    """One recording cut into constant-state pieces, participants in canonical order."""

    starts: np.ndarray
    ends: np.ndarray
    states: np.ndarray
    """``(n_pieces, n_participants)`` int8: SILENT / SPEAKING / UNKNOWN."""
    participant_ids: tuple[str, ...]
    resolved: tuple[bool, ...]

    @property
    def count(self) -> int:
        """Number of pieces."""
        return len(self.starts)

    @property
    def n_participants(self) -> int:
        """Number of participants (wearer first)."""
        return len(self.participant_ids)

    @property
    def all_known(self) -> np.ndarray:
        """Pieces where every participant's state is known."""
        return (self.states != UNKNOWN).all(axis=1)

    @property
    def speaking_count(self) -> np.ndarray:
        """Participants speaking in each piece (meaningful where ``all_known``)."""
        return (self.states == SPEAKING).sum(axis=1)


def build_pieces(recording: RecordingFacts) -> Pieces:
    """Cut the recording window into pieces of constant per-participant state."""
    order = participant_order(recording.participants)
    participants = [recording.participants[i] for i in order]
    window = (recording.start_s, recording.end_s)
    shared_unknown = [(item.start_s, item.end_s) for item in recording.unknown]
    speech = []
    unknown = []
    boundaries = {window[0], window[1]}
    for participant in participants:
        spoken = merge(
            np.asarray(
                [(a.start_s, a.end_s) for a in participant.speech], dtype=float
            ).reshape(-1, 2)
        )
        hidden = merge(
            np.asarray(
                shared_unknown + [(a.start_s, a.end_s) for a in participant.unknown],
                dtype=float,
            ).reshape(-1, 2)
        )
        speech.append(spoken)
        unknown.append(hidden)
        for intervals in (spoken, hidden):
            for value in intervals.reshape(-1):
                boundaries.add(float(min(max(value, window[0]), window[1])))
    edges = np.asarray(sorted(boundaries), dtype=float)
    starts, ends = edges[:-1], edges[1:]
    keep = ends - starts > 0
    starts, ends = starts[keep], ends[keep]
    middles = (starts + ends) / 2.0
    states = np.zeros((len(starts), len(participants)), dtype=np.int8)
    for column, (spoken, hidden) in enumerate(zip(speech, unknown, strict=True)):
        states[_inside(middles, spoken), column] = SPEAKING
        states[_inside(middles, hidden), column] = UNKNOWN
    return Pieces(
        starts=starts,
        ends=ends,
        states=states,
        participant_ids=tuple(p.participant_id for p in participants),
        resolved=tuple(p.resolved for p in participants),
    )


def _inside(points: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Which points fall inside the union of sorted, merged ``[start, end)`` intervals."""
    if len(intervals) == 0:
        return np.zeros(len(points), dtype=bool)
    position = np.searchsorted(intervals[:, 0], points, side="right") - 1
    valid = position >= 0
    result = np.zeros(len(points), dtype=bool)
    result[valid] = points[valid] < intervals[position[valid], 1]
    return result


def floor_holder(pieces: Pieces) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous holder per piece and whether it is known."""
    count = pieces.speaking_count
    holder = np.where(count == 0, NONE, MULTIPLE).astype(np.int16)
    single = count == 1
    holder[single] = np.argmax(pieces.states[single] == SPEAKING, axis=1)
    return holder, pieces.all_known


def last_unique_speaker(pieces: Pieces) -> np.ndarray:
    """Last participant who spoke alone, per piece (``NO_SPEAKER`` when unknown)."""
    holder, known = floor_holder(pieces)
    result = np.full(pieces.count, NO_SPEAKER, dtype=np.int16)
    current = NO_SPEAKER
    for index in range(pieces.count):
        if not known[index]:
            current = NO_SPEAKER
        elif holder[index] >= 0:
            current = int(holder[index])
        result[index] = current
    return result


@dataclass(frozen=True)
class Events:
    """Onsets/offsets and floor changes of one recording, as column arrays."""

    time_s: np.ndarray
    kind: np.ndarray
    """``"onset"`` / ``"offset"``."""
    participant: np.ndarray
    piece: np.ndarray
    """Index of the piece starting at the event."""


def speaker_events(pieces: Pieces) -> Events:
    """Every observable onset and offset (both adjacent states known)."""
    if pieces.count < 2:
        empty = np.empty(0)
        return Events(empty, empty.astype(object), empty.astype(int), empty.astype(int))
    before, after = pieces.states[:-1], pieces.states[1:]
    onset = (before == SILENT) & (after == SPEAKING)
    offset = (before == SPEAKING) & (after == SILENT)
    rows, columns = np.nonzero(onset | offset)
    order = np.lexsort((columns, rows))
    rows, columns = rows[order], columns[order]
    return Events(
        time_s=pieces.starts[rows + 1],
        kind=np.where(onset[rows, columns], "onset", "offset").astype(object),
        participant=columns.astype(np.int64),
        piece=(rows + 1).astype(np.int64),
    )


@dataclass(frozen=True)
class FloorChange:
    """One change of the last unique speaker between two known participants."""

    time_s: float
    previous_holder: int
    next_holder: int
    previous_run_end_s: float
    next_run_start_s: float

    @property
    def fto_s(self) -> float:
        """Floor-transfer offset: negative = overlap, positive = gap."""
        return self.next_run_start_s - self.previous_run_end_s


def floor_changes(pieces: Pieces, last_unique: np.ndarray) -> list[FloorChange]:
    """Changes of the last unique speaker, with the runs that frame them."""
    changes: list[FloorChange] = []
    speaking = pieces.states == SPEAKING
    for index in range(1, pieces.count):
        before, after = int(last_unique[index - 1]), int(last_unique[index])
        if before < 0 or after < 0 or before == after:
            continue
        cursor = index
        while cursor > 0 and speaking[cursor - 1, after]:
            cursor -= 1
        next_start = pieces.starts[cursor]
        back = index - 1
        while back >= 0 and not speaking[back, before]:
            back -= 1
        previous_end = pieces.ends[back] if back >= 0 else np.nan
        changes.append(
            FloorChange(
                time_s=float(pieces.starts[index]),
                previous_holder=before,
                next_holder=after,
                previous_run_end_s=float(previous_end),
                next_run_start_s=float(next_start),
            )
        )
    return changes


def transfer_kind(fto_s: float) -> str:
    """``gap``, ``overlap`` or ``no_gap_no_overlap`` for a floor-transfer offset."""
    if abs(fto_s) <= EPSILON_S:
        return "no_gap_no_overlap"
    return "gap" if fto_s > 0 else "overlap"


def transfer_locality(fto_s: float, config: LabelConfig) -> str:
    """``local``, ``long_gap`` or ``long_overlap`` (flags, never filters)."""
    if fto_s > config.max_local_gap_s:
        return "long_gap"
    if fto_s < -config.max_local_overlap_s:
        return "long_overlap"
    return "local"


@dataclass(frozen=True)
class Run:
    """Maximal continuous speech of one participant."""

    participant: int
    start_s: float
    end_s: float
    start_censored: bool
    end_censored: bool
    first_piece: int
    last_piece: int

    @property
    def duration_s(self) -> float:
        """Run length in seconds."""
        return self.end_s - self.start_s


def speech_runs(pieces: Pieces) -> list[Run]:
    """Every participant's runs, sorted by (start, participant).

    A run is censored at an end touching UNKNOWN or the recording edge: its
    true extent may be larger than what is observed.
    """
    runs: list[Run] = []
    for participant in range(pieces.n_participants):
        column = pieces.states[:, participant]
        speaking = column == SPEAKING
        if not speaking.any():
            continue
        padded = np.concatenate([[False], speaking, [False]])
        changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for first, stop in zip(changes[::2], changes[1::2], strict=True):
            last = stop - 1
            runs.append(
                Run(
                    participant=participant,
                    start_s=float(pieces.starts[first]),
                    end_s=float(pieces.ends[last]),
                    start_censored=bool(first == 0 or column[first - 1] == UNKNOWN),
                    end_censored=bool(
                        last == pieces.count - 1 or column[last + 1] == UNKNOWN
                    ),
                    first_piece=int(first),
                    last_piece=int(last),
                )
            )
    return sorted(runs, key=lambda run: (run.start_s, run.participant))


@dataclass(frozen=True)
class Overlap:
    """A pairwise overlap: a run of ``entering`` starting inside a run of ``initial``."""

    initial: int
    entering: int
    start_s: float
    end_s: float
    overlap_type: str
    """``simultaneous_onset``, ``within``, ``between`` or ``undetermined``."""
    previous_floor_holder: int
    """Last unique speaker just before the entry (``NO_SPEAKER`` when unknown)."""
    valid: bool


def overlaps(
    pieces: Pieces,
    runs: list[Run],
    last_unique: np.ndarray,
    config: LabelConfig,
) -> list[Overlap]:
    """Pairwise overlaps between runs of different participants.

    For runs ``I`` and ``E`` with ``I.start <= E.start < I.end`` (equal starts
    are counted once, with the lower participant index as initial), the type
    is ``simultaneous_onset`` if ``E.start - I.start <= tolerance``, else
    ``within`` if ``E`` ends no later than ``I``, else ``between``. A type that
    depends on a censored run end is ``undetermined`` and invalid.
    """
    starts = np.asarray([run.start_s for run in runs], dtype=float)
    result: list[Overlap] = []
    for initial in runs:
        low = int(np.searchsorted(starts, initial.start_s - EPSILON_S, side="left"))
        high = int(np.searchsorted(starts, initial.end_s - EPSILON_S, side="left"))
        for entering in runs[low:high]:
            if entering.participant == initial.participant:
                continue
            same_start = abs(entering.start_s - initial.start_s) <= EPSILON_S
            if entering.start_s < initial.start_s - EPSILON_S:
                continue
            if same_start and entering.participant < initial.participant:
                continue
            simultaneous = (
                entering.start_s - initial.start_s
                <= config.simultaneous_onset_tolerance_s + EPSILON_S
            )
            if simultaneous:
                kind = "simultaneous_onset"
                valid = not (initial.start_censored or entering.start_censored)
            elif initial.end_censored or entering.end_censored:
                kind, valid = "undetermined", False
            elif entering.end_s <= initial.end_s + EPSILON_S:
                kind, valid = "within", True
            else:
                kind, valid = "between", True
            before = _piece_before(pieces, entering.start_s)
            result.append(
                Overlap(
                    initial=initial.participant,
                    entering=entering.participant,
                    start_s=entering.start_s,
                    end_s=min(initial.end_s, entering.end_s),
                    overlap_type=kind,
                    previous_floor_holder=int(last_unique[before])
                    if before >= 0
                    else NO_SPEAKER,
                    valid=valid and not entering.start_censored,
                )
            )
    return sorted(result, key=lambda item: (item.start_s, item.initial, item.entering))


def _piece_before(pieces: Pieces, time_s: float) -> int:
    """Index of the piece ending at ``time_s`` (-1 at the recording start)."""
    return int(np.searchsorted(pieces.ends, time_s + EPSILON_S, side="right")) - 1


@dataclass(frozen=True)
class Silence:
    """A maximal joint silence."""

    start_s: float
    end_s: float
    silence_type: str
    """``pause``, ``gap`` or ``undetermined``."""
    previous_speaker: int
    next_speaker: int
    valid: bool


def silences(pieces: Pieces) -> list[Silence]:
    """Joint silences, typed from the single speaker before and after.

    A silence bounded by UNKNOWN or the recording edge, or where several
    participants stop (start) together, is ``undetermined``.
    """
    silent = pieces.all_known & (pieces.speaking_count == 0)
    padded = np.concatenate([[False], silent, [False]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    speaking = pieces.states == SPEAKING
    result: list[Silence] = []
    for first, stop in zip(changes[::2], changes[1::2], strict=True):
        last = stop - 1
        before = _single_speaker(pieces, speaking, first - 1)
        after = _single_speaker(pieces, speaking, last + 1)
        if before >= 0 and after >= 0:
            kind = "pause" if before == after else "gap"
        else:
            kind = "undetermined"
        result.append(
            Silence(
                start_s=float(pieces.starts[first]),
                end_s=float(pieces.ends[last]),
                silence_type=kind,
                previous_speaker=before,
                next_speaker=after,
                valid=kind != "undetermined",
            )
        )
    return result


def _single_speaker(pieces: Pieces, speaking: np.ndarray, index: int) -> int:
    """The only speaker of a known piece, else ``NO_SPEAKER``."""
    if index < 0 or index >= pieces.count or not pieces.all_known[index]:
        return NO_SPEAKER
    active = np.flatnonzero(speaking[index])
    return int(active[0]) if len(active) == 1 else NO_SPEAKER


@dataclass(frozen=True)
class Turn:
    """Runs of one participant joined across short, fully observed pauses."""

    participant: int
    start_s: float
    end_s: float
    run_count: int
    pauses: tuple[float, ...]
    start_censored: bool
    end_censored: bool
    previous_participant: int = NO_SPEAKER
    next_participant: int = NO_SPEAKER
    fto_from_previous_s: float = float("nan")

    @property
    def duration_s(self) -> float:
        """Turn length in seconds."""
        return self.end_s - self.start_s


def turns(pieces: Pieces, runs: list[Run], config: LabelConfig) -> list[Turn]:
    """Pause-closed turns, sorted by (start, participant), with their neighbours.

    Rule (``LABEL_RULES_VERSION`` 1): two consecutive runs of a participant
    join when the pause between them is at most ``turn_max_internal_pause_s``
    and contains no UNKNOWN time for that participant. Other participants'
    speech inside the pause does not break the turn. Neighbours are the turns
    immediately before/after in start order; the FTO is recorded only when the
    previous turn belongs to another participant.
    """
    known = pieces.states != UNKNOWN
    cumulative_unknown = [
        np.concatenate([[0.0], np.cumsum(~known[:, p] * (pieces.ends - pieces.starts))])
        for p in range(pieces.n_participants)
    ]
    by_participant: dict[int, list[Run]] = {}
    for run in runs:
        by_participant.setdefault(run.participant, []).append(run)
    joined: list[Turn] = []
    for participant, own in by_participant.items():
        group = [own[0]]
        for run in own[1:]:
            pause = run.start_s - group[-1].end_s
            hidden = (
                cumulative_unknown[participant][run.first_piece]
                - cumulative_unknown[participant][group[-1].last_piece + 1]
            )
            if pause <= config.turn_max_internal_pause_s + EPSILON_S and hidden <= 0:
                group.append(run)
            else:
                joined.append(_turn(participant, group))
                group = [run]
        joined.append(_turn(participant, group))
    joined.sort(key=lambda turn: (turn.start_s, turn.participant))
    result: list[Turn] = []
    for index, turn in enumerate(joined):
        previous = joined[index - 1] if index > 0 else None
        following = joined[index + 1] if index + 1 < len(joined) else None
        fto = (
            turn.start_s - previous.end_s
            if previous is not None and previous.participant != turn.participant
            else float("nan")
        )
        result.append(
            Turn(
                participant=turn.participant,
                start_s=turn.start_s,
                end_s=turn.end_s,
                run_count=turn.run_count,
                pauses=turn.pauses,
                start_censored=turn.start_censored,
                end_censored=turn.end_censored,
                previous_participant=previous.participant if previous else NO_SPEAKER,
                next_participant=following.participant if following else NO_SPEAKER,
                fto_from_previous_s=fto,
            )
        )
    return result


def _turn(participant: int, group: list[Run]) -> Turn:
    return Turn(
        participant=participant,
        start_s=group[0].start_s,
        end_s=group[-1].end_s,
        run_count=len(group),
        pauses=tuple(b.start_s - a.end_s for a, b in pairwise(group)),
        start_censored=group[0].start_censored,
        end_censored=group[-1].end_censored,
    )


@dataclass(frozen=True)
class OnsetContext:
    """The primitives describing the situation of one onset."""

    previous_unique_speaker: int
    """``NO_SPEAKER`` when unknown."""
    others_active_before: bool | None
    others_active_at_onset: bool | None
    silence_before_s: float | None
    simultaneous_other_onset: bool
    onset_type: str

    @property
    def context_valid(self) -> bool:
        """Whether the onset type could be determined."""
        return self.onset_type != "undetermined"


def simultaneous_onsets(events: Events, tolerance_s: float) -> np.ndarray:
    """Per event: an onset with another participant's onset within ``tolerance_s``."""
    result = np.zeros(len(events.time_s), dtype=bool)
    onsets = np.flatnonzero(events.kind == "onset")
    if len(onsets) < 2:
        return result
    order = onsets[np.argsort(events.time_s[onsets], kind="stable")]
    times = events.time_s[order]
    who = events.participant[order]
    low = np.searchsorted(times, times - tolerance_s - EPSILON_S, side="left")
    high = np.searchsorted(times, times + tolerance_s + EPSILON_S, side="right")
    for position, (first, stop) in enumerate(zip(low, high, strict=True)):
        if stop - first > 1 and (who[first:stop] != who[position]).any():
            result[order[position]] = True
    return result


def onset_context(
    pieces: Pieces,
    last_unique: np.ndarray,
    events: Events,
    event_index: int,
    *,
    simultaneous: bool,
) -> OnsetContext:
    """Context of the onset ``events[event_index]`` (see :data:`ONSET_TYPES`).

    * ``overlap`` — another participant speaks just before the onset;
    * ``after_silence`` — joint silence before, the speaker was the last
      unique speaker (self-resumption);
    * ``floor_transfer`` — joint silence before, another participant was;
    * ``undetermined`` — anything unknown decides it.
    """
    participant = int(events.participant[event_index])
    piece = int(events.piece[event_index])
    others = [p for p in range(pieces.n_participants) if p != participant]
    before = _others_active(pieces, piece - 1, others)
    at = _others_active(pieces, piece, others)
    previous = int(last_unique[piece - 1])
    silence_before: float | None
    if before is None:
        silence_before = None
    elif before:
        silence_before = 0.0
    else:
        silence_before = _silence_before(pieces, piece)
    if before is None:
        kind = "undetermined"
    elif before:
        kind = "overlap"
    elif previous == NO_SPEAKER or silence_before is None:
        kind = "undetermined"
    elif previous == participant:
        kind = "after_silence"
    else:
        kind = "floor_transfer"
    return OnsetContext(
        previous_unique_speaker=previous,
        others_active_before=before,
        others_active_at_onset=at,
        silence_before_s=silence_before,
        simultaneous_other_onset=simultaneous,
        onset_type=kind,
    )


def _others_active(pieces: Pieces, piece: int, others: list[int]) -> bool | None:
    """Three-valued: does any other participant speak in ``piece``?"""
    if piece < 0 or not others:
        return None if piece < 0 else False
    states = pieces.states[piece, others]
    if (states == SPEAKING).any():
        return True
    if (states == UNKNOWN).any():
        return None
    return False


def _silence_before(pieces: Pieces, piece: int) -> float | None:
    """Duration of the joint silence ending at ``pieces.starts[piece]`` (None if censored)."""
    cursor = piece - 1
    while (
        cursor >= 0 and pieces.all_known[cursor] and pieces.speaking_count[cursor] == 0
    ):
        cursor -= 1
    if cursor < 0 or not pieces.all_known[cursor]:
        return None
    return float(pieces.starts[piece] - pieces.ends[cursor])


@dataclass(frozen=True)
class ObservedSpans:
    """Maximal spans where every participant is known: the censoring bounds."""

    starts: np.ndarray
    ends: np.ndarray

    def containing(self, times: np.ndarray, *, left_limit: bool) -> np.ndarray:
        """Span index containing each time, or -1.

        With ``left_limit`` a time equal to a span end belongs to that span
        (the state just before the instant), which is how a reference
        instant at a cell end is observed.
        """
        times = np.asarray(times, dtype=float)
        if len(self.starts) == 0:
            return np.full(len(times), -1, dtype=np.int64)
        if left_limit:
            index = np.searchsorted(self.starts, times - EPSILON_S, side="left") - 1
            ok = (index >= 0) & (
                times <= self.ends[np.clip(index, 0, None)] + EPSILON_S
            )
            ok &= times > self.starts[np.clip(index, 0, None)] + EPSILON_S
        else:
            index = np.searchsorted(self.starts, times + EPSILON_S, side="right") - 1
            ok = (index >= 0) & (times < self.ends[np.clip(index, 0, None)] - EPSILON_S)
        return np.where(ok, index, -1).astype(np.int64)


def observed_spans(pieces: Pieces) -> ObservedSpans:
    """Maximal runs of pieces in which every participant is known."""
    known = pieces.all_known
    padded = np.concatenate([[False], known, [False]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    firsts, stops = changes[::2], changes[1::2]
    return ObservedSpans(pieces.starts[firsts], pieces.ends[stops - 1])


@dataclass
class RecordingStructure:
    """Everything this module derives for one recording."""

    recording: RecordingFacts
    pieces: Pieces
    holder: np.ndarray
    holder_known: np.ndarray
    last_unique: np.ndarray
    events: Events
    contexts: dict[int, OnsetContext]
    """Onset context by index into ``events``."""
    floor_changes: list[FloorChange]
    runs: list[Run]
    overlaps: list[Overlap]
    silences: list[Silence]
    turns: list[Turn]
    spans: ObservedSpans
    notes: dict[str, Any] = field(default_factory=dict)


def derive(recording: RecordingFacts, config: LabelConfig) -> RecordingStructure:
    """Run every derivation of this module on one recording."""
    pieces = build_pieces(recording)
    holder, known = floor_holder(pieces)
    last_unique = last_unique_speaker(pieces)
    events = speaker_events(pieces)
    simultaneous = simultaneous_onsets(events, config.simultaneous_onset_tolerance_s)
    contexts = {
        int(index): onset_context(
            pieces,
            last_unique,
            events,
            int(index),
            simultaneous=bool(simultaneous[index]),
        )
        for index in np.flatnonzero(events.kind == "onset")
    }
    runs = speech_runs(pieces)
    return RecordingStructure(
        recording=recording,
        pieces=pieces,
        holder=holder,
        holder_known=known,
        last_unique=last_unique,
        events=events,
        contexts=contexts,
        floor_changes=floor_changes(pieces, last_unique),
        runs=runs,
        overlaps=overlaps(pieces, runs, last_unique, config),
        silences=silences(pieces),
        turns=turns(pieces, runs, config),
        spans=observed_spans(pieces),
    )


__all__ = [
    "EPSILON_S",
    "LABEL_RULES_VERSION",
    "MULTIPLE",
    "NONE",
    "NO_SPEAKER",
    "ONSET_TYPES",
    "SILENT",
    "SPEAKING",
    "UNKNOWN",
    "Events",
    "FloorChange",
    "LabelConfig",
    "ObservedSpans",
    "OnsetContext",
    "Overlap",
    "Pieces",
    "RecordingStructure",
    "Run",
    "Silence",
    "Turn",
    "build_pieces",
    "derive",
    "floor_changes",
    "floor_holder",
    "last_unique_speaker",
    "observed_spans",
    "onset_context",
    "overlaps",
    "silences",
    "simultaneous_onsets",
    "speaker_events",
    "speech_runs",
    "transfer_kind",
    "transfer_locality",
    "turns",
]
