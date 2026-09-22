"""Continuous focal vocal state -> regular decision grid of logged vocal actions.

The production input is the *control* focal voice-state timeline, which is the
native one requantized at the same Δ this module grids on; the same functions
grid the native timeline unchanged, which is how the two are compared.

One slot is the half-open control step ``[t_k, t_k + Δ)`` with ``t_k = k · Δ``
on the recording's canonical clock. Each slot carries the wearer's state just
before ``t_k`` and, when exactly one resolved ``SILENT``/``SPEAKING``
transition falls inside it, the action that transition logs. Everything the
vocabulary cannot express — unknown state, unknown time inside the slot, more
than one resolved transition, an undefined left state — is masked, never
relabelled.

Native timestamps are never rounded: ``event_time_s`` stays exact and
``tau_s = event_time_s - t_k`` records where in the step the event happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

ACTION_SCHEMA_VERSION = 2
"""Bumped when the vocabulary, the slot convention, the mask policy or the source layer changes.

Version 2 grids the *control* focal voice state (native state requantized at
Δ by sub-step silence bridging) instead of the native state directly.
"""

DECISION_STEP_S = 0.100
"""Δ: the control step. Not a video frame rate, an audio period or a horizon."""

BOUNDARY_TOLERANCE_S = 1e-9
"""Time within which a native timestamp counts as being exactly on a grid point.

``t / Δ`` is not exact in binary floating point (``0.3 / 0.1`` is below 3), so
a nanosecond tolerance keeps an event annotated at a grid point in the slot
that starts there. Native annotations have millisecond resolution at best, so
this can never move a genuine event.
"""


class VoiceState(StrEnum):
    """States of the native focal voice-state timeline."""

    SPEAKING = "SPEAKING"
    SILENT = "SILENT"
    UNKNOWN = "UNKNOWN"


class Action(StrEnum):
    """The complete v0 vocal action space."""

    NO_EVENT = "NO_EVENT"
    """Hold the current vocal state for this decision step."""
    ONSET = "ONSET"
    OFFSET = "OFFSET"


class MaskReason(StrEnum):
    """Why a slot carries no action."""

    RECORDING_START = "recording_start"
    """The state strictly before ``t_k`` is outside the timeline."""
    UNKNOWN_STATE = "unknown_state"
    """The state before ``t_k`` is UNKNOWN."""
    UNKNOWN_WITHIN_SLOT = "unknown_within_slot"
    """The slot intersects an UNKNOWN region (a transition to/from UNKNOWN)."""
    COMPOUND_TRANSITION = "compound_transition"
    """More than one resolved transition falls inside the slot."""
    INVALID_TIMELINE = "invalid_timeline"
    """The source timeline of this recording is unusable (gap, overlap, empty)."""


_STATE_CODES = {
    str(VoiceState.SILENT): 0,
    str(VoiceState.SPEAKING): 1,
    str(VoiceState.UNKNOWN): 2,
}
SILENT_CODE, SPEAKING_CODE, UNKNOWN_CODE = 0, 1, 2


def slot_index(times: np.ndarray, step_s: float = DECISION_STEP_S) -> np.ndarray:
    """Index of the slot each time belongs to, with ``[t_k, t_k + Δ)`` semantics.

    A time exactly on ``t_k`` (within :data:`BOUNDARY_TOLERANCE_S`) belongs to
    slot ``k``; a time exactly on ``t_k + Δ`` belongs to slot ``k + 1``.
    """
    return np.floor(
        (np.asarray(times, dtype=float) + BOUNDARY_TOLERANCE_S) / step_s
    ).astype(np.int64)


def within_step_offset(
    event_time_s: np.ndarray,
    decision_time_s: np.ndarray,
    step_s: float = DECISION_STEP_S,
) -> np.ndarray:
    """``tau`` of each event inside its step, kept inside ``[0, step)``.

    ``k · Δ`` is not the exact grid point in binary floating point, so the raw
    difference can land an ULP below zero for an event annotated exactly on a
    boundary. Residues within :data:`BOUNDARY_TOLERANCE_S` are snapped back
    into the half-open range; anything larger is left alone so the build's
    invariant check fails loudly. ``event_time_s`` itself is never modified.
    """
    tau = np.asarray(event_time_s, dtype=float) - np.asarray(
        decision_time_s, dtype=float
    )
    below = (tau < 0.0) & (tau >= -BOUNDARY_TOLERANCE_S)
    above = (tau >= step_s) & (tau <= step_s + BOUNDARY_TOLERANCE_S)
    tau = np.where(below, 0.0, tau)
    return np.where(above, np.nextafter(step_s, 0.0), tau)


@dataclass(frozen=True)
class RecordingTimeline:
    """One recording's contiguous native state intervals, in canonical seconds."""

    starts_s: np.ndarray
    ends_s: np.ndarray
    states: np.ndarray
    """State codes: 0 SILENT, 1 SPEAKING, 2 UNKNOWN."""

    @classmethod
    def from_states(
        cls,
        starts_s: list[float] | np.ndarray,
        ends_s: list[float] | np.ndarray,
        states: list[str],
    ) -> RecordingTimeline:
        """Build from state names, in timeline order."""
        return cls(
            np.asarray(starts_s, dtype=float),
            np.asarray(ends_s, dtype=float),
            np.asarray([_STATE_CODES[str(state)] for state in states], dtype=np.int8),
        )

    def validity_error(self) -> str | None:
        """Why this timeline cannot be gridded, or ``None`` when it is usable."""
        if len(self.starts_s) == 0:
            return "empty timeline"
        if np.any(self.ends_s <= self.starts_s):
            return "non-increasing interval"
        if len(self.starts_s) > 1 and not np.allclose(
            self.starts_s[1:], self.ends_s[:-1], rtol=0.0, atol=BOUNDARY_TOLERANCE_S
        ):
            return "gap or overlap between intervals"
        return None

    @property
    def start_s(self) -> float:
        """First covered instant."""
        return float(self.starts_s[0])

    @property
    def end_s(self) -> float:
        """Last covered instant."""
        return float(self.ends_s[-1])


@dataclass(frozen=True)
class Transition:
    """One state change at an interior boundary of the timeline."""

    time_s: np.ndarray
    before: np.ndarray
    after: np.ndarray

    @property
    def resolved(self) -> np.ndarray:
        """Mask of SILENT<->SPEAKING changes: the only ones that log an action."""
        return (self.before != UNKNOWN_CODE) & (self.after != UNKNOWN_CODE)

    @property
    def involves_unknown(self) -> np.ndarray:
        """Mask of changes to or from UNKNOWN: masking evidence, never an action."""
        return ~self.resolved


def transitions_of(timeline: RecordingTimeline) -> Transition:
    """State changes at interior interval boundaries (adjacent states differ)."""
    if len(timeline.starts_s) < 2:
        empty_time = np.empty(0, dtype=float)
        empty_code = np.empty(0, dtype=np.int8)
        return Transition(empty_time, empty_code, empty_code)
    before, after = timeline.states[:-1], timeline.states[1:]
    changed = before != after
    return Transition(timeline.ends_s[:-1][changed], before[changed], after[changed])


@dataclass(frozen=True)
class GridBounds:
    """The complete slots of one recording and the partial time dropped at each end."""

    first_index: int
    slot_count: int
    leading_dropped_s: float
    trailing_dropped_s: float

    @property
    def indices(self) -> np.ndarray:
        """Decision indices of the complete slots."""
        return np.arange(
            self.first_index, self.first_index + self.slot_count, dtype=np.int64
        )


def grid_bounds(
    timeline: RecordingTimeline, step_s: float = DECISION_STEP_S
) -> GridBounds:
    """Complete slots entirely inside the timeline; partial ends are dropped."""
    first = int(np.ceil((timeline.start_s - BOUNDARY_TOLERANCE_S) / step_s))
    last = int(np.floor((timeline.end_s + BOUNDARY_TOLERANCE_S) / step_s)) - 1
    count = max(0, last - first + 1)
    if count == 0:
        return GridBounds(first, 0, 0.0, timeline.end_s - timeline.start_s)
    return GridBounds(
        first,
        count,
        max(0.0, first * step_s - timeline.start_s),
        max(0.0, timeline.end_s - (last + 1) * step_s),
    )


@dataclass(frozen=True)
class ActionGrid:
    """Column arrays of one recording's slots, plus its compound-slot records."""

    decision_index: np.ndarray
    decision_time_s: np.ndarray
    slot_end_s: np.ndarray
    focal_state_before: np.ndarray
    """State codes; :data:`UNKNOWN_CODE` where the slot is masked for that reason."""
    action: np.ndarray
    """Action codes: 0 NO_EVENT, 1 ONSET, 2 OFFSET, -1 masked."""
    action_valid: np.ndarray
    event_time_s: np.ndarray
    tau_s: np.ndarray
    mask_reason: np.ndarray
    """Mask-reason codes indexing :data:`MASK_REASONS`; -1 when the slot is valid."""
    resolved_transitions_in_slot: np.ndarray
    bounds: GridBounds
    transitions: Transition


ACTIONS: tuple[str, ...] = (str(Action.NO_EVENT), str(Action.ONSET), str(Action.OFFSET))
MASK_REASONS: tuple[str, ...] = (
    str(MaskReason.RECORDING_START),
    str(MaskReason.UNKNOWN_STATE),
    str(MaskReason.UNKNOWN_WITHIN_SLOT),
    str(MaskReason.COMPOUND_TRANSITION),
    str(MaskReason.INVALID_TIMELINE),
)
STATE_NAMES: tuple[str, ...] = (
    str(VoiceState.SILENT),
    str(VoiceState.SPEAKING),
    str(VoiceState.UNKNOWN),
)


def build_action_grid(
    timeline: RecordingTimeline, step_s: float = DECISION_STEP_S
) -> ActionGrid:
    """Grid one recording: state before each step, the action it logs, or a mask.

    Linear in the number of intervals and slots: states before the steps come
    from one ``searchsorted``, transitions are bucketed by integer slot index.
    """
    bounds = grid_bounds(timeline, step_s)
    indices = bounds.indices
    decision_time = indices * step_s
    count = len(indices)
    action = np.full(count, -1, dtype=np.int8)
    mask_reason = np.full(count, -1, dtype=np.int8)
    event_time = np.full(count, np.nan)
    tau = np.full(count, np.nan)
    resolved_counts = np.zeros(count, dtype=np.int32)
    transitions = transitions_of(timeline)

    if count == 0:
        return ActionGrid(
            indices,
            decision_time,
            decision_time + step_s,
            np.empty(0, dtype=np.int8),
            action,
            np.zeros(0, dtype=bool),
            event_time,
            tau,
            mask_reason,
            resolved_counts,
            bounds,
            transitions,
        )

    # State strictly before t_k: the interval whose end is the first one at or
    # after t_k. A step at the very first instant has no such interval.
    before_position = np.searchsorted(
        timeline.ends_s, decision_time - BOUNDARY_TOLERANCE_S, side="left"
    )
    has_left_state = decision_time > timeline.start_s + BOUNDARY_TOLERANCE_S
    before_position = np.clip(before_position, 0, len(timeline.states) - 1)
    state_before = timeline.states[before_position].astype(np.int8)

    resolved_slots = (
        slot_index(transitions.time_s[transitions.resolved], step_s)
        - bounds.first_index
    )
    inside = (resolved_slots >= 0) & (resolved_slots < count)
    resolved_slots_inside = resolved_slots[inside]
    resolved_counts = np.bincount(resolved_slots_inside, minlength=count).astype(
        np.int32
    )

    unknown_slots = (
        slot_index(transitions.time_s[transitions.involves_unknown], step_s)
        - bounds.first_index
    )
    unknown_inside = unknown_slots[(unknown_slots >= 0) & (unknown_slots < count)]
    has_unknown_transition = np.zeros(count, dtype=bool)
    has_unknown_transition[unknown_inside] = True

    # Exactly one resolved transition: the slot logs the action it carries.
    single = resolved_counts == 1
    resolved_times = transitions.time_s[transitions.resolved][inside]
    resolved_after = transitions.after[transitions.resolved][inside]
    single_slot_of_transition = np.isin(resolved_slots_inside, np.flatnonzero(single))
    target_slots = resolved_slots_inside[single_slot_of_transition]
    event_time[target_slots] = resolved_times[single_slot_of_transition]
    tau[target_slots] = within_step_offset(
        event_time[target_slots], decision_time[target_slots], step_s
    )
    action[target_slots] = np.where(
        resolved_after[single_slot_of_transition] == SPEAKING_CODE,
        ACTIONS.index(str(Action.ONSET)),
        ACTIONS.index(str(Action.OFFSET)),
    )
    action[(resolved_counts == 0)] = ACTIONS.index(str(Action.NO_EVENT))

    # Masking, most specific reason first; a masked slot keeps no action.
    masked = np.zeros(count, dtype=bool)
    for reason, condition in (
        (MaskReason.RECORDING_START, ~has_left_state),
        (MaskReason.UNKNOWN_STATE, state_before == UNKNOWN_CODE),
        (MaskReason.UNKNOWN_WITHIN_SLOT, has_unknown_transition),
        (MaskReason.COMPOUND_TRANSITION, resolved_counts > 1),
    ):
        selected = condition & ~masked
        mask_reason[selected] = MASK_REASONS.index(str(reason))
        masked |= selected
    action[masked] = -1
    event_time[masked] = np.nan
    tau[masked] = np.nan
    return ActionGrid(
        indices,
        decision_time,
        decision_time + step_s,
        state_before,
        action,
        ~masked,
        event_time,
        tau,
        mask_reason,
        resolved_counts,
        bounds,
        transitions,
    )


def masked_grid(
    timeline: RecordingTimeline, reason: MaskReason, step_s: float = DECISION_STEP_S
) -> ActionGrid:
    """Every complete slot masked with one reason, for an unusable timeline."""
    bounds = (
        grid_bounds(timeline, step_s)
        if len(timeline.starts_s)
        else GridBounds(0, 0, 0.0, 0.0)
    )
    count = bounds.slot_count
    return ActionGrid(
        bounds.indices,
        bounds.indices * step_s,
        bounds.indices * step_s + step_s,
        np.full(count, UNKNOWN_CODE, dtype=np.int8),
        np.full(count, -1, dtype=np.int8),
        np.zeros(count, dtype=bool),
        np.full(count, np.nan),
        np.full(count, np.nan),
        np.full(count, MASK_REASONS.index(str(reason)), dtype=np.int8),
        np.zeros(count, dtype=np.int32),
        bounds,
        transitions_of(timeline),
    )


@dataclass(frozen=True)
class SubDeltaGap:
    """One native SILENT gap shorter than Δ, framed by resolved transitions."""

    gap_duration_s: float
    offset_time_s: float
    onset_time_s: float
    offset_slot_index: int
    onset_slot_index: int
    cross_slot: bool
    within_grid: bool


def sub_delta_gaps(
    timeline: RecordingTimeline, bounds: GridBounds, step_s: float = DECISION_STEP_S
) -> list[SubDeltaGap]:
    """SPEAKING -> SILENT -> SPEAKING gaps shorter than Δ, and how the grid sees them.

    ``cross_slot`` means the closing OFFSET and the reopening ONSET fall in two
    different slots and are therefore both representable; ``same_slot`` means
    they collide in one slot, which v0 masks as a compound transition. This is
    a measurement: nothing is merged.
    """
    states, starts, ends = timeline.states, timeline.starts_s, timeline.ends_s
    if len(states) < 3:
        return []
    interior = np.arange(1, len(states) - 1)
    candidate = (
        (states[interior] == SILENT_CODE)
        & (states[interior - 1] == SPEAKING_CODE)
        & (states[interior + 1] == SPEAKING_CODE)
        & ((ends[interior] - starts[interior]) < step_s - BOUNDARY_TOLERANCE_S)
    )
    selected = interior[candidate]
    offset_slots = slot_index(starts[selected], step_s)
    onset_slots = slot_index(ends[selected], step_s)
    last_index = bounds.first_index + bounds.slot_count - 1
    return [
        SubDeltaGap(
            gap_duration_s=float(ends[index] - starts[index]),
            offset_time_s=float(starts[index]),
            onset_time_s=float(ends[index]),
            offset_slot_index=int(offset_slot),
            onset_slot_index=int(onset_slot),
            cross_slot=bool(offset_slot != onset_slot),
            within_grid=bool(
                bounds.slot_count > 0
                and offset_slot >= bounds.first_index
                and onset_slot <= last_index
            ),
        )
        for index, offset_slot, onset_slot in zip(
            selected, offset_slots, onset_slots, strict=True
        )
    ]


__all__ = [
    "ACTIONS",
    "ACTION_SCHEMA_VERSION",
    "BOUNDARY_TOLERANCE_S",
    "DECISION_STEP_S",
    "MASK_REASONS",
    "STATE_NAMES",
    "Action",
    "ActionGrid",
    "GridBounds",
    "MaskReason",
    "RecordingTimeline",
    "SubDeltaGap",
    "Transition",
    "VoiceState",
    "build_action_grid",
    "grid_bounds",
    "masked_grid",
    "slot_index",
    "sub_delta_gaps",
    "transitions_of",
    "within_step_offset",
]
