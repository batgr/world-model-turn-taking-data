"""Slot semantics of the v0 vocal action grid, on synthetic canonical timelines."""

from itertools import pairwise

import numpy as np
import pytest

from conv_wm.data.vocal.action_grid import (
    ACTIONS,
    DECISION_STEP_S,
    MASK_REASONS,
    STATE_NAMES,
    Action,
    MaskReason,
    RecordingTimeline,
    build_action_grid,
    grid_bounds,
    masked_grid,
    slot_index,
    sub_delta_gaps,
)

STEP = DECISION_STEP_S


def _timeline(*intervals):
    """``(start, end, state)`` triples, in timeline order."""
    starts = [item[0] for item in intervals]
    ends = [item[1] for item in intervals]
    states = [item[2] for item in intervals]
    return RecordingTimeline.from_states(starts, ends, states)


def _slots(timeline):
    """Readable ``(index, state_before, action, tau, mask_reason)`` per slot."""
    grid = build_action_grid(timeline)
    return [
        (
            int(grid.decision_index[i]),
            STATE_NAMES[int(grid.focal_state_before[i])],
            ACTIONS[int(grid.action[i])] if grid.action[i] >= 0 else None,
            None if np.isnan(grid.tau_s[i]) else round(float(grid.tau_s[i]), 6),
            MASK_REASONS[int(grid.mask_reason[i])]
            if grid.mask_reason[i] >= 0
            else None,
        )
        for i in range(len(grid.decision_index))
    ]


def test_stable_silent_slots_are_no_event():
    slots = _slots(_timeline((0.0, 1.0, "SILENT")))

    assert slots[1:] == [(k, "SILENT", "NO_EVENT", None, None) for k in range(1, 10)]


def test_stable_speaking_slots_are_no_event():
    slots = _slots(_timeline((0.0, 0.05, "SILENT"), (0.05, 1.0, "SPEAKING")))

    assert slots[1:] == [(k, "SPEAKING", "NO_EVENT", None, None) for k in range(1, 10)]


def test_single_silent_to_speaking_is_onset_with_tau():
    slots = _slots(_timeline((0.0, 0.34, "SILENT"), (0.34, 1.0, "SPEAKING")))

    assert slots[3] == (3, "SILENT", str(Action.ONSET), 0.04, None)
    assert slots[2] == (2, "SILENT", "NO_EVENT", None, None)
    assert slots[4] == (4, "SPEAKING", "NO_EVENT", None, None)


def test_single_speaking_to_silent_is_offset_with_tau():
    slots = _slots(
        _timeline(
            (0.0, 0.05, "SILENT"), (0.05, 0.62, "SPEAKING"), (0.62, 1.0, "SILENT")
        )
    )

    assert slots[6] == (6, "SPEAKING", str(Action.OFFSET), 0.02, None)
    assert slots[7] == (7, "SILENT", "NO_EVENT", None, None)


def test_event_exactly_on_a_grid_point_has_tau_zero_in_that_slot():
    slots = _slots(_timeline((0.0, 0.3, "SILENT"), (0.3, 1.0, "SPEAKING")))

    assert slots[3] == (3, "SILENT", str(Action.ONSET), 0.0, None)
    assert slots[2] == (2, "SILENT", "NO_EVENT", None, None)
    assert slot_index(np.array([0.3])).tolist() == [3]


def test_event_exactly_on_the_slot_end_belongs_to_the_next_slot():
    # 0.4 is both slot 3's end and slot 4's start: it must land in slot 4.
    slots = _slots(_timeline((0.0, 0.4, "SILENT"), (0.4, 1.0, "SPEAKING")))

    assert slots[3] == (3, "SILENT", "NO_EVENT", None, None)
    assert slots[4] == (4, "SILENT", str(Action.ONSET), 0.0, None)
    assert slot_index(np.array([0.4 - 1e-12, 0.4])).tolist() == [4, 4]


def test_offset_and_onset_in_one_slot_is_masked_compound():
    timeline = _timeline(
        (0.0, 0.05, "SILENT"),
        (0.05, 0.32, "SPEAKING"),
        (0.32, 0.36, "SILENT"),
        (0.36, 1.0, "SPEAKING"),
    )

    slots = _slots(timeline)

    assert slots[3] == (3, "SPEAKING", None, None, str(MaskReason.COMPOUND_TRANSITION))
    assert build_action_grid(timeline).resolved_transitions_in_slot[3] == 2


def test_offset_at_slot_end_and_onset_in_the_next_slot_are_two_valid_actions():
    timeline = _timeline(
        (0.0, 0.05, "SILENT"),
        (0.05, 0.395, "SPEAKING"),
        (0.395, 0.405, "SILENT"),
        (0.405, 1.0, "SPEAKING"),
    )

    slots = _slots(timeline)

    assert slots[3] == (3, "SPEAKING", str(Action.OFFSET), 0.095, None)
    assert slots[4] == (4, "SILENT", str(Action.ONSET), 0.005, None)


def test_sub_delta_gap_crossing_a_slot_boundary_is_not_merged():
    # A 10 ms gap: both events are representable, so nothing is masked.
    timeline = _timeline(
        (0.0, 0.05, "SILENT"),
        (0.05, 0.395, "SPEAKING"),
        (0.395, 0.405, "SILENT"),
        (0.405, 1.0, "SPEAKING"),
    )
    grid = build_action_grid(timeline)

    [gap] = sub_delta_gaps(timeline, grid.bounds)

    assert gap.gap_duration_s == pytest.approx(0.01)
    assert (gap.offset_slot_index, gap.onset_slot_index) == (3, 4)
    assert gap.cross_slot and gap.within_grid
    assert grid.action_valid[3] and grid.action_valid[4]
    assert [ACTIONS[int(grid.action[k])] for k in (3, 4)] == [
        str(Action.OFFSET),
        str(Action.ONSET),
    ]


def test_sub_delta_gap_inside_one_slot_is_compound_and_measured_as_same_slot():
    timeline = _timeline(
        (0.0, 0.05, "SILENT"),
        (0.05, 0.32, "SPEAKING"),
        (0.32, 0.36, "SILENT"),
        (0.36, 1.0, "SPEAKING"),
    )
    grid = build_action_grid(timeline)

    [gap] = sub_delta_gaps(timeline, grid.bounds)

    assert gap.gap_duration_s == pytest.approx(0.04)
    assert (gap.offset_slot_index, gap.onset_slot_index) == (3, 3)
    assert not gap.cross_slot
    assert not grid.action_valid[3]


def test_unknown_state_before_the_step_masks_the_slot():
    slots = _slots(_timeline((0.0, 0.45, "UNKNOWN"), (0.45, 1.0, "SILENT")))

    assert slots[1] == (1, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    assert slots[4] == (4, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    assert slots[5] == (5, "SILENT", "NO_EVENT", None, None)


def test_unknown_starting_inside_a_slot_masks_it():
    slots = _slots(
        _timeline((0.0, 0.25, "SILENT"), (0.25, 0.5, "UNKNOWN"), (0.5, 1.0, "SILENT"))
    )

    assert slots[2] == (2, "SILENT", None, None, str(MaskReason.UNKNOWN_WITHIN_SLOT))
    assert slots[3] == (3, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    assert slots[4] == (4, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    # the UNKNOWN region ends exactly at t_5, so the state *before* that step is
    # still UNKNOWN and the slot stays masked; slot 6 is the first usable one
    assert slots[5] == (5, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    assert slots[6] == (6, "SILENT", "NO_EVENT", None, None)


def test_unknown_to_speaking_is_not_an_onset():
    slots = _slots(_timeline((0.0, 0.25, "UNKNOWN"), (0.25, 1.0, "SPEAKING")))

    assert slots[2] == (2, "UNKNOWN", None, None, str(MaskReason.UNKNOWN_STATE))
    assert all(slot[2] != str(Action.ONSET) for slot in slots)
    assert slots[3] == (3, "SPEAKING", "NO_EVENT", None, None)


def test_speaking_to_unknown_is_not_an_offset():
    slots = _slots(_timeline((0.0, 0.25, "SPEAKING"), (0.25, 1.0, "UNKNOWN")))

    assert slots[2] == (2, "SPEAKING", None, None, str(MaskReason.UNKNOWN_WITHIN_SLOT))
    assert all(slot[2] != str(Action.OFFSET) for slot in slots)


def test_first_slot_without_a_left_state_is_masked_recording_start():
    slots = _slots(_timeline((0.0, 1.0, "SILENT")))

    assert slots[0] == (0, "SILENT", None, None, str(MaskReason.RECORDING_START))
    # a timeline starting mid-grid has a left state for its first complete slot
    shifted = _slots(_timeline((0.05, 1.0, "SILENT")))
    assert shifted[0] == (1, "SILENT", "NO_EVENT", None, None)


def test_incomplete_trailing_slot_is_dropped_and_reported():
    bounds = grid_bounds(_timeline((0.0, 0.97, "SILENT")))

    assert bounds.first_index == 0
    assert bounds.slot_count == 9  # slots 0..8; [0.9, 1.0) is incomplete
    assert bounds.trailing_dropped_s == pytest.approx(0.07)
    assert 0.0 <= bounds.trailing_dropped_s < STEP
    assert bounds.leading_dropped_s == 0.0

    leading = grid_bounds(_timeline((0.04, 1.0, "SILENT")))
    assert leading.first_index == 1
    assert 0.0 <= leading.leading_dropped_s < STEP


def test_slots_have_no_gap_no_duplicate_and_an_exact_grid_time():
    grid = build_action_grid(
        _timeline(
            (0.0, 0.33, "SILENT"), (0.33, 0.61, "SPEAKING"), (0.61, 2.0, "SILENT")
        )
    )

    indices = grid.decision_index
    assert len(set(indices.tolist())) == len(indices)
    for previous, current in pairwise(indices.tolist()):
        assert current == previous + 1
    assert np.allclose(grid.decision_time_s, indices * STEP)
    assert np.allclose(grid.slot_end_s - grid.decision_time_s, STEP)


def test_build_is_deterministic():
    timeline = _timeline(
        (0.0, 0.33, "SILENT"), (0.33, 0.61, "SPEAKING"), (0.61, 1.0, "SILENT")
    )

    first, second = build_action_grid(timeline), build_action_grid(timeline)

    assert np.array_equal(first.action, second.action)
    assert np.array_equal(first.mask_reason, second.mask_reason)
    assert np.allclose(first.tau_s, second.tau_s, equal_nan=True)


def test_masked_grid_marks_every_slot_of_an_unusable_timeline():
    timeline = _timeline((0.0, 0.5, "SILENT"), (0.6, 1.0, "SILENT"))  # gap at 0.5

    assert timeline.validity_error() == "gap or overlap between intervals"
    grid = masked_grid(timeline, MaskReason.INVALID_TIMELINE)
    assert not grid.action_valid.any()
    assert (grid.action == -1).all()
    assert (
        grid.mask_reason == MASK_REASONS.index(str(MaskReason.INVALID_TIMELINE))
    ).all()


def test_the_action_space_has_exactly_three_actions_and_five_mask_reasons():
    assert ACTIONS == ("NO_EVENT", "ONSET", "OFFSET")
    assert set(MASK_REASONS) == {
        "recording_start",
        "unknown_state",
        "unknown_within_slot",
        "compound_transition",
        "invalid_timeline",
    }


def test_tau_of_an_event_on_a_grid_point_is_exactly_zero_without_touching_the_event():
    # k * 0.1 overshoots the annotated instant by an ULP for several k
    timeline = _timeline((0.0, 0.7, "SILENT"), (0.7, 1.5, "SPEAKING"))

    grid = build_action_grid(timeline)
    position = int(np.flatnonzero(grid.decision_index == 7)[0])

    assert 7 * STEP > 0.7  # the float grid point is above the annotated time
    assert grid.tau_s[position] == 0.0
    assert grid.event_time_s[position] == 0.7
    assert np.all(grid.tau_s[np.isfinite(grid.tau_s)] >= 0.0)
    assert np.all(grid.tau_s[np.isfinite(grid.tau_s)] < STEP)
