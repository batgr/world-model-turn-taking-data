"""Pure sub-step silence bridging semantics (no dataset, no files)."""

from copy import deepcopy

import pytest

from conv_wm.data.vocal.action_grid import (
    DECISION_STEP_S,
    RecordingTimeline,
    slot_index,
)
from conv_wm.data.vocal.control_state import (
    CONTROL_STATE_SCHEMA_VERSION,
    TransformKind,
    build_control_timeline,
    resolved_transition_count,
)
from conv_wm.data.vocal.native_state import (
    NativeStateInterval,
    SourceKind,
    VoiceState,
)

STEP = DECISION_STEP_S


def _native(timeline, *, recording="rec-1"):
    """Native intervals from ``(start, end, state)`` triples, ids on SPEAKING."""
    return [
        NativeStateInterval(
            dataset="synthetic",
            recording_id=recording,
            sync_group_id="group-1",
            view_id="view-1",
            wearer_id="0",
            canonical_start_s=start,
            canonical_end_s=end,
            voice_state=VoiceState(state),
            source_kind=SourceKind.INVALID_OR_MISSING
            if state == "UNKNOWN"
            else SourceKind.EGOCOM_TRANSCRIPT,
            source_annotation_id=f"token#{index}" if state == "SPEAKING" else None,
            annotation_schema_version="test-v1",
        )
        for index, (start, end, state) in enumerate(timeline)
    ]


def _states(rows):
    return [
        (
            pytest.approx(row.canonical_start_s),
            pytest.approx(row.canonical_end_s),
            str(row.voice_state),
        )
        for row in rows
    ]


def _control(timeline, **kwargs):
    return build_control_timeline(_native(timeline, **kwargs))


def test_sub_step_silence_between_two_speaking_becomes_continuous_speaking():
    control, gaps = _control(
        [
            (0.0, 1.0, "SILENT"),
            (1.0, 1.5, "SPEAKING"),
            (1.5, 1.54, "SILENT"),
            (1.54, 2.0, "SPEAKING"),
        ]
    )

    assert _states(control) == [(0.0, 1.0, "SILENT"), (1.0, 2.0, "SPEAKING")]
    assert control[1].transform_kind == str(TransformKind.SUB_STEP_SILENCE_BRIDGE)
    assert control[1].bridged_gap_count == 1
    assert control[1].bridged_gap_total_duration_s == pytest.approx(0.04)
    assert [gap.gap_duration_s for gap in gaps] == [pytest.approx(0.04)]


def test_a_gap_of_exactly_one_step_is_kept():
    control, gaps = _control(
        [
            (0.0, 1.0, "SPEAKING"),
            (1.0, 1.0 + STEP, "SILENT"),
            (1.0 + STEP, 2.0, "SPEAKING"),
        ]
    )

    assert [str(row.voice_state) for row in control] == [
        "SPEAKING",
        "SILENT",
        "SPEAKING",
    ]
    assert gaps == []
    assert all(row.transform_kind is None for row in control)


def test_a_gap_longer_than_one_step_is_kept():
    control, gaps = _control(
        [(0.0, 1.0, "SPEAKING"), (1.0, 1.101, "SILENT"), (1.101, 2.0, "SPEAKING")]
    )

    assert len(control) == 3 and gaps == []


def test_a_short_speaking_burst_between_two_silences_is_kept():
    control, gaps = _control(
        [(0.0, 1.0, "SILENT"), (1.0, 1.02, "SPEAKING"), (1.02, 2.0, "SILENT")]
    )

    assert _states(control) == [
        (0.0, 1.0, "SILENT"),
        (1.0, 1.02, "SPEAKING"),
        (1.02, 2.0, "SILENT"),
    ]
    assert gaps == []


def test_nothing_is_bridged_across_unknown():
    control, gaps = _control(
        [
            (0.0, 1.0, "SPEAKING"),
            (1.0, 1.01, "UNKNOWN"),
            (1.01, 2.0, "SPEAKING"),
            (2.0, 2.01, "SILENT"),
            (2.01, 3.0, "UNKNOWN"),
        ]
    )

    assert [str(row.voice_state) for row in control] == [
        "SPEAKING",
        "UNKNOWN",
        "SPEAKING",
        "SILENT",
        "UNKNOWN",
    ]
    assert gaps == []


def test_successive_micro_gaps_chain_into_one_deterministic_interval():
    timeline = [
        (0.0, 0.5, "SPEAKING"),
        (0.5, 0.52, "SILENT"),
        (0.52, 0.9, "SPEAKING"),
        (0.9, 0.93, "SILENT"),
        (0.93, 1.4, "SPEAKING"),
        (1.4, 1.41, "SILENT"),
        (1.41, 2.0, "SPEAKING"),
    ]

    control, gaps = _control(timeline)
    again, again_gaps = _control(timeline)

    assert _states(control) == [(0.0, 2.0, "SPEAKING")]
    assert control[0].bridged_gap_count == 3
    assert control[0].bridged_gap_total_duration_s == pytest.approx(0.06)
    assert control[0].source_native_index_first == 0
    assert control[0].source_native_index_last == 6
    assert control == again and gaps == again_gaps


def test_bridged_gap_provenance_is_kept():
    control, gaps = _control(
        [
            (0.0, 0.5, "SPEAKING"),
            (0.5, 0.52, "SILENT"),
            (0.52, 0.9, "SPEAKING"),
        ]
    )

    [gap] = gaps
    assert (gap.native_index, gap.recording_id) == (1, "rec-1")
    assert (gap.gap_start_s, gap.gap_end_s) == (0.5, 0.52)
    assert gap.gap_duration_s == pytest.approx(0.02)
    assert control[0].source_annotation_id == "token#0|token#2"
    assert control[0].control_state_schema_version == CONTROL_STATE_SCHEMA_VERSION
    assert control[0].annotation_schema_version == "test-v1"


def test_total_duration_and_window_are_unchanged():
    timeline = [
        (0.0, 0.5, "SPEAKING"),
        (0.5, 0.52, "SILENT"),
        (0.52, 0.9, "SPEAKING"),
        (0.9, 1.5, "SILENT"),
        (1.5, 2.0, "SPEAKING"),
    ]
    native = _native(timeline)

    control, _ = build_control_timeline(native)

    assert control[0].canonical_start_s == native[0].canonical_start_s
    assert control[-1].canonical_end_s == native[-1].canonical_end_s
    assert sum(row.duration_s for row in control) == pytest.approx(
        sum(row.duration_s for row in native)
    )


def test_the_native_intervals_are_never_modified():
    native = _native(
        [(0.0, 0.5, "SPEAKING"), (0.5, 0.52, "SILENT"), (0.52, 0.9, "SPEAKING")]
    )
    before = deepcopy(native)

    build_control_timeline(native)

    assert native == before


def test_the_same_physical_micro_gap_no_longer_depends_on_the_grid_phase():
    """A 20 ms pause is bridged wherever it sits; the native grid saw two cases."""

    def timeline(offset):
        return [
            (offset, offset + 0.5, "SPEAKING"),
            (offset + 0.5, offset + 0.52, "SILENT"),
            (offset + 0.52, offset + 1.0, "SPEAKING"),
        ]

    aligned, shifted = timeline(0.0), timeline(0.09)

    def native_slots(rows):
        recording = RecordingTimeline.from_states(
            [start for start, _, _ in rows],
            [end for _, end, _ in rows],
            [state for _, _, state in rows],
        )
        return slot_index(recording.ends_s[:-1])

    # the native grid resolves the same pause differently depending on phase:
    # aligned, both events collide in one slot (a masked compound); shifted,
    # they land in two slots and both are representable.
    aligned_slots, shifted_slots = native_slots(aligned), native_slots(shifted)
    assert aligned_slots[0] == aligned_slots[1]
    assert shifted_slots[0] != shifted_slots[1]

    for rows in (aligned, shifted):
        control, gaps = _control(rows)
        assert [str(row.voice_state) for row in control] == ["SPEAKING"]
        assert len(gaps) == 1


def test_resolved_transitions_exclude_unknown():
    assert resolved_transition_count(["SILENT", "SPEAKING", "SILENT"]) == 2
    assert resolved_transition_count(["SPEAKING", "UNKNOWN", "SILENT"]) == 0
    assert resolved_transition_count(["SILENT"]) == 0
