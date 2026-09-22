"""Pure native-state timeline semantics (no dataset, no files)."""

from itertools import pairwise

import pytest

from conv_wm.data.vocal.native_state import (
    NATIVE_STATE_SCHEMA_VERSION,
    NativeAnnotation,
    NativeFocalRecording,
    SourceKind,
    VoiceState,
    build_native_timeline,
    state_durations,
)


def _recording(*, speaking=(), unknown=(), start=0.0, end=10.0):
    return NativeFocalRecording(
        dataset="synthetic",
        recording_id="rec-1",
        sync_group_id="group-1",
        view_id="view-1",
        wearer_id="0",
        start_s=start,
        end_s=end,
        speaking=tuple(NativeAnnotation(s, e, i) for s, e, i in speaking),
        unknown=tuple(NativeAnnotation(s, e, i) for s, e, i in unknown),
        source_kind=SourceKind.EGO4D_VOICE_SEGMENTS,
        annotation_schema_version="test-v1",
    )


def _states(rows):
    return [(r.canonical_start_s, r.canonical_end_s, str(r.voice_state)) for r in rows]


def test_native_segment_is_speaking_and_the_rest_is_silent():
    rows = build_native_timeline(_recording(speaking=[(2.0, 4.0, "vs#1")]))

    assert _states(rows) == [
        (0.0, 2.0, "SILENT"),
        (2.0, 4.0, "SPEAKING"),
        (4.0, 10.0, "SILENT"),
    ]
    assert rows[1].source_annotation_id == "vs#1"
    assert rows[1].source_kind == SourceKind.EGO4D_VOICE_SEGMENTS
    assert rows[0].source_annotation_id is None
    assert rows[0].native_state_schema_version == NATIVE_STATE_SCHEMA_VERSION


def test_missing_region_is_unknown_even_inside_a_native_segment():
    rows = build_native_timeline(
        _recording(speaking=[(2.0, 6.0, "vs#1")], unknown=[(5.0, 7.0, "missing#1")])
    )

    assert _states(rows) == [
        (0.0, 2.0, "SILENT"),
        (2.0, 5.0, "SPEAKING"),
        (5.0, 7.0, "UNKNOWN"),
        (7.0, 10.0, "SILENT"),
    ]
    assert rows[2].source_kind == SourceKind.INVALID_OR_MISSING
    assert rows[2].source_annotation_id == "missing#1"


def test_native_segment_spanning_an_internal_pause_stays_speaking():
    # One native episode [10, 13]; no acoustic evidence is consulted.
    rows = build_native_timeline(_recording(speaking=[(10.0, 13.0, "vs#1")], end=20.0))

    assert _states(rows) == [
        (0.0, 10.0, "SILENT"),
        (10.0, 13.0, "SPEAKING"),
        (13.0, 20.0, "SILENT"),
    ]


def test_segments_touching_or_exceeding_the_window_are_clipped():
    rows = build_native_timeline(
        _recording(speaking=[(-1.0, 1.0, "vs#1"), (9.5, 12.0, "vs#2")], end=10.0)
    )

    assert _states(rows) == [
        (0.0, 1.0, "SPEAKING"),
        (1.0, 9.5, "SILENT"),
        (9.5, 10.0, "SPEAKING"),
    ]


def test_overlapping_native_segments_keep_both_provenance_ids():
    rows = build_native_timeline(
        _recording(speaking=[(1.0, 3.0, "vs#1"), (2.0, 4.0, "vs#2")])
    )

    speaking = [r for r in rows if r.voice_state == VoiceState.SPEAKING]
    assert [
        (r.canonical_start_s, r.canonical_end_s, r.source_annotation_id)
        for r in speaking
    ] == [
        (1.0, 2.0, "vs#1"),
        (2.0, 3.0, "vs#1|vs#2"),
        (3.0, 4.0, "vs#2"),
    ]


def test_adjacent_identical_intervals_merge_and_word_gaps_stay_silent():
    rows = build_native_timeline(
        _recording(speaking=[(1.0, 2.0, "w#1"), (2.0, 3.0, "w#1"), (3.05, 4.0, "w#2")])
    )

    assert _states(rows) == [
        (0.0, 1.0, "SILENT"),
        (1.0, 3.0, "SPEAKING"),
        (3.0, 3.05, "SILENT"),
        (3.05, 4.0, "SPEAKING"),
        (4.0, 10.0, "SILENT"),
    ]


@pytest.mark.parametrize(
    "recording",
    [
        _recording(),
        _recording(speaking=[(0.0, 10.0, "vs#1")]),
        _recording(unknown=[(0.0, 10.0, "media_missing")]),
        _recording(
            speaking=[(1.0, 2.0, "a"), (1.5, 6.0, "b"), (8.0, 9.0, "c")],
            unknown=[(0.5, 1.2, "m"), (9.5, 10.0, "n")],
        ),
    ],
)
def test_timeline_is_exhaustive_sorted_and_non_overlapping(recording):
    rows = build_native_timeline(recording)

    assert rows[0].canonical_start_s == recording.start_s
    assert rows[-1].canonical_end_s == recording.end_s
    for previous, current in pairwise(rows):
        assert previous.canonical_end_s == current.canonical_start_s
    assert all(row.duration_s > 0 for row in rows)
    assert sum(state_durations(rows).values()) == pytest.approx(recording.duration_s)


def test_unknown_never_becomes_silent_and_all_unknown_window_is_one_row():
    rows = build_native_timeline(_recording(unknown=[(0.0, 10.0, "clip_invalid")]))

    assert _states(rows) == [(0.0, 10.0, "UNKNOWN")]
    assert state_durations(rows) == {"SPEAKING": 0.0, "SILENT": 0.0, "UNKNOWN": 10.0}


def test_build_is_deterministic():
    recording = _recording(
        speaking=[(3.0, 4.0, "b"), (1.0, 2.0, "a")], unknown=[(5.0, 6.0, "m")]
    )

    assert [r.to_row() for r in build_native_timeline(recording)] == [
        r.to_row() for r in build_native_timeline(recording)
    ]


def test_empty_or_reversed_intervals_are_rejected():
    with pytest.raises(ValueError):
        NativeAnnotation(2.0, 2.0, "empty")
    with pytest.raises(ValueError):
        _recording(start=5.0, end=5.0)
