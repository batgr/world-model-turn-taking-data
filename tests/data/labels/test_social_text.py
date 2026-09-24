"""Native social (LAM/TTM, face tracks) and text labels on synthetic recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from label_helpers import recording

from conv_wm.data.labels.facts import SOCIAL_COLUMNS, TOKEN_COLUMNS, TRACK_COLUMNS
from conv_wm.data.labels.gridding import GridFrame
from conv_wm.data.labels.social import social_grid, social_segment_rows
from conv_wm.data.labels.text import (
    DISCOURSE_MARKERS_FINAL,
    closing_punctuation,
    discourse_marker,
    text_rows,
    token_class,
)
from conv_wm.data.labels.timeline import LabelConfig, derive

FPS = 30.0


def frame(end_s: float = 6.0) -> GridFrame:
    return GridFrame(np.arange(round(end_s / 0.1), dtype=np.int64), 0.1, 3)


def social_rows(rows):
    return pd.DataFrame(
        [
            {
                "recording_id": "r1",
                "participant_id": pid,
                "person": person,
                "kind": kind,
                "start_s": start,
                "end_s": end,
                "start_frame": int(start * FPS),
                "end_frame": int(end * FPS),
                "is_at_me": at_me,
                "annotation_target": target,
                "source_row": i,
            }
            for i, (pid, person, kind, start, end, at_me, target) in enumerate(rows)
        ],
        columns=list(SOCIAL_COLUMNS),
    )


def track_rows(pid, frames, box=(10.0, 20.0, 30.0, 40.0)):
    return pd.DataFrame(
        [
            {
                "recording_id": "r1",
                "participant_id": pid,
                "track_id": f"t-{pid}",
                "frame": f,
                "time_s": f / FPS,
                "x": box[0],
                "y": box[1],
                "width": box[2],
                "height": box[3],
            }
            for f in frames
        ],
        columns=list(TRACK_COLUMNS),
    )


@pytest.fixture
def social_case():
    facts = recording(
        {"w": [(0.5, 1.0)], "1": [(3.0, 3.5)], "2": [(4.2, 4.6)]},
        end=6.0,
        unknown=[(5.5, 6.0)],
    )
    structure = derive(facts, LabelConfig())
    social = social_rows(
        [
            ("1", "1", "looking", 1.0, 1.5, True, None),
            ("1", "1", "talking", 3.0, 3.5, False, "0"),
            (None, "-1", "talking", 4.0, 5.0, True, None),
        ]
    )
    tracks = pd.concat(
        [track_rows("1", range(30, 60)), track_rows("2", range(120, 150))],
        ignore_index=True,
    )
    return structure, social, tracks


def value(columns, name, row):
    return columns[name][row].as_py()


def test_looking_is_true_in_positive_segments_false_only_where_tracked(social_case):
    structure, social, tracks = social_case
    columns = social_grid(structure, frame(), social, tracks)
    ids = value(columns, "participant_ids", 0)
    assert ids == ["w", "1", "2"]
    # cell 12 = [1.2, 1.3): participant 1 tracked and looking
    assert value(columns, "looking_at_wearer_subframes", 12)[1] == [True, True, True]
    # cell 17: tracked, no positive segment -> False
    assert value(columns, "looking_at_wearer_subframes", 17)[1] == [False, False, False]
    # cell 25: not tracked -> not annotated, never negative
    assert value(columns, "looking_at_wearer_subframes", 25)[1] == [None, None, None]
    assert value(columns, "looking_at_wearer_valid_subframes", 25)[1] == [
        False,
        False,
        False,
    ]
    # the wearer is never tracked in their own view
    assert value(columns, "looking_at_wearer_subframes", 12)[0] == [None, None, None]


def test_talking_to_wearer_only_inside_native_talking_segments(social_case):
    structure, social, tracks = social_case
    columns = social_grid(structure, frame(), social, tracks)
    assert value(columns, "talking_to_wearer_subframes", 31)[1] == [False, False, False]
    assert value(columns, "talking_to_wearer_subframes", 20)[1] == [None, None, None]


def test_unresolved_rows_are_not_painted(social_case):
    structure, social, tracks = social_case
    columns = social_grid(structure, frame(), social, tracks)
    # the -1 talking segment at [4, 5) belongs to nobody on the grid
    assert all(
        v == [None, None, None]
        for v in value(columns, "talking_to_wearer_subframes", 42)
    )


def test_anyone_aggregates_are_three_valued(social_case):
    structure, social, tracks = social_case
    columns = social_grid(structure, frame(), social, tracks)
    assert value(columns, "anyone_looking_at_wearer_subframes", 12) == [True] * 3
    assert value(columns, "anyone_looking_at_wearer_subframes", 17) == [False] * 3
    assert value(columns, "anyone_looking_at_wearer_subframes", 25) == [None] * 3
    assert value(columns, "anyone_talking_to_wearer_subframes", 31) == [False] * 3
    # participant 2 speaks at [4.2, 4.6) without a talking segment: a hidden positive
    assert value(columns, "anyone_talking_to_wearer_subframes", 43) == [None] * 3
    assert value(columns, "anyone_talking_to_wearer_valid_subframes", 43) == [False] * 3


def test_face_tracks_and_boxes(social_case):
    structure, social, tracks = social_case
    columns = social_grid(structure, frame(), social, tracks)
    assert value(columns, "face_tracked_subframes", 40)[2] == [True, True, True]
    assert value(columns, "face_tracked_subframes", 40)[1] == [False, False, False]
    boxes = value(columns, "face_track_bbox", 40)
    assert boxes[0] is None and boxes[1] is None
    assert boxes[2] == pytest.approx([10.0, 20.0, 30.0, 40.0])
    # UNKNOWN recording time is null, even for track presence
    assert value(columns, "face_tracked_subframes", 56)[2] == [None, None, None]


def test_social_segments_keep_unresolved_rows_and_the_raw_target(social_case):
    structure, social, tracks = social_case
    rows = social_segment_rows(structure, social, tracks, FPS)
    segments = [r for r in rows if r["segment_type"] == "social_segment"]
    unresolved = [r for r in segments if not r["resolved"]]
    assert len(unresolved) == 1 and unresolved[0]["person"] == "-1"
    assert unresolved[0]["participant_index"] is None
    talking = next(r for r in segments if r["kind"] == "talking" and r["resolved"])
    assert talking["annotation_target"] == "0" and talking["is_at_me"] is False
    track = next(
        r
        for r in rows
        if r["segment_type"] == "face_track" and r["participant_id"] == "2"
    )
    assert (track["first_frame"], track["last_frame"], track["frame_count"]) == (
        120,
        149,
        30,
    )
    assert track["end_s"] == pytest.approx(150 / FPS)


def token_frame(rows):
    return pd.DataFrame(
        [
            {
                "recording_id": "r1",
                "participant_id": pid,
                "unit": unit,
                "text": text,
                "start_s": start,
                "end_s": end,
                "timing_valid": start is not None,
                "source_row": i,
            }
            for i, (pid, unit, text, start, end) in enumerate(rows)
        ],
        columns=list(TOKEN_COLUMNS),
    )


def test_token_classes():
    assert token_class("Okay") == "lexical"
    assert token_class("?") == "punctuation"
    assert token_class("[laughter]") == "special"
    assert token_class("  ") == "empty"


def test_discourse_markers_prefer_multi_word_entries():
    assert (
        discourse_marker(["you", "know", "what"], ("you know", "you"), initial=True)
        == "you know"
    )
    assert (
        discourse_marker(
            ["it", "is", "you", "know"], DISCOURSE_MARKERS_FINAL, initial=False
        )
        == "you know"
    )
    assert closing_punctuation("really?") == "?"
    assert closing_punctuation("really") is None


def test_word_tokens_are_associated_to_turns_and_give_cues():
    facts = recording({"w": [(1.0, 1.6)], "x": [(2.0, 2.5)]})
    structure = derive(facts, LabelConfig())
    tokens = token_frame(
        [
            ("w", "word", "So", 1.0, 1.2),
            ("w", "word", "really", 1.2, 1.6),
            ("w", "word", "?", 1.2, 1.6),
            ("x", "word", "Yeah", 2.0, 2.5),
            ("x", "word", "right", None, None),
        ]
    )
    events, units = text_rows(structure, tokens)
    assert [e["turn_id"] for e in events] == [0, 0, 0, 1, None]
    untimed = events[-1]
    assert untimed["time_s"] is None and untimed["timing_valid"] is False
    first = units[0]
    assert first["unit"] == "turn" and first["word_count"] == 2
    assert first["final_punctuation"] == "?" and first["interrogative_cue"] is True
    assert first["initial_discourse_marker"] == "so"
    assert first["final_token"] == "really"
    assert first["words_per_second"] == pytest.approx(2 / 0.6)
    assert (
        units[1]["interrogative_cue"] is False and units[1]["final_punctuation"] is None
    )


def test_utterance_units_never_claim_a_speech_rate():
    facts = recording({"w": [(1.0, 3.0)]})
    structure = derive(facts, LabelConfig())
    tokens = token_frame([("w", "utterance", "well what do you think?", 1.0, 3.0)])
    events, units = text_rows(structure, tokens)
    assert events[0]["turn_id"] == 0
    (unit,) = units
    assert unit["unit"] == "utterance" and unit["speech_rate_valid"] is False
    assert unit["words_per_second"] is None
    assert unit["interrogative_cue"] is True
    assert unit["initial_discourse_marker"] == "well"
    assert unit["final_discourse_marker"] is None


def test_tables_build_with_their_schemas(social_case):
    from conv_wm.data.labels.social import SOCIAL_SEGMENT_SCHEMA
    from conv_wm.data.labels.text import TEXT_UNIT_SCHEMA, TOKEN_EVENT_SCHEMA

    structure, social, tracks = social_case
    table = pa.Table.from_pylist(
        [
            {n: r.get(n) for n in SOCIAL_SEGMENT_SCHEMA.names}
            for r in social_segment_rows(structure, social, tracks, FPS)
        ],
        schema=SOCIAL_SEGMENT_SCHEMA,
    )
    frame_ = table.to_pandas()
    assert (frame_.end_s > frame_.start_s).all()
    tokens = token_frame([("w", "word", "hi", 0.5, 0.8)])
    events, units = text_rows(structure, tokens)
    pa.Table.from_pylist(events, schema=TOKEN_EVENT_SCHEMA)
    pa.Table.from_pylist(units, schema=TEXT_UNIT_SCHEMA)


def test_missing_token_text_is_empty_never_the_word_nan():
    facts = recording({"w": [(1.0, 1.6)]})
    structure = derive(facts, LabelConfig())
    tokens = token_frame(
        [("w", "word", "okay", 1.0, 1.2), ("w", "word", float("nan"), 1.2, 1.6)]
    )
    events, units = text_rows(structure, tokens)
    assert events[1]["text"] is None and events[1]["token_class"] == "empty"
    assert units[0]["word_count"] == 1 and units[0]["final_token"] == "okay"
    assert token_class(None) == "empty"
