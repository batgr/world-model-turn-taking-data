"""The speech extractor's tables on synthetic conversations with known answers."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from label_helpers import recording

from conv_wm.data.labels.catalog import extractor_labels
from conv_wm.data.labels.registry import Extractor, Table
from conv_wm.data.labels.speech import GridKeys, derive_structures, speech_tables
from conv_wm.data.labels.timeline import LabelConfig

STEP = 0.1
CONFIG = LabelConfig()


def keys(end_s: float = 10.0, start_s: float = 0.0) -> GridKeys:
    first = round(start_s / STEP)
    last = round(end_s / STEP)
    index = np.arange(first, last, dtype=np.int64)
    return GridKeys(index, index * STEP)


def tables_for(recordings, grid, config: LabelConfig = CONFIG):
    structures = derive_structures(recordings, grid, config)
    return speech_tables(structures, grid, config, step_s=STEP).tables


def build(facts, config: LabelConfig = CONFIG, grid: GridKeys | None = None):
    grid_keys = {facts.recording_id: grid or keys(facts.end_s, facts.start_s)}
    return tables_for([facts], grid_keys, config)


def cell(grid: pa.Table, column: str, index: int):
    row = pc.index(grid.column("decision_index"), index).as_py()
    return grid.column(column)[row].as_py()


def test_every_available_speech_label_has_its_columns():
    tables = build(recording({"w": [(1, 2)], "x": [(2.5, 3)]}))
    for spec in extractor_labels(Extractor.SPEECH):
        assert spec.table is not None
        names = tables[spec.table].column_names
        missing = set(spec.columns) - set(names)
        assert not missing, (spec.name, missing)


def test_grid_rows_are_exactly_the_action_grid_keys():
    grid = build(recording({"w": [(1, 2)]}))[Table.GRID]
    assert grid.column("decision_index").to_pylist() == list(range(100))
    assert grid.column("decision_time_s").to_pylist()[3] == 3 * STEP
    assert (
        not grid.select(["recording_id", "decision_index"])
        .to_pandas()
        .duplicated()
        .any()
    )


def test_speaker_activity_on_subframes_uses_native_times():
    # w speaks [1.02, 1.05): only the first subframe of cell 10 ([1.0, 1.0333))
    grid = build(recording({"w": [(1.02, 1.05)], "x": []}))[Table.GRID]
    assert cell(grid, "ego_speaking_subframes", 10) == [True, True, False]
    assert cell(grid, "speaker_activity_subframes", 10) == [
        [True, True, False],
        [False, False, False],
    ]
    assert cell(grid, "ego_speaking", 10) is True
    assert cell(grid, "ego_speaking", 11) is False


def test_unknown_is_null_never_silent():
    facts = recording({"w": [], "x": []}, unknown=[(2.0, 2.05)])
    grid = build(facts)[Table.GRID]
    assert cell(grid, "ego_speaking_subframes", 20) == [None, None, False]
    assert cell(grid, "ego_speaking", 20) is None
    assert cell(grid, "others_active", 20) is None
    assert cell(grid, "joint_speech_state_occupancy", 20) is None
    assert cell(grid, "floor_holder_subframes", 20) == [None, None, -1]
    assert cell(grid, "joint_speech_state_occupancy", 21) == [1.0, 0.0, 0.0, 0.0]


def test_joint_state_and_floor_holder_codes():
    facts = recording({"w": [(1.0, 1.2)], "x": [(1.1, 1.3)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "joint_speech_state_subframes", 10) == [1, 1, 1]
    assert cell(grid, "joint_speech_state_subframes", 11) == [3, 3, 3]
    assert cell(grid, "joint_speech_state_subframes", 12) == [2, 2, 2]
    assert cell(grid, "floor_holder_subframes", 11) == [-2, -2, -2]
    assert cell(grid, "floor_holder_subframes", 12) == [1, 1, 1]
    assert cell(grid, "active_speaker_count_subframes", 11) == [2, 2, 2]
    occupancy = cell(grid, "joint_speech_state_occupancy", 10)
    assert occupancy == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-6)


def test_onsets_and_offsets_land_in_the_subframe_of_their_native_time():
    facts = recording({"w": [(1.04, 1.5)], "x": []})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "ego_onset_subframes", 10) == [False, True, False]
    assert cell(grid, "ego_offset_subframes", 15) == [True, False, False]
    assert cell(grid, "other_onset_subframes", 10) == [False, False, False]


def test_an_event_on_a_boundary_belongs_to_the_later_subframe():
    grid = build(recording({"w": [(1.1, 1.5)]}))[Table.GRID]
    assert cell(grid, "ego_onset_subframes", 11) == [True, False, False]
    assert cell(grid, "ego_onset_subframes", 10) == [False, False, False]


def test_transitions_are_not_observable_at_the_recording_start_or_near_unknown():
    facts = recording({"w": [(0.0, 1.0)]}, unknown=[(5.0, 5.5)])
    grid = build(facts)[Table.GRID]
    assert cell(grid, "ego_onset_subframes", 0) == [None, False, False]
    assert cell(grid, "speaker_transition_valid_subframes", 0) == [[False, True, True]]
    assert cell(grid, "ego_offset_subframes", 55) == [None, False, False]


def test_floor_change_subframes_and_holders():
    facts = recording({"w": [(1.0, 2.0)], "x": [(2.5, 3.0)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "floor_change_subframes", 25) == [True, False, False]
    assert cell(grid, "previous_floor_holder_subframes", 25) == [0, None, None]
    assert cell(grid, "next_floor_holder_subframes", 25) == [1, None, None]
    # before anyone spoke alone the floor is unknown
    assert cell(grid, "floor_change_subframes", 5) == [None, None, None]


def test_ego_onset_context_subframes():
    facts = recording({"w": [(1.0, 2.0), (3.0, 3.5)], "x": [(2.2, 2.6)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "ego_onset_type_subframes", 30) == [
        2,
        None,
        None,
    ]  # floor_transfer
    assert cell(grid, "previous_unique_speaker_subframes", 30) == [1, None, None]
    assert cell(grid, "ego_was_last_unique_speaker_subframes", 30) == [
        False,
        None,
        None,
    ]
    assert cell(grid, "silence_duration_before_ego_onset_subframes", 30)[
        0
    ] == pytest.approx(0.4)
    assert cell(grid, "ego_onset_context_valid_subframes", 30) == [True, False, False]
    # the first onset follows a silence censored by the recording start
    assert cell(grid, "ego_onset_type_subframes", 10) == [0, None, None]
    assert cell(grid, "ego_onset_context_valid_subframes", 10) == [False, False, False]


def test_overlap_subframes_by_type():
    facts = recording({"w": [(1.0, 3.0)], "x": [(1.5, 1.6)], "y": [(2.5, 4.0)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "within_overlap_subframes", 15) == [True, True, True]
    assert cell(grid, "between_overlap_subframes", 15) == [False, False, False]
    assert cell(grid, "between_overlap_subframes", 25) == [True, True, True]


def test_time_to_next_ego_onset_and_censoring():
    facts = recording({"w": [(2.0, 3.0)], "x": [(5.0, 6.0)]}, end=8.0)
    grid = build(facts)[Table.GRID]
    # reference is the cell end: cell 9 ends at 1.0
    assert cell(grid, "time_to_next_ego_onset", 9) == pytest.approx(1.0)
    assert cell(grid, "time_to_next_ego_onset_valid", 9) is True
    # the event at exactly the cell end is future: 0 s, valid
    assert cell(grid, "time_to_next_ego_onset", 19) == pytest.approx(0.0, abs=1e-6)
    # after the only wearer onset nothing more happens: censored, not zero
    assert cell(grid, "time_to_next_ego_onset", 40) is None
    assert cell(grid, "time_to_next_ego_onset_valid", 40) is False
    assert cell(grid, "time_to_observation_end", 40) == pytest.approx(8.0 - 4.1)
    assert cell(grid, "time_since_ego_offset", 40) == pytest.approx(1.1)
    assert cell(grid, "time_since_ego_offset_valid", 5) is False


def test_timing_is_censored_by_unknown_regions():
    facts = recording({"w": [(1.0, 2.0), (6.0, 7.0)]}, unknown=[(4.0, 5.0)])
    grid = build(facts)[Table.GRID]
    assert (
        cell(grid, "time_to_next_ego_onset_valid", 25) is False
    )  # next onset beyond UNKNOWN
    assert cell(grid, "time_to_next_ego_onset", 44) is None  # reference inside UNKNOWN
    assert cell(grid, "time_since_observation_start", 44) is None
    assert (
        cell(grid, "time_since_ego_offset_valid", 55) is False
    )  # offset before UNKNOWN


def test_silence_duration_elapsed_and_censored():
    facts = recording({"w": [(1.0, 2.0)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "silence_duration", 24) == pytest.approx(0.5)
    assert cell(grid, "silence_duration", 5) is None  # started at the recording start
    assert cell(grid, "silence_duration", 15) is None  # someone is speaking


def test_next_speaker_with_three_participants():
    facts = recording({"w": [(1.0, 2.0)], "a": [(2.5, 3.0)], "b": [(3.5, 4.0)]})
    grid = build(facts)[Table.GRID]
    ids = cell(grid, "participant_ids", 0)
    assert ids == ["w", "a", "b"]
    assert cell(grid, "next_speaker", 20) == 1
    assert cell(grid, "next_speaker", 30) == 2
    assert cell(grid, "floor_transfer_target", 20) == 1
    assert cell(grid, "next_speaker_valid", 45) is False  # nobody speaks again
    assert cell(grid, "current_speaker_continues", 20) is False


def test_simultaneous_next_onsets_are_a_tie_not_a_guess():
    facts = recording({"w": [(2.0, 3.0)], "x": [(2.0, 2.5)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "next_speaker_valid", 10) is False


def test_current_speaker_continues_after_a_pause():
    facts = recording({"w": [(1.0, 2.0), (2.5, 3.0)], "x": [(4.0, 5.0)]})
    grid = build(facts)[Table.GRID]
    assert cell(grid, "current_speaker_continues", 21) is True
    assert cell(grid, "current_speaker_continues", 30) is False


def test_future_targets_per_horizon_and_censoring():
    config = LabelConfig(future_horizons_s=(0.2, 1.0))
    facts = recording({"w": [(1.0, 2.0)], "x": [(2.5, 3.0)]}, end=3.2)
    grid = build(facts, config)[Table.GRID]
    # r = 0.9 (end of cell 8): the wearer starts 0.1 s later
    assert cell(grid, "future_ego_onset", 8) == [True, True]
    assert cell(grid, "future_ego_activity", 8) == [True, True]
    assert cell(grid, "future_next_speaker", 8) == [0, 0]
    assert cell(grid, "future_speaker_activity", 8) == [[True, False], [True, False]]
    # r = 2.1: nobody within 0.2 s, x within 1 s
    assert cell(grid, "future_next_speaker", 20) == [-1, 1]
    assert cell(grid, "future_other_onset", 20) == [False, True]
    # r = 3.0: the recording ends at 3.2, only the 1 s horizon is censored
    assert cell(grid, "future_ego_activity", 29) == [False, None]
    assert cell(grid, "future_ego_activity", 30) == [None, None]


def test_future_floor_holder_and_overlap():
    config = LabelConfig(future_horizons_s=(0.5,))
    facts = recording({"w": [(1.0, 2.0)], "x": [(1.5, 2.5)]})
    grid = build(facts, config)[Table.GRID]
    assert cell(grid, "future_overlap", 10) == [True]
    assert cell(grid, "future_floor_holder", 10) == [-2]  # at 1.6 both speak
    assert cell(grid, "future_floor_holder", 16) == [1]  # at 2.2 only x


def test_events_table_carries_native_times_and_context():
    facts = recording({"w": [(1.0, 2.0), (3.0, 3.5)], "x": [(2.2, 2.6)]})
    events = build(facts)[Table.EVENTS].to_pandas()
    onsets = events[events.event_type == "onset"]
    assert onsets.time_s.tolist() == [1.0, 2.2, 3.0]
    last = onsets.iloc[-1]
    assert (
        last.onset_type == "floor_transfer" and last.previous_unique_speaker_index == 1
    )
    changes = events[events.event_type == "floor_change"]
    assert changes.fto_s.round(6).tolist() == [0.2, 0.4]
    assert set(changes.locality) == {"local"}
    assert events.time_s.is_monotonic_increasing


def test_segments_table_types_and_intervals():
    facts = recording({"w": [(1.0, 2.0)], "x": [(1.5, 3.0)]})
    segments = build(facts)[Table.SEGMENTS].to_pandas()
    assert set(segments.segment_type) == {"speech_run", "turn", "overlap", "silence"}
    assert (segments.end_s > segments.start_s).all()
    assert np.allclose(segments.duration_s, segments.end_s - segments.start_s)
    overlap = segments[segments.segment_type == "overlap"].iloc[0]
    assert overlap.overlap_type == "between" and overlap.duration_s == pytest.approx(
        0.5
    )
    assert not segments.duplicated(["recording_id", "segment_type", "segment_id"]).any()


def test_participants_table_profiles():
    facts = recording(
        {"w": [(1.0, 2.0)], "x": [(2.5, 3.0)]},
        participant_metadata={"w": {"native_speaker": True, "is_host": False}},
    )
    participants = build(facts)[Table.PARTICIPANTS].to_pandas()
    ego = participants.iloc[0]
    assert ego.participant_id == "w" and ego.is_ego and ego.native_speaker is True
    assert participants.iloc[1].native_speaker is None or math.isnan(
        participants.iloc[1].native_speaker
    )
    assert ego.speaking_time_s == pytest.approx(1.0)
    assert ego.floor_yields == 1 and participants.iloc[1].floor_takes == 1
    assert participants.iloc[1].fto_take_mean_s == pytest.approx(0.5)
    assert len(ego.interaction_profile_features) == len(
        ego.interaction_profile_feature_names
    )


def test_recordings_table_metadata():
    facts = recording({"w": [], "x": []}, metadata={"background_fan": True})
    recordings = build(facts)[Table.RECORDINGS].to_pylist()
    assert recordings[0]["participant_ids"] == ["w", "x"]
    assert recordings[0]["wearer_index"] == 0
    assert recordings[0]["background_fan"] is True
    assert recordings[0]["background_music"] is None


def test_a_grid_recording_without_facts_is_an_error():
    facts = recording({"w": []})
    with pytest.raises(ValueError, match="no label facts"):
        tables_for([facts], {"r1": keys(), "other": keys()})


def test_grids_of_different_participant_counts_concatenate():
    first = recording({"w": [(1, 2)]}, recording_id="a")
    second = recording({"w": [(1, 2)], "x": [(3, 4)], "y": []}, recording_id="b")
    tables = tables_for([second, first], {"a": keys(), "b": keys()})
    grid = tables[Table.GRID]
    assert grid.column("recording_id").to_pylist()[0] == "a"
    widths = {len(v) for v in grid.column("speaker_activity").to_pylist()}
    assert widths == {1, 3}
