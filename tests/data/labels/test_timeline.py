"""Continuous-time derivations on hand-made conversations with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest
from label_helpers import recording

from conv_wm.data.labels.facts import ParticipantFacts, RecordingFacts
from conv_wm.data.labels.timeline import (
    MULTIPLE,
    NO_SPEAKER,
    NONE,
    SPEAKING,
    UNKNOWN,
    LabelConfig,
    derive,
    transfer_kind,
    transfer_locality,
)

CONFIG = LabelConfig()


def onsets(structure):
    events = structure.events
    return [
        (round(float(t), 6), int(p))
        for t, k, p in zip(events.time_s, events.kind, events.participant, strict=True)
        if k == "onset"
    ]


def context_of(structure, time_s, participant):
    events = structure.events
    for index, context in structure.contexts.items():
        if (
            math.isclose(events.time_s[index], time_s)
            and events.participant[index] == participant
        ):
            return context
    raise AssertionError("no such onset")


def test_silence_only_has_no_event_and_no_floor():
    structure = derive(recording({"w": [], "x": []}), CONFIG)
    assert structure.pieces.count == 1
    assert len(structure.events.time_s) == 0
    assert structure.holder.tolist() == [NONE]
    assert structure.last_unique.tolist() == [NO_SPEAKER]
    assert [s.silence_type for s in structure.silences] == ["undetermined"]
    assert structure.turns == [] and structure.floor_changes == []


def test_the_wearer_is_always_participant_zero():
    facts = recording({"10": [(1, 2)], "2": [(3, 4)], "w": [(5, 6)]})
    structure = derive(facts, CONFIG)
    assert structure.pieces.participant_ids == ("w", "2", "10")


def test_single_speaker_runs_turns_and_pause():
    facts = recording({"w": [(1.0, 2.0), (2.2, 3.0), (5.0, 6.0)]})
    structure = derive(facts, CONFIG)
    assert [(r.start_s, r.end_s) for r in structure.runs] == [(1, 2), (2.2, 3), (5, 6)]
    # 0.2 s <= 0.3 s pause closure joins the first two runs; 2 s does not.
    assert [(t.start_s, t.end_s, t.run_count) for t in structure.turns] == [
        (1.0, 3.0, 2),
        (5.0, 6.0, 1),
    ]
    assert structure.turns[0].pauses == pytest.approx((0.2,))
    types = [(s.silence_type, s.valid) for s in structure.silences]
    # leading silence censored by the recording start, then two pauses, then the tail
    assert types == [
        ("undetermined", False),
        ("pause", True),
        ("pause", True),
        ("undetermined", False),
    ]


def test_clean_transfer_is_a_floor_change_with_a_positive_fto():
    facts = recording({"w": [(1.0, 2.0)], "x": [(2.5, 4.0)]})
    structure = derive(facts, CONFIG)
    (change,) = structure.floor_changes
    assert (change.previous_holder, change.next_holder) == (0, 1)
    assert change.time_s == 2.5
    assert change.fto_s == pytest.approx(0.5)
    assert transfer_kind(change.fto_s) == "gap"
    gap = next(s for s in structure.silences if s.start_s == 2.0)
    assert (gap.silence_type, gap.previous_speaker, gap.next_speaker) == ("gap", 0, 1)
    assert context_of(structure, 2.5, 1).onset_type == "floor_transfer"


def test_overlapping_transfer_has_a_negative_fto_at_the_first_speakers_offset():
    facts = recording({"w": [(1.0, 3.0)], "x": [(2.5, 4.0)]})
    structure = derive(facts, CONFIG)
    (change,) = structure.floor_changes
    assert change.time_s == 3.0  # x becomes the sole speaker when w stops
    assert change.fto_s == pytest.approx(-0.5)
    assert transfer_kind(change.fto_s) == "overlap"
    (overlap,) = structure.overlaps
    assert (overlap.initial, overlap.entering, overlap.overlap_type) == (
        0,
        1,
        "between",
    )
    assert (overlap.start_s, overlap.end_s) == (2.5, 3.0)
    assert overlap.previous_floor_holder == 0
    context = context_of(structure, 2.5, 1)
    assert context.onset_type == "overlap"
    assert context.others_active_before is True
    assert context.silence_before_s == 0.0


def test_within_overlap_keeps_the_floor():
    facts = recording({"w": [(1.0, 4.0)], "x": [(2.0, 2.5)]})
    structure = derive(facts, CONFIG)
    (overlap,) = structure.overlaps
    assert overlap.overlap_type == "within"
    assert structure.floor_changes == []


def test_within_overlap_ending_exactly_with_the_initial_run_is_within():
    facts = recording({"w": [(1.0, 3.0)], "x": [(2.0, 3.0)]})
    (overlap,) = derive(facts, CONFIG).overlaps
    assert overlap.overlap_type == "within"


def test_simultaneous_onset_after_silence():
    facts = recording({"w": [(1.0, 3.0)], "x": [(1.02, 2.0)]})
    structure = derive(facts, CONFIG)
    (overlap,) = structure.overlaps
    assert overlap.overlap_type == "simultaneous_onset"
    ego = context_of(structure, 1.0, 0)
    assert ego.simultaneous_other_onset is True
    assert ego.others_active_before is False
    assert ego.others_active_at_onset is False  # x starts 20 ms later
    # nobody spoke before: the recording start censors the silence
    assert ego.onset_type == "undetermined" and ego.silence_before_s is None


def test_exactly_equal_starts_are_counted_once_with_the_lower_index_initial():
    facts = recording({"w": [(1.0, 2.0)], "x": [(1.0, 3.0)]})
    (overlap,) = derive(facts, CONFIG).overlaps
    assert (overlap.initial, overlap.entering) == (0, 1)
    assert overlap.overlap_type == "simultaneous_onset"


def test_onset_after_silence_is_a_self_resumption():
    facts = recording({"w": [(1.0, 2.0), (3.0, 4.0)], "x": [(0.2, 0.5)]})
    structure = derive(facts, CONFIG)
    context = context_of(structure, 3.0, 0)
    assert context.onset_type == "after_silence"
    assert context.previous_unique_speaker == 0
    assert context.silence_before_s == pytest.approx(1.0)


def test_three_speakers_floor_and_holders():
    facts = recording(
        {"w": [(1.0, 2.0)], "a": [(2.5, 3.5)], "b": [(3.2, 5.0)]},
    )
    structure = derive(facts, CONFIG)
    ids = structure.pieces.participant_ids
    assert ids == ("w", "a", "b")
    changes = [(c.previous_holder, c.next_holder) for c in structure.floor_changes]
    assert changes == [(0, 1), (1, 2)]
    overlap_piece = np.flatnonzero(
        (structure.pieces.starts >= 3.2) & (structure.pieces.ends <= 3.5)
    )[0]
    assert structure.holder[overlap_piece] == MULTIPLE
    assert structure.last_unique[overlap_piece] == 1


def test_unknown_region_blocks_events_and_resets_the_floor():
    facts = recording(
        {"w": [(1.0, 2.0), (4.5, 5.0)], "x": [(2.5, 3.5)]},
        unknown=[(3.0, 4.0)],
    )
    structure = derive(facts, CONFIG)
    states = structure.pieces.states
    unknown_piece = np.flatnonzero(structure.pieces.starts == 3.0)[0]
    assert (states[unknown_piece] == UNKNOWN).all()
    # x's offset at 3.5 is hidden by UNKNOWN: not an event
    offsets = [
        (float(t), int(p))
        for t, k, p in zip(
            structure.events.time_s,
            structure.events.kind,
            structure.events.participant,
            strict=True,
        )
        if k == "offset"
    ]
    assert (3.5, 1) not in offsets and (3.0, 1) not in offsets
    assert structure.last_unique[unknown_piece] == NO_SPEAKER
    after = np.flatnonzero(structure.pieces.starts == 4.0)[0]
    assert structure.last_unique[after] == NO_SPEAKER  # nobody spoke alone yet
    # the onset at 4.5 follows a silence censored by UNKNOWN
    assert context_of(structure, 4.5, 0).onset_type == "undetermined"
    x_run = next(r for r in structure.runs if r.participant == 1)
    assert x_run.end_censored and x_run.end_s == 3.0
    assert structure.spans.starts.tolist() == [0.0, 4.0]
    assert structure.spans.ends.tolist() == [3.0, 10.0]


def test_speech_in_progress_at_the_recording_start_is_not_an_onset():
    facts = recording({"w": [(0.0, 1.0)], "x": [(1.5, 2.0)]})
    structure = derive(facts, CONFIG)
    assert (0.0, 0) not in onsets(structure)
    assert structure.runs[0].start_censored
    assert structure.runs[-1].end_censored is False


def test_speech_until_the_recording_end_is_censored():
    facts = recording({"w": [(9.0, 12.0)]})
    run = derive(facts, CONFIG).runs[0]
    assert run.end_s == 10.0 and run.end_censored


def test_a_participant_who_never_speaks_is_still_a_participant():
    facts = recording({"w": [(1.0, 2.0)], "silent": []})
    structure = derive(facts, CONFIG)
    assert structure.pieces.participant_ids == ("w", "silent")
    assert not (structure.pieces.states[:, 1] == SPEAKING).any()


def test_a_participant_specific_unknown_region():
    facts = recording(
        {"w": [(1.0, 2.0)], "x": [(3.0, 4.0)]}, participant_unknown={"x": [(0.0, 5.0)]}
    )
    structure = derive(facts, CONFIG)
    assert (structure.pieces.states[structure.pieces.starts < 5.0, 1] == UNKNOWN).all()
    assert (0.0, 1) not in onsets(structure)
    assert structure.floor_changes == []


def test_unresolved_speech_is_its_own_last_participant():
    facts = recording(
        {"w": [(1.0, 2.0)], "x": [(4.0, 5.0)], "<unresolved>": [(2.5, 3.0)]},
        unresolved=["<unresolved>"],
    )
    structure = derive(facts, CONFIG)
    assert structure.pieces.participant_ids == ("w", "x", "<unresolved>")
    assert structure.pieces.resolved == (True, True, False)


def test_variable_participant_count_is_supported():
    for count in (1, 2, 5):
        speech = {"w": [(0.5, 1.0)]}
        speech.update({f"p{i}": [(1.0 + i, 1.5 + i)] for i in range(count - 1)})
        structure = derive(recording(speech), CONFIG)
        assert structure.pieces.n_participants == count


def test_turn_neighbours_and_fto():
    facts = recording({"w": [(1.0, 2.0), (4.0, 5.0)], "x": [(2.4, 3.0)]})
    turns = derive(facts, CONFIG).turns
    assert [t.participant for t in turns] == [0, 1, 0]
    assert turns[1].previous_participant == 0 and turns[1].next_participant == 0
    assert turns[1].fto_from_previous_s == pytest.approx(0.4)
    assert math.isnan(turns[0].fto_from_previous_s)


def test_a_turn_does_not_close_a_pause_across_unknown_time():
    facts = recording({"w": [(1.0, 2.0), (2.2, 3.0)]}, unknown=[(2.05, 2.1)])
    turns = derive(facts, CONFIG).turns
    assert len(turns) == 2


def test_overlap_type_is_undetermined_when_an_end_is_censored():
    facts = recording({"w": [(8.0, 10.0)], "x": [(9.0, 10.0)]})
    (overlap,) = derive(facts, CONFIG).overlaps
    assert overlap.overlap_type == "undetermined" and not overlap.valid


def test_transfer_locality_flags_long_transitions_without_dropping_them():
    assert transfer_locality(3.0, CONFIG) == "long_gap"
    assert transfer_locality(-3.0, CONFIG) == "long_overlap"
    assert transfer_locality(0.2, CONFIG) == "local"


def test_config_rejects_unsorted_horizons():
    with pytest.raises(ValueError, match="increasing"):
        LabelConfig(future_horizons_s=(1.0, 0.5))


def test_config_from_mapping_rejects_unknown_settings():
    with pytest.raises(ValueError, match="unknown labels settings"):
        LabelConfig.from_mapping({"horizons": [1]})


def test_config_round_trips():
    config = LabelConfig.from_mapping({"future_horizons_s": [0.2, 2.0]})
    assert config.future_horizons_s == (0.2, 2.0)
    assert LabelConfig.from_mapping(config.to_dict()) == config


def test_facts_reject_a_missing_wearer():
    with pytest.raises(ValueError, match="wearer"):
        RecordingFacts(
            dataset="toy",
            recording_id="r",
            conversation_id="c",
            view_id="v",
            wearer_id="w",
            start_s=0.0,
            end_s=1.0,
            participants=(ParticipantFacts("x", False),),
        )
