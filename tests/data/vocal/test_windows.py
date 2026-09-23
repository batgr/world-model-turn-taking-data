"""Pure anchor and window semantics (no dataset, no files)."""

import numpy as np
import pytest

from conv_wm.data.vocal.action_grid import ACTIONS, Action
from conv_wm.data.vocal.windows import (
    EVENT_ACTIONS,
    FUTURE_STEPS,
    MASKED_INDEX,
    MAX_CONTEXT_STEPS,
    MIN_CONTEXT_STEPS,
    SampleClass,
    WindowSpec,
    classify,
    is_trainable,
    segment_anchors,
    segment_starts,
)

SPEC = WindowSpec()
NO_EVENT = ACTIONS.index(str(Action.NO_EVENT))
ONSET = ACTIONS.index(str(Action.ONSET))


def _anchors(count, *, actions=None, valid=None, spec=SPEC):
    return segment_anchors(
        action_valid=np.ones(count, dtype=bool) if valid is None else np.asarray(valid),
        action_codes=np.full(count, NO_EVENT)
        if actions is None
        else np.asarray(actions),
        spec=spec,
    )


def test_the_event_vocabulary_is_derived_from_the_action_space():
    assert str(Action.NO_EVENT) not in EVENT_ACTIONS
    assert set(EVENT_ACTIONS) == {str(Action.ONSET), str(Action.OFFSET)}
    assert set(EVENT_ACTIONS) | {str(Action.NO_EVENT)} == set(ACTIONS)


def test_an_anchor_needs_the_minimum_context_before_it():
    anchors = _anchors(100)

    # positions 0..8 have fewer than 10 steps up to and including themselves
    assert anchors.position.min() == MIN_CONTEXT_STEPS - 1
    assert anchors.max_context_steps.min() == MIN_CONTEXT_STEPS


def test_an_anchor_needs_the_full_future_after_it():
    count = 100
    anchors = _anchors(count)

    assert anchors.position.max() == count - 1 - FUTURE_STEPS
    assert len(anchors.position) == count - FUTURE_STEPS - MIN_CONTEXT_STEPS + 1


def test_a_segment_too_short_for_one_window_yields_no_anchor():
    assert len(_anchors(MIN_CONTEXT_STEPS + FUTURE_STEPS - 1).position) == 0
    assert len(_anchors(MIN_CONTEXT_STEPS + FUTURE_STEPS).position) == 1


def test_the_available_context_grows_with_the_position_and_is_capped():
    anchors = _anchors(200)

    steps = dict(zip(anchors.position, anchors.max_context_steps, strict=True))
    assert steps[MIN_CONTEXT_STEPS - 1] == MIN_CONTEXT_STEPS
    assert steps[20] == 21
    assert steps[MAX_CONTEXT_STEPS - 1] == MAX_CONTEXT_STEPS
    assert steps[150] == MAX_CONTEXT_STEPS
    assert anchors.max_context_steps.max() == MAX_CONTEXT_STEPS


def test_segments_split_on_a_hole_in_the_decision_index():
    assert segment_starts(np.array([4, 5, 6, 7])).tolist() == [0]
    assert segment_starts(np.array([4, 5, 9, 10])).tolist() == [0, 2]
    assert segment_starts(np.array([], dtype=np.int64)).tolist() == []


def test_a_future_holding_an_action_is_an_event_and_the_rest_is_background():
    actions = np.full(60, NO_EVENT)
    actions[40] = ONSET
    anchors = _anchors(60, actions=actions)

    classes = dict(
        zip(anchors.position, classify(anchors.future_event_count), strict=True)
    )
    # the ONSET at step 40 is in the future of anchors 30..39
    assert classes[39] == str(SampleClass.EVENT)
    assert classes[30] == str(SampleClass.EVENT)
    assert classes[29] == str(SampleClass.BACKGROUND)
    assert classes[40] == str(SampleClass.BACKGROUND)  # the action is now behind it


def test_a_masked_action_is_neither_an_event_nor_valid():
    actions = np.full(60, NO_EVENT)
    actions[40] = MASKED_INDEX
    valid = np.ones(60, dtype=bool)
    valid[40] = False

    anchors = _anchors(60, actions=actions, valid=valid)

    events = dict(zip(anchors.position, anchors.future_event_count, strict=True))
    ratios = dict(zip(anchors.position, anchors.future_valid_ratio, strict=True))
    assert events[39] == 0
    assert ratios[39] == pytest.approx(0.9)
    assert ratios[29] == pytest.approx(1.0)


def test_validity_thresholds_select_trainable_anchors():
    valid = np.ones(80, dtype=bool)
    valid[:2] = False  # the first two steps are masked
    anchors = _anchors(80, valid=valid)

    trainable = dict(zip(anchors.position, is_trainable(anchors, SPEC), strict=True))
    ratios = dict(zip(anchors.position, anchors.context_valid_ratio, strict=True))
    assert ratios[9] == pytest.approx(0.8)
    assert not trainable[9]  # 8/10 is below the 0.90 threshold
    assert ratios[19] == pytest.approx(0.9)
    assert trainable[19]  # 18/20 is exactly at it
    assert trainable[69]  # far enough that the two masked steps are out of reach


def test_the_window_spec_refuses_an_impossible_geometry():
    with pytest.raises(ValueError, match="min_context_steps"):
        WindowSpec(min_context_steps=50, max_context_steps=10)
    with pytest.raises(ValueError, match="future_steps"):
        WindowSpec(future_steps=0)
    with pytest.raises(ValueError, match="min_future_valid_ratio"):
        WindowSpec(min_future_valid_ratio=1.5)


def test_the_prototype_geometry_is_one_to_five_seconds_of_context():
    assert (SPEC.min_context_seconds, SPEC.max_context_seconds) == (1.0, 5.0)
    assert SPEC.future_seconds == pytest.approx(1.0)
