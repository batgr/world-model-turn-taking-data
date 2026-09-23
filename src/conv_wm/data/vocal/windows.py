"""Vocal action grid -> valid training windows (pure, no torch, no files).

One window is anchored on a decision slot ``t`` of the grid:

```text
context = [t - L + 1, ..., t]        L ∈ [min_context_steps, max_context_steps]
future  = [t + 1, ..., t + H]        H = future_steps
```

Steps are grid slots of :data:`~conv_wm.data.vocal.action_grid.DECISION_STEP_S`
(100 ms, so 10 Hz); the prototype defaults are 1 s of minimum context, 5 s of
maximum context and a 1 s future horizon.

A window never crosses a session boundary, and never crosses a discontinuity
inside a session either: anchors are computed per *segment*, a maximal run of
consecutive ``decision_index`` values of one recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from conv_wm.data.vocal.action_grid import ACTIONS, DECISION_STEP_S, Action

WINDOW_SCHEMA_VERSION = 1
"""Bumped when the index columns or the anchor-validity semantics change."""

MIN_CONTEXT_STEPS = 10
"""1 s of context at 100 ms per step."""
MAX_CONTEXT_STEPS = 50
"""5 s of context."""
FUTURE_STEPS = 10
"""1 s of future horizon. Fixed in the prototype."""
MIN_CONTEXT_VALID_RATIO = 0.90
MIN_FUTURE_VALID_RATIO = 1.0

MASKED_INDEX = -1
"""Code for a slot the action grid masked, and for a padded step after collation.

The two are distinguished by the mask: a padded step has ``context_mask``
false, a masked real step has ``context_mask`` true and ``context_valid`` false.
"""

ACTION_INDEX: dict[str, int] = {name: index for index, name in enumerate(ACTIONS)}
"""``NO_EVENT`` 0, ``ONSET`` 1, ``OFFSET`` 2 — the action grid's own order."""

EVENT_ACTIONS: tuple[str, ...] = tuple(
    name for name in ACTIONS if name != str(Action.NO_EVENT)
)
"""The actions that log a vocal event: everything but "hold the current state".

Derived from the action vocabulary rather than listed, so a change to
:class:`~conv_wm.data.vocal.action_grid.Action` reaches this layer.
"""
EVENT_ACTION_INDICES: tuple[int, ...] = tuple(
    ACTION_INDEX[name] for name in EVENT_ACTIONS
)

GRID_SORT_COLUMNS = ("recording_id", "decision_index")
"""The canonical row order of the grid that ``anchor_row`` addresses."""


class SampleClass(StrEnum):
    """Binary sampling class of a window, from its future horizon."""

    EVENT = "event"
    """The future contains at least one ``ONSET`` or ``OFFSET``."""
    BACKGROUND = "background"
    """The future contains no vocal event."""


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry and the validity thresholds an anchor must satisfy."""

    min_context_steps: int = MIN_CONTEXT_STEPS
    max_context_steps: int = MAX_CONTEXT_STEPS
    future_steps: int = FUTURE_STEPS
    min_context_valid_ratio: float = MIN_CONTEXT_VALID_RATIO
    min_future_valid_ratio: float = MIN_FUTURE_VALID_RATIO
    feature_columns: tuple[str, ...] = ()
    """Extra per-step numeric grid columns to expose as ``context_features``.

    Empty in v0: the vocal action grid carries no multimodal feature column, so
    a window's per-step observation is its state and action sequence.
    """

    def __post_init__(self) -> None:
        if not 1 <= self.min_context_steps <= self.max_context_steps:
            raise ValueError(
                "expected 1 <= min_context_steps <= max_context_steps, got "
                f"{self.min_context_steps} and {self.max_context_steps}"
            )
        if self.future_steps < 1:
            raise ValueError(f"future_steps must be positive, got {self.future_steps}")
        for name in ("min_context_valid_ratio", "min_future_valid_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def min_context_seconds(self) -> float:
        """Minimum context duration in seconds."""
        return self.min_context_steps * DECISION_STEP_S

    @property
    def max_context_seconds(self) -> float:
        """Maximum context duration in seconds."""
        return self.max_context_steps * DECISION_STEP_S

    @property
    def future_seconds(self) -> float:
        """Future horizon in seconds."""
        return self.future_steps * DECISION_STEP_S


def canonical_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """The grid in the row order ``anchor_row`` addresses (stable, reindexed)."""
    return grid.sort_values(list(GRID_SORT_COLUMNS), kind="stable", ignore_index=True)


def encode_actions(actions: pd.Series) -> np.ndarray:
    """Action names -> codes, :data:`MASKED_INDEX` where the slot is masked."""
    return (
        actions.map(ACTION_INDEX)
        .astype("Int64")
        .fillna(MASKED_INDEX)
        .to_numpy(np.int64)
    )


def segment_starts(decision_index: np.ndarray) -> np.ndarray:
    """Positions where a run of consecutive decision indices begins.

    The action grid emits contiguous slots per recording, so there is normally
    one segment per session; a hole would otherwise let a window silently span
    missing time, which this splits instead.
    """
    if len(decision_index) == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(decision_index) != 1) + 1
    return np.concatenate([[0], breaks]).astype(np.int64)


@dataclass(frozen=True)
class SegmentAnchors:
    """Structurally valid anchors of one segment, as column arrays.

    ``position`` is relative to the segment; ratios are measured over the
    largest context this anchor allows and over the full future horizon.
    """

    position: np.ndarray
    max_context_steps: np.ndarray
    context_valid_ratio: np.ndarray
    future_valid_ratio: np.ndarray
    future_event_count: np.ndarray


def segment_anchors(
    *,
    action_valid: np.ndarray,
    action_codes: np.ndarray,
    spec: WindowSpec,
) -> SegmentAnchors:
    """Anchors of one segment that have enough context and a complete future.

    Structural validity only: an anchor needs ``min_context_steps`` slots up to
    and including itself and ``future_steps`` slots after it, all inside this
    segment. The validity ratios are measured but not thresholded here, so the
    index can report how many anchors a threshold rejects.
    """
    count = len(action_valid)
    horizon = spec.future_steps
    position = np.arange(count, dtype=np.int64)
    structural = (position + 1 >= spec.min_context_steps) & (position + horizon < count)
    position = position[structural]
    if not len(position):
        empty_int = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=float)
        return SegmentAnchors(empty_int, empty_int, empty_float, empty_float, empty_int)
    available = np.minimum(position + 1, spec.max_context_steps)
    valid_cumulative = np.concatenate([[0], np.cumsum(action_valid.astype(np.int64))])
    is_event = np.isin(action_codes, EVENT_ACTION_INDICES)
    event_cumulative = np.concatenate([[0], np.cumsum(is_event.astype(np.int64))])
    context_valid = (
        valid_cumulative[position + 1] - valid_cumulative[position + 1 - available]
    )
    future_stop = position + 1 + horizon
    future_valid = valid_cumulative[future_stop] - valid_cumulative[position + 1]
    return SegmentAnchors(
        position=position,
        max_context_steps=available,
        context_valid_ratio=context_valid / available,
        future_valid_ratio=future_valid / horizon,
        future_event_count=(
            event_cumulative[future_stop] - event_cumulative[position + 1]
        ),
    )


def classify(future_event_count: np.ndarray) -> np.ndarray:
    """``event`` where the future horizon logs at least one action, else ``background``."""
    return np.where(
        future_event_count > 0, str(SampleClass.EVENT), str(SampleClass.BACKGROUND)
    )


def is_trainable(anchors: SegmentAnchors, spec: WindowSpec) -> np.ndarray:
    """Anchors whose measured validity satisfies both configured thresholds."""
    return (anchors.context_valid_ratio >= spec.min_context_valid_ratio) & (
        anchors.future_valid_ratio >= spec.min_future_valid_ratio
    )


__all__ = [
    "ACTION_INDEX",
    "EVENT_ACTIONS",
    "EVENT_ACTION_INDICES",
    "FUTURE_STEPS",
    "GRID_SORT_COLUMNS",
    "MASKED_INDEX",
    "MAX_CONTEXT_STEPS",
    "MIN_CONTEXT_STEPS",
    "WINDOW_SCHEMA_VERSION",
    "SampleClass",
    "SegmentAnchors",
    "WindowSpec",
    "canonical_grid",
    "classify",
    "encode_actions",
    "is_trainable",
    "segment_anchors",
    "segment_starts",
]
