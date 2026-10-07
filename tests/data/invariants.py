"""Invariants of the built layers, checked on test builds (not at build time).

Each function re-derives a promise of one layer from its output, independently
of the code that produced it.
"""

from itertools import pairwise

import numpy as np
import pandas as pd

from conv_wm.data.pipeline.control_state import _RecordingResult
from conv_wm.data.pipeline.model_ready import split_leakage
from conv_wm.data.vocal.action_grid import (
    ACTIONS,
    BOUNDARY_TOLERANCE_S,
    DECISION_STEP_S,
    STATE_NAMES,
    Action,
)
from conv_wm.data.vocal.native_state import VoiceState
from conv_wm.data.vocal.windows import WindowSpec, canonical_grid


def check_action_grid(grid: pd.DataFrame, step_s: float = DECISION_STEP_S) -> None:
    """Every contract of the action grid holds."""
    if grid.empty:
        return
    valid = grid.loc[grid["action_valid"]]
    masked = grid.loc[~grid["action_valid"]]
    problems: list[str] = []
    if not valid["action"].isin(ACTIONS).all():
        problems.append("a valid slot carries an action outside the vocabulary")
    if masked["action"].notna().any() or masked["mask_reason"].isna().any():
        problems.append("a masked slot carries an action or no mask reason")
    if valid["mask_reason"].notna().any():
        problems.append("a valid slot carries a mask reason")
    no_event = valid.loc[valid["action"].eq(str(Action.NO_EVENT))]
    if no_event["event_time_s"].notna().any() or no_event["tau_s"].notna().any():
        problems.append("NO_EVENT carries an event time")
    if not no_event["focal_state_before"].isin(STATE_NAMES[:2]).all():
        problems.append("NO_EVENT with an UNKNOWN state before")
    for action, state in ((Action.ONSET, "SILENT"), (Action.OFFSET, "SPEAKING")):
        rows = valid.loc[valid["action"].eq(str(action))]
        if not rows["focal_state_before"].eq(state).all():
            problems.append(f"{action} from a state other than {state}")
        tau = rows["tau_s"].to_numpy(dtype=float)
        if len(tau) and (np.any(tau < 0.0) or np.any(tau >= step_s)):
            problems.append(f"{action} tau outside [0, {step_s})")
        event = rows["event_time_s"].to_numpy(dtype=float)
        if len(event) and not np.allclose(
            event - rows["decision_time_s"].to_numpy(dtype=float), tau
        ):
            problems.append(f"{action} event time inconsistent with tau")
    if grid.duplicated(["recording_id", "decision_index"]).any():
        problems.append("duplicate (recording_id, decision_index)")
    expected = grid["decision_index"].to_numpy(dtype=float) * step_s
    if not np.allclose(
        grid["decision_time_s"].to_numpy(dtype=float), expected, atol=1e-9
    ):
        problems.append("decision_time_s is not decision_index * step")
    increasing = grid.groupby("recording_id", sort=False)["decision_index"].apply(
        lambda values: bool(values.is_monotonic_increasing)
    )
    if not increasing.all():
        problems.append("decision_index is not strictly increasing within a recording")
    if problems:
        raise ValueError("action grid invariants violated: " + "; ".join(problems))


def check_control_partition(
    results: list[_RecordingResult], *, tolerance_s: float = BOUNDARY_TOLERANCE_S
) -> None:
    """The control timeline must be a requantization of the native one.

    Bridging may only remove interior boundaries, so the control intervals must
    partition the native ones in order and keep the same span and total
    duration. A degenerate native timeline (a gap or an overlap) is copied
    through unchanged; refusing it is the action grid's job, which masks that
    recording as ``invalid_timeline``.
    """
    problems: list[str] = []
    for result in results:
        native, control = result.native, result.control
        if not control:
            problems.append(f"{result.recording_id}: empty control timeline")
            continue
        if (
            abs(control[0].canonical_start_s - native[0].canonical_start_s)
            > tolerance_s
            or abs(control[-1].canonical_end_s - native[-1].canonical_end_s)
            > tolerance_s
        ):
            problems.append(
                f"{result.recording_id}: control window differs from native"
            )
        covered = sum(row.duration_s for row in control)
        if abs(covered - sum(row.duration_s for row in native)) > tolerance_s:
            problems.append(f"{result.recording_id}: total duration changed")
        indices = [
            (row.source_native_index_first, row.source_native_index_last)
            for row in control
        ]
        expected = [(0, indices[0][1])] + [
            (previous[1] + 1, current[1]) for previous, current in pairwise(indices)
        ]
        if indices != expected or indices[-1][1] != len(native) - 1:
            problems.append(
                f"{result.recording_id}: control intervals do not partition the native ones"
            )
        if any(
            row.voice_state != VoiceState.SPEAKING and row.transform_kind is not None
            for row in control
        ):
            problems.append(
                f"{result.recording_id}: a non-SPEAKING interval was bridged"
            )
        bridged = sum(row.bridged_gap_count for row in control)
        if bridged != len(result.gaps):
            problems.append(
                f"{result.recording_id}: bridged gap provenance is incomplete"
            )
        if any(
            row.source_native_index_last < row.source_native_index_first
            for row in control
        ):
            problems.append(f"{result.recording_id}: reversed native index range")
    if problems:
        raise ValueError(
            "control focal voice state invariants violated: " + "; ".join(problems[:10])
        )


def model_ready_contract(
    index: pd.DataFrame, grid: pd.DataFrame, spec: WindowSpec
) -> dict[str, bool]:
    """The index's promises, re-derived from the grid it was built from.

    Cheap and independent of the code that produced the index, so a regression
    in anchor generation shows up as a failed check in the report rather than
    as a silently wrong dataset.
    """
    if index.empty:
        return dict.fromkeys(
            (
                "no_recording_in_multiple_splits",
                "every_anchor_has_minimum_context",
                "every_anchor_has_full_future",
                "no_anchor_crosses_a_recording_boundary",
            ),
            True,
        )
    ordered = canonical_grid(grid)
    bounds = ordered.groupby("recording_id", observed=True)["decision_index"].agg(
        ["min", "max"]
    )
    recording = index["recording_id"]
    first = recording.map(bounds["min"]).to_numpy()
    last = recording.map(bounds["max"]).to_numpy()
    anchor = index["anchor_idx"].to_numpy()
    context_start = anchor - index["max_context_steps"].to_numpy() + 1
    future_end = anchor + index["future_steps"].to_numpy()
    rows = ordered["recording_id"].to_numpy()
    return {
        "no_recording_in_multiple_splits": bool(
            index.groupby("recording_id", observed=True)["split"]
            .nunique(dropna=False)
            .le(1)
            .all()
        )
        and not split_leakage(index),
        "every_anchor_has_minimum_context": bool(
            (index["max_context_steps"] >= spec.min_context_steps).all()
            and (index["max_context_steps"] <= spec.max_context_steps).all()
        ),
        "every_anchor_has_full_future": bool((future_end <= last).all()),
        "no_anchor_crosses_a_recording_boundary": bool(
            (context_start >= first).all()
            and (future_end <= last).all()
            and np.array_equal(
                rows[index["anchor_row"].to_numpy()], recording.to_numpy()
            )
        ),
    }
