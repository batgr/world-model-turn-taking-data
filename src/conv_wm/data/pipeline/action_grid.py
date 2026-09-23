"""Build the v0 vocal action grid from the control focal voice-state layer.

The production grid is sampled from the *control* timeline — the native one
requantized at Δ by sub-step silence bridging
(:mod:`conv_wm.data.pipeline.control_state`). This module never reads a
dataset's annotations, never touches audio and never merges intervals itself:
the only transform lives one layer up, and this build records which one ran.

The native timeline is loaded as well and gridded with the same functions, but
only to produce the before/after comparison in the report: it is never written
as a production table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.pipeline.control_state import (
    NATIVE_BUILD_COMMAND,
    config_checksum,
    selected_datasets,
    supported_datasets,
)
from conv_wm.data.pipeline.control_state import (
    PROCESSED_ROOT as CONTROL_PROCESSED_ROOT,
)
from conv_wm.data.pipeline.control_state import (
    REPORT_FILE as CONTROL_REPORT_FILE,
)
from conv_wm.data.pipeline.control_state import (
    REPORT_ROOT as CONTROL_REPORT_ROOT,
)
from conv_wm.data.pipeline.control_state import (
    TIMELINE_TABLE as CONTROL_TIMELINE_TABLE,
)
from conv_wm.data.pipeline_inputs import (
    CheckedArtifact,
    artifact_reference,
    load_checked_artifact,
    sha256_file,
)
from conv_wm.data.vocal.action_grid import (
    ACTION_SCHEMA_VERSION,
    ACTIONS,
    DECISION_STEP_S,
    MASK_REASONS,
    STATE_NAMES,
    Action,
    ActionGrid,
    MaskReason,
    RecordingTimeline,
    SubDeltaGap,
    build_action_grid,
    masked_grid,
    slot_index,
    sub_delta_gaps,
)
from conv_wm.data.vocal.control_state import CONTROL_STATE_SCHEMA_VERSION
from conv_wm.data.vocal.native_state import NATIVE_STATE_SCHEMA_VERSION
from conv_wm.provenance import collect_provenance
from conv_wm.reports import JsonDict, write_summary, write_table

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 2
PROCESSED_ROOT = Path("vocal_action_grid")
REPORT_ROOT = Path("vocal_action_grid")
GRID_TABLE = "vocal_action_grid.parquet"
SUMMARY_TABLE = "summary.parquet"
COMPOUND_TABLE = "compound_slots.parquet"
SUB_DELTA_TABLE = "sub_delta_gap_analysis.parquet"
REPORT_FILE = "report.json"

IDENTITY_COLUMNS = ("dataset", "recording_id", "sync_group_id", "view_id", "wearer_id")
CONTROL_BUILD_COMMAND = "conv-wm build control-focal-voice-state"


@dataclass(frozen=True)
class VocalActionGridOutputs:
    """Tables and report written for one dataset."""

    dataset: str
    grid: pd.DataFrame
    summary: pd.DataFrame
    compound_slots: pd.DataFrame
    sub_delta_gaps: pd.DataFrame
    report: JsonDict
    grid_path: Path
    report_path: Path


def load_control_state_input(cfg: DictConfig, dataset: str) -> CheckedArtifact:
    """Read one dataset's control timeline, refusing an input its report disowns."""
    paths = pipeline_paths(cfg)
    return load_checked_artifact(
        dataset=dataset,
        table_path=(
            paths.processed / CONTROL_PROCESSED_ROOT / dataset / CONTROL_TIMELINE_TABLE
        ),
        report_path=paths.reports / CONTROL_REPORT_ROOT / dataset / CONTROL_REPORT_FILE,
        schema_key="control_state_schema_version",
        expected_version=CONTROL_STATE_SCHEMA_VERSION,
        artifact_key="timeline",
        produce_with=f"{CONTROL_BUILD_COMMAND} --dataset {dataset}",
    )


def load_native_timeline_for_comparison(source: CheckedArtifact) -> pd.DataFrame:
    """The native timeline the control layer was built from, for the diagnostic grid.

    The path and checksum come from the control report, so the comparison can
    only ever be made against the exact bytes that produced the control layer.
    """
    reference = source.report["input_artifacts"]["native_focal_voice_state_timeline"]
    path = Path(reference["path"])
    if not path.exists():
        raise MissingPrerequisiteError(
            path, produce_with=f"{NATIVE_BUILD_COMMAND} --dataset {source.dataset}"
        )
    digest = sha256_file(path)
    if digest != reference["sha256"]:
        raise ValueError(
            f"{path}: checksum {digest} does not match the one recorded in "
            f"{source.report_path} ({reference['sha256']}); rerun "
            f"{CONTROL_BUILD_COMMAND} --dataset {source.dataset}"
        )
    return pd.read_parquet(path)


@dataclass
class _RecordingResult:
    """One recording's grid plus what the report needs to describe it."""

    identity: dict[str, object]
    grid: ActionGrid
    source_transition_count: int
    transitions_outside_grid: int
    gaps: list[SubDeltaGap]
    invalid_reason: str | None


def _recording_results(timeline: pd.DataFrame) -> list[_RecordingResult]:
    """Grid every recording of one dataset's timeline, in stable recording order."""
    ordered = timeline.sort_values(
        ["recording_id", "canonical_start_s"], kind="stable", ignore_index=True
    )
    recording_ids = ordered["recording_id"].to_numpy()
    starts = ordered["canonical_start_s"].to_numpy(dtype=float)
    ends = ordered["canonical_end_s"].to_numpy(dtype=float)
    state_names = ordered["voice_state"].to_numpy()
    boundaries = np.flatnonzero(recording_ids[1:] != recording_ids[:-1]) + 1
    slices = zip(
        np.concatenate([[0], boundaries]),
        np.concatenate([boundaries, [len(ordered)]]),
        strict=True,
    )
    results: list[_RecordingResult] = []
    for first, last in slices:
        row = ordered.iloc[first]
        identity = {column: row[column] for column in IDENTITY_COLUMNS}
        recording = RecordingTimeline.from_states(
            starts[first:last], ends[first:last], list(state_names[first:last])
        )
        invalid_reason = recording.validity_error()
        if invalid_reason is not None:
            grid = masked_grid(recording, MaskReason.INVALID_TIMELINE)
            results.append(_RecordingResult(identity, grid, 0, 0, [], invalid_reason))
            continue
        grid = build_action_grid(recording)
        resolved_count = int(grid.transitions.resolved.sum())
        represented_or_masked = int(grid.resolved_transitions_in_slot.sum())
        results.append(
            _RecordingResult(
                identity,
                grid,
                resolved_count,
                resolved_count - represented_or_masked,
                sub_delta_gaps(recording, grid.bounds),
                None,
            )
        )
    return results


def _grid_table(results: list[_RecordingResult], dataset: str) -> pd.DataFrame:
    """Assemble the slot table from per-recording column arrays (no per-row Python)."""
    if not results:
        return pd.DataFrame(
            columns=[
                *IDENTITY_COLUMNS,
                "decision_index",
                "decision_time_s",
                "slot_end_s",
                "focal_state_before",
                "action",
                "action_valid",
                "event_time_s",
                "tau_s",
                "mask_reason",
                "source_control_state_schema_version",
                "source_native_state_schema_version",
                "action_schema_version",
            ]
        )
    counts = [len(result.grid.decision_index) for result in results]
    state_lookup = np.array([*STATE_NAMES, None], dtype=object)
    action_lookup = np.array([*ACTIONS, None], dtype=object)
    reason_lookup = np.array([*MASK_REASONS, None], dtype=object)
    frame = pd.DataFrame(
        {
            "decision_index": np.concatenate([r.grid.decision_index for r in results]),
            "decision_time_s": np.concatenate(
                [r.grid.decision_time_s for r in results]
            ),
            "slot_end_s": np.concatenate([r.grid.slot_end_s for r in results]),
            "focal_state_before": state_lookup[
                np.concatenate([r.grid.focal_state_before for r in results])
            ],
            "action": action_lookup[np.concatenate([r.grid.action for r in results])],
            "action_valid": np.concatenate([r.grid.action_valid for r in results]),
            "event_time_s": np.concatenate([r.grid.event_time_s for r in results]),
            "tau_s": np.concatenate([r.grid.tau_s for r in results]),
            "mask_reason": reason_lookup[
                np.concatenate([r.grid.mask_reason for r in results])
            ],
        }
    )
    for column in IDENTITY_COLUMNS:
        values = np.array([result.identity[column] for result in results], dtype=object)
        frame[column] = pd.Series(np.repeat(values, counts), dtype="string")
    frame["dataset"] = pd.Series(dataset, index=frame.index, dtype="string")
    frame["source_control_state_schema_version"] = CONTROL_STATE_SCHEMA_VERSION
    frame["source_native_state_schema_version"] = NATIVE_STATE_SCHEMA_VERSION
    frame["action_schema_version"] = ACTION_SCHEMA_VERSION
    for column in ("focal_state_before", "action", "mask_reason"):
        frame[column] = frame[column].astype("string")
    return frame[
        [
            *IDENTITY_COLUMNS,
            "decision_index",
            "decision_time_s",
            "slot_end_s",
            "focal_state_before",
            "action",
            "action_valid",
            "event_time_s",
            "tau_s",
            "mask_reason",
            "source_control_state_schema_version",
            "source_native_state_schema_version",
            "action_schema_version",
        ]
    ]


def compound_pattern(state_before: str, states_after: list[str]) -> str:
    """The state chain a compound slot walks through, e.g. ``SILENT-SPEAKING-SILENT``.

    ``SILENT-SPEAKING-SILENT`` is a short speech burst, which the control layer
    deliberately keeps; ``SPEAKING-SILENT-SPEAKING`` is a sub-step silence,
    which sub-step silence bridging removes.
    """
    return "-".join([state_before, *states_after])


def _compound_table(results: list[_RecordingResult]) -> pd.DataFrame:
    """One row per slot holding more than one resolved transition."""
    rows: list[dict[str, object]] = []
    for result in results:
        grid = result.grid
        compound = np.flatnonzero(grid.resolved_transitions_in_slot > 1)
        if not len(compound):
            continue
        resolved = grid.transitions.resolved
        times = grid.transitions.time_s[resolved]
        after = grid.transitions.after[resolved]
        # Select by slot index, exactly as the counting does: a comparison
        # against the float slot bounds would drop an event sitting on t_k.
        transition_slots = slot_index(times)
        for position in compound:
            start = grid.decision_time_s[position]
            inside = transition_slots == grid.decision_index[position]
            rows.append(
                {
                    "dataset": result.identity["dataset"],
                    "recording_id": result.identity["recording_id"],
                    "decision_index": int(grid.decision_index[position]),
                    "decision_time_s": float(start),
                    "transition_count": int(
                        grid.resolved_transitions_in_slot[position]
                    ),
                    "state_before": STATE_NAMES[int(grid.focal_state_before[position])],
                    "event_times_s": [float(value) for value in times[inside]],
                    "event_types": [
                        str(Action.ONSET) if code == 1 else str(Action.OFFSET)
                        for code in after[inside]
                    ],
                    "state_after": STATE_NAMES[int(after[inside][-1])]
                    if inside.any()
                    else None,
                    "pattern": compound_pattern(
                        STATE_NAMES[int(grid.focal_state_before[position])],
                        [STATE_NAMES[int(code)] for code in after[inside]],
                    ),
                }
            )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "dataset",
            "recording_id",
            "decision_index",
            "decision_time_s",
            "transition_count",
            "state_before",
            "event_times_s",
            "event_types",
            "state_after",
            "pattern",
        ],
    )


def _sub_delta_table(results: list[_RecordingResult]) -> pd.DataFrame:
    """One row per native SILENT gap shorter than Δ (measurement only)."""
    rows = [
        {
            "dataset": result.identity["dataset"],
            "recording_id": result.identity["recording_id"],
            "gap_duration_s": gap.gap_duration_s,
            "offset_time_s": gap.offset_time_s,
            "onset_time_s": gap.onset_time_s,
            "offset_slot_index": gap.offset_slot_index,
            "onset_slot_index": gap.onset_slot_index,
            "cross_slot": gap.cross_slot,
            "within_grid": gap.within_grid,
        }
        for result in results
        for gap in result.gaps
    ]
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "dataset",
            "recording_id",
            "gap_duration_s",
            "offset_time_s",
            "onset_time_s",
            "offset_slot_index",
            "onset_slot_index",
            "cross_slot",
            "within_grid",
        ],
    )


def _summary_table(results: list[_RecordingResult], dataset: str) -> pd.DataFrame:
    """One row per recording: slot counts, action counts, masks, dropped time."""
    rows: list[dict[str, object]] = []
    for result in results:
        grid = result.grid
        valid = grid.action_valid
        actions = grid.action
        reasons = grid.mask_reason
        row: dict[str, object] = {
            "dataset": dataset,
            "recording_id": result.identity["recording_id"],
            "sync_group_id": result.identity["sync_group_id"],
            "view_id": result.identity["view_id"],
            "wearer_id": result.identity["wearer_id"],
            "total_slots": len(actions),
            "valid_action_slots": int(valid.sum()),
            "masked_slots": int((~valid).sum()),
            "no_event_count": int((actions == 0).sum()),
            "onset_count": int((actions == 1).sum()),
            "offset_count": int((actions == 2).sum()),
            "source_transition_count": result.source_transition_count,
            "represented_transition_count": int(
                ((actions == 1) | (actions == 2)).sum()
            ),
            "compound_slot_count": int((grid.resolved_transitions_in_slot > 1).sum()),
            "compound_transition_count": int(
                grid.resolved_transitions_in_slot[
                    grid.resolved_transitions_in_slot > 1
                ].sum()
            ),
            "transitions_outside_grid_count": result.transitions_outside_grid,
            "sub_delta_gap_count": len(result.gaps),
            "sub_delta_gap_cross_slot_count": sum(
                gap.cross_slot for gap in result.gaps
            ),
            "sub_delta_gap_same_slot_count": sum(
                not gap.cross_slot for gap in result.gaps
            ),
            "leading_duration_dropped_s": grid.bounds.leading_dropped_s,
            "trailing_duration_dropped_s": grid.bounds.trailing_dropped_s,
            "invalid_timeline_reason": result.invalid_reason,
        }
        for index, name in enumerate(MASK_REASONS):
            row[f"masked_{name}_count"] = int((reasons == index).sum())
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(
        "recording_id", kind="stable", ignore_index=True
    )


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    keys = ("min", "q05", "median", "q95", "max")
    quantiles = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {key: float(value) for key, value in zip(keys, quantiles, strict=True)}


SHORT_BURST_PATTERN = "SILENT-SPEAKING-SILENT"
"""The compound pattern a short speech burst produces; kept, never filtered."""

SUB_STEP_SILENCE_PATTERN = "SPEAKING-SILENT-SPEAKING"
"""The compound pattern sub-step silence bridging removes upstream."""


def _pattern_counts(compound: pd.DataFrame) -> dict[str, int]:
    """How many compound slots each remaining state pattern accounts for."""
    if compound.empty:
        return {}
    counts = compound["pattern"].value_counts()
    return {str(name): int(count) for name, count in counts.sort_index().items()}


def _transition_count_histogram(compound: pd.DataFrame) -> dict[str, int]:
    """How many compound slots hold 2, 3, ... resolved transitions."""
    if compound.empty:
        return {}
    counts = compound["transition_count"].to_numpy(dtype=int)
    values, occurrences = np.unique(counts, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, occurrences, strict=True)
    }


def _statistics(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    compound: pd.DataFrame,
    gaps: pd.DataFrame,
) -> dict[str, object]:
    """The measurement this build exists to produce."""
    action = grid["action"]
    state = grid["focal_state_before"]
    valid = grid["action_valid"].to_numpy(dtype=bool)
    no_event = action.eq(str(Action.NO_EVENT))
    onset = action.eq(str(Action.ONSET))
    offset = action.eq(str(Action.OFFSET))
    total = len(grid)
    source_transitions = int(summary["source_transition_count"].sum())
    represented = int(summary["represented_transition_count"].sum())
    outside = int(summary["transitions_outside_grid_count"].sum())
    within_grid = gaps.loc[gaps["within_grid"]] if len(gaps) else gaps
    return {
        "total_slots": total,
        "valid_action_slots": int(valid.sum()),
        "masked_slots": int((~valid).sum()),
        "valid_action_ratio": float(valid.mean()) if total else None,
        "action_counts": {
            str(Action.NO_EVENT): int(no_event.sum()),
            str(Action.ONSET): int(onset.sum()),
            str(Action.OFFSET): int(offset.sum()),
        },
        "no_event_given_state": {
            str(name): int((no_event & state.eq(str(name))).sum())
            for name in STATE_NAMES[:2]
        },
        "mask_counts_by_reason": {
            name: int(grid["mask_reason"].eq(name).sum()) for name in MASK_REASONS
        },
        "source_transition_count": source_transitions,
        "represented_transition_count": represented,
        "masked_transition_count": source_transitions - represented - outside,
        "transitions_outside_grid_count": outside,
        "compound_slot_count": len(compound),
        "compound_transition_count": int(summary["compound_transition_count"].sum()),
        "compound_transitions_per_slot": _transition_count_histogram(compound),
        "compound_patterns": _pattern_counts(compound),
        "compound_pattern_summary": {
            SHORT_BURST_PATTERN: int(compound["pattern"].eq(SHORT_BURST_PATTERN).sum())
            if len(compound)
            else 0,
            SUB_STEP_SILENCE_PATTERN: int(
                compound["pattern"].eq(SUB_STEP_SILENCE_PATTERN).sum()
            )
            if len(compound)
            else 0,
            "other": int(
                (
                    ~compound["pattern"].isin(
                        [SHORT_BURST_PATTERN, SUB_STEP_SILENCE_PATTERN]
                    )
                ).sum()
            )
            if len(compound)
            else 0,
        },
        "tau_s_onset": _quantiles(grid.loc[onset, "tau_s"].to_numpy(dtype=float)),
        "tau_s_offset": _quantiles(grid.loc[offset, "tau_s"].to_numpy(dtype=float)),
        "recording_count": len(summary),
        "recordings_with_compound_slot": int(
            (summary["compound_slot_count"] > 0).sum()
        ),
        "recordings_with_unknown": int(
            (
                summary["masked_unknown_state_count"]
                + summary["masked_unknown_within_slot_count"]
                > 0
            ).sum()
        ),
        "recordings_with_invalid_timeline": int(
            summary["invalid_timeline_reason"].notna().sum()
        ),
        "leading_duration_dropped_s": float(
            summary["leading_duration_dropped_s"].sum()
        ),
        "trailing_duration_dropped_s": float(
            summary["trailing_duration_dropped_s"].sum()
        ),
        "max_trailing_duration_dropped_s": float(
            summary["trailing_duration_dropped_s"].max()
        )
        if len(summary)
        else None,
        "sub_delta_gap_analysis": {
            "sub_delta_gap_count": len(gaps),
            "sub_delta_gap_cross_slot_count": int(gaps["cross_slot"].sum())
            if len(gaps)
            else 0,
            "sub_delta_gap_same_slot_count": int((~gaps["cross_slot"]).sum())
            if len(gaps)
            else 0,
            "sub_delta_gap_cross_slot_ratio": float(gaps["cross_slot"].mean())
            if len(gaps)
            else None,
            "sub_delta_gap_same_slot_ratio": float((~gaps["cross_slot"]).mean())
            if len(gaps)
            else None,
            "within_grid_count": len(within_grid),
            "within_grid_cross_slot_count": int(within_grid["cross_slot"].sum())
            if len(within_grid)
            else 0,
            "within_grid_same_slot_count": int((~within_grid["cross_slot"]).sum())
            if len(within_grid)
            else 0,
            "gap_duration_s": _quantiles(gaps["gap_duration_s"].to_numpy(dtype=float))
            if len(gaps)
            else None,
        },
    }


def check_invariants(grid: pd.DataFrame, step_s: float = DECISION_STEP_S) -> None:
    """Fail the build if any contract of the action grid is violated."""
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


def _dataset_tables(
    timeline: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Grid one timeline and assemble its four tables (grid, summary, compound, gaps)."""
    results = _recording_results(timeline)
    grid = _grid_table(results, dataset)
    check_invariants(grid)
    return (
        grid,
        _summary_table(results, dataset),
        _compound_table(results),
        _sub_delta_table(results),
    )


COMPARED_STATISTICS = (
    "total_slots",
    "valid_action_slots",
    "masked_slots",
    "valid_action_ratio",
    "action_counts",
    "mask_counts_by_reason",
    "source_transition_count",
    "represented_transition_count",
    "masked_transition_count",
    "compound_slot_count",
    "compound_transition_count",
    "compound_patterns",
    "compound_pattern_summary",
    "recordings_with_compound_slot",
    "sub_delta_gap_analysis",
)
"""Statistics reported for both the native and the control grid."""


def _native_comparison(
    native_timeline: pd.DataFrame, control: dict[str, object], dataset: str
) -> dict[str, object]:
    """Grid the native timeline too, so the report states what bridging changed.

    The native grid is a measurement only: it is never written as a table. Its
    slot count is identical to the control one — bridging removes interval
    boundaries, never time — so the two columns are directly comparable.
    """
    native = _statistics(*_dataset_tables(native_timeline, dataset))
    return {
        "native": {key: native[key] for key in COMPARED_STATISTICS},
        "control": {key: control[key] for key in COMPARED_STATISTICS},
        "delta": {
            key: int(control[key]) - int(native[key])  # type: ignore[arg-type]
            for key in (
                "compound_slot_count",
                "compound_transition_count",
                "source_transition_count",
                "represented_transition_count",
                "masked_transition_count",
                "valid_action_slots",
            )
        },
        "note": (
            "native is the diagnostic grid of the untransformed native timeline; "
            "control is the production grid written by this build"
        ),
    }


def _build_dataset(
    source: CheckedArtifact,
    *,
    cfg: DictConfig,
    manifest_path: Path,
    native_timeline: pd.DataFrame,
    command: str,
) -> VocalActionGridOutputs:
    paths = pipeline_paths(cfg)
    grid, summary, compound, gaps = _dataset_tables(source.table, source.dataset)

    processed_dir = paths.processed / PROCESSED_ROOT / source.dataset
    report_dir = paths.reports / REPORT_ROOT / source.dataset
    grid_path = write_table(processed_dir / GRID_TABLE, grid)
    summary_path = write_table(report_dir / SUMMARY_TABLE, summary)
    compound_path = write_table(report_dir / COMPOUND_TABLE, compound)
    gaps_path = write_table(report_dir / SUB_DELTA_TABLE, gaps)

    provenance = collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    control_report = source.report
    statistics = _statistics(grid, summary, compound, gaps)
    transform = control_report.get("transform", {})
    control_statistics = control_report.get("statistics", {})
    payload: JsonDict = {
        "build_type": "vocal_action_grid",
        "schema_version": REPORT_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "source_control_state_schema_version": CONTROL_STATE_SCHEMA_VERSION,
        "source_native_state_schema_version": NATIVE_STATE_SCHEMA_VERSION,
        "decision_step_s": DECISION_STEP_S,
        "boundary_convention": "half-open [t_k, t_k + step)",
        "action_space": list(ACTIONS),
        "mask_reasons": list(MASK_REASONS),
        "dataset": source.dataset,
        "created_at": provenance.generated_at_utc,
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
        "command": command,
        "config_checksum": config_checksum(cfg),
        "lineage_chain": list(control_report.get("lineage_chain", [])),
        "control_transform": {
            **transform,
            "bridged_gap_count": control_statistics.get("bridged_gap_count"),
            "bridged_gap_total_duration_s": control_statistics.get(
                "bridged_gap_total_duration_s"
            ),
            "bridged_gap_duration_s": control_statistics.get("bridged_gap_duration_s"),
            "bridged_gap_duration_histogram_s": control_statistics.get(
                "bridged_gap_duration_histogram_s"
            ),
            "native_transition_count": control_statistics.get(
                "native_transition_count"
            ),
            "control_transition_count": control_statistics.get(
                "control_transition_count"
            ),
        },
        "input_artifacts": {
            "control_focal_voice_state_timeline": {
                "path": str(source.table_path),
                "sha256": source.table_sha256,
            },
            "control_focal_voice_state_report": {
                "path": str(source.report_path),
                "sha256": source.report_sha256,
            },
            "native_focal_voice_state_timeline": control_report.get(
                "input_artifacts", {}
            ).get("native_focal_voice_state_timeline"),
            "dataset_manifest": artifact_reference(manifest_path),
        },
        "source_annotation_versions": control_report.get(
            "source_annotation_versions", {}
        ),
        "uv_lock_checksum": sha256_file(uv_lock) if uv_lock.exists() else None,
        "output_artifacts": {
            "grid": artifact_reference(grid_path),
            "summary": artifact_reference(summary_path),
            "compound_slots": artifact_reference(compound_path),
            "sub_delta_gap_analysis": artifact_reference(gaps_path),
        },
        "statistics": statistics,
        "native_grid_comparison": _native_comparison(
            native_timeline, statistics, source.dataset
        ),
        "action_semantics": {
            "NO_EVENT": "hold the current vocal state for this decision step",
            "ONSET": "the single resolved SILENT -> SPEAKING transition of this step",
            "OFFSET": "the single resolved SPEAKING -> SILENT transition of this step",
            "masked": "action is null and action_valid is false; never a fourth action",
            "tau_s": "event_time_s - t_k, the position of the event inside the step",
        },
        "policy": {
            "native_timestamps_rounded": False,
            "sub_delta_gaps_merged_in_this_build": False,
            "sub_delta_gaps_bridged_upstream": True,
            "short_speaking_bursts_filtered": False,
            "compound_slots_relabelled": False,
            "unknown_transitions_are_actions": False,
            "partial_trailing_slot_emitted": False,
            "acoustic_evidence_used": False,
        },
        "limitations": [
            (
                "Actions are observational logged behaviour proxies derived from native "
                "annotations, not randomized causal interventions."
            ),
            (
                "Dataset conventions differ upstream and are not harmonized here: Ego4D "
                "SPEAKING is a native vocal episode, EgoCom SPEAKING is a transcript-derived "
                "speaker interval."
            ),
            *control_report.get("limitations", []),
        ],
    }
    report = write_summary(report_dir / REPORT_FILE, payload, provenance=provenance)
    return VocalActionGridOutputs(
        source.dataset,
        grid,
        summary,
        compound,
        gaps,
        report,
        grid_path,
        report_dir / REPORT_FILE,
    )


def run_vocal_action_grid_build(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    command: str | None = None,
) -> list[VocalActionGridOutputs]:
    """Build the vocal action grid of every selected dataset."""
    manifest_path = Path(cfg.manifest.output)
    if not manifest_path.exists():
        raise MissingPrerequisiteError(
            manifest_path, produce_with="conv-wm audit manifest"
        )
    invoked = command or f"conv-wm build vocal-action-grid --dataset {dataset}"
    outputs: list[VocalActionGridOutputs] = []
    for name in selected_datasets(dataset):
        source = load_control_state_input(cfg, name)
        native_timeline = load_native_timeline_for_comparison(source)
        logger.info(
            "%s: %d control intervals (%d native)",
            name,
            len(source.table),
            len(native_timeline),
        )
        outputs.append(
            _build_dataset(
                source,
                cfg=cfg,
                manifest_path=manifest_path,
                native_timeline=native_timeline,
                command=invoked,
            )
        )
    return outputs


def format_outputs(outputs: list[VocalActionGridOutputs]) -> str:
    """One line per dataset with the headline counts and the report path."""
    lines = []
    for output in outputs:
        statistics = output.report["statistics"]
        counts = statistics["action_counts"]
        comparison = output.report["native_grid_comparison"]
        lines.append(
            f"{output.dataset}: {statistics['total_slots']} slots, "
            f"valid {statistics['valid_action_ratio']:.4f}, "
            f"NO_EVENT {counts['NO_EVENT']} ONSET {counts['ONSET']} "
            f"OFFSET {counts['OFFSET']}, compound slots "
            f"{comparison['native']['compound_slot_count']} -> "
            f"{statistics['compound_slot_count']}, transitions "
            f"{comparison['native']['source_transition_count']} -> "
            f"{statistics['source_transition_count']} "
            f"-> {output.report_path}"
        )
    return "\n".join(lines)


__all__ = [
    "VocalActionGridOutputs",
    "check_invariants",
    "compound_pattern",
    "format_outputs",
    "load_control_state_input",
    "load_native_timeline_for_comparison",
    "run_vocal_action_grid_build",
    "supported_datasets",
]
