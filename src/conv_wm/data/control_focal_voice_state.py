"""Build the control focal voice-state layer between the native state and the grid.

One artifact per dataset: the native timeline requantized to the controller's
resolution by a single rule — sub-step silence bridging — plus the provenance
of every bridged gap and a report measuring the layer before and after.

```text
native annotation -> native focal state -> control focal state -> action grid
```

The native Parquet files are inputs only: this build never rewrites them, and
it refuses a native artifact whose checksum no longer matches its own report.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data.native_focal_voice_state import (
    PROCESSED_ROOT as NATIVE_PROCESSED_ROOT,
)
from conv_wm.data.native_focal_voice_state import (
    REPORT_FILE as NATIVE_REPORT_FILE,
)
from conv_wm.data.native_focal_voice_state import (
    REPORT_ROOT as NATIVE_REPORT_ROOT,
)
from conv_wm.data.native_focal_voice_state import (
    TIMELINE_TABLE as NATIVE_TIMELINE_TABLE,
)
from conv_wm.data.native_focal_voice_state import (
    selected_datasets,
    supported_datasets,
)
from conv_wm.data.pipeline_inputs import (
    CheckedArtifact,
    artifact_reference,
    load_checked_artifact,
    sha256_file,
)
from conv_wm.data.vocal.action_grid import BOUNDARY_TOLERANCE_S, DECISION_STEP_S
from conv_wm.data.vocal.control_state import (
    CONTROL_STATE_SCHEMA_VERSION,
    BridgedGap,
    ControlStateInterval,
    TransformKind,
    build_control_timeline,
    resolved_transition_count,
)
from conv_wm.data.vocal.native_state import (
    NATIVE_STATE_SCHEMA_VERSION,
    NativeStateInterval,
    SourceKind,
    VoiceState,
)
from conv_wm.provenance import collect_provenance
from conv_wm.reports import JsonDict, write_summary, write_table

REPORT_SCHEMA_VERSION = 1
PROCESSED_ROOT = Path("control_focal_voice_state")
REPORT_ROOT = Path("control_focal_voice_state")
TIMELINE_TABLE = "control_focal_voice_intervals.parquet"
SUMMARY_TABLE = "summary.parquet"
BRIDGED_GAPS_TABLE = "bridged_gaps.parquet"
REPORT_FILE = "report.json"

NATIVE_BUILD_COMMAND = "conv-wm build native-focal-voice-state"
LINEAGE_CHAIN = (
    "native annotation",
    "native focal voice state",
    "control focal voice state",
    "vocal action grid",
)


@dataclass(frozen=True)
class ControlFocalVoiceStateOutputs:
    """Tables and report written for one dataset."""

    dataset: str
    timeline: pd.DataFrame
    summary: pd.DataFrame
    bridged_gaps: pd.DataFrame
    report: JsonDict
    timeline_path: Path
    summary_path: Path
    bridged_gaps_path: Path
    report_path: Path


def config_checksum(cfg: DictConfig) -> str:
    """SHA-256 of the fully resolved configuration this build ran with."""
    return hashlib.sha256(
        OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    ).hexdigest()


def load_native_state_input(cfg: DictConfig, dataset: str) -> CheckedArtifact:
    """Read one dataset's native timeline, refusing an input its report disowns."""
    paths = pipeline_paths(cfg)
    return load_checked_artifact(
        dataset=dataset,
        table_path=(
            paths.processed / NATIVE_PROCESSED_ROOT / dataset / NATIVE_TIMELINE_TABLE
        ),
        report_path=paths.reports / NATIVE_REPORT_ROOT / dataset / NATIVE_REPORT_FILE,
        schema_key="native_state_schema_version",
        expected_version=NATIVE_STATE_SCHEMA_VERSION,
        artifact_key="timeline",
        produce_with=f"{NATIVE_BUILD_COMMAND} --dataset {dataset}",
    )


def native_intervals_by_recording(
    timeline: pd.DataFrame,
) -> list[tuple[str, list[NativeStateInterval]]]:
    """Native rows grouped per recording, each in canonical timeline order.

    Columns are read as arrays rather than row by row: the native timelines
    hold hundreds of thousands of intervals and this build must stay seconds.
    """
    ordered = timeline.sort_values(
        ["recording_id", "canonical_start_s"], kind="stable", ignore_index=True
    )
    columns = {
        name: ordered[name].to_numpy()
        for name in (
            "dataset",
            "recording_id",
            "sync_group_id",
            "view_id",
            "wearer_id",
            "canonical_start_s",
            "canonical_end_s",
            "voice_state",
            "source_kind",
            "source_annotation_id",
            "annotation_schema_version",
            "native_state_schema_version",
        )
    }
    rows = [
        NativeStateInterval(
            dataset=str(dataset),
            recording_id=str(recording_id),
            sync_group_id=None if pd.isna(sync_group_id) else str(sync_group_id),
            view_id=str(view_id),
            wearer_id=str(wearer_id),
            canonical_start_s=float(start_s),
            canonical_end_s=float(end_s),
            voice_state=VoiceState(str(voice_state)),
            source_kind=SourceKind(str(source_kind)),
            source_annotation_id=None if pd.isna(annotation_id) else str(annotation_id),
            annotation_schema_version=str(annotation_version),
            native_state_schema_version=int(state_version),
        )
        for (
            dataset,
            recording_id,
            sync_group_id,
            view_id,
            wearer_id,
            start_s,
            end_s,
            voice_state,
            source_kind,
            annotation_id,
            annotation_version,
            state_version,
        ) in zip(*columns.values(), strict=True)
    ]
    identifiers = columns["recording_id"]
    boundaries = np.flatnonzero(identifiers[1:] != identifiers[:-1]) + 1
    return [
        (str(identifiers[first]), rows[first:last])
        for first, last in zip(
            np.concatenate([[0], boundaries]),
            np.concatenate([boundaries, [len(rows)]]),
            strict=True,
        )
    ]


@dataclass
class _RecordingResult:
    """One recording's native input, its control timeline and its bridged gaps."""

    recording_id: str
    native: list[NativeStateInterval]
    control: list[ControlStateInterval]
    gaps: list[BridgedGap]


def _transform(
    timeline: pd.DataFrame, *, step_s: float = DECISION_STEP_S
) -> list[_RecordingResult]:
    results: list[_RecordingResult] = []
    for recording_id, native in native_intervals_by_recording(timeline):
        control, gaps = build_control_timeline(native, step_s=step_s)
        results.append(_RecordingResult(recording_id, native, control, gaps))
    return results


def _durations(
    rows: list[NativeStateInterval] | list[ControlStateInterval],
) -> dict[str, float]:
    totals = {str(state): 0.0 for state in VoiceState}
    for row in rows:
        totals[str(row.voice_state)] += row.duration_s
    return totals


def _summary_table(results: list[_RecordingResult], dataset: str) -> pd.DataFrame:
    """One row per recording: intervals and transitions before and after bridging."""
    rows: list[dict[str, object]] = []
    for result in results:
        native_states = [str(row.voice_state) for row in result.native]
        control_states = [str(row.voice_state) for row in result.control]
        native_durations = _durations(result.native)
        control_durations = _durations(result.control)
        first = result.control[0]
        rows.append(
            {
                "dataset": dataset,
                "recording_id": result.recording_id,
                "sync_group_id": first.sync_group_id,
                "view_id": first.view_id,
                "wearer_id": first.wearer_id,
                "canonical_start_s": first.canonical_start_s,
                "canonical_end_s": result.control[-1].canonical_end_s,
                "native_interval_count": len(result.native),
                "control_interval_count": len(result.control),
                "native_transition_count": resolved_transition_count(native_states),
                "control_transition_count": resolved_transition_count(control_states),
                "bridged_gap_count": len(result.gaps),
                "bridged_gap_total_duration_s": sum(
                    gap.gap_duration_s for gap in result.gaps
                ),
                "bridged_interval_count": sum(
                    row.transform_kind is not None for row in result.control
                ),
                "native_speaking_duration_s": native_durations[
                    str(VoiceState.SPEAKING)
                ],
                "control_speaking_duration_s": control_durations[
                    str(VoiceState.SPEAKING)
                ],
                "native_silent_duration_s": native_durations[str(VoiceState.SILENT)],
                "control_silent_duration_s": control_durations[str(VoiceState.SILENT)],
                "unknown_duration_s": control_durations[str(VoiceState.UNKNOWN)],
                "window_duration_s": result.control[-1].canonical_end_s
                - first.canonical_start_s,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        "recording_id", kind="stable", ignore_index=True
    )


def _timeline_table(results: list[_RecordingResult]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [row.to_row() for result in results for row in result.control],
        columns=list(ControlStateInterval.__dataclass_fields__),
    )


def _bridged_gaps_table(results: list[_RecordingResult]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [gap.to_row() for result in results for gap in result.gaps],
        columns=list(BridgedGap.__dataclass_fields__),
    )


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    keys = ("min", "q05", "median", "q95", "max")
    quantiles = np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {key: float(value) for key, value in zip(keys, quantiles, strict=True)}


def _duration_histogram(
    values: np.ndarray, *, step_s: float = DECISION_STEP_S, bins: int = 10
) -> dict[str, int]:
    """Bridged-gap durations bucketed in Δ/``bins`` slices of ``[0, Δ)``."""
    if not len(values):
        return {}
    width = step_s / bins
    indices = np.clip(np.floor(values / width).astype(int), 0, bins - 1)
    counts = np.bincount(indices, minlength=bins)
    return {
        f"[{index * width:.3f},{(index + 1) * width:.3f})": int(count)
        for index, count in enumerate(counts)
        if count
    }


def _statistics(
    results: list[_RecordingResult],
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    step_s: float,
) -> dict[str, object]:
    """The before/after measurement this layer exists to produce."""
    durations = (
        gaps["gap_duration_s"].to_numpy(dtype=float) if len(gaps) else np.empty(0)
    )
    native_transitions = int(summary["native_transition_count"].sum())
    control_transitions = int(summary["control_transition_count"].sum())
    native_window = float(summary["window_duration_s"].sum())
    return {
        "recording_count": len(summary),
        "recordings_with_bridged_gap": int((summary["bridged_gap_count"] > 0).sum()),
        "native_interval_count": int(summary["native_interval_count"].sum()),
        "control_interval_count": int(summary["control_interval_count"].sum()),
        "native_transition_count": native_transitions,
        "control_transition_count": control_transitions,
        "removed_transition_count": native_transitions - control_transitions,
        "bridged_gap_count": len(gaps),
        "bridged_gap_total_duration_s": float(durations.sum()),
        "bridged_gap_duration_s": _quantiles(durations),
        "bridged_gap_duration_histogram_s": _duration_histogram(
            durations, step_s=step_s
        ),
        "bridged_interval_count": int(summary["bridged_interval_count"].sum()),
        "max_bridged_gaps_in_one_interval": max(
            (row.bridged_gap_count for result in results for row in result.control),
            default=0,
        ),
        "native_speaking_duration_s": float(
            summary["native_speaking_duration_s"].sum()
        ),
        "control_speaking_duration_s": float(
            summary["control_speaking_duration_s"].sum()
        ),
        "native_silent_duration_s": float(summary["native_silent_duration_s"].sum()),
        "control_silent_duration_s": float(summary["control_silent_duration_s"].sum()),
        "unknown_duration_s": float(summary["unknown_duration_s"].sum()),
        "window_duration_s": native_window,
    }


def check_invariants(
    results: list[_RecordingResult], *, tolerance_s: float = BOUNDARY_TOLERANCE_S
) -> None:
    """Fail the build rather than write a control timeline that is not a requantization.

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


def _build_dataset(
    source: CheckedArtifact,
    *,
    cfg: DictConfig,
    command: str,
    step_s: float = DECISION_STEP_S,
) -> ControlFocalVoiceStateOutputs:
    paths = pipeline_paths(cfg)
    results = _transform(source.table, step_s=step_s)
    check_invariants(results)
    timeline = _timeline_table(results)
    summary = _summary_table(results, source.dataset)
    gaps = _bridged_gaps_table(results)

    processed_dir = paths.processed / PROCESSED_ROOT / source.dataset
    report_dir = paths.reports / REPORT_ROOT / source.dataset
    timeline_path = write_table(processed_dir / TIMELINE_TABLE, timeline)
    summary_path = write_table(report_dir / SUMMARY_TABLE, summary)
    gaps_path = write_table(report_dir / BRIDGED_GAPS_TABLE, gaps)

    provenance = collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    native_report = source.report
    payload: JsonDict = {
        "build_type": "control_focal_voice_state",
        "schema_version": REPORT_SCHEMA_VERSION,
        "control_state_schema_version": CONTROL_STATE_SCHEMA_VERSION,
        "native_state_schema_version": NATIVE_STATE_SCHEMA_VERSION,
        "decision_step_s": step_s,
        "dataset": source.dataset,
        "created_at": provenance.generated_at_utc,
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
        "command": command,
        "config_checksum": config_checksum(cfg),
        "lineage_chain": list(LINEAGE_CHAIN),
        "transform": {
            "transform_kind": str(TransformKind.SUB_STEP_SILENCE_BRIDGE),
            "pattern": "SPEAKING -> SILENT (g < decision_step_s) -> SPEAKING",
            "becomes": "SPEAKING continuous",
            "condition": "gap_duration_s < decision_step_s",
            "condition_is_strict": True,
            "comparison_tolerance_s": BOUNDARY_TOLERANCE_S,
            "decision_step_s": step_s,
            "rationale": (
                "quantization to the controller's temporal resolution, not a "
                "correction of the annotation"
            ),
            "speech_bursts_filtered": False,
            "native_timestamps_modified": False,
            "native_artifact_modified": False,
        },
        "input_artifacts": {
            "native_focal_voice_state_timeline": {
                "path": str(source.table_path),
                "sha256": source.table_sha256,
            },
            "native_focal_voice_state_report": {
                "path": str(source.report_path),
                "sha256": source.report_sha256,
            },
        },
        "source_annotation_versions": {
            "annotation_schema_version": native_report.get(
                "source_dataset_version", {}
            ).get("annotation_schema_version"),
            "cleaning_rule_version": native_report.get(
                "source_dataset_version", {}
            ).get("cleaning_rule_version"),
            "native_annotations": native_report.get("input_artifacts", {}).get(
                "native_annotations"
            ),
            "native_state_git_commit": native_report.get("git_commit"),
            "native_state_created_at": native_report.get("created_at"),
        },
        "uv_lock_checksum": sha256_file(uv_lock) if uv_lock.exists() else None,
        "output_artifacts": {
            "timeline": artifact_reference(timeline_path),
            "summary": artifact_reference(summary_path),
            "bridged_gaps": artifact_reference(gaps_path),
        },
        "statistics": _statistics(results, summary, gaps, step_s=step_s),
        "state_semantics": {
            "SPEAKING": (
                "native SPEAKING, possibly extended over sub-step silences the "
                "controller cannot represent"
            ),
            "SILENT": "native SILENT lasting at least one decision step",
            "UNKNOWN": "native UNKNOWN, never bridged across",
        },
        "limitations": [
            (
                "The control timeline is a requantization of the native timeline at "
                "decision_step_s: silences shorter than one control step are absorbed "
                "into the surrounding speech and are no longer recoverable from this "
                "layer alone (bridged_gaps.parquet keeps their provenance)."
            ),
            (
                "Short SPEAKING bursts are deliberately kept: the symmetric rule is "
                "not applied, so bursts can still put two transitions in one slot."
            ),
            *native_report.get("limitations", []),
        ],
    }
    report = write_summary(report_dir / REPORT_FILE, payload, provenance=provenance)
    return ControlFocalVoiceStateOutputs(
        source.dataset,
        timeline,
        summary,
        gaps,
        report,
        timeline_path,
        summary_path,
        gaps_path,
        report_dir / REPORT_FILE,
    )


def run_control_focal_voice_state_build(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    command: str | None = None,
    stream: TextIO | None = sys.stderr,
) -> list[ControlFocalVoiceStateOutputs]:
    """Build the control focal voice-state artifact of every selected dataset."""
    invoked = command or f"conv-wm build control-focal-voice-state --dataset {dataset}"
    outputs: list[ControlFocalVoiceStateOutputs] = []
    for name in selected_datasets(dataset):
        source = load_native_state_input(cfg, name)
        if stream is not None:
            print(
                f"control focal voice state {name}: {len(source.table)} native intervals",
                file=stream,
            )
        outputs.append(_build_dataset(source, cfg=cfg, command=invoked))
    return outputs


def format_outputs(outputs: list[ControlFocalVoiceStateOutputs]) -> str:
    """One line per dataset with the headline before/after counts and the report path."""
    lines = []
    for output in outputs:
        statistics = output.report["statistics"]
        lines.append(
            f"{output.dataset}: intervals "
            f"{statistics['native_interval_count']} -> "
            f"{statistics['control_interval_count']}, transitions "
            f"{statistics['native_transition_count']} -> "
            f"{statistics['control_transition_count']}, bridged "
            f"{statistics['bridged_gap_count']} gaps "
            f"({statistics['bridged_gap_total_duration_s']:.2f} s) "
            f"-> {output.report_path}"
        )
    return "\n".join(lines)


__all__ = [
    "ControlFocalVoiceStateOutputs",
    "check_invariants",
    "config_checksum",
    "format_outputs",
    "load_native_state_input",
    "run_control_focal_voice_state_build",
    "selected_datasets",
    "supported_datasets",
]
