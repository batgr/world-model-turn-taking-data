"""Build the v0 native focal voice-state layer and its lineage report.

One artifact per dataset: the exhaustive ``SPEAKING`` / ``SILENT`` /
``UNKNOWN`` timeline of every (view, wearer) pair, a per-recording summary and
a report answering which native annotations, code revision and schema produced
the timeline. No acoustic model is involved; the build only reads cleaned
annotation tables and the media metadata audit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.vocal.native_source import NativeFocalVoiceSource
from conv_wm.data.vocal.native_state import (
    NATIVE_STATE_SCHEMA_VERSION,
    NativeStateInterval,
    VoiceState,
    build_native_timeline,
    state_durations,
)
from conv_wm.provenance import collect_provenance
from conv_wm.reports import JsonDict, write_summary, write_table

REPORT_SCHEMA_VERSION = 1
PROCESSED_ROOT = Path("native_focal_voice_state")
REPORT_ROOT = Path("native_focal_voice_state")
TIMELINE_TABLE = "focal_voice_intervals.parquet"
SUMMARY_TABLE = "summary.parquet"
REPORT_FILE = "report.json"
COVERAGE_AUDIT_REPORT = Path("vocal_annotation_coverage") / "report.json"
SHORT_SILENCE_S = 0.1
"""Silent gaps shorter than the future 100 ms decision step are counted."""


@dataclass(frozen=True)
class NativeFocalVoiceStateOutputs:
    """Tables and report written for one dataset."""

    dataset: str
    timeline: pd.DataFrame
    summary: pd.DataFrame
    report: JsonDict
    timeline_path: Path
    summary_path: Path
    report_path: Path


def supported_datasets() -> tuple[str, ...]:
    """Registered datasets that provide a native focal voice adapter."""
    return tuple(spec.name for spec in datasets.iter_specs() if spec.native_focal_voice)


def selected_datasets(dataset: str) -> tuple[str, ...]:
    """Expand the CLI selector (``all`` or one supported dataset)."""
    supported = supported_datasets()
    if dataset == "all":
        return supported
    if dataset not in supported:
        raise ValueError(
            f"dataset must be one of {[*supported, 'all']}, got {dataset!r}"
        )
    return (dataset,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with=command)
    return path


def _summary_row(
    recording_id: str, rows: list[NativeStateInterval], source: NativeFocalVoiceSource
) -> dict[str, object]:
    durations = state_durations(rows)
    first = rows[0]
    total = first.canonical_start_s
    valid = sum(row.duration_s for row in rows if row.voice_state != VoiceState.UNKNOWN)
    window = rows[-1].canonical_end_s - total
    return {
        "dataset": source.dataset,
        "recording_id": recording_id,
        "sync_group_id": first.sync_group_id,
        "view_id": first.view_id,
        "wearer_id": first.wearer_id,
        "canonical_start_s": total,
        "canonical_end_s": rows[-1].canonical_end_s,
        "window_duration_s": window,
        "valid_duration_s": valid,
        "speaking_duration_s": durations[str(VoiceState.SPEAKING)],
        "silent_duration_s": durations[str(VoiceState.SILENT)],
        "unknown_duration_s": durations[str(VoiceState.UNKNOWN)],
        "speaking_ratio": durations[str(VoiceState.SPEAKING)] / valid
        if valid
        else None,
        "unknown_ratio": durations[str(VoiceState.UNKNOWN)] / window,
        "speaking_interval_count": sum(
            row.voice_state == VoiceState.SPEAKING for row in rows
        ),
        "silent_interval_count": sum(
            row.voice_state == VoiceState.SILENT for row in rows
        ),
        "short_silent_interval_count": sum(
            row.voice_state == VoiceState.SILENT and row.duration_s < SHORT_SILENCE_S
            for row in rows
        ),
        "unknown_interval_count": sum(
            row.voice_state == VoiceState.UNKNOWN for row in rows
        ),
        "annotation_schema_version": first.annotation_schema_version,
    }


def _dataset_statistics(
    summary: pd.DataFrame, timeline: pd.DataFrame
) -> dict[str, object]:
    valid = float(summary["valid_duration_s"].sum())
    window = float(summary["window_duration_s"].sum())
    speaking = float(summary["speaking_duration_s"].sum())
    silent = float(summary["silent_duration_s"].sum())
    unknown = float(summary["unknown_duration_s"].sum())
    return {
        "recording_count": len(summary),
        "recordings_with_unknown_time": int((summary["unknown_duration_s"] > 0).sum()),
        "window_duration_s": window,
        "valid_duration_s": valid,
        "speaking_duration_s": speaking,
        "silent_duration_s": silent,
        "unknown_duration_s": unknown,
        "speaking_ratio": speaking / valid if valid else None,
        "unknown_ratio": unknown / window if window else None,
        "native_speaking_interval_count": int(summary["speaking_interval_count"].sum()),
        "silent_interval_count": int(summary["silent_interval_count"].sum()),
        "short_silent_interval_count": int(
            summary["short_silent_interval_count"].sum()
        ),
        "interval_count": len(timeline),
        "unknown_source_ids": _unknown_source_counts(timeline),
    }


def _unknown_source_counts(timeline: pd.DataFrame) -> dict[str, int]:
    """How many UNKNOWN intervals each declared reason (row ids stripped) explains."""
    counts: dict[str, int] = {}
    unknown = timeline.loc[timeline["voice_state"].eq(str(VoiceState.UNKNOWN))]
    for value in unknown["source_annotation_id"].dropna():
        for source_id in str(value).split("|"):
            reason = source_id.split("#", 1)[0]
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _coverage_audit_reference(
    reports_root: Path, dataset: str
) -> dict[str, object] | None:
    """Point at the vocal annotation coverage audit without recomputing it."""
    path = reports_root / COVERAGE_AUDIT_REPORT
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    by_dataset = report.get("coverage_statistics_by_dataset", {})
    statistics = by_dataset.get(dataset, {}) if isinstance(by_dataset, dict) else {}
    return {
        **_artifact(path),
        "created_at": report.get("created_at"),
        "git_commit": report.get("git_commit"),
        "focal_annotation_coverage_ratio": statistics.get(
            "focal_annotation_coverage_ratio"
        ),
        "focal_measurable_recording_count": statistics.get(
            "focal_measurable_recording_count"
        ),
        "focal_specific_status": statistics.get("focal_specific_status"),
    }


def _build_dataset(
    source: NativeFocalVoiceSource,
    *,
    cfg: DictConfig,
    manifest_path: Path,
    media_path: Path,
    command: str,
) -> NativeFocalVoiceStateOutputs:
    paths = pipeline_paths(cfg)
    timeline_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for recording in sorted(source.recordings, key=lambda item: item.recording_id):
        rows = build_native_timeline(recording)
        timeline_rows.extend(row.to_row() for row in rows)
        summary_rows.append(_summary_row(recording.recording_id, rows, source))
    timeline = pd.DataFrame.from_records(
        timeline_rows, columns=list(NativeStateInterval.__dataclass_fields__)
    )
    summary = pd.DataFrame.from_records(summary_rows)

    processed_dir = paths.processed / PROCESSED_ROOT / source.dataset
    report_dir = paths.reports / REPORT_ROOT / source.dataset
    timeline_path = write_table(processed_dir / TIMELINE_TABLE, timeline)
    summary_path = write_table(report_dir / SUMMARY_TABLE, summary)

    provenance = collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    payload: JsonDict = {
        "build_type": "native_focal_voice_state",
        "schema_version": REPORT_SCHEMA_VERSION,
        "native_state_schema_version": NATIVE_STATE_SCHEMA_VERSION,
        "dataset": source.dataset,
        "created_at": provenance.generated_at_utc,
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
        "command": command,
        "source_dataset_version": {
            "raw_annotations_root": str(cfg.datasets[source.dataset].raw),
            "annotation_schema_version": source.annotation_schema_version,
            "cleaning_rule_version": source.cleaning_rule_version,
        },
        "input_artifacts": {
            "dataset_manifest": _artifact(manifest_path),
            "media_metadata": _artifact(media_path),
            "native_annotations": [_artifact(path) for path in source.annotation_paths],
        },
        "uv_lock_checksum": _sha256(uv_lock) if uv_lock.exists() else None,
        "output_artifacts": {
            "timeline": _artifact(timeline_path),
            "summary": _artifact(summary_path),
        },
        "statistics": {
            **_dataset_statistics(summary, timeline),
            "adapter": source.statistics,
        },
        "state_semantics": {
            "SPEAKING": "inside a native annotation interval of the wearer",
            "SILENT": "valid time outside every native wearer annotation",
            "UNKNOWN": "invalid clip, explicitly missing annotation, or media not covering the window",
        },
        "limitations": list(source.limitations),
        "vocal_annotation_coverage_audit": _coverage_audit_reference(
            paths.reports, source.dataset
        ),
    }
    report = write_summary(report_dir / REPORT_FILE, payload, provenance=provenance)
    return NativeFocalVoiceStateOutputs(
        source.dataset,
        timeline,
        summary,
        report,
        timeline_path,
        summary_path,
        report_dir / REPORT_FILE,
    )


def run_native_focal_voice_state_build(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    command: str | None = None,
    stream: TextIO | None = sys.stderr,
) -> list[NativeFocalVoiceStateOutputs]:
    """Build the native focal voice-state artifact of every selected dataset."""
    paths = pipeline_paths(cfg)
    manifest_path = _require(Path(cfg.manifest.output), "conv-wm audit manifest")
    media_path = _require(paths.reports / MEDIA_METADATA_TABLE, "conv-wm audit media")
    media = pd.read_parquet(media_path)
    invoked = command or f"conv-wm build native-focal-voice-state --dataset {dataset}"
    outputs: list[NativeFocalVoiceStateOutputs] = []
    for name in selected_datasets(dataset):
        loader = datasets.get(name).native_focal_voice
        assert loader is not None
        source = loader(cfg, media)
        if stream is not None:
            print(
                f"native focal voice state {name}: {len(source.recordings)} recordings",
                file=stream,
            )
        outputs.append(
            _build_dataset(
                source,
                cfg=cfg,
                manifest_path=manifest_path,
                media_path=media_path,
                command=invoked,
            )
        )
    return outputs


def format_outputs(outputs: list[NativeFocalVoiceStateOutputs]) -> str:
    """One line per dataset with the headline durations and the report path."""
    lines = []
    for output in outputs:
        statistics = output.report["statistics"]
        lines.append(
            f"{output.dataset}: {statistics['recording_count']} recordings, "
            f"speaking {statistics['speaking_duration_s']:.0f} s, "
            f"silent {statistics['silent_duration_s']:.0f} s, "
            f"unknown {statistics['unknown_duration_s']:.0f} s -> {output.report_path}"
        )
    return "\n".join(lines)


__all__ = [
    "NativeFocalVoiceStateOutputs",
    "format_outputs",
    "run_native_focal_voice_state_build",
    "selected_datasets",
    "supported_datasets",
]
