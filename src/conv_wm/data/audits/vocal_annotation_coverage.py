"""Population audit of acoustic and focal-speaker annotation coverage.

The audit detects acoustic voice activity, attempts wearer attribution only with
dataset-supported evidence, and measures annotation overlap. It never rewrites
annotations or creates vocal actions.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.vocal.audio import DecodedAudio, decode_audio
from conv_wm.data.vocal.config import (
    DEFAULT_COVERAGE_CONFIG_PATH,
    VocalCoverageConfig,
    load_coverage_config,
)
from conv_wm.data.vocal.coverage import RecordingAssessment, assess_recording
from conv_wm.data.vocal.detector import VoiceActivityDetector, build_detector
from conv_wm.data.vocal.identity import IdentityPipeline
from conv_wm.data.vocal.records import (
    CoverageStatus,
    DetectedSegment,
    FocalRecording,
    FocalVoiceActivityStatus,
    RecordingCoverageSummary,
    SegmentAssessment,
)
from conv_wm.data.vocal.sources import DatasetRecordingSet, load_recording_sets
from conv_wm.provenance import ReportProvenance, collect_provenance
from conv_wm.reports import JsonDict, write_summary, write_table

OUTPUT_DIR = Path("vocal_annotation_coverage")
SUMMARY_TABLE = OUTPUT_DIR / "summary.parquet"
UNCOVERED_SEGMENTS_TABLE = OUTPUT_DIR / "uncovered_segments.parquet"
REPORT_FILE = OUTPUT_DIR / "report.json"
REPORT_SCHEMA_VERSION = 1
SUPPORTED_DATASETS = ("ego4d", "egocom")

Decoder = Callable[..., DecodedAudio]


@dataclass(frozen=True)
class VocalCoverageOutputs:
    """Tables and report written by one audit run."""

    summary: pd.DataFrame
    uncovered_segments: pd.DataFrame
    report: JsonDict
    output_dir: Path

    @property
    def report_path(self) -> Path:
        """Location of the machine-readable report."""
        return self.output_dir / REPORT_FILE.name


def selected_datasets(dataset: str) -> tuple[str, ...]:
    """Expand the CLI dataset selector and reject unsupported values."""
    if dataset == "all":
        return SUPPORTED_DATASETS
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {[*SUPPORTED_DATASETS, 'all']}, got {dataset!r}"
        )
    return (dataset,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_artifacts() -> list[Path]:
    """Source files whose content determines this audit's scientific output."""
    return sorted(
        [
            PROJECT_ROOT / "pyproject.toml",
            PROJECT_ROOT / "src" / "conv_wm" / "cli.py",
            PROJECT_ROOT
            / "src"
            / "conv_wm"
            / "data"
            / "audits"
            / "vocal_annotation_coverage.py",
            *(PROJECT_ROOT / "src" / "conv_wm" / "data" / "vocal").glob("*.py"),
        ]
    )


def _combined_checksum(paths: list[Path]) -> str:
    """Hash paths and contents in stable order, including uncommitted source files."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with=command)
    return path


def _progress(
    items: Iterable[RecordingAssessment],
    *,
    total: int,
    stream: TextIO | None,
) -> list[RecordingAssessment]:
    results: list[RecordingAssessment] = []
    for index, result in enumerate(items, start=1):
        if stream is not None:
            print(
                f"\rvocal annotation coverage {index}/{total}",
                end="",
                file=stream,
                flush=True,
            )
        results.append(result)
    if stream is not None and total:
        print(file=stream)
    return results


def _failed(
    recording: FocalRecording, error: Exception, detector: VoiceActivityDetector
) -> RecordingAssessment:
    return RecordingAssessment(
        [],
        RecordingCoverageSummary.failed(
            recording,
            f"{type(error).__name__}: {error}",
            detection_method=detector.name,
        ),
    )


def _assess_decoded(
    recording: FocalRecording,
    decoded: DecodedAudio,
    other_audio: dict[str, DecodedAudio],
    *,
    detector: VoiceActivityDetector,
    identity: IdentityPipeline,
    config: VocalCoverageConfig,
    detected_segments: list[DetectedSegment] | None = None,
) -> RecordingAssessment:
    segments = (
        detected_segments if detected_segments is not None else detector.detect(decoded)
    )
    context = identity.prepare(recording, decoded, other_audio)
    return assess_recording(
        recording,
        segments,
        attribute=lambda segment: identity.attribute(context, segment),
        config=config.coverage,
        detection_method=detector.name,
        identity_method=identity.name,
        identity_confidence=context.confidence,
        focal_activity_measurable=context.can_attribute_unannotated_focal,
        identity_diagnostics=context.notes,
        decoded_duration_error_s=decoded.duration_error_s,
    )


def _audit_individual_recordings(
    source: DatasetRecordingSet,
    *,
    detector: VoiceActivityDetector,
    decoder: Decoder,
    config: VocalCoverageConfig,
) -> Iterable[RecordingAssessment]:
    identity = IdentityPipeline(source.identity_methods, config.identity)
    for recording in source.recordings:
        try:
            decoded = decoder(
                recording.audio, sample_rate_hz=config.detector.sample_rate_hz
            )
            yield _assess_decoded(
                recording,
                decoded,
                {},
                detector=detector,
                identity=identity,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - one file must not abort the population
            yield _failed(recording, exc, detector)


def _audit_synchronized_groups(
    source: DatasetRecordingSet,
    *,
    detector: VoiceActivityDetector,
    decoder: Decoder,
    config: VocalCoverageConfig,
) -> Iterable[RecordingAssessment]:
    """Decode/detect each EgoCom view once, then assess every wearer in its group."""
    identity = IdentityPipeline(source.identity_methods, config.identity)
    groups: dict[str, list[FocalRecording]] = {}
    for recording in source.recordings:
        groups.setdefault(recording.sync_group_id or recording.recording_id, []).append(
            recording
        )
    for group_id in sorted(groups):
        recordings = sorted(groups[group_id], key=lambda item: item.view_id)
        decoded: dict[str, DecodedAudio] = {}
        detected: dict[str, list[DetectedSegment]] = {}
        errors: dict[str, Exception] = {}
        for recording in recordings:
            try:
                audio = decoder(
                    recording.audio, sample_rate_hz=config.detector.sample_rate_hz
                )
                decoded[recording.view_id] = audio
                detected[recording.view_id] = detector.detect(audio)
            except Exception as exc:  # noqa: BLE001 - explicit population failure row
                errors[recording.view_id] = exc
        for recording in recordings:
            if recording.view_id in errors:
                yield _failed(recording, errors[recording.view_id], detector)
                continue
            other_audio = {
                view_id: audio
                for view_id, audio in decoded.items()
                if view_id != recording.view_id
            }
            try:
                yield _assess_decoded(
                    recording,
                    decoded[recording.view_id],
                    other_audio,
                    detector=detector,
                    identity=identity,
                    config=config,
                    detected_segments=detected[recording.view_id],
                )
            except Exception as exc:  # noqa: BLE001 - explicit population failure row
                yield _failed(recording, exc, detector)


def audit_recording_sets(
    sources: list[DatasetRecordingSet],
    *,
    detector: VoiceActivityDetector,
    decoder: Decoder,
    config: VocalCoverageConfig,
    stream: TextIO | None = sys.stderr,
) -> list[RecordingAssessment]:
    """Run selected recording sets in stable order with explicit failure rows."""
    total = sum(len(source.recordings) for source in sources)

    def results() -> Iterable[RecordingAssessment]:
        for source in sorted(sources, key=lambda item: item.dataset):
            if source.dataset == "egocom":
                yield from _audit_synchronized_groups(
                    source, detector=detector, decoder=decoder, config=config
                )
            else:
                yield from _audit_individual_recordings(
                    source, detector=detector, decoder=decoder, config=config
                )

    return _progress(results(), total=total, stream=stream)


def _finite_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric[np.isfinite(numeric)].sum())


def _quantiles(values: pd.Series) -> dict[str, float] | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return {
        name: float(numeric.quantile(q))
        for name, q in (
            ("min", 0.0),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def coverage_statistics(
    summaries: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    short_segment_max_duration_s: float,
) -> dict[str, object]:
    """Aggregate a population without applying a PASS/WARN/FAIL threshold."""
    successful = summaries.loc[summaries["audit_ok"]]
    measurable = successful.loc[
        successful["focal_voice_activity_status"].eq(
            str(FocalVoiceActivityStatus.MEASURABLE)
        )
    ]
    acoustic_duration = _finite_sum(successful["detected_acoustic_voice_duration_s"])
    unresolved_duration = _finite_sum(successful["unresolved_identity_duration_s"])
    focal_duration = _finite_sum(measurable["detected_focal_voice_duration_s"])
    covered_duration = _finite_sum(measurable["covered_focal_duration_s"])
    uncovered_duration = _finite_sum(measurable["uncovered_focal_voice_duration_s"])
    uncovered = segments.loc[
        segments["coverage_status"].eq(str(CoverageStatus.UNCOVERED))
    ]
    unresolved = segments.loc[
        segments["coverage_status"].isin(
            [
                str(CoverageStatus.UNRESOLVED_IDENTITY),
                str(CoverageStatus.ANNOTATION_UNAVAILABLE),
            ]
        )
    ]
    short_count = int(uncovered["duration_s"].lt(short_segment_max_duration_s).sum())
    measurable_count = len(measurable)
    if measurable_count == 0:
        focal_status = "audit inconclusive for focal action coverage"
    elif measurable_count < len(successful):
        focal_status = "measurable for a subset of recordings and resolved segments"
    else:
        focal_status = "measurable for resolved segments; unresolved intervals retained"
    return {
        "recording_count": len(summaries),
        "successful_recording_count": len(successful),
        "failed_recording_count": int((~summaries["audit_ok"]).sum()),
        "focal_measurable_recording_count": measurable_count,
        "focal_specific_status": focal_status,
        "audited_duration_s": _finite_sum(successful["audited_duration_s"]),
        "detected_acoustic_voice_duration_s": acoustic_duration,
        "annotated_focal_voice_duration_s": _finite_sum(
            successful["annotated_focal_voice_duration_s"]
        ),
        "detected_focal_voice_duration_s": focal_duration if measurable_count else None,
        "focal_annotation_coverage_ratio": covered_duration / focal_duration
        if focal_duration > 0
        else None,
        "covered_focal_duration_s": covered_duration if measurable_count else None,
        "uncovered_focal_voice_duration_s": uncovered_duration
        if measurable_count
        else None,
        "uncovered_focal_segment_count": len(uncovered),
        "uncovered_short_segment_ratio": short_count / len(uncovered)
        if len(uncovered)
        else None,
        "uncovered_segment_duration_quantiles_s": _quantiles(uncovered["duration_s"]),
        "unresolved_identity_duration_s": unresolved_duration,
        "unresolved_identity_ratio": unresolved_duration / acoustic_duration
        if acoustic_duration > 0
        else None,
        "unresolved_identity_segment_count": len(unresolved),
        "unannotated_any_speaker_duration_s": _finite_sum(
            successful["unannotated_any_speaker_duration_s"]
        ),
    }


def _bucket_statistics(segments: pd.DataFrame, labels: tuple[str, ...]) -> JsonDict:
    uncovered = segments.loc[
        segments["coverage_status"].eq(str(CoverageStatus.UNCOVERED))
    ]
    return {
        label: {
            "segment_count": int(uncovered["duration_bucket"].eq(label).sum()),
            "duration_s": float(
                uncovered.loc[
                    uncovered["duration_bucket"].eq(label), "duration_s"
                ].sum()
            ),
        }
        for label in labels
    }


def _examples(
    segments: pd.DataFrame, status: str, limit: int = 5
) -> list[dict[str, object]]:
    subset = segments.loc[segments["coverage_status"].eq(status)].sort_values(
        ["duration_s", "dataset", "recording_id", "canonical_start_s"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    columns = [
        "dataset",
        "recording_id",
        "sync_group_id",
        "view_id",
        "wearer_id",
        "canonical_start_s",
        "canonical_end_s",
        "duration_s",
        "annotation_overlap_ratio",
        "identity_method",
        "identity_confidence",
        "coverage_status",
    ]
    records = (
        subset.head(limit)[columns].replace({np.nan: None}).to_dict(orient="records")
    )
    return [{str(key): value for key, value in row.items()} for row in records]


def build_report(
    summaries: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    sources: list[DatasetRecordingSet],
    config: VocalCoverageConfig,
    detector: VoiceActivityDetector,
    manifest_path: Path,
    media_path: Path,
    command: str,
    provenance: ReportProvenance | None = None,
) -> tuple[JsonDict, Mapping[str, object]]:
    """Build report payload and parameters, including exact input checksums."""
    provenance = provenance or collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    implementation = _implementation_artifacts()
    by_dataset = {
        source.dataset: coverage_statistics(
            summaries.loc[summaries["dataset"].eq(source.dataset)],
            segments.loc[segments["dataset"].eq(source.dataset)],
            short_segment_max_duration_s=config.coverage.short_segment_max_duration_s,
        )
        for source in sources
    }
    input_artifacts: dict[str, object] = {
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "media_metadata": {"path": str(media_path), "sha256": _sha256(media_path)},
        "annotations": {
            source.dataset: [
                {"path": str(path), "sha256": _sha256(path)}
                for path in source.annotation_paths
            ]
            for source in sources
        },
    }
    limitations = [
        "Silero VAD detects acoustic speech-like activity in mixed audio; it does not identify a speaker.",
        "Coverage tolerance changes overlap classification only; boundary offsets remain separate alignment measurements.",
        "The raw manifest was generated without per-media content hashes; its artifact SHA-256 binds the inventory, not every media byte.",
        "No missing annotation is inferred from unresolved speaker identity.",
        "No annotation, media, pseudo-label or ONSET/OFFSET/NO_EVENT action is created or corrected.",
        *[note for source in sources for note in source.limitations],
    ]
    payload: JsonDict = {
        "audit_type": "vocal_annotation_coverage",
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": provenance.generated_at_utc,
        "git_commit": provenance.git_commit,
        "dataset_manifest_checksum": input_artifacts["dataset_manifest"]["sha256"],  # type: ignore[index]
        "configuration_checksum": config.checksum(),
        "implementation_checksum": _combined_checksum(implementation),
        "implementation_artifacts": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(path),
            }
            for path in implementation
        ],
        "uv_lock_checksum": _sha256(uv_lock) if uv_lock.exists() else None,
        "datasets": [source.dataset for source in sources],
        "detector_name": detector.name,
        "detector_version": detector.version(),
        "identity_method": {
            source.dataset: "+".join(source.identity_methods) for source in sources
        },
        "recording_count": len(summaries),
        "audited_duration_s": _finite_sum(summaries["audited_duration_s"]),
        "coverage_statistics": coverage_statistics(
            summaries,
            segments,
            short_segment_max_duration_s=config.coverage.short_segment_max_duration_s,
        ),
        "coverage_statistics_by_dataset": by_dataset,
        "coverage_statistics_by_duration_bucket": _bucket_statistics(
            segments, config.coverage.bucket_labels()
        ),
        "unresolved_identity_duration_s": _finite_sum(
            summaries["unresolved_identity_duration_s"]
        ),
        "unresolved_identity_ratio": coverage_statistics(
            summaries,
            segments,
            short_segment_max_duration_s=config.coverage.short_segment_max_duration_s,
        )["unresolved_identity_ratio"],
        "focal_voice_activity_status_counts": dict(
            Counter(summaries["focal_voice_activity_status"].astype(str))
        ),
        "annotation_provenance": {
            source.dataset: source.annotation_provenance for source in sources
        },
        "media_provenance": {
            "temporal_authority": "native audio PTS",
            "decoder": "ffmpeg mono 16 kHz float32 with aresample async PTS preservation",
            "media_metadata_artifact": str(media_path),
        },
        "input_artifacts": input_artifacts,
        "command": command,
        "representative_examples": {
            "uncovered": _examples(segments, str(CoverageStatus.UNCOVERED)),
            "partially_covered": _examples(
                segments, str(CoverageStatus.PARTIALLY_COVERED)
            ),
            "unresolved_identity": _examples(
                segments, str(CoverageStatus.UNRESOLVED_IDENTITY)
            ),
        },
        "decision_policy": {
            "pass_warn_fail_applied": False,
            "A": "High focal coverage and rare short omissions would support candidacy for v0 action construction.",
            "B": "Non-negligible measurable focal omissions would motivate a separate completion or pseudo-annotation step first.",
            "C": "Without reliable focal attribution, the result is audit inconclusive for focal action coverage, never evidence of completeness.",
        },
        "limitations": list(dict.fromkeys(limitations)),
    }
    parameters = config.to_dict()
    parameters["configuration_path"] = config.source_path
    return payload, parameters


def _summary_table(results: list[RecordingAssessment]) -> pd.DataFrame:
    rows = [result.summary.to_row() for result in results]
    return pd.DataFrame.from_records(rows).sort_values(
        ["dataset", "recording_id", "view_id"], kind="stable", ignore_index=True
    )


def _candidate_table(results: list[RecordingAssessment]) -> pd.DataFrame:
    excluded = {CoverageStatus.COVERED, CoverageStatus.NOT_FOCAL}
    rows = [
        segment.to_row()
        for result in results
        for segment in result.segments
        if segment.coverage_status not in excluded
    ]
    if not rows:
        fields = list(SegmentAssessment.__dataclass_fields__)
        return pd.DataFrame(
            columns=[field for field in fields if field != "identity_evidence"]
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["dataset", "recording_id", "canonical_start_s", "canonical_end_s"],
        kind="stable",
        ignore_index=True,
    )


def run_vocal_annotation_coverage_audit(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    coverage_config_path: Path | None = None,
    detector: VoiceActivityDetector | None = None,
    decoder: Decoder = decode_audio,
    command: str | None = None,
    stream: TextIO | None = sys.stderr,
) -> VocalCoverageOutputs:
    """Run the configured audit and write deterministic population artifacts."""
    paths = pipeline_paths(cfg)
    manifest_path = _require(Path(cfg.manifest.output), "conv-wm audit manifest")
    media_path = _require(paths.reports / MEDIA_METADATA_TABLE, "conv-wm audit media")
    media = pd.read_parquet(media_path)
    config = load_coverage_config(coverage_config_path)
    active_detector = detector or build_detector(config.detector)
    datasets = selected_datasets(dataset)
    sources = load_recording_sets(cfg, media, datasets)
    results = audit_recording_sets(
        sources,
        detector=active_detector,
        decoder=decoder,
        config=config,
        stream=stream,
    )
    summary_table = _summary_table(results)
    candidate_table = _candidate_table(results)
    invoked = command or (
        "conv-wm audit vocal-annotation-coverage "
        f"--dataset {dataset} --coverage-config {coverage_config_path or DEFAULT_COVERAGE_CONFIG_PATH}"
    )
    provenance = collect_provenance()
    payload, parameters = build_report(
        summary_table,
        candidate_table,
        sources=sources,
        config=config,
        detector=active_detector,
        manifest_path=manifest_path,
        media_path=media_path,
        command=invoked,
        provenance=provenance,
    )
    output_dir = paths.reports / OUTPUT_DIR
    write_table(output_dir / SUMMARY_TABLE.name, summary_table)
    write_table(output_dir / UNCOVERED_SEGMENTS_TABLE.name, candidate_table)
    report = write_summary(
        output_dir / REPORT_FILE.name,
        payload,
        parameters=parameters,
        provenance=provenance,
    )
    return VocalCoverageOutputs(summary_table, candidate_table, report, output_dir)


__all__ = [
    "REPORT_FILE",
    "SUMMARY_TABLE",
    "UNCOVERED_SEGMENTS_TABLE",
    "VocalCoverageOutputs",
    "audit_recording_sets",
    "build_report",
    "coverage_statistics",
    "run_vocal_annotation_coverage_audit",
    "selected_datasets",
]
