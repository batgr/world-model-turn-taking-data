"""Audio timeline audit over the whole media population.

Reads the media metadata table, scans every audio stream's packets, interprets
the events with dataset knowledge, validates selected windows by decoding and
writes the report artifacts. Scientific logic lives in
:mod:`conv_wm.data.media.audio`; dataset knowledge in :mod:`conv_wm.data.datasets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.audits.population import run_population
from conv_wm.data.datasets.spec import DatasetSpec
from conv_wm.data.media.audio import (
    INTERPRETED_EVENT_COLUMNS,
    AudioFileTimelineRecord,
    AudioTimelineParameters,
    DecodeValidationCase,
    InterpretedAudioEvent,
    TimelineStatus,
    analyze_audio_packets,
    interpret_events,
    interpret_file,
    interpreted_events_frame,
    probe_audio_packets,
    validate_decoded_window,
)
from conv_wm.data.media.audio.decode import select_decode_validation_cases
from conv_wm.data.media.audio.summary import (
    build_audio_timeline_summary,
    build_decode_validation_summary,
)
from conv_wm.data.media.metadata import MediaFileInfo, media_files
from conv_wm.reports import write_summary, write_table

OUTPUT_DIR = Path("temporal") / "audio_timeline"
MAX_WORKERS = 4

DECODED_SAMPLE_REFERENCES: dict[str, tuple[int, str]] = {
    "aac": (1024, "targeted_show_frames_mode"),
}
"""Decoded samples per packet by codec, with the provenance of that number.

The population is AAC-LC only; targeted ``-show_frames`` validation established
1024 as the decoded-frame mode. Any new codec must be validated the same way
before it is added here.
"""


@dataclass(frozen=True)
class AudioTimelineOutputs:
    """Artifacts written by one run."""

    files: pd.DataFrame
    events: pd.DataFrame
    decode_validation: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        """Location of the population summary."""
        return self.output_dir / "audio_packet_timeline_summary.json"


def parameters_for(
    codec: str, **overrides: float
) -> tuple[AudioTimelineParameters, str]:
    """Analysis parameters for ``codec`` plus the provenance of its decoded reference."""
    try:
        reference, source = DECODED_SAMPLE_REFERENCES[codec]
    except KeyError as exc:
        raise ValueError(
            f"No validated decoded-sample reference for audio codec: {codec}"
        ) from exc
    return AudioTimelineParameters(
        reference_decoded_samples=reference, **overrides
    ), source


def audit_audio_file(
    info: MediaFileInfo,
    *,
    raw_root: Path,
    spec: DatasetSpec,
) -> tuple[AudioFileTimelineRecord, list[InterpretedAudioEvent]]:
    """Scan, analyse and interpret one file's audio stream."""
    parameters, source = parameters_for(info.audio_codec)
    file = _identity_record(info, parameters=parameters, source=source)
    analysis = analyze_audio_packets(
        probe_audio_packets(raw_root / info.relative_path), file=file
    )
    skip_samples = (
        analysis.file.codec_boundary.skip_samples if analysis.file.codec_boundary else 0
    )
    events = interpret_events(
        analysis.events,
        sample_rate_hz=info.audio_sample_rate_hz,
        reference_decoded_samples=parameters.reference_decoded_samples,
        skip_samples=skip_samples,
        boundary_grid=spec.audio.known_boundary_grid,
    )
    return interpret_file(analysis.file, events), events


def failed_audio_file(
    info: MediaFileInfo, error: Exception
) -> tuple[AudioFileTimelineRecord, list[InterpretedAudioEvent]]:
    """Explicit failed record for a file whose probe or analysis raised."""
    parameters, source = parameters_for(info.audio_codec)
    record = AudioFileTimelineRecord.failed(
        dataset=info.dataset,
        relative_path=info.relative_path,
        audio_codec=info.audio_codec,
        sample_rate_hz=info.audio_sample_rate_hz,
        time_base=info.audio_time_base,
        audio_duration_sec=info.audio_duration_sec,
        video_avg_frame_rate=info.video_avg_frame_rate,
        reference_decoded_samples_source=source,
        parameters=parameters,
        error=str(error),
    )
    return interpret_file(record, []), []


def _identity_record(
    info: MediaFileInfo, *, parameters: AudioTimelineParameters, source: str
) -> AudioFileTimelineRecord:
    return AudioFileTimelineRecord(
        dataset=info.dataset,
        relative_path=info.relative_path,
        audio_codec=info.audio_codec,
        sample_rate_hz=info.audio_sample_rate_hz,
        time_base=info.audio_time_base,
        audio_duration_sec=info.audio_duration_sec,
        video_avg_frame_rate=info.video_avg_frame_rate,
        reference_decoded_samples_source=source,
        probe_ok=True,
        probe_error=None,
        timeline_status=TimelineStatus.MEASURED,
        parameters=parameters,
    )


def check_codecs_supported(items: list[MediaFileInfo]) -> None:
    """Refuse to analyse codecs without a validated decoded-frame reference."""
    unsupported = {item.audio_codec for item in items} - set(DECODED_SAMPLE_REFERENCES)
    if unsupported:
        raise ValueError(
            f"Audio codecs need decoded-sample validation: {sorted(unsupported)}"
        )


def load_media_files(reports_root: Path) -> list[MediaFileInfo]:
    """Typed media items from the media metadata audit, which must have run."""
    path = reports_root / MEDIA_METADATA_TABLE
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with="conv-wm audit media")
    return media_files(pd.read_parquet(path))


def run_audio_timeline_audit(
    cfg: DictConfig,
    *,
    max_workers: int = MAX_WORKERS,
) -> AudioTimelineOutputs:
    """Run the audit on the configured corpus and write its artifacts."""
    paths = pipeline_paths(cfg)
    output_dir = paths.reports / OUTPUT_DIR
    items = load_media_files(paths.reports)
    check_codecs_supported(items)

    results = run_population(
        items,
        lambda info: audit_audio_file(
            info, raw_root=paths.raw, spec=datasets.get(info.dataset)
        ),
        on_error=failed_audio_file,
        label="audio packet scan",
        max_workers=max_workers,
    )
    files = (
        pd.DataFrame.from_records([record.to_row() for record, _ in results])
        .sort_values(["dataset", "relative_path"])
        .reset_index(drop=True)
    )
    events = _events_table(results)

    cases = select_decode_validation_cases(
        files, events, extra_cases=_dataset_decode_cases(files)
    )
    files_by_path = files.set_index("relative_path")
    decode_validation = pd.DataFrame.from_records(
        [
            validate_decoded_window(
                case,
                path=paths.raw / case.relative_path,
                file_row=files_by_path.loc[case.relative_path],
            ).to_row()
            for case in cases
        ]
    )

    summary = build_audio_timeline_summary(
        files,
        events,
        threshold_comparison=AudioTimelineParameters(
            reference_decoded_samples=1
        ).threshold_comparison,
        dataset_sections=_dataset_sections(files, decode_validation),
    )
    parameters = {
        "decoded_sample_references": {
            codec: {"samples": samples, "source": source}
            for codec, (samples, source) in DECODED_SAMPLE_REFERENCES.items()
        },
        "analysis": parameters_for("aac")[0].to_dict(),
        "datasets": {
            name: {"known_boundary_grid": _grid_dict(datasets.get(name))}
            for name in sorted(files["dataset"].unique())
        },
    }
    write_table(output_dir / "audio_packet_timeline_files.parquet", files)
    write_table(output_dir / "audio_packet_timeline_events.parquet", events)
    write_table(output_dir / "audio_decode_validation.parquet", decode_validation)
    outputs = AudioTimelineOutputs(
        files, events, decode_validation, summary, output_dir
    )
    write_summary(outputs.summary_path, summary, parameters=parameters)
    write_summary(
        output_dir / "audio_decode_validation_summary.json",
        build_decode_validation_summary(decode_validation),
        parameters=parameters,
    )
    return outputs


def _events_table(
    results: list[tuple[AudioFileTimelineRecord, list[InterpretedAudioEvent]]],
) -> pd.DataFrame:
    frames = []
    for record, events in results:
        if not events:
            continue
        frame = interpreted_events_frame(events)
        frame.insert(0, "sample_rate_hz", record.sample_rate_hz)
        frame.insert(0, "audio_codec", record.audio_codec)
        frame.insert(0, "relative_path", record.relative_path)
        frame.insert(0, "dataset", record.dataset)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["dataset", "relative_path", "audio_codec", "sample_rate_hz"]
            + list(INTERPRETED_EVENT_COLUMNS)
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["dataset", "relative_path", "event_time_sec", "event_type"])
        .reset_index(drop=True)
    )


def _dataset_decode_cases(files: pd.DataFrame) -> list[DecodeValidationCase]:
    cases: list[DecodeValidationCase] = []
    for name, subset in files.loc[files["probe_ok"]].groupby("dataset"):
        selector = datasets.get(str(name)).audio.extra_decode_cases
        if selector is not None:
            cases.extend(selector(subset))
    return cases


def _dataset_sections(
    files: pd.DataFrame, decode_validation: pd.DataFrame
) -> dict[str, dict[str, object]]:
    sections: dict[str, dict[str, object]] = {}
    for name, subset in files.loc[files["probe_ok"]].groupby("dataset"):
        builder = datasets.get(str(name)).audio.summary_section
        if builder is not None:
            decoded = (
                decode_validation.loc[decode_validation["dataset"].eq(name)]
                if not decode_validation.empty
                else decode_validation
            )
            sections[str(name)] = dict(builder(subset, decoded))
    return sections


def _grid_dict(spec: DatasetSpec) -> dict[str, object] | None:
    grid = spec.audio.known_boundary_grid
    if grid is None:
        return None
    return {
        "period_sec": grid.period_sec,
        "tolerance_packets": grid.tolerance_packets,
        "category": str(grid.category),
    }
