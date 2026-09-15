"""Video timeline audit: sampled windows for every file, full scan for suspects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.audits.audio_timeline import load_media_files
from conv_wm.data.audits.population import run_population
from conv_wm.data.media.metadata import MediaFileInfo
from conv_wm.data.media.video_timeline import (
    VideoTimelineClass,
    VideoWindowRecord,
    analyze_frame_timeline,
    audit_video_windows,
    known_boundaries_within,
    probe_video_timestamps,
)
from conv_wm.reports import write_summary, write_table

OUTPUT_DIR = Path("temporal") / "video_timeline"
WINDOW_SEC = 10.0


@dataclass(frozen=True)
class VideoTimelineOutputs:
    """Artifacts written by one run."""

    sampled: pd.DataFrame
    file_summary: pd.DataFrame
    full_scan: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        """Location of the population summary."""
        return self.output_dir / "summary.json"


def sample_file(info: MediaFileInfo, *, raw_root: Path) -> list[VideoWindowRecord]:
    """Sampled windows of one file, including the dataset's known boundaries."""
    grid = datasets.get(info.dataset).audio.known_boundary_grid
    return audit_video_windows(
        dataset=info.dataset,
        relative_path=info.relative_path,
        path=raw_root / info.relative_path,
        duration_sec=info.video_duration_sec,
        fps=info.video_avg_frame_rate,
        time_base=info.video_time_base,
        window_sec=WINDOW_SEC,
        known_boundaries_sec=known_boundaries_within(
            info.video_duration_sec, period_sec=grid.period_sec if grid else None
        ),
    )


def failed_file(info: MediaFileInfo, error: Exception) -> list[VideoWindowRecord]:
    """Single failed window record for a file that could not be sampled at all."""
    return [
        VideoWindowRecord(
            info.dataset, info.relative_path, "all", "", False, str(error), None
        )
    ]


def summarize_files(sampled: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window rows into one row per file."""
    return sampled.groupby(["dataset", "relative_path"], as_index=False).agg(
        n_windows=("window", "count"),
        all_probes_ok=("probe_ok", "all"),
        all_windows_cfr_consistent=(
            "timeline_class",
            lambda values: bool(
                (values == str(VideoTimelineClass.CFR_CONSISTENT)).all()
            ),
        ),
        min_timestamp_coverage=("timestamp_coverage", "min"),
        max_delta_error_ticks=("max_delta_error_ticks", "max"),
        max_cumulative_drift_sec=("max_abs_cumulative_drift_sec", "max"),
        total_duplicate_timestamps=("n_duplicate_timestamps", "sum"),
        total_non_positive_deltas=("n_non_positive_deltas", "sum"),
    )


def full_scan(info: MediaFileInfo, *, raw_root: Path) -> dict[str, object]:
    """Analyse every packet of a suspect file."""
    metrics = analyze_frame_timeline(
        probe_video_timestamps(raw_root / info.relative_path),
        fps=info.video_avg_frame_rate,
        time_base=info.video_time_base,
    )
    return {
        "dataset": info.dataset,
        "relative_path": info.relative_path,
        "probe_ok": True,
        "probe_error": None,
        **metrics.to_row(),
    }


def failed_full_scan(info: MediaFileInfo, error: Exception) -> dict[str, object]:
    """Explicit failed record for a suspect that could not be scanned."""
    return {
        "dataset": info.dataset,
        "relative_path": info.relative_path,
        "probe_ok": False,
        "probe_error": str(error),
        "timeline_class": str(VideoTimelineClass.PROBE_ERROR),
    }


def build_video_timeline_summary(
    file_summary: pd.DataFrame, sampled: pd.DataFrame, full_scan_table: pd.DataFrame
) -> dict[str, object]:
    """Population counts: consistent files, suspects and full-scan outcomes."""
    n_consistent = int(file_summary["all_windows_cfr_consistent"].sum())
    n_irregular = (
        0
        if full_scan_table.empty
        else int(
            (
                full_scan_table["timeline_class"]
                != str(VideoTimelineClass.CFR_CONSISTENT)
            ).sum()
        )
    )
    return {
        "audit_type": "video_timeline",
        "n_files": len(file_summary),
        "n_sampled_windows": len(sampled),
        "window_counts": {
            str(name): int(count)
            for name, count in sampled["window"].value_counts().items()
        },
        "n_files_sampled_consistent": n_consistent,
        "n_sampled_suspects": int(len(file_summary) - n_consistent),
        "n_full_scanned": len(full_scan_table),
        "n_irregular_after_full_scan": n_irregular,
        "datasets": {
            str(dataset): {
                "n_files": len(group),
                "n_files_sampled_consistent": int(
                    group["all_windows_cfr_consistent"].sum()
                ),
            }
            for dataset, group in file_summary.groupby("dataset")
        },
        "interpretation": {
            "video_time_source": "native_pts",
            "fps_role": "describes cadence; never replaces timestamps",
            "known_boundary_windows": "one extra window per declared dataset boundary",
        },
    }


def run_video_timeline_audit(
    cfg: DictConfig, *, max_workers: int = 4
) -> VideoTimelineOutputs:
    """Run the audit on the configured corpus and write its artifacts."""
    paths = pipeline_paths(cfg)
    output_dir = paths.reports / OUTPUT_DIR
    items = load_media_files(paths.reports)
    window_lists = run_population(
        items,
        lambda info: sample_file(info, raw_root=paths.raw),
        on_error=failed_file,
        label="video window scan",
        max_workers=max_workers,
    )
    sampled = pd.DataFrame.from_records(
        [record.to_row() for windows in window_lists for record in windows]
    )
    file_summary = summarize_files(sampled)
    suspects = set(
        file_summary.loc[~file_summary["all_windows_cfr_consistent"], "relative_path"]
    )
    suspect_items = [info for info in items if info.relative_path in suspects]
    full_scan_rows = run_population(
        suspect_items,
        lambda info: full_scan(info, raw_root=paths.raw),
        on_error=failed_full_scan,
        label="video full scan",
        max_workers=max_workers,
    )
    full_scan_table = pd.DataFrame.from_records(full_scan_rows)
    summary = build_video_timeline_summary(file_summary, sampled, full_scan_table)

    write_table(output_dir / "sampled_timeline.parquet", sampled)
    write_table(output_dir / "file_summary.parquet", file_summary)
    if not full_scan_table.empty:
        write_table(output_dir / "full_scan_suspects.parquet", full_scan_table)
    outputs = VideoTimelineOutputs(
        sampled, file_summary, full_scan_table, summary, output_dir
    )
    write_summary(outputs.summary_path, summary, parameters={"window_sec": WINDOW_SEC})
    return outputs
