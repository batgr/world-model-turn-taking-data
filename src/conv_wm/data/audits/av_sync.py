"""Technical A/V alignment audit from container/stream metadata.

Measures start offsets and end/duration deltas between the audio and video
streams of each file. These are container-timeline facts; they do not measure
perceptual lip-sync and no correction is derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.media.av_sync import compute_av_sync_metadata
from conv_wm.reports import write_summary, write_table

OUTPUT_DIR = Path("temporal") / "av_sync"


@dataclass(frozen=True)
class AvSyncOutputs:
    """Artifacts written by one run."""

    files: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        """Location of the population summary."""
        return self.output_dir / "av_sync_summary.json"


def build_av_sync_summary(sync: pd.DataFrame) -> dict[str, object]:
    """Start offsets and end/duration deltas by dataset, with the interpretation."""
    datasets: dict[str, object] = {}
    for dataset, group in sync.groupby("dataset"):
        datasets[str(dataset)] = {
            "n_files": len(group),
            "n_zero_start_offsets": int(group["av_start_offset_sec"].eq(0).sum()),
            "max_abs_start_offset_sec": float(group["av_start_offset_sec"].abs().max()),
            "av_end_delta_sec": {
                "min": float(group["av_end_delta_sec"].min()),
                "median": float(group["av_end_delta_sec"].median()),
                "max": float(group["av_end_delta_sec"].max()),
                "max_abs": float(group["abs_av_end_delta_sec"].max()),
            },
        }
    return {
        "audit_type": "av_technical_alignment",
        "n_files": len(sync),
        "n_zero_start_offsets": int(sync["av_start_offset_sec"].eq(0).sum()),
        "all_start_offsets_zero": bool(sync["av_start_offset_sec"].eq(0).all()),
        "datasets": datasets,
        "interpretation": {
            "audio_time_source": "native_pts",
            "video_time_source": "native_pts",
            "global_av_offset_correction": False,
            "duration_delta_is_sync_failure": False,
            "perceptual_av_sync": "deferred",
            "fine_av_drift_correction": "deferred",
            "downstream_constraint": (
                "absolute audio time must come from native PTS; naively concatenated "
                "decoded PCM drifts by the file's cumulative PCM-vs-PTS offset"
            ),
        },
    }


def run_av_sync_audit(cfg: DictConfig) -> AvSyncOutputs:
    """Derive A/V alignment measurements from the media metadata table."""
    paths = pipeline_paths(cfg)
    metadata_path = paths.reports / MEDIA_METADATA_TABLE
    if not metadata_path.exists():
        raise MissingPrerequisiteError(
            metadata_path, produce_with="conv-wm audit media"
        )
    sync = compute_av_sync_metadata(pd.read_parquet(metadata_path)).sort_values(
        ["dataset", "relative_path"]
    )
    summary = build_av_sync_summary(sync)
    output_dir = paths.reports / OUTPUT_DIR
    write_table(output_dir / "av_sync_files.parquet", sync)
    outputs = AvSyncOutputs(sync, summary, output_dir)
    write_summary(outputs.summary_path, summary)
    return outputs
