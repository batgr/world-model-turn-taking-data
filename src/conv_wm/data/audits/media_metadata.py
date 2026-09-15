"""Media metadata audit: probe every media file's container and streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.population import run_population
from conv_wm.data.media.ffprobe import FFprobeError, get_ffprobe_path
from conv_wm.data.media.metadata import derive_media_qc, extract_media_metadata
from conv_wm.reports import write_summary, write_table

OUTPUT_DIR = Path("temporal") / "media_metadata"
MEDIA_METADATA_TABLE = OUTPUT_DIR / "media_metadata.parquet"
QC_COLUMNS = (
    "qc_probe_ok",
    "qc_has_video",
    "qc_has_audio",
    "qc_positive_video_duration",
    "qc_positive_audio_duration",
)


@dataclass(frozen=True)
class MediaItem:
    """One media file from the raw manifest."""

    dataset: str
    relative_path: str


@dataclass(frozen=True)
class MediaMetadataOutputs:
    """Artifacts written by one run."""

    metadata: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path

    @property
    def summary_path(self) -> Path:
        """Location of the population summary."""
        return self.output_dir / "summary.json"


def load_media_items(manifest_path: Path) -> list[MediaItem]:
    """Video and audio files listed by the raw manifest, which must exist."""
    path = manifest_path
    if not path.exists():
        raise MissingPrerequisiteError(path, produce_with="conv-wm audit manifest")
    manifest = pd.read_parquet(path)
    media = manifest.loc[manifest["file_type"].isin(["video", "audio"])]
    return [
        MediaItem(dataset=str(row["dataset"]), relative_path=str(row["relative_path"]))
        for row in media.to_dict(orient="records")
    ]


def probe_item(item: MediaItem, *, raw_root: Path) -> dict[str, object]:
    """Probe one file; the record carries ``probe_ok``/``probe_error``."""
    record = extract_media_metadata(
        dataset=item.dataset,
        relative_path=item.relative_path,
        path=raw_root / item.relative_path,
    )
    record["probe_error"] = None
    return record


def failed_item(item: MediaItem, error: Exception) -> dict[str, object]:
    """Explicit failed record for a file ffprobe could not read."""
    return {
        "dataset": item.dataset,
        "relative_path": item.relative_path,
        "probe_ok": False,
        "probe_error": str(error),
    }


def build_media_metadata_summary(metadata: pd.DataFrame) -> dict[str, object]:
    """Population counts and the overall QC verdict."""
    return {
        "audit_type": "media_metadata",
        "n_files": len(metadata),
        "n_probe_ok": int(metadata["qc_probe_ok"].sum()),
        "n_with_video": int(metadata["qc_has_video"].sum()),
        "n_with_audio": int(metadata["qc_has_audio"].sum()),
        "datasets": {
            str(dataset): {
                "n_files": len(group),
                "n_probe_ok": int(group["qc_probe_ok"].sum()),
                "n_with_video": int(group["qc_has_video"].sum()),
                "n_with_audio": int(group["qc_has_audio"].sum()),
            }
            for dataset, group in metadata.groupby("dataset")
        },
        "all_qc_passed": bool(metadata[list(QC_COLUMNS)].all(axis=None)),
    }


def run_media_metadata_audit(
    cfg: DictConfig, *, max_workers: int = 4
) -> MediaMetadataOutputs:
    """Run the audit on the configured corpus and write its artifacts."""
    get_ffprobe_path()  # fail early and clearly when ffprobe is missing
    paths = pipeline_paths(cfg)
    items = load_media_items(Path(cfg.manifest.output))
    records = run_population(
        items,
        lambda item: probe_item(item, raw_root=paths.raw),
        on_error=failed_item,
        label="media probe",
        max_workers=max_workers,
    )
    metadata = pd.DataFrame.from_records(records)
    if {"audio_duration_sec", "video_duration_sec"}.issubset(metadata.columns):
        metadata["av_duration_delta_sec"] = (
            metadata["audio_duration_sec"] - metadata["video_duration_sec"]
        )
    metadata = derive_media_qc(metadata)
    summary = build_media_metadata_summary(metadata)
    output_dir = paths.reports / OUTPUT_DIR
    write_table(paths.reports / MEDIA_METADATA_TABLE, metadata)
    outputs = MediaMetadataOutputs(metadata, summary, output_dir)
    write_summary(outputs.summary_path, summary)
    return outputs


__all__ = [
    "MEDIA_METADATA_TABLE",
    "FFprobeError",
    "MediaItem",
    "MediaMetadataOutputs",
    "build_media_metadata_summary",
    "run_media_metadata_audit",
]
