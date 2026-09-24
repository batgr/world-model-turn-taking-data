"""Physical check of the media manifest against a locally available raw corpus.

The manifest stores corpus-root-relative paths only. When the raw corpus is
mounted, this audit joins ``raw root / corpus directory / path`` and verifies
that every referenced file exists. The resolved locations are never written
back into the manifest; the report lists missing files by relative path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data.pipeline.media_manifest import (
    REPORT_ROOT,
    load_media_manifest,
    selected_datasets,
)
from conv_wm.reports import JsonDict, write_summary

CHECK_FILE = "file_check.json"


@dataclass(frozen=True)
class MediaFileCheckOutputs:
    """The verdict for one dataset."""

    dataset: str
    summary: JsonDict
    summary_path: Path

    @property
    def passed(self) -> bool:
        """Whether every referenced file exists."""
        return self.summary["status"] == "PASS"


def check_media_files(cfg: DictConfig, dataset: str) -> MediaFileCheckOutputs:
    """Verify that every path of the checked manifest of ``dataset`` exists locally."""
    manifest = load_media_manifest(cfg, dataset)
    corpus_directory = manifest.report["corpus_directory"]
    root = pipeline_paths(cfg).raw / (corpus_directory or "")
    checked = 0
    missing: list[str] = []
    for column in ("video_path", "audio_path"):
        for relative in manifest.table[column].dropna().astype(str):
            checked += 1
            if not (root / relative).is_file():
                missing.append(relative)
    summary: JsonDict = {
        "audit_type": "media_manifest_files",
        "dataset": dataset,
        "media_manifest_sha256": manifest.table_sha256,
        "corpus_directory": corpus_directory,
        "records": len(manifest.table),
        "paths_checked": checked,
        "paths_missing": len(missing),
        "missing_examples": sorted(missing)[:20],
        "status": "PASS" if root.is_dir() and not missing else "FAIL",
        "corpus_root_found": root.is_dir(),
    }
    path = pipeline_paths(cfg).reports / REPORT_ROOT / dataset / CHECK_FILE
    return MediaFileCheckOutputs(dataset, write_summary(path, summary), path)


def run_media_file_check(
    cfg: DictConfig, *, dataset: str = "all"
) -> list[MediaFileCheckOutputs]:
    """Run the physical check for every selected dataset."""
    return [check_media_files(cfg, name) for name in selected_datasets(dataset)]


__all__ = ["MediaFileCheckOutputs", "check_media_files", "run_media_file_check"]
