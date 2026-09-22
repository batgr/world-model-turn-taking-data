"""Shared fixtures for the three v0 vocal state layers.

The native focal voice-state artifact is written by hand — with the report that
vouches for its checksum — so the control and action-grid builds can be tested
without running the annotation pipeline.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from conv_wm.data.control_focal_voice_state import (
    run_control_focal_voice_state_build,
)
from conv_wm.data.vocal.native_state import (
    NATIVE_STATE_SCHEMA_VERSION,
    NativeStateInterval,
    SourceKind,
    VoiceState,
)

# A native timeline holding one representable silence, one cross-slot micro-gap,
# one same-slot micro-gap, one short speech burst and an UNKNOWN region.
TIMELINE = [
    (0.0, 0.34, "SILENT"),
    (0.34, 0.62, "SPEAKING"),
    (0.62, 1.0, "SILENT"),
    (1.0, 1.395, "SPEAKING"),
    (1.395, 1.405, "SILENT"),  # 10 ms gap across the 1.4 s boundary
    (1.405, 1.82, "SPEAKING"),
    (1.82, 1.86, "SILENT"),  # 40 ms gap inside slot 18
    (1.86, 2.4, "SPEAKING"),
    (2.4, 2.9, "UNKNOWN"),
    (2.9, 3.5, "SILENT"),
    (3.5, 3.52, "SPEAKING"),  # a 20 ms burst: kept, never filtered
    (3.52, 3.97, "SILENT"),
]


def config_for(tmp_path: Path):
    reports = tmp_path / "reports"
    return OmegaConf.create(
        {
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(tmp_path / "interim"),
                "validated": str(tmp_path / "validated"),
                "processed": str(tmp_path / "processed"),
                "model_ready": str(tmp_path / "model_ready"),
                "reports": str(reports),
            },
            "datasets": {"ego4d": {}, "egocom": {}},
            "manifest": {"output": str(reports / "manifest" / "raw_manifest.parquet")},
        }
    )


def native_row(dataset, recording_id, start, end, state):
    kind = (
        SourceKind.EGO4D_VOICE_SEGMENTS
        if dataset == "ego4d"
        else SourceKind.EGOCOM_TRANSCRIPT
    )
    return NativeStateInterval(
        dataset=dataset,
        recording_id=recording_id,
        sync_group_id=None if dataset == "ego4d" else "group-1",
        view_id=f"view-{recording_id}",
        wearer_id="0",
        canonical_start_s=start,
        canonical_end_s=end,
        voice_state=VoiceState(state),
        source_kind=SourceKind.INVALID_OR_MISSING if state == "UNKNOWN" else kind,
        source_annotation_id=None if state == "SILENT" else f"native#{start}",
        annotation_schema_version=f"{dataset}-test-v1",
    ).to_row()


def write_native_state(cfg, dataset: str, timeline=TIMELINE, *, recording="rec-1"):
    """Write a native focal voice-state artifact and the report that vouches for it."""
    processed = Path(cfg.paths.processed) / "native_focal_voice_state" / dataset
    reports = Path(cfg.paths.reports) / "native_focal_voice_state" / dataset
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(
        [native_row(dataset, recording, *item) for item in timeline]
    )
    path = processed / "focal_voice_intervals.parquet"
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (reports / "report.json").write_text(
        json.dumps(
            {
                "build_type": "native_focal_voice_state",
                "dataset": dataset,
                "native_state_schema_version": NATIVE_STATE_SCHEMA_VERSION,
                "git_commit": "abc123",
                "created_at": "2026-09-22T00:00:00+00:00",
                "output_artifacts": {"timeline": {"path": str(path), "sha256": digest}},
                "input_artifacts": {
                    "native_annotations": [{"path": "native.parquet", "sha256": "dead"}]
                },
                "source_dataset_version": {
                    "annotation_schema_version": f"{dataset}-test-v1",
                    "cleaning_rule_version": f"{dataset}-cleaning-v1",
                },
                "limitations": [f"{dataset} upstream limitation"],
            }
        )
    )
    manifest = Path(cfg.manifest.output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"relative_path": ["x"]}).to_parquet(manifest, index=False)
    return path


def build_control_state(cfg, dataset="all"):
    """Run the control build quietly, as the grid tests need it as a prerequisite."""
    return run_control_focal_voice_state_build(
        cfg, dataset=dataset, stream=io.StringIO()
    )


__all__ = [
    "TIMELINE",
    "build_control_state",
    "config_for",
    "native_row",
    "write_native_state",
]
