"""Shared fixtures for the three v0 vocal state layers.

The native focal voice-state artifact is written by hand — with the report that
vouches for its checksum — so the control and action-grid builds can be tested
without running the annotation pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from conv_wm.data.pipeline.action_grid import run_vocal_action_grid_build
from conv_wm.data.pipeline.control_state import (
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
            "datasets": {
                "ego4d": {
                    "interim": str(tmp_path / "interim" / "ego4d"),
                    "files": {"clips_clean": "clips_clean.parquet"},
                },
                "egocom": {
                    "interim": str(tmp_path / "interim" / "egocom"),
                    "raw": str(tmp_path / "raw" / "EgoCom" / "annotations"),
                    "files": {
                        "video_info_clean": "video_info_clean.parquet",
                        "ground_truth_clean": "ground_truth_clean.parquet",
                    },
                },
            },
            "manifest": {"output": str(reports / "manifest" / "raw_manifest.parquet")},
        }
    )


def native_row(dataset, recording_id, start, end, state, conversation="group-1"):
    kind = (
        SourceKind.EGO4D_VOICE_SEGMENTS
        if dataset == "ego4d"
        else SourceKind.EGOCOM_TRANSCRIPT
    )
    return NativeStateInterval(
        dataset=dataset,
        recording_id=recording_id,
        sync_group_id=None if dataset == "ego4d" else conversation,
        view_id=f"view-{recording_id}",
        wearer_id="0",
        canonical_start_s=start,
        canonical_end_s=end,
        voice_state=VoiceState(state),
        source_kind=SourceKind.INVALID_OR_MISSING if state == "UNKNOWN" else kind,
        source_annotation_id=None if state == "SILENT" else f"native#{start}",
        annotation_schema_version=f"{dataset}-test-v1",
    ).to_row()


def write_native_state(
    cfg,
    dataset: str,
    timeline=TIMELINE,
    *,
    recording="rec-1",
    conversation="group-1",
    annotation_paths: list[Path] | None = None,
):
    """Write a native focal voice-state artifact and the report that vouches for it."""
    processed = Path(cfg.paths.processed) / "native_focal_voice_state" / dataset
    reports = Path(cfg.paths.reports) / "native_focal_voice_state" / dataset
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(
        [
            native_row(dataset, recording, *item, conversation=conversation)
            for item in timeline
        ]
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
                    "native_annotations": [
                        {
                            "path": str(item),
                            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                        }
                        for item in annotation_paths
                    ]
                    if annotation_paths
                    else [{"path": "native.parquet", "sha256": "dead"}]
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
    return run_control_focal_voice_state_build(cfg, dataset=dataset)


def long_timeline(
    *, speech_s: float = 1.5, pause_s: float = 2.5, cycles: int = 12
) -> list[tuple[float, float, str]]:
    """A long alternating SILENT/SPEAKING timeline, for window-sized fixtures.

    One cycle is ``pause_s`` of silence then ``speech_s`` of speech, so the
    action grid holds one OFFSET and one ONSET per cycle with long stretches of
    NO_EVENT between them — enough to produce both event and background windows.
    """
    timeline: list[tuple[float, float, str]] = []
    time_s = 0.0
    for _ in range(cycles):
        timeline.append((time_s, time_s + pause_s, "SILENT"))
        time_s += pause_s
        timeline.append((time_s, time_s + speech_s, "SPEAKING"))
        time_s += speech_s
    return timeline


SPLIT_TABLES = {
    "ego4d": ("clips_clean.parquet", "clip_uid"),
    "egocom": ("video_info_clean.parquet", "video_name"),
}


def write_recording_splits(cfg, dataset: str, splits: dict[str, str]) -> Path:
    """Write the cleaned upstream table the model-ready stage reads splits from."""
    table, key = SPLIT_TABLES[dataset]
    path = Path(cfg.datasets[dataset].interim) / table
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            key: pd.Series(list(splits), dtype="string"),
            "split": pd.Series(list(splits.values()), dtype="string"),
        }
    ).to_parquet(path, index=False)
    return path


def write_media_metadata(cfg, rows: list[tuple[str, str, int]]) -> Path:
    """Write the media metadata audit table: ``(dataset, relative_path, n_audio)``."""
    path = (
        Path(cfg.paths.reports)
        / "temporal"
        / "media_metadata"
        / "media_metadata.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": [row[0] for row in rows],
            "relative_path": [row[1] for row in rows],
            "probe_ok": [True] * len(rows),
            "n_audio_streams": [row[2] for row in rows],
        }
    ).to_parquet(path, index=False)
    return path


def write_egocom_interim(
    cfg,
    recordings: dict[str, tuple[str, str, list[tuple[float, float, str]]]],
    *,
    others: dict[str, list[tuple[str, float, float]]] | None = None,
) -> list[Path]:
    """Cleaned EgoCom tables consistent with hand-written wearer timelines.

    ``recordings`` maps a POV video to ``(conversation part, split, timeline)``;
    the wearer (speaker 0) gets one timed token per SPEAKING interval, so the
    label facts reproduce the timeline exactly. ``others`` adds tokens of other
    speakers ``(speaker, start, end)`` per conversation part. Media metadata
    covering every video's audio is written as well.
    """
    interim = Path(cfg.datasets.egocom.interim)
    interim.mkdir(parents=True, exist_ok=True)
    info_rows = []
    words = []
    for video, (part, split, timeline) in recordings.items():
        info_rows.append(
            {
                "video_id": len(info_rows),
                "conversation_id": part,
                "video_speaker_id": 0,
                "num_speakers": 2,
                "duration_seconds": round(timeline[-1][1]),
                "native_speaker": True,
                "speaker_is_host": False,
                "video_name": video,
                "background_fan": False,
                "background_music": None,
                "split": split,
            }
        )
        for start, end, state in timeline:
            if state == "SPEAKING" and not any(
                w["conversation_id"] == part and w["startTime"] == start for w in words
            ):
                words.append(
                    {
                        "conversation_id": part,
                        "speaker_id": 0,
                        "startTime": start,
                        "endTime": end,
                        "word": "hello",
                    }
                )
    for part, items in (others or {}).items():
        for speaker, start, end in items:
            words.append(
                {
                    "conversation_id": part,
                    "speaker_id": int(speaker),
                    "startTime": start,
                    "endTime": end,
                    "word": "yes",
                }
            )
    info_path = interim / "video_info_clean.parquet"
    pd.DataFrame(info_rows).to_parquet(info_path, index=False)
    transcript = pd.DataFrame(
        words, columns=["conversation_id", "speaker_id", "startTime", "endTime", "word"]
    )
    transcript.insert(0, "source_row", range(len(transcript)))
    transcript_path = interim / "ground_truth_clean.parquet"
    transcript.to_parquet(transcript_path, index=False)
    media_path = (
        Path(cfg.paths.reports)
        / "temporal"
        / "media_metadata"
        / "media_metadata.parquet"
    )
    media_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": ["egocom"] * len(recordings),
            "relative_path": [f"EgoCom/240p/{video}.MP4" for video in recordings],
            "probe_ok": [True] * len(recordings),
            "n_audio_streams": [1] * len(recordings),
            "audio_start_time_sec": [0.0] * len(recordings),
            "audio_duration_sec": [
                timeline[-1][1] + 1.0 for _, _, timeline in recordings.values()
            ],
        }
    ).to_parquet(media_path, index=False)
    return [info_path, transcript_path]


def build_action_grid(cfg, dataset="all"):
    """Run the control and grid builds quietly, as the model-ready stage needs both."""
    build_control_state(cfg, dataset)
    return run_vocal_action_grid_build(cfg, dataset=dataset)


__all__ = [
    "TIMELINE",
    "build_action_grid",
    "build_control_state",
    "config_for",
    "long_timeline",
    "native_row",
    "write_egocom_interim",
    "write_media_metadata",
    "write_native_state",
    "write_recording_splits",
]
