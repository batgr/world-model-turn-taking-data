"""Dataset adapters and the native focal voice-state build, on synthetic tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from conv_wm.data.datasets.ego4d_native_voice import load_ego4d_native_voice
from conv_wm.data.datasets.egocom_native_voice import load_egocom_native_voice
from conv_wm.data.pipeline.native_state import (
    run_native_focal_voice_state_build,
    supported_datasets,
)
from conv_wm.data.vocal.native_state import (
    NativeStateInterval,
    SourceKind,
    build_native_timeline,
)


def _config(tmp_path: Path):
    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    return OmegaConf.create(
        {
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(interim),
                "validated": str(tmp_path / "validated"),
                "processed": str(tmp_path / "processed"),
                "model_ready": str(tmp_path / "model_ready"),
                "reports": str(reports),
            },
            "datasets": {
                "ego4d": {
                    "raw": str(tmp_path / "raw" / "Ego4D"),
                    "interim": str(interim / "ego4d"),
                    "files": {
                        "clips_clean": "clips.parquet",
                        "persons_clean": "persons.parquet",
                        "voice_segments_clean": "voice.parquet",
                        "missing_voice_segments_clean": "missing.parquet",
                    },
                },
                "egocom": {
                    "raw": str(tmp_path / "raw" / "EgoCom"),
                    "interim": str(interim / "egocom"),
                    "files": {
                        "video_info_clean": "video_info.parquet",
                        "ground_truth_clean": "ground_truth.parquet",
                    },
                },
            },
            "manifest": {"output": str(reports / "manifest" / "raw_manifest.parquet")},
        }
    )


def _media(rows):
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "dataset",
            "relative_path",
            "probe_ok",
            "audio_start_time_sec",
            "audio_duration_sec",
        ],
    )


def _write_ego4d(cfg, *, valid=True, missing=(), media_start=10.0, media_duration=20.0):
    root = Path(cfg.datasets.ego4d.interim)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "split": ["train"],
            "clip_uid": ["clip-1"],
            "source_clip_uid": ["source-1"],
            "video_uid": ["video-1"],
            "video_start_sec": [10.0],
            "video_end_sec": [30.0],
            "video_start_frame": [300],
            "video_end_frame": [900],
            "clip_start_sec": [0],
            "clip_end_sec": [20.0],
            "clip_start_frame": [0],
            "clip_end_frame": [600],
            "valid": [valid],
        }
    ).to_parquet(root / "clips.parquet")
    pd.DataFrame(
        {
            "clip_uid": ["clip-1", "clip-1"],
            "person_id": ["0", "1"],
            "is_camera_wearer": [True, False],
        }
    ).to_parquet(root / "persons.parquet")
    pd.DataFrame(
        {
            "clip_uid": ["clip-1", "clip-1", "clip-1"],
            "start_time": [2.0, 2.5, 8.0],
            "end_time": [5.0, 3.0, 9.0],
            "start_frame": [60, 75, 240],
            "end_frame": [150, 90, 270],
            "video_start_time": [12.0, 12.5, 18.0],
            "video_end_time": [15.0, 13.0, 19.0],
            "video_start_frame": [360, 375, 540],
            "video_end_frame": [450, 390, 570],
            "person_id": ["0", "0", "1"],
        }
    ).to_parquet(root / "voice.parquet")
    pd.DataFrame.from_records(
        list(missing), columns=["clip_uid", "person_id", "start_time", "end_time"]
    ).astype({"start_time": float, "end_time": float}).to_parquet(
        root / "missing.parquet"
    )
    return _media([("ego4d", "Ego4D/video-1.mp4", True, media_start, media_duration)])


def _write_egocom(cfg, *, media_rows=None):
    root = Path(cfg.datasets.egocom.interim)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "video_name": ["vid-1", "vid-2"],
            "conversation_id": ["part-1", "part-1"],
            "cid": ["conv", "conv"],
            "video_speaker_id": [1, 2],
            "duration_seconds": [10, 10],
        }
    ).to_parquet(root / "video_info.parquet")
    pd.DataFrame(
        {
            "source_row": [0, 1, 2, 3, 4, 5],
            "conversation_id": ["part-1"] * 6,
            "speaker_id": [1, 1, 3, 2, 1, 1],
            "startTime": [1.0, 1.0, 4.0, 6.0, None, 8.5],
            "endTime": [2.0, 2.0, 5.0, 7.0, None, 9.0],
            "word": ["hi", ".", "third", "yes", "", "ok"],
        }
    ).to_parquet(root / "ground_truth.parquet")
    return _media(
        media_rows
        if media_rows is not None
        else [
            ("egocom", "EgoCom/vid-1.mp4", True, 0.0, 10.0),
            ("egocom", "EgoCom/vid-2.mp4", True, 0.0, 10.0),
        ]
    )


def _states(rows):
    return [(r.canonical_start_s, r.canonical_end_s, str(r.voice_state)) for r in rows]


# --- Ego4D ---------------------------------------------------------------


def test_ego4d_wearer_voice_segments_become_speaking_and_others_are_ignored(tmp_path):
    cfg = _config(tmp_path)
    media = _write_ego4d(cfg)

    source = load_ego4d_native_voice(cfg, media)
    [recording] = source.recordings
    rows = build_native_timeline(recording)

    assert recording.wearer_id == "0"
    assert recording.source_kind == SourceKind.EGO4D_VOICE_SEGMENTS
    # overlapping wearer segments [2,5] and [2.5,3] are one episode; person 1's
    # segment [8,9] leaves the wearer SILENT
    assert _states(rows) == [
        (0.0, 2.0, "SILENT"),
        (2.0, 2.5, "SPEAKING"),
        (2.5, 3.0, "SPEAKING"),
        (3.0, 5.0, "SPEAKING"),
        (5.0, 20.0, "SILENT"),
    ]
    assert (
        rows[2].source_annotation_id == "voice_segments_clean#0|voice_segments_clean#1"
    )
    assert source.statistics["clips_with_missing_voice_regions"] == 0


def test_ego4d_missing_voice_region_and_invalid_clip_are_unknown(tmp_path):
    cfg = _config(tmp_path)
    media = _write_ego4d(
        cfg, missing=[("clip-1", "0", 6.0, 7.0), ("clip-1", "1", 9.0, 9.5)]
    )
    [recording] = load_ego4d_native_voice(cfg, media).recordings
    rows = build_native_timeline(recording)
    assert (6.0, 7.0, "UNKNOWN") in _states(rows)
    assert (9.0, 9.5, "UNKNOWN") not in _states(rows)  # other person's gap

    media = _write_ego4d(cfg, valid=False)
    source = load_ego4d_native_voice(cfg, media)
    [recording] = source.recordings
    assert _states(build_native_timeline(recording)) == [(0.0, 20.0, "UNKNOWN")]
    assert source.statistics["invalid_clips"] == 1


def test_ego4d_media_not_covering_the_clip_is_unknown_at_the_edges(tmp_path):
    cfg = _config(tmp_path)
    # audio covers video time [12, 27] while the clip needs [10, 30]
    media = _write_ego4d(cfg, media_start=12.0, media_duration=15.0)
    [recording] = load_ego4d_native_voice(cfg, media).recordings
    states = _states(build_native_timeline(recording))

    assert states[0] == (0.0, 2.0, "UNKNOWN")
    assert states[-1] == (17.0, 20.0, "UNKNOWN")
    assert (5.0, 17.0, "SILENT") in states

    missing_media = load_ego4d_native_voice(cfg, _media([]))
    assert missing_media.statistics["clips_without_usable_media"] == 1
    assert _states(build_native_timeline(missing_media.recordings[0])) == [
        (0.0, 20.0, "UNKNOWN")
    ]


# --- EgoCom --------------------------------------------------------------


def test_egocom_wearer_words_are_speaking_and_other_speakers_leave_focal_silent(
    tmp_path,
):
    cfg = _config(tmp_path)
    media = _write_egocom(cfg)

    source = load_egocom_native_voice(cfg, media)
    by_view = {r.view_id: r for r in source.recordings}
    first = build_native_timeline(by_view["vid-1"])
    second = build_native_timeline(by_view["vid-2"])

    assert by_view["vid-1"].wearer_id == "1" and by_view["vid-2"].wearer_id == "2"
    assert by_view["vid-1"].sync_group_id == "part-1"
    # speaker 1: "hi" + "." share [1,2]; untimed row ignored; "ok" at [8.5,9]
    assert _states(first) == [
        (0.0, 1.0, "SILENT"),
        (1.0, 2.0, "SPEAKING"),
        (2.0, 8.5, "SILENT"),
        (8.5, 9.0, "SPEAKING"),
        (9.0, 10.0, "SILENT"),
    ]
    assert first[1].source_annotation_id == "ground_truth_clean#0|ground_truth_clean#1"
    # speaker 3 has no POV and speaker 1 talks: wearer 2 is SILENT there
    assert _states(second) == [
        (0.0, 6.0, "SILENT"),
        (6.0, 7.0, "SPEAKING"),
        (7.0, 10.0, "SILENT"),
    ]
    assert source.statistics["transcript_speakers_without_pov"] == 1
    assert all(r.source_kind == SourceKind.EGOCOM_TRANSCRIPT for r in first)


def test_egocom_invalid_or_partial_media_is_unknown(tmp_path):
    cfg = _config(tmp_path)
    media = _write_egocom(
        cfg,
        media_rows=[
            ("egocom", "EgoCom/vid-1.mp4", False, None, None),
            ("egocom", "EgoCom/vid-2.mp4", True, 0.5, 9.0),
        ],
    )

    source = load_egocom_native_voice(cfg, media)
    by_view = {r.view_id: r for r in source.recordings}

    assert _states(build_native_timeline(by_view["vid-1"])) == [(0.0, 10.0, "UNKNOWN")]
    states = _states(build_native_timeline(by_view["vid-2"]))
    assert states[0] == (0.0, 0.5, "UNKNOWN")
    assert states[-1] == (9.5, 10.0, "UNKNOWN")
    assert source.statistics["videos_without_usable_media"] == 1


# --- build ---------------------------------------------------------------


def _prepare_build(tmp_path):
    cfg = _config(tmp_path)
    media = pd.concat([_write_ego4d(cfg), _write_egocom(cfg)], ignore_index=True)
    reports = Path(cfg.paths.reports)
    (reports / "manifest").mkdir(parents=True)
    pd.DataFrame({"relative_path": ["x"]}).to_parquet(cfg.manifest.output)
    (reports / "temporal" / "media_metadata").mkdir(parents=True)
    media.to_parquet(reports / "temporal" / "media_metadata" / "media_metadata.parquet")
    return cfg


def test_build_writes_one_artifact_per_dataset_with_shared_schema_and_lineage(tmp_path):
    cfg = _prepare_build(tmp_path)

    outputs = run_native_focal_voice_state_build(cfg)

    assert (
        [o.dataset for o in outputs]
        == list(supported_datasets())
        == ["ego4d", "egocom"]
    )
    columns = list(NativeStateInterval.__dataclass_fields__)
    for output in outputs:
        assert list(output.timeline.columns) == columns
        assert output.timeline_path.exists() and output.report_path.exists()
        report = json.loads(output.report_path.read_text())
        assert report["output_artifacts"]["timeline"]["sha256"]
        assert report["input_artifacts"]["dataset_manifest"]["sha256"]
        assert report["input_artifacts"]["native_annotations"]
        assert report["native_state_schema_version"] == 1
        assert {"git_commit", "git_dirty", "command", "created_at"} <= set(report)
        assert report["source_dataset_version"]["annotation_schema_version"]
        statistics = report["statistics"]
        assert statistics["window_duration_s"] == pytest.approx(
            statistics["speaking_duration_s"]
            + statistics["silent_duration_s"]
            + statistics["unknown_duration_s"]
        )
    ego4d, egocom = outputs
    assert (
        ego4d.report["statistics"]["adapter"]["clips_with_missing_voice_regions"] == 0
    )
    assert "83.49" in " ".join(egocom.report["limitations"])


def test_build_is_deterministic_and_needs_no_acoustic_model(tmp_path):
    cfg = _prepare_build(tmp_path)

    first = run_native_focal_voice_state_build(cfg)
    second = run_native_focal_voice_state_build(cfg)

    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a.timeline, b.timeline)
        assert (
            a.report["output_artifacts"]["timeline"]["sha256"]
            == b.report["output_artifacts"]["timeline"]["sha256"]
        )
    loaded = set(sys.modules)
    assert not {
        name
        for name in loaded
        if name.split(".")[0] in {"torch", "silero_vad", "onnxruntime", "pyannote"}
    }


def test_build_selects_a_single_dataset(tmp_path):
    cfg = _prepare_build(tmp_path)

    [output] = run_native_focal_voice_state_build(cfg, dataset="egocom")

    assert output.dataset == "egocom"
    assert not (
        Path(cfg.paths.processed) / "native_focal_voice_state" / "ego4d"
    ).exists()
    with pytest.raises(ValueError):
        run_native_focal_voice_state_build(cfg, dataset="nope")
