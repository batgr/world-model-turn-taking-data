"""The media manifest build: adapters, strictness, staleness and the file check."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from state_layers import (
    build_action_grid,
    config_for,
    long_timeline,
    write_media_metadata,
    write_native_state,
    write_recording_splits,
)

from conv_wm import cli
from conv_wm.data import datasets
from conv_wm.data.audits.media_manifest import check_media_files
from conv_wm.data.datasets.spec import DatasetSpec
from conv_wm.data.media.media_manifest import (
    MediaFile,
    MediaManifestError,
    MediaRecord,
    MediaSource,
)
from conv_wm.data.pipeline.media_manifest import run_media_manifest_build
from conv_wm.data.pipeline.model_ready import run_model_ready_build


def _grid(cfg, dataset, recording="rec-1", timeline=None):
    write_native_state(cfg, dataset, timeline or long_timeline(), recording=recording)
    build_action_grid(cfg, dataset)


def test_egocom_recordings_resolve_to_the_video_named_by_their_id(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "egocom", recording="vid_001__day_1__con_1__person_1")
    write_media_metadata(
        cfg,
        [
            ("egocom", "EgoCom/240p/20min/vid_001__day_1__con_1__person_1.MP4", 1),
            ("egocom", "EgoCom/240p/20min/vid_002__day_1__con_1__person_2.MP4", 1),
        ],
    )

    [output] = run_media_manifest_build(cfg, dataset="egocom")
    row = output.table.iloc[0]

    assert output.passed
    assert len(output.table) == 1
    assert row["recording_id"] == "vid_001__day_1__con_1__person_1"
    assert row["video_path"] == "240p/20min/vid_001__day_1__con_1__person_1.MP4"
    assert pd.isna(row["audio_path"])
    assert bool(row["video_has_audio"]) is True
    assert row["media_offset_s"] == 0.0
    assert output.report["corpus_directory"] == "EgoCom"
    assert (
        output.table_path
        == Path(cfg.paths.model_ready) / "egocom" / "media_manifest.parquet"
    )


def test_ego4d_clips_resolve_to_their_source_video_with_the_clip_offset(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "ego4d", recording="clip-a")
    clips = Path(cfg.datasets.ego4d.interim) / "clips_clean.parquet"
    clips.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "clip_uid": ["clip-a", "clip-b"],
            "video_uid": ["video-9", "video-9"],
            # Full-scale timeline, one frame early: must not be used.
            "video_start_sec": [30.4987, 330.4987],
            "video_start_frame": [915, 9915],
            "clip_start_sec": [0, 0],
            "clip_start_frame": [0, 0],
        }
    ).to_parquet(clips, index=False)
    write_media_metadata(cfg, [("ego4d", "Ego4D/v2/video_540ss/video-9.mp4", 1)])

    [output] = run_media_manifest_build(cfg, dataset="ego4d")
    row = output.table.iloc[0]

    assert output.passed
    assert row["recording_id"] == "clip-a"
    assert row["video_path"] == "v2/video_540ss/video-9.mp4"
    assert pd.isna(row["audio_path"])
    assert row["media_offset_s"] == pytest.approx(30.5)
    assert output.report["input_artifacts"]["adapter_tables"][0]["sha256"]


def test_a_labelled_recording_without_media_fails_the_build(tmp_path, capsys):
    cfg = config_for(tmp_path)
    _grid(cfg, "egocom", recording="rec-1")
    write_media_metadata(cfg, [("egocom", "EgoCom/240p/other.MP4", 1)])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg))

    code = cli.main(
        ["--config", str(config_path), "build", "media-manifest", "--dataset", "egocom"]
    )

    assert code == cli.EXIT_AUDIT_FAILED
    report = json.loads(
        (
            Path(cfg.paths.reports) / "media_manifest" / "egocom" / "report.json"
        ).read_text()
    )
    assert report["status"] == "FAIL"
    assert report["coverage"]["unresolved_recordings"] == {"rec-1": "media_missing"}
    assert report["coverage"]["unresolved_with_valid_slots"] == ["rec-1"]
    assert "labelled recordings without media" in capsys.readouterr().out


def test_a_fully_masked_recording_without_media_is_reported_not_fatal(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "egocom", recording="rec-1", timeline=[(0.0, 8.0, "UNKNOWN")])
    write_media_metadata(cfg, [])

    [output] = run_media_manifest_build(cfg, dataset="egocom")

    assert output.passed
    assert output.table.empty
    assert output.report["coverage"]["recordings_without_resolved_media"] == 1


def test_model_ready_references_the_manifest_and_refuses_a_stale_one(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "egocom", recording="rec-1")
    write_recording_splits(cfg, "egocom", {"rec-1": "train"})
    write_media_metadata(cfg, [("egocom", "EgoCom/240p/rec-1.MP4", 1)])
    run_media_manifest_build(cfg, dataset="egocom")

    [output] = run_model_ready_build(cfg, dataset="egocom")
    metadata = json.loads(output.metadata_path.read_text())
    assert metadata["files"]["media_manifest"] == "media_manifest.parquet"
    assert metadata["media"]["coverage"]["recordings_in_media_manifest"] == 1
    assert output.report["input_artifacts"]["media_manifest"]["sha256"]

    _grid(cfg, "egocom", recording="rec-1", timeline=long_timeline(cycles=10))
    with pytest.raises(ValueError, match="another action grid"):
        run_model_ready_build(cfg, dataset="egocom")


def test_the_file_check_resolves_paths_below_the_local_corpus_root(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "egocom", recording="rec-1")
    write_media_metadata(cfg, [("egocom", "EgoCom/240p/rec-1.MP4", 1)])
    [build] = run_media_manifest_build(cfg, dataset="egocom")

    missing = check_media_files(cfg, "egocom")
    assert not missing.passed
    assert missing.summary["missing_examples"] == ["240p/rec-1.MP4"]

    media_file = Path(cfg.paths.raw) / "EgoCom" / "240p" / "rec-1.MP4"
    media_file.parent.mkdir(parents=True)
    media_file.write_bytes(b"")
    present = check_media_files(cfg, "egocom")
    assert present.passed
    assert present.summary["paths_checked"] == 1
    # the check never writes the resolved location back into the manifest
    assert pd.read_parquet(build.table_path).equals(build.table)
    assert str(tmp_path) not in json.dumps(present.summary["missing_examples"])


def _moodlab_media(cfg, media, recording_ids):
    """A third corpus with its own layout: cameras and separate microphones."""
    del cfg, media
    return MediaSource(
        dataset="moodlab",
        records=tuple(
            MediaRecord(
                rid,
                MediaFile(f"MoodLab/cams/{rid}_cam.mp4", has_audio_stream=False),
                MediaFile(f"MoodLab/mics/{rid}.flac", has_audio_stream=None),
                media_offset_s=1.5,
            )
            for rid in recording_ids
        ),
    )


@pytest.fixture
def moodlab():
    spec = datasets.register(
        DatasetSpec(
            name="moodlab",
            native_focal_voice=lambda cfg, media: None,  # type: ignore[arg-type,return-value]
            media_records=_moodlab_media,
        )
    )
    try:
        yield spec
    finally:
        datasets.unregister("moodlab")


def test_a_third_dataset_provides_media_through_the_generic_contract(tmp_path, moodlab):
    cfg = config_for(tmp_path)
    _grid(cfg, "moodlab", recording="s1")
    write_media_metadata(cfg, [])

    [output] = run_media_manifest_build(cfg, dataset="moodlab")
    row = output.table.iloc[0]

    assert output.passed
    assert (row["video_path"], row["audio_path"]) == ("cams/s1_cam.mp4", "mics/s1.flac")
    assert output.report["coverage"]["both"] == 1
    assert output.report["corpus_directory"] == "MoodLab"


def test_an_adapter_inventing_a_recording_is_rejected(tmp_path):
    def inventing(cfg, media, recording_ids):
        return _moodlab_media(cfg, media, [*recording_ids, "ghost"])

    datasets.register(
        DatasetSpec(
            name="moodlab",
            native_focal_voice=lambda cfg, media: None,  # type: ignore[arg-type,return-value]
            media_records=inventing,
        )
    )
    try:
        cfg = config_for(tmp_path)
        _grid(cfg, "moodlab", recording="s1")
        write_media_metadata(cfg, [])
        with pytest.raises(MediaManifestError, match="moodlab/ghost"):
            run_media_manifest_build(cfg, dataset="moodlab")
    finally:
        datasets.unregister("moodlab")


def test_ego4d_offset_comes_from_frame_indices_not_the_release_seconds(tmp_path):
    # Real Ego4D clip 108cebaa...: its first frame is video frame 0, but the
    # release's video_start_sec (full-scale timeline) is -0.0013 s.
    cfg = config_for(tmp_path)
    _grid(cfg, "ego4d", recording="clip-a")
    clips = Path(cfg.datasets.ego4d.interim) / "clips_clean.parquet"
    clips.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "clip_uid": ["clip-a"],
            "video_uid": ["video-9"],
            "video_start_sec": [-0.0013021333333333301],
            "video_start_frame": [0],
            "clip_start_sec": [0],
            "clip_start_frame": [0],
        }
    ).to_parquet(clips, index=False)
    write_media_metadata(cfg, [("ego4d", "Ego4D/v2/video_540ss/video-9.mp4", 1)])

    [output] = run_media_manifest_build(cfg, dataset="ego4d")

    assert output.table.iloc[0]["media_offset_s"] == 0.0
