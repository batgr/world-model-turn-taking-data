from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from conv_wm.data.vocal.sources import (
    load_ego4d_recordings,
    load_egocom_recordings,
)


def _config(tmp_path: Path):
    interim = tmp_path / "interim"
    return OmegaConf.create(
        {
            "paths": {
                "raw": str(tmp_path / "raw"),
                "interim": str(interim),
                "validated": str(tmp_path / "validated"),
                "processed": str(tmp_path / "processed"),
                "model_ready": str(tmp_path / "model_ready"),
                "reports": str(tmp_path / "reports"),
            },
            "datasets": {
                "ego4d": {
                    "interim": str(interim / "ego4d"),
                    "files": {
                        "clips_clean": "clips.parquet",
                        "persons_clean": "persons.parquet",
                        "voice_segments_clean": "voice.parquet",
                    },
                },
                "egocom": {
                    "interim": str(interim / "egocom"),
                    "files": {
                        "video_info_clean": "video_info.parquet",
                        "ground_truth_clean": "ground_truth.parquet",
                    },
                },
            },
        }
    )


def test_ego4d_adapter_keeps_clip_time_and_media_pts_mapping(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "interim" / "ego4d"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "clip_uid": ["clip"],
            "video_uid": ["video"],
            "video_start_sec": [-0.1],
            "video_end_sec": [9.9],
        }
    ).to_parquet(root / "clips.parquet")
    pd.DataFrame(
        {
            "clip_uid": ["clip", "clip"],
            "person_id": ["0", "1"],
            "is_camera_wearer": [True, False],
        }
    ).to_parquet(root / "persons.parquet")
    pd.DataFrame(
        {
            "clip_uid": ["clip", "clip"],
            "person_id": ["0", "1"],
            "start_time": [0.05, 2.0],
            "end_time": [1.0, 3.0],
        }
    ).to_parquet(root / "voice.parquet")
    media = pd.DataFrame(
        {
            "dataset": ["ego4d"],
            "relative_path": ["Ego4D/video.mp4"],
            "probe_ok": [True],
            "audio_start_time_sec": [0.0],
            "audio_duration_sec": [10.0],
        }
    )

    [recording] = load_ego4d_recordings(cfg, media).recordings

    assert recording.recording_id == "clip"
    assert recording.view_id == "video"
    assert recording.wearer_id == "0"
    assert recording.audio.start_s == 0.0
    assert recording.audio.canonical_offset_s == 0.1
    assert recording.canonical_start_s == 0.1
    assert recording.focal_annotation.tolist() == [[0.1, 1.0]]
    assert recording.other_annotation.tolist() == [[2.0, 3.0]]


def test_egocom_adapter_preserves_synchronized_views_and_wearers(tmp_path):
    cfg = _config(tmp_path)
    root = tmp_path / "interim" / "egocom"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "video_name": ["view-1", "view-2"],
            "conversation_id": ["group", "group"],
            "video_speaker_id": [1, 2],
            "duration_seconds": [10, 10],
        }
    ).to_parquet(root / "video_info.parquet")
    pd.DataFrame(
        {
            "conversation_id": ["group", "group"],
            "speaker_id": [1, 2],
            "startTime": [1.0, 3.0],
            "endTime": [2.0, 4.0],
        }
    ).to_parquet(root / "ground_truth.parquet")
    media = pd.DataFrame(
        {
            "dataset": ["egocom", "egocom"],
            "relative_path": ["EgoCom/view-1.mp4", "EgoCom/view-2.mp4"],
            "probe_ok": [True, True],
            "audio_start_time_sec": [0.0, 0.0],
            "audio_duration_sec": [9.5, 9.7],
        }
    )

    recordings = load_egocom_recordings(cfg, media).recordings

    assert [recording.view_id for recording in recordings] == ["view-1", "view-2"]
    assert [recording.wearer_id for recording in recordings] == ["1", "2"]
    assert {recording.sync_group_id for recording in recordings} == {"group"}
    assert [source.view_id for source in recordings[0].other_views] == ["view-2"]
    assert recordings[0].focal_annotation.tolist() == [[1.0, 2.0]]
    assert recordings[0].other_annotation.tolist() == [[3.0, 4.0]]
