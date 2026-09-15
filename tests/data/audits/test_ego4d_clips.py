import pandas as pd
import pandera.errors as pa_errors
import pytest

from conv_wm.data.datasets.ego4d import EGO4D_CLIPS_SCHEMA


def make_valid_clips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "val"],
            "clip_uid": ["clip-1", "clip-2"],
            "source_clip_uid": ["source-1", "source-2"],
            "video_uid": ["video-1", "video-1"],
            "video_start_sec": [10.0, 20.0],
            "video_end_sec": [310.0, 320.0],
            "video_start_frame": [300, 600],
            "video_end_frame": [9300, 9600],
            "clip_start_sec": [0, 0],
            "clip_end_sec": [300.0, 300.0],
            "clip_start_frame": [0, 0],
            "clip_end_frame": [9000, 9000],
        }
    )


def test_ego4d_clips_schema_accepts_valid_table():
    df = make_valid_clips()

    validated = EGO4D_CLIPS_SCHEMA.validate(df, lazy=True)

    assert len(validated) == 2


def test_ego4d_clips_schema_rejects_duplicate_clip_uid():
    df = make_valid_clips()
    df.loc[1, "clip_uid"] = "clip-1"

    with pytest.raises(pa_errors.SchemaErrors):
        EGO4D_CLIPS_SCHEMA.validate(df, lazy=True)


def test_ego4d_clips_schema_rejects_unknown_split():
    df = make_valid_clips()
    df.loc[0, "split"] = "unknown"

    with pytest.raises(pa_errors.SchemaErrors):
        EGO4D_CLIPS_SCHEMA.validate(df, lazy=True)


def test_ego4d_clips_schema_does_not_require_unique_source_clip_uid():
    df = make_valid_clips()
    df.loc[1, "source_clip_uid"] = "source-1"

    validated = EGO4D_CLIPS_SCHEMA.validate(df, lazy=True)

    assert len(validated) == 2
