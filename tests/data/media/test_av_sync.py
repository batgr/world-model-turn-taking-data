import pandas as pd

from conv_wm.data.media.av_sync import compute_av_sync_metadata


def test_identical_starts_have_zero_av_offset():
    metadata = pd.DataFrame(
        {
            "audio_start_time_sec": [0.0],
            "video_start_time_sec": [0.0],
            "audio_duration_sec": [10.0],
            "video_duration_sec": [10.0],
        }
    )

    result = compute_av_sync_metadata(metadata)

    assert result.loc[0, "av_start_offset_sec"] == 0.0
    assert result.loc[0, "av_end_delta_sec"] == 0.0


def test_shifted_audio_start_changes_start_and_end_offsets():
    metadata = pd.DataFrame(
        {
            "audio_start_time_sec": [0.25],
            "video_start_time_sec": [0.0],
            "audio_duration_sec": [10.0],
            "video_duration_sec": [10.0],
        }
    )

    result = compute_av_sync_metadata(metadata)

    assert result.loc[0, "av_start_offset_sec"] == 0.25
    assert result.loc[0, "av_end_delta_sec"] == 0.25


def test_duration_delta_retains_audio_minus_video_sign():
    metadata = pd.DataFrame(
        {
            "audio_start_time_sec": [0.0],
            "video_start_time_sec": [0.0],
            "audio_duration_sec": [9.5],
            "video_duration_sec": [10.0],
        }
    )

    result = compute_av_sync_metadata(metadata)

    assert result.loc[0, "av_duration_delta_sec"] == -0.5
    assert result.loc[0, "av_end_delta_sec"] == -0.5
    assert result.loc[0, "abs_av_end_delta_sec"] == 0.5
