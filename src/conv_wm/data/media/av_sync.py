"""Container-timeline audio/video offsets derived from stream metadata."""

from __future__ import annotations

import pandas as pd


def compute_av_sync_metadata(
    media_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Derive container-timeline A/V offsets from the media metadata table.

    Adds ``av_start_offset_sec`` (audio minus video start), stream end times,
    ``av_end_delta_sec`` and its absolute value. Duration differences are
    measurements, not perceptual synchronization failures.
    """
    required = {
        "audio_start_time_sec",
        "video_start_time_sec",
        "audio_duration_sec",
        "video_duration_sec",
    }

    missing = required - set(media_metadata.columns)

    if missing:
        raise ValueError(f"Missing required media metadata columns: {sorted(missing)}")

    result = media_metadata.copy()

    result["av_start_offset_sec"] = (
        result["audio_start_time_sec"] - result["video_start_time_sec"]
    )

    result["av_duration_delta_sec"] = (
        result["audio_duration_sec"] - result["video_duration_sec"]
    )

    result["audio_end_time_sec"] = (
        result["audio_start_time_sec"] + result["audio_duration_sec"]
    )

    result["video_end_time_sec"] = (
        result["video_start_time_sec"] + result["video_duration_sec"]
    )

    result["av_end_delta_sec"] = (
        result["audio_end_time_sec"] - result["video_end_time_sec"]
    )

    result["abs_av_end_delta_sec"] = result["av_end_delta_sec"].abs()

    return result
