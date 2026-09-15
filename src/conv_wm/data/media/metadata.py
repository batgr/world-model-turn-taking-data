from fractions import Fraction
from pathlib import Path

import pandas as pd

from conv_wm.data.media.ffprobe import probe_media


def extract_media_metadata(
    *,
    dataset: str,
    relative_path: str,
    path: Path,
) -> dict:
    probe = probe_media(path)

    record = normalize_probe(
        dataset=dataset,
        relative_path=relative_path,
        probe=probe,
    )

    streams = probe.get("streams", [])

    record.update(
        {
            "probe_ok": True,
            "n_streams": len(streams),
            "n_video_streams": sum(
                stream.get("codec_type") == "video" for stream in streams
            ),
            "n_audio_streams": sum(
                stream.get("codec_type") == "audio" for stream in streams
            ),
        }
    )

    return record


def derive_media_qc(metadata: pd.DataFrame) -> pd.DataFrame:
    qc = metadata.copy()

    qc["qc_probe_ok"] = qc["probe_ok"]

    qc["qc_has_video"] = qc["n_video_streams"].fillna(0) >= 1

    qc["qc_has_audio"] = qc["n_audio_streams"].fillna(0) >= 1

    qc["qc_positive_video_duration"] = qc["video_duration_sec"].fillna(0) > 0

    qc["qc_positive_audio_duration"] = qc["audio_duration_sec"].fillna(0) > 0

    return qc


def parse_fraction(value: str | None) -> float | None:
    if value in (None, "0/0", "N/A"):
        return None

    return float(Fraction(value))


def get_stream(
    probe: dict,
    codec_type: str,
) -> dict | None:
    return next(
        (
            stream
            for stream in probe["streams"]
            if stream.get("codec_type") == codec_type
        ),
        None,
    )


def normalize_probe(
    dataset: str,
    relative_path: str,
    probe: dict,
) -> dict:
    format_info = probe["format"]

    video = get_stream(probe, "video")
    audio = get_stream(probe, "audio")

    return {
        "dataset": dataset,
        "relative_path": relative_path,
        # Container
        "container_format": format_info.get("format_name"),
        "container_duration_sec": (
            float(format_info["duration"]) if format_info.get("duration") else None
        ),
        "container_start_time_sec": (
            float(format_info["start_time"]) if format_info.get("start_time") else None
        ),
        "container_size_bytes": (
            int(format_info["size"]) if format_info.get("size") else None
        ),
        "container_bit_rate": (
            int(format_info["bit_rate"]) if format_info.get("bit_rate") else None
        ),
        # Video
        "video_present": video is not None,
        "video_codec": video.get("codec_name") if video else None,
        "video_width": video.get("width") if video else None,
        "video_height": video.get("height") if video else None,
        "video_pixel_format": video.get("pix_fmt") if video else None,
        "video_r_frame_rate": (
            parse_fraction(video.get("r_frame_rate")) if video else None
        ),
        "video_avg_frame_rate": (
            parse_fraction(video.get("avg_frame_rate")) if video else None
        ),
        "video_time_base": video.get("time_base") if video else None,
        "video_start_pts": video.get("start_pts") if video else None,
        "video_start_time_sec": (
            float(video["start_time"]) if video and video.get("start_time") else None
        ),
        "video_duration_sec": (
            float(video["duration"]) if video and video.get("duration") else None
        ),
        "video_nb_frames": (
            int(video["nb_frames"]) if video and video.get("nb_frames") else None
        ),
        # Audio
        "audio_present": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_sample_rate_hz": (
            int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None
        ),
        "audio_channels": audio.get("channels") if audio else None,
        "audio_channel_layout": audio.get("channel_layout") if audio else None,
        "audio_time_base": audio.get("time_base") if audio else None,
        "audio_start_pts": audio.get("start_pts") if audio else None,
        "audio_start_time_sec": (
            float(audio["start_time"]) if audio and audio.get("start_time") else None
        ),
        "audio_duration_sec": (
            float(audio["duration"]) if audio and audio.get("duration") else None
        ),
    }
