"""Container/stream metadata of media files and the typed row boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd

from conv_wm.data.media.ffprobe import probe_media


@dataclass(frozen=True)
class MediaFileInfo:
    """Stream facts of one media file, read once from the media metadata table.

    Audits receive this object instead of a pandas row so that a renamed column
    fails loudly at the boundary (``from_row``) rather than deep in analysis.
    """

    dataset: str
    relative_path: str
    audio_codec: str
    audio_sample_rate_hz: int
    audio_time_base: str
    audio_duration_sec: float
    video_codec: str
    video_avg_frame_rate: float
    video_time_base: str
    video_duration_sec: float

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> MediaFileInfo:
        """Build from one row of the media metadata table (dict-like access)."""
        return cls(
            dataset=str(row["dataset"]),
            relative_path=str(row["relative_path"]),
            audio_codec=str(row["audio_codec"]),
            audio_sample_rate_hz=int(row["audio_sample_rate_hz"]),  # type: ignore[call-overload]
            audio_time_base=str(row["audio_time_base"]),
            audio_duration_sec=float(row["audio_duration_sec"]),  # type: ignore[arg-type]
            video_codec=str(row["video_codec"]),
            video_avg_frame_rate=float(row["video_avg_frame_rate"]),  # type: ignore[arg-type]
            video_time_base=str(row["video_time_base"]),
            video_duration_sec=float(row["video_duration_sec"]),  # type: ignore[arg-type]
        )


MEDIA_FILE_INFO_COLUMNS: tuple[str, ...] = (
    "dataset",
    "relative_path",
    "audio_codec",
    "audio_sample_rate_hz",
    "audio_time_base",
    "audio_duration_sec",
    "video_codec",
    "video_avg_frame_rate",
    "video_time_base",
    "video_duration_sec",
)


def media_files(metadata: pd.DataFrame) -> list[MediaFileInfo]:
    """Typed rows of a media metadata table; raises on missing columns."""
    missing = set(MEDIA_FILE_INFO_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"Media metadata table is missing columns: {sorted(missing)}")
    return [
        MediaFileInfo.from_row({str(key): value for key, value in row.items()})
        for row in metadata.to_dict(orient="records")
    ]


def extract_media_metadata(
    *,
    dataset: str,
    relative_path: str,
    path: Path,
) -> dict:
    """Probe one file and return its normalized metadata row plus stream counts."""
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
    """Add boolean ``qc_*`` columns: probe success, stream presence, positive durations."""
    qc = metadata.copy()
    qc["qc_probe_ok"] = qc["probe_ok"].fillna(False).astype(bool)
    for column in (
        "n_video_streams",
        "n_audio_streams",
        "video_duration_sec",
        "audio_duration_sec",
    ):
        if column not in qc.columns:
            qc[column] = float("nan")
    qc["qc_has_video"] = qc["n_video_streams"].fillna(0) >= 1
    qc["qc_has_audio"] = qc["n_audio_streams"].fillna(0) >= 1
    qc["qc_positive_video_duration"] = qc["video_duration_sec"].fillna(0) > 0
    qc["qc_positive_audio_duration"] = qc["audio_duration_sec"].fillna(0) > 0
    return qc


def parse_fraction(value: str | None) -> float | None:
    """Float value of an ffprobe rational such as ``"30000/1001"``; ``None`` if unset."""
    if value in (None, "0/0", "N/A"):
        return None

    return float(Fraction(value))


def get_stream(
    probe: dict,
    codec_type: str,
) -> dict | None:
    """First stream of ``codec_type`` (``"video"``/``"audio"``) in a probe payload."""
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
    """Flatten container, first-video and first-audio stream facts into one row."""
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
