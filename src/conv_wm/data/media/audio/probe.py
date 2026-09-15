"""ffprobe access for audio streams: packets, decoded frames and side data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conv_wm.data.media.ffprobe import probe_json


def probe_audio_packets(path: Path) -> list[dict[str, object]]:
    """Return every packet of the first audio stream without decoding.

    Each packet carries ``pts``, ``pts_time``, ``duration``, ``duration_time`` and,
    where present, ``Skip Samples`` side data.
    """
    payload = probe_json(
        path,
        [
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            (
                "packet=pts,pts_time,duration,duration_time:"
                "packet_side_data=side_data_type,skip_samples,discard_padding,"
                "skip_reason,discard_reason"
            ),
        ],
    )
    return list(payload.get("packets", []))


def probe_audio_frames(
    path: Path,
    *,
    read_intervals: str | None = None,
) -> pd.DataFrame:
    """Decode the first audio stream (optionally one interval) and return frames.

    Decoding is far slower than packet probing; use it for targeted validation of
    the decoded frame size, not for population scans.
    """
    command = ["-select_streams", "a:0"]
    if read_intervals is not None:
        command += ["-read_intervals", read_intervals]
    command += [
        "-show_frames",
        "-show_entries",
        "frame=pts,pts_time,duration,duration_time,nb_samples",
    ]
    return pd.DataFrame(probe_json(path, command).get("frames", []))


def extract_skip_samples(packets: list[dict[str, object]]) -> pd.DataFrame:
    """Return one row per packet carrying FFmpeg ``Skip Samples`` side data."""
    records: list[dict[str, object]] = []
    for packet_index, packet in enumerate(packets):
        side_data_list = packet.get("side_data_list", [])
        if not isinstance(side_data_list, list):
            continue
        for side_data in side_data_list:
            if side_data.get("side_data_type") != "Skip Samples":
                continue
            records.append(
                {
                    "packet_index": packet_index,
                    "pts": packet.get("pts"),
                    "pts_time": packet.get("pts_time"),
                    "skip_samples": int(side_data.get("skip_samples", 0)),
                    "discard_padding": int(side_data.get("discard_padding", 0)),
                    "skip_reason": side_data.get("skip_reason"),
                    "discard_reason": side_data.get("discard_reason"),
                }
            )
    return pd.DataFrame.from_records(records)
