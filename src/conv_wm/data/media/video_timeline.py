from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from conv_wm.data.media.ffprobe import probe_json


def _fraction_to_float(value: str | float) -> float:
    if isinstance(value, str):
        return float(Fraction(value))

    return float(value)


def probe_video_timestamps(
    path: Path,
    *,
    read_intervals: str | None = None,
) -> pd.DataFrame:
    """Extract presentation timestamps for the first video stream.

    Packets are read from the container without decoding, which is much faster
    than ``-show_frames``. Packets come out in decode order, so the result is
    sorted by presentation timestamp; packets without a pts are kept at the end
    and counted as untimestamped by ``analyze_frame_timeline``.
    """

    command = [
        "-select_streams",
        "v:0",
    ]

    if read_intervals is not None:
        command += [
            "-read_intervals",
            read_intervals,
        ]

    command += [
        "-show_packets",
        "-show_entries",
        "packet=pts,pts_time,dts,dts_time",
    ]
    payload = probe_json(path, command)

    packets = pd.DataFrame(payload.get("packets", []))

    if "pts_time" not in packets.columns:
        return packets

    packets["pts_time"] = pd.to_numeric(packets["pts_time"], errors="coerce")

    return packets.sort_values(
        "pts_time",
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def analyze_frame_timeline(
    frames: pd.DataFrame,
    *,
    fps: float,
    time_base: str,
) -> dict[str, object]:
    """Analyze video presentation timestamps sorted in presentation order."""

    timestamps = pd.to_numeric(
        frames.get("pts_time", pd.Series(dtype=float)),
        errors="coerce",
    )

    valid = timestamps.dropna()

    base_result: dict[str, object] = {
        "n_frames": len(frames),
        "n_timestamped": len(valid),
        "timestamp_coverage": (float(len(valid) / len(frames)) if len(frames) else 0.0),
        "first_timestamp_sec": None,
        "last_timestamp_sec": None,
        "monotonic": False,
        "n_duplicate_timestamps": 0,
        "n_non_positive_deltas": 0,
        "expected_delta_sec": None,
        "delta_min_sec": None,
        "delta_median_sec": None,
        "delta_max_sec": None,
        "delta_std_sec": None,
        "max_delta_error_sec": None,
        "max_delta_error_ticks": None,
        "max_abs_cumulative_drift_sec": None,
    }

    if len(valid) < 2:
        return {
            **base_result,
            "timeline_class": "insufficient_data",
        }

    deltas = valid.diff().dropna()

    expected_delta = 1.0 / fps
    time_base_sec = _fraction_to_float(time_base)

    # Work relative to the first observed presentation timestamp.
    relative_timestamps = valid - valid.iloc[0]

    frame_indices = np.arange(len(valid), dtype=float)
    expected_timestamps = frame_indices / fps

    cumulative_drift = relative_timestamps.to_numpy() - expected_timestamps

    delta_error = deltas - expected_delta
    delta_error_ticks = delta_error.abs() / time_base_sec

    n_non_positive = int((deltas <= 0).sum())

    cfr_consistent = (
        n_non_positive == 0
        and bool(valid.is_monotonic_increasing)
        and int(valid.duplicated().sum()) == 0
        and float(delta_error_ticks.max()) <= 1.0 + 1e-9
    )

    return {
        **base_result,
        "timeline_class": ("cfr_consistent" if cfr_consistent else "vfr_or_irregular"),
        "first_timestamp_sec": float(valid.iloc[0]),
        "last_timestamp_sec": float(valid.iloc[-1]),
        "monotonic": bool(valid.is_monotonic_increasing),
        "n_duplicate_timestamps": int(valid.duplicated().sum()),
        "n_non_positive_deltas": n_non_positive,
        "expected_delta_sec": expected_delta,
        "delta_min_sec": float(deltas.min()),
        "delta_median_sec": float(deltas.median()),
        "delta_max_sec": float(deltas.max()),
        "delta_std_sec": float(deltas.std()),
        "max_delta_error_sec": float(delta_error.abs().max()),
        "max_delta_error_ticks": float(delta_error_ticks.max()),
        "max_abs_cumulative_drift_sec": float(np.abs(cumulative_drift).max()),
    }


def make_timeline_windows(
    duration_sec: float,
    *,
    window_sec: float = 10.0,
) -> dict[str, str]:
    """Build start/middle/end ffprobe intervals."""

    if duration_sec <= window_sec:
        return {
            "full_short": f"%+{duration_sec:.6f}",
        }

    middle_start = max(
        0.0,
        duration_sec / 2.0 - window_sec / 2.0,
    )

    end_start = max(
        0.0,
        duration_sec - window_sec,
    )

    return {
        "start": f"%+{window_sec:.6f}",
        "middle": (f"{middle_start:.6f}%+{window_sec:.6f}"),
        "end": (f"{end_start:.6f}%+{window_sec:.6f}"),
    }


def audit_sampled_video_timeline(
    *,
    dataset: str,
    relative_path: str,
    path: Path,
    duration_sec: float,
    fps: float,
    time_base: str,
    window_sec: float = 10.0,
) -> list[dict[str, object]]:
    """Audit several temporal regions of one video."""

    records: list[dict[str, object]] = []

    windows = make_timeline_windows(
        duration_sec,
        window_sec=window_sec,
    )

    for window_name, interval in windows.items():
        try:
            frames = probe_video_timestamps(
                path,
                read_intervals=interval,
            )

            analysis = analyze_frame_timeline(
                frames,
                fps=fps,
                time_base=time_base,
            )

            records.append(
                {
                    "dataset": dataset,
                    "relative_path": relative_path,
                    "window": window_name,
                    "requested_interval": interval,
                    "probe_ok": True,
                    "probe_error": None,
                    **analysis,
                }
            )

        # One bad file must not abort the audit: record the failure and go on.
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "dataset": dataset,
                    "relative_path": relative_path,
                    "window": window_name,
                    "requested_interval": interval,
                    "probe_ok": False,
                    "probe_error": str(exc),
                    "timeline_class": "probe_error",
                }
            )

    return records
