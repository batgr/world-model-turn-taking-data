"""Video timeline measurements from container packet timestamps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from conv_wm.data.media.ffprobe import probe_json


class VideoTimelineClass(StrEnum):
    """Verdict of one analysed window or file."""

    CFR_CONSISTENT = "cfr_consistent"
    """Every PTS step matches ``1 / fps`` within one time-base tick."""
    VFR_OR_IRREGULAR = "vfr_or_irregular"
    INSUFFICIENT_DATA = "insufficient_data"
    PROBE_ERROR = "probe_error"


def probe_video_timestamps(
    path: Path,
    *,
    read_intervals: str | None = None,
) -> pd.DataFrame:
    """Presentation timestamps of the first video stream, sorted by ``pts_time``.

    Packets are read without decoding. They arrive in decode order, so the
    result is sorted by presentation time; packets without a PTS are kept last
    and counted as untimestamped by :func:`analyze_frame_timeline`.
    """
    command = ["-select_streams", "v:0"]
    if read_intervals is not None:
        command += ["-read_intervals", read_intervals]
    command += ["-show_packets", "-show_entries", "packet=pts,pts_time,dts,dts_time"]
    packets = pd.DataFrame(probe_json(path, command).get("packets", []))
    if "pts_time" not in packets.columns:
        return packets
    packets["pts_time"] = pd.to_numeric(packets["pts_time"], errors="coerce")
    return packets.sort_values(
        "pts_time", kind="stable", na_position="last"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class VideoTimelineMetrics:
    """Cadence of presentation timestamps against the nominal frame rate.

    Deltas are consecutive PTS differences in seconds; ``max_delta_error_ticks``
    is the largest deviation from ``1 / fps`` expressed in time-base ticks, and
    ``max_abs_cumulative_drift_sec`` compares each timestamp with
    ``first + index / fps``.
    """

    timeline_class: VideoTimelineClass
    n_frames: int
    n_timestamped: int
    timestamp_coverage: float
    first_timestamp_sec: float | None = None
    last_timestamp_sec: float | None = None
    monotonic: bool = False
    n_duplicate_timestamps: int = 0
    n_non_positive_deltas: int = 0
    expected_delta_sec: float | None = None
    delta_min_sec: float | None = None
    delta_median_sec: float | None = None
    delta_max_sec: float | None = None
    delta_std_sec: float | None = None
    max_delta_error_sec: float | None = None
    max_delta_error_ticks: float | None = None
    max_abs_cumulative_drift_sec: float | None = None

    def to_row(self) -> dict[str, object]:
        """Flatten into the Parquet column layout."""
        row = asdict(self)
        row["timeline_class"] = str(self.timeline_class)
        return row


def analyze_frame_timeline(
    frames: pd.DataFrame,
    *,
    fps: float,
    time_base: str,
) -> VideoTimelineMetrics:
    """Measure whether sorted presentation timestamps follow a constant frame rate."""
    timestamps = pd.to_numeric(
        frames.get("pts_time", pd.Series(dtype=float)), errors="coerce"
    )
    valid = timestamps.dropna()
    coverage = float(len(valid) / len(frames)) if len(frames) else 0.0
    if len(valid) < 2:
        return VideoTimelineMetrics(
            timeline_class=VideoTimelineClass.INSUFFICIENT_DATA,
            n_frames=len(frames),
            n_timestamped=len(valid),
            timestamp_coverage=coverage,
        )
    deltas = valid.diff().dropna()
    expected_delta = 1.0 / fps
    time_base_sec = float(Fraction(time_base))
    expected_timestamps = np.arange(len(valid), dtype=float) / fps
    cumulative_drift = (valid - valid.iloc[0]).to_numpy() - expected_timestamps
    delta_error = deltas - expected_delta
    delta_error_ticks = delta_error.abs() / time_base_sec
    n_non_positive = int((deltas <= 0).sum())
    monotonic = bool(valid.is_monotonic_increasing)
    duplicates = int(valid.duplicated().sum())
    consistent = (
        n_non_positive == 0
        and monotonic
        and duplicates == 0
        and float(delta_error_ticks.max()) <= 1.0 + 1e-9
    )
    return VideoTimelineMetrics(
        timeline_class=(
            VideoTimelineClass.CFR_CONSISTENT
            if consistent
            else VideoTimelineClass.VFR_OR_IRREGULAR
        ),
        n_frames=len(frames),
        n_timestamped=len(valid),
        timestamp_coverage=coverage,
        first_timestamp_sec=float(valid.iloc[0]),
        last_timestamp_sec=float(valid.iloc[-1]),
        monotonic=monotonic,
        n_duplicate_timestamps=duplicates,
        n_non_positive_deltas=n_non_positive,
        expected_delta_sec=expected_delta,
        delta_min_sec=float(deltas.min()),
        delta_median_sec=float(deltas.median()),
        delta_max_sec=float(deltas.max()),
        delta_std_sec=float(deltas.std()),
        max_delta_error_sec=float(delta_error.abs().max()),
        max_delta_error_ticks=float(delta_error_ticks.max()),
        max_abs_cumulative_drift_sec=float(np.abs(cumulative_drift).max()),
    )


def make_timeline_windows(
    duration_sec: float,
    *,
    window_sec: float = 10.0,
    known_boundaries_sec: tuple[float, ...] = (),
) -> dict[str, str]:
    """ffprobe ``-read_intervals`` specs for the start, middle and end of a stream.

    ``known_boundaries_sec`` adds one window centred on each declared boundary
    that falls inside the stream, named ``boundary_<k>``; sampling only the
    start/middle/end of a stream would otherwise miss deterministic joins.
    """
    if duration_sec <= window_sec:
        return {"full_short": f"%+{duration_sec:.6f}"}
    windows = {
        "start": f"%+{window_sec:.6f}",
        "middle": f"{max(0.0, duration_sec / 2.0 - window_sec / 2.0):.6f}%+{window_sec:.6f}",
        "end": f"{max(0.0, duration_sec - window_sec):.6f}%+{window_sec:.6f}",
    }
    for index, boundary in enumerate(sorted(known_boundaries_sec), start=1):
        if 0.0 < boundary < duration_sec:
            start = max(0.0, boundary - window_sec / 2.0)
            windows[f"boundary_{index}"] = f"{start:.6f}%+{window_sec:.6f}"
    return windows


def known_boundaries_within(
    duration_sec: float, *, period_sec: float | None
) -> tuple[float, ...]:
    """Multiples of ``period_sec`` strictly inside ``(0, duration_sec)``."""
    if period_sec is None or period_sec <= 0:
        return ()
    count = int(duration_sec // period_sec) + 1
    return tuple(
        k * period_sec for k in range(1, count) if k * period_sec < duration_sec
    )


@dataclass(frozen=True)
class VideoWindowRecord:
    """Result of one probed window (one Parquet row)."""

    dataset: str
    relative_path: str
    window: str
    requested_interval: str
    probe_ok: bool
    probe_error: str | None
    metrics: VideoTimelineMetrics | None

    def to_row(self) -> dict[str, object]:
        """Flatten into the Parquet column layout."""
        row: dict[str, object] = {
            "dataset": self.dataset,
            "relative_path": self.relative_path,
            "window": self.window,
            "requested_interval": self.requested_interval,
            "probe_ok": self.probe_ok,
            "probe_error": self.probe_error,
        }
        if self.metrics is not None:
            row.update(self.metrics.to_row())
        else:
            row["timeline_class"] = str(VideoTimelineClass.PROBE_ERROR)
        return row


def audit_video_windows(
    *,
    dataset: str,
    relative_path: str,
    path: Path,
    duration_sec: float,
    fps: float,
    time_base: str,
    window_sec: float = 10.0,
    known_boundaries_sec: tuple[float, ...] = (),
) -> list[VideoWindowRecord]:
    """Probe and analyse each sampled window of one video stream."""
    records: list[VideoWindowRecord] = []
    windows = make_timeline_windows(
        duration_sec, window_sec=window_sec, known_boundaries_sec=known_boundaries_sec
    )
    for window_name, interval in windows.items():
        try:
            metrics = analyze_frame_timeline(
                probe_video_timestamps(path, read_intervals=interval),
                fps=fps,
                time_base=time_base,
            )
        except Exception as exc:  # noqa: BLE001 - one bad window must not abort the audit
            records.append(
                VideoWindowRecord(
                    dataset, relative_path, window_name, interval, False, str(exc), None
                )
            )
            continue
        records.append(
            VideoWindowRecord(
                dataset, relative_path, window_name, interval, True, None, metrics
            )
        )
    return records
